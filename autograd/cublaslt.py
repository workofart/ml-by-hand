"""Minimal ctypes binding for cuBLASLt matmul with BIAS / RELU_BIAS epilogue.

Targets the bf16 GPT-2 path on CUDA 13. cuBLAS's classic GEMM path requires a
separate bias-add kernel after the matmul, which doubles the wall time for
small-K matmuls (raw matmul ~451us vs matmul+bias ~810us on Q/K/V/O). The
cuBLASLt EPILOGUE_BIAS/RELU_BIAS fuses the bias (and optional ReLU) into the
same kernel, recovering ~355us per Q/K/V/O call and ~1340us per FC1 call.

Lazy singleton (`get_lt()`); falls back to None if libcublasLt.so or CUDA
runtime isn't available, which keeps the CPU backend tests untouched.

For shape layout: cuBLASLt is column-major. To compute row-major
`D = A @ B + bias` with our row-major `A:(M,K)`, `B:(K,N)`, `bias:(N,)`,
we flip the cuBLAS args so cuBLAS computes `D^T(N,M) = B^T(N,K) * A^T(K,M)`
using transA = transB = N on the raw buffers (the transpose is implicit in
how cuBLAS reinterprets row-major as column-major).
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- cuBLASLt enums (CUDA 13) ----
CUBLAS_COMPUTE_32F = 68
CUDA_R_16BF = 14
CUDA_R_32F = 0

CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_DESC_TRANSB = 4
CUBLASLT_MATMUL_DESC_EPILOGUE = 7
CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8
CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

CUBLASLT_EPILOGUE_DEFAULT = 1
CUBLASLT_EPILOGUE_RELU = 2
CUBLASLT_EPILOGUE_BIAS = 4
CUBLASLT_EPILOGUE_RELU_BIAS = 6  # RELU | BIAS
CUBLASLT_EPILOGUE_BGRADA = 256  # bias gradient w.r.t. A side
CUBLASLT_EPILOGUE_BGRADB = 512  # bias gradient w.r.t. B side

CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1


class _HeurResult(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_uint64 * 8),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _try_load_cublaslt() -> Optional[ctypes.CDLL]:
    for libname in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            return ctypes.CDLL(libname)
        except OSError:
            continue
    return None


def _bind(lib: ctypes.CDLL) -> None:
    handle_t = ctypes.c_void_p
    desc_t = ctypes.c_void_p
    layout_t = ctypes.c_void_p
    pref_t = ctypes.c_void_p

    lib.cublasLtCreate.argtypes = [ctypes.POINTER(handle_t)]
    lib.cublasLtCreate.restype = ctypes.c_int

    lib.cublasLtMatmulDescCreate.argtypes = [
        ctypes.POINTER(desc_t),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cublasLtMatmulDescCreate.restype = ctypes.c_int

    lib.cublasLtMatmulDescSetAttribute.argtypes = [
        desc_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.cublasLtMatmulDescSetAttribute.restype = ctypes.c_int

    lib.cublasLtMatrixLayoutCreate.argtypes = [
        ctypes.POINTER(layout_t),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    lib.cublasLtMatrixLayoutCreate.restype = ctypes.c_int

    lib.cublasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(pref_t)]
    lib.cublasLtMatmulPreferenceCreate.restype = ctypes.c_int

    lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [
        pref_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.cublasLtMatmulPreferenceSetAttribute.restype = ctypes.c_int

    lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [
        handle_t,
        desc_t,
        layout_t,
        layout_t,
        layout_t,
        layout_t,
        pref_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.cublasLtMatmulAlgoGetHeuristic.restype = ctypes.c_int

    lib.cublasLtMatmul.argtypes = [
        handle_t,
        desc_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        layout_t,
        ctypes.c_void_p,
        layout_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        layout_t,
        ctypes.c_void_p,
        layout_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.cublasLtMatmul.restype = ctypes.c_int


def _check(status: int, where: str) -> None:
    if status != 0:
        raise RuntimeError(f"cuBLASLt error at {where}: status={status}")


class LtMatmulBias:
    """Compute row-major D = A @ B + bias (optionally ReLU'd) on bf16.

    Plans are cached per (M, N, K, with_relu); the bias pointer is reset
    on every call so we can pass arbitrary tensors without rebuilding.
    """

    _WORKSPACE_BYTES = 64 * 1024 * 1024

    def __init__(self, lib: ctypes.CDLL):
        import cupy as cp  # local import; module must be CuPy-only

        self._cp = cp
        self._lib = lib
        h = ctypes.c_void_p()
        _check(lib.cublasLtCreate(ctypes.byref(h)), "create")
        self._handle = h
        self._workspace = cp.empty((self._WORKSPACE_BYTES,), dtype=cp.uint8)
        self._cache: Dict[Any, Tuple[Any, ...]] = {}

    def _build_plan(
        self,
        M: int,
        N: int,
        K: int,
        with_relu: bool,
        with_bias: bool,
    ):
        lib = self._lib
        desc = ctypes.c_void_p()
        _check(
            lib.cublasLtMatmulDescCreate(
                ctypes.byref(desc), CUBLAS_COMPUTE_32F, CUDA_R_32F
            ),
            "descCreate",
        )

        op_n = ctypes.c_int32(CUBLAS_OP_N)
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                ctypes.byref(op_n),
                ctypes.sizeof(op_n),
            ),
            "transA",
        )
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                ctypes.byref(op_n),
                ctypes.sizeof(op_n),
            ),
            "transB",
        )

        if with_bias:
            ep_val = (
                CUBLASLT_EPILOGUE_RELU_BIAS if with_relu else CUBLASLT_EPILOGUE_BIAS
            )
        else:
            ep_val = CUBLASLT_EPILOGUE_RELU if with_relu else CUBLASLT_EPILOGUE_DEFAULT
        ep = ctypes.c_uint32(ep_val)
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc, CUBLASLT_MATMUL_DESC_EPILOGUE, ctypes.byref(ep), ctypes.sizeof(ep)
            ),
            "epilogue",
        )

        if with_bias:
            bias_dt = ctypes.c_int32(CUDA_R_16BF)
            _check(
                lib.cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    ctypes.byref(bias_dt),
                    ctypes.sizeof(bias_dt),
                ),
                "biasDtype",
            )

        # Layouts: row-major (M,N) computed via col-major (N,M) = (N,K)*(K,M).
        # bf16 inputs, bf16 output; the fp32 accumulator lives inside the
        # CUBLAS_COMPUTE_32F descriptor (PyTorch-autocast pattern).
        ay = ctypes.c_void_p()  # cuBLAS A := our B, shape (N,K) col-major ld=N
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(ay), CUDA_R_16BF, N, K, N),
            "AlayoutCreate",
        )
        by = ctypes.c_void_p()  # cuBLAS B := our A, shape (K,M) col-major ld=K
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(by), CUDA_R_16BF, K, M, K),
            "BlayoutCreate",
        )
        cy = (
            ctypes.c_void_p()
        )  # cuBLAS C: (N,M) col-major ld=N (== our D row-major (M,N))
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(cy), CUDA_R_16BF, N, M, N),
            "ClayoutCreate",
        )
        dy = ctypes.c_void_p()
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(dy), CUDA_R_16BF, N, M, N),
            "DlayoutCreate",
        )

        pref = ctypes.c_void_p()
        _check(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "prefCreate")
        ws = ctypes.c_size_t(self._WORKSPACE_BYTES)
        _check(
            lib.cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(ws),
                ctypes.sizeof(ws),
            ),
            "prefSetWs",
        )

        heur = (_HeurResult * 1)()
        returned = ctypes.c_int(0)
        _check(
            lib.cublasLtMatmulAlgoGetHeuristic(
                self._handle,
                desc,
                ay,
                by,
                cy,
                dy,
                pref,
                1,
                heur,
                ctypes.byref(returned),
            ),
            "heuristic",
        )
        if returned.value == 0:
            raise RuntimeError(
                f"cuBLASLt no algo for shape ({M},{N},{K}) relu={with_relu}"
            )
        return desc, ay, by, cy, dy, heur

    def matmul_bias(
        self,
        A,
        B,
        bias=None,
        with_relu: bool = False,
    ):
        """A:(M,K) row-major bf16, B:(K,N) row-major bf16 → D:(M,N) bf16.

        bias:(N,) optional bf16; added via the cuBLASLt BIAS epilogue. Uses
        the CUBLAS_COMPUTE_32F descriptor so the accumulator runs in fp32
        (PyTorch-autocast pattern) while inputs/outputs stay bf16.
        """
        cp = self._cp
        M, K = int(A.shape[0]), int(A.shape[1])
        K2, N = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise ValueError("matmul shape mismatch")

        with_bias = bias is not None
        key = (M, N, K, with_relu, with_bias)
        plan = self._cache.get(key)
        if plan is None:
            plan = self._build_plan(M, N, K, with_relu, with_bias)
            self._cache[key] = plan
        desc, ay, by, cy, dy, heur = plan

        D = cp.empty((M, N), dtype="bfloat16")

        if with_bias:
            bias_ptr = ctypes.c_void_p(int(bias.data.ptr))
            _check(
                self._lib.cublasLtMatmulDescSetAttribute(
                    desc,
                    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    ctypes.byref(bias_ptr),
                    ctypes.sizeof(bias_ptr),
                ),
                "biasPtr",
            )

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        stream = cp.cuda.get_current_stream()
        _check(
            self._lib.cublasLtMatmul(
                self._handle,
                desc,
                ctypes.byref(alpha),
                ctypes.c_void_p(int(B.data.ptr)),
                ay,
                ctypes.c_void_p(int(A.data.ptr)),
                by,
                ctypes.byref(beta),
                ctypes.c_void_p(int(D.data.ptr)),
                cy,
                ctypes.c_void_p(int(D.data.ptr)),
                dy,
                ctypes.cast(ctypes.byref(heur), ctypes.c_void_p),
                ctypes.c_void_p(int(self._workspace.data.ptr)),
                ctypes.c_size_t(self._WORKSPACE_BYTES),
                ctypes.c_void_p(int(stream.ptr)),
            ),
            "matmul",
        )
        return D

    def _build_dw_bgrada_plan(self, BT: int, K: int, N: int):
        """Plan for dW(K,N) = X.T(K,BT) @ grad(BT,N) with BGRADA epilogue.

        BGRADA writes dbias = column_sum(grad) (= sum over BT axis) into the
        BIAS_POINTER buffer. In cuBLAS col-major view:
            D_col(N,K) = grad_col(N,BT) @ X_col(K,BT)^T
            cuBLAS A = grad (op=N), cuBLAS B = X (op=T)
        BGRADA refers to cuBLAS A which is grad, so column_sum(grad) is what
        we want for the bias gradient of `Y = X @ W + b`.

        bf16 inputs / bf16 outputs (dW and dbias both bf16); fp32 accumulator.
        """
        lib = self._lib
        desc = ctypes.c_void_p()
        _check(
            lib.cublasLtMatmulDescCreate(
                ctypes.byref(desc), CUBLAS_COMPUTE_32F, CUDA_R_32F
            ),
            "dwDescCreate",
        )
        op_n = ctypes.c_int32(CUBLAS_OP_N)
        op_t = ctypes.c_int32(CUBLAS_OP_T)
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                ctypes.byref(op_n),
                ctypes.sizeof(op_n),
            ),
            "dwTransA",
        )
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                ctypes.byref(op_t),
                ctypes.sizeof(op_t),
            ),
            "dwTransB",
        )
        ep = ctypes.c_uint32(CUBLASLT_EPILOGUE_BGRADA)
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc, CUBLASLT_MATMUL_DESC_EPILOGUE, ctypes.byref(ep), ctypes.sizeof(ep)
            ),
            "dwEpilogue",
        )
        bias_dt = ctypes.c_int32(CUDA_R_16BF)
        _check(
            lib.cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                ctypes.byref(bias_dt),
                ctypes.sizeof(bias_dt),
            ),
            "dwBiasDtype",
        )

        # cuBLAS A = grad: shape (N, BT) col-major, ld=N
        ay = ctypes.c_void_p()
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(ay), CUDA_R_16BF, N, BT, N),
            "dwAlayout",
        )
        # cuBLAS B = X: shape (K, BT) col-major, ld=K, op=T -> (BT, K)
        by = ctypes.c_void_p()
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(by), CUDA_R_16BF, K, BT, K),
            "dwBlayout",
        )
        # cuBLAS D = dW: col-major (N, K) ld=N == row-major (K, N) ld=N
        cy = ctypes.c_void_p()
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(cy), CUDA_R_16BF, N, K, N),
            "dwClayout",
        )
        dy = ctypes.c_void_p()
        _check(
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(dy), CUDA_R_16BF, N, K, N),
            "dwDlayout",
        )

        pref = ctypes.c_void_p()
        _check(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "dwPref")
        ws = ctypes.c_size_t(self._WORKSPACE_BYTES)
        _check(
            lib.cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(ws),
                ctypes.sizeof(ws),
            ),
            "dwPrefWs",
        )

        heur = (_HeurResult * 1)()
        returned = ctypes.c_int(0)
        _check(
            lib.cublasLtMatmulAlgoGetHeuristic(
                self._handle,
                desc,
                ay,
                by,
                cy,
                dy,
                pref,
                1,
                heur,
                ctypes.byref(returned),
            ),
            "dwHeuristic",
        )
        if returned.value == 0:
            raise RuntimeError(f"cuBLASLt no algo for dW shape (K={K},N={N},BT={BT})")
        return desc, ay, by, cy, dy, heur

    def matmul_dW_bgrada(self, X, grad):
        """Compute dW = X.T @ grad and dbias = column_sum(grad) in one call.

        X: (BT, K) row-major bf16 (contiguous)
        grad: (BT, N) row-major bf16 (contiguous)
        Returns (dW (K, N), dbias (N,)), both bf16. fp32 accumulator.
        """
        cp = self._cp
        BT, K = int(X.shape[0]), int(X.shape[1])
        BT2, N = int(grad.shape[0]), int(grad.shape[1])
        if BT != BT2:
            raise ValueError("matmul_dW_bgrada BT mismatch")

        key = ("dW_bgrada", BT, K, N)
        plan = self._cache.get(key)
        if plan is None:
            plan = self._build_dw_bgrada_plan(BT, K, N)
            self._cache[key] = plan
        desc, ay, by, cy, dy, heur = plan

        dW = cp.empty((K, N), dtype="bfloat16")
        dbias = cp.empty((N,), dtype="bfloat16")

        bias_ptr = ctypes.c_void_p(int(dbias.data.ptr))
        _check(
            self._lib.cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                ctypes.byref(bias_ptr),
                ctypes.sizeof(bias_ptr),
            ),
            "dwBiasPtr",
        )

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        stream = cp.cuda.get_current_stream()
        _check(
            self._lib.cublasLtMatmul(
                self._handle,
                desc,
                ctypes.byref(alpha),
                ctypes.c_void_p(int(grad.data.ptr)),
                ay,
                ctypes.c_void_p(int(X.data.ptr)),
                by,
                ctypes.byref(beta),
                ctypes.c_void_p(int(dW.data.ptr)),
                cy,
                ctypes.c_void_p(int(dW.data.ptr)),
                dy,
                ctypes.cast(ctypes.byref(heur), ctypes.c_void_p),
                ctypes.c_void_p(int(self._workspace.data.ptr)),
                ctypes.c_size_t(self._WORKSPACE_BYTES),
                ctypes.c_void_p(int(stream.ptr)),
            ),
            "dwMatmul",
        )
        return dW, dbias


_LT_SINGLETON: Optional[LtMatmulBias] = None
_LT_TRIED: bool = False


def get_lt() -> Optional[LtMatmulBias]:
    """Return a process-global cuBLASLt wrapper, or None if unavailable.

    Used by `LinearAffine`/`LinearRelu` forward; safe to call without CuPy
    (returns None).
    """
    global _LT_SINGLETON, _LT_TRIED
    if _LT_TRIED:
        return _LT_SINGLETON
    _LT_TRIED = True
    try:
        import cupy  # noqa: F401
    except Exception:
        return None
    lib = _try_load_cublaslt()
    if lib is None:
        logger.info("cuBLASLt library not found; falling back to matmul+bias kernel")
        return None
    try:
        _bind(lib)
        _LT_SINGLETON = LtMatmulBias(lib)
    except Exception as e:  # pragma: no cover - hardware/runtime dependent
        logger.warning("cuBLASLt init failed (%s); falling back", e)
        _LT_SINGLETON = None
    return _LT_SINGLETON


def can_use_lt(*arrays) -> bool:
    """All arrays must be CuPy bf16 contiguous, 2-D shape compatible."""
    if get_lt() is None:
        return False
    try:
        import cupy as cp
    except Exception:
        return False
    for a in arrays:
        if not isinstance(a, cp.ndarray):
            return False
        if str(a.dtype) != "bfloat16":
            return False
        if not a.flags.c_contiguous:
            return False
    return True
