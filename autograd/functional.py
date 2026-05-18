from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Optional, Tuple, Union

from autograd.backend import (
    IS_CUPY,
    LOW_PRECISION_FLOAT_DTYPES,
    Array,
    ArrayLike,
    xp,
)
from autograd.tensor import (
    Function,
    Tensor,
    _matmul_autocast,
    _matmul_autocast_dW_bgrad,
)

logger = logging.getLogger(__name__)


########### Activation Functions ###############
def relu(x: Tensor) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the ReLU function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([-1, 0, 2])
        >>> y = relu(x) # Expected output: [0, 0, 2]
    """
    return Relu.apply(x)


def linear_relu(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return LinearRelu.apply(x, weight, bias)


def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return LinearAffine.apply(x, weight, bias)


def sigmoid(x: Tensor) -> Tensor:
    """
    Applies the sigmoid activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the sigmoid function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 2])
        >>> y = sigmoid(x) # Expected output: [0.5, ~0.88]
    """
    return Sigmoid.apply(x)


def softmax(x: Tensor) -> Tensor:
    """
    Applies the softmax activation function.

    Args:
        x (Tensor): The input tensor containing logits.

    Returns:
        Tensor: The tensor with softmax probabilities.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> import cupy as np
        >>> x = Tensor(xp.array([2.0, 1.0, 0.1]))
        >>> y = softmax(x) # Expected output: probabilities that sum to 1
    """
    return Softmax.apply(x)


def causal_softmax(x: Tensor) -> Tensor:
    return CausalSoftmax.apply(x)


def layer_norm_affine(
    x: Tensor,
    gain: Tensor,
    bias: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    return LayerNormAffine.apply(x, gain, bias, epsilon=epsilon)


def _cupy_row_grid(rows: int) -> Tuple[int, int]:
    grid_x = min(rows, 1024)
    return grid_x, (rows + grid_x - 1) // grid_x


@lru_cache(maxsize=1)
def _cupy_relu_kernels() -> Any:
    module = xp.RawModule(
        code=r"""
        #include <cuda_bf16.h>

        extern "C" __global__ void relu_backward_bf16(
            const __nv_bfloat16* grad,
            const __nv_bfloat16* x,
            __nv_bfloat16* out,
            const long long size
        ) {
            long long idx = blockIdx.x * blockDim.x + threadIdx.x;
            long long stride = blockDim.x * gridDim.x;
            for (; idx < size; idx += stride) {
                out[idx] = __bfloat162float(x[idx]) > 0.0f ? grad[idx] : __float2bfloat16(0.0f);
            }
        }
        """,
        options=("--std=c++11",),
    )
    return module.get_function("relu_backward_bf16")


def _cupy_relu_backward(grad: Array, x: Array) -> Optional[Array]:
    if (
        not IS_CUPY
        or not hasattr(xp, "bfloat16")
        or grad.dtype != xp.bfloat16
        or x.dtype != xp.bfloat16
        or grad.shape != x.shape
        or not grad.flags.c_contiguous
        or not x.flags.c_contiguous
    ):
        return None

    out = xp.empty_like(grad)
    threads = 256
    blocks = min((int(grad.size) + threads - 1) // threads, 1024)
    kernel = _cupy_relu_kernels()
    kernel((blocks,), (threads,), (grad, x, out, int(grad.size)))
    return out


@lru_cache(maxsize=1)
def _cupy_linear_relu_kernels() -> Tuple[Any, Any, Any]:
    module = xp.RawModule(
        code=r"""
        #include <cuda_bf16.h>

        extern "C" __global__ void linear_bias_bf16(
            __nv_bfloat16* y,
            const __nv_bfloat16* bias,
            const long long size,
            const int cols
        ) {
            long long idx = blockIdx.x * blockDim.x + threadIdx.x;
            long long stride = blockDim.x * gridDim.x;
            for (; idx < size; idx += stride) {
                float value = __bfloat162float(y[idx]) + __bfloat162float(bias[idx % cols]);
                y[idx] = __float2bfloat16(value);
            }
        }

        extern "C" __global__ void linear_relu_bias_bf16(
            __nv_bfloat16* y,
            const __nv_bfloat16* bias,
            const long long size,
            const int cols
        ) {
            long long idx = blockIdx.x * blockDim.x + threadIdx.x;
            long long stride = blockDim.x * gridDim.x;
            for (; idx < size; idx += stride) {
                float value = __bfloat162float(y[idx]) + __bfloat162float(bias[idx % cols]);
                y[idx] = __float2bfloat16(value > 0.0f ? value : 0.0f);
            }
        }

        extern "C" __global__ void linear_relu_backward_bf16(
            const __nv_bfloat16* grad,
            const __nv_bfloat16* out,
            __nv_bfloat16* grad_act,
            const long long size
        ) {
            long long idx = blockIdx.x * blockDim.x + threadIdx.x;
            long long stride = blockDim.x * gridDim.x;
            for (; idx < size; idx += stride) {
                grad_act[idx] = __bfloat162float(out[idx]) > 0.0f
                    ? grad[idx]
                    : __float2bfloat16(0.0f);
            }
        }
        """,
        options=("--std=c++11",),
    )
    return (
        module.get_function("linear_bias_bf16"),
        module.get_function("linear_relu_bias_bf16"),
        module.get_function("linear_relu_backward_bf16"),
    )


def _cupy_linear_relu_backward(grad: Array, out: Array) -> Optional[Array]:
    if (
        not IS_CUPY
        or not hasattr(xp, "bfloat16")
        or grad.dtype != xp.bfloat16
        or out.dtype != xp.bfloat16
        or grad.shape != out.shape
        or not grad.flags.c_contiguous
        or not out.flags.c_contiguous
    ):
        return None

    grad_act = xp.empty_like(grad)
    _, _, backward_kernel = _cupy_linear_relu_kernels()
    threads = 256
    blocks = min((int(grad.size) + threads - 1) // threads, 1024)
    backward_kernel((blocks,), (threads,), (grad, out, grad_act, int(grad.size)))
    return grad_act


@lru_cache(maxsize=1)
def _cupy_layer_norm_kernels() -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    module = xp.RawModule(
        code=r"""
        #include <cuda_bf16.h>

        extern "C" __global__ void layer_norm_forward_row(
            const float* x,
            const float* gain,
            const float* bias,
            float* y,
            float* x_hat,
            float* rstd,
            const int rows,
            const int cols,
            const float epsilon
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const float* xr = x + ((long long)row) * cols;
            float* yr = y + ((long long)row) * cols;
            float* hr = x_hat + ((long long)row) * cols;

            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += xr[col];
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float mean = smem[0] / cols;
            __syncthreads();
            float var_sum = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                float centered = xr[col] - mean;
                var_sum += centered * centered;
            }
            smem[tid] = var_sum;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float inv = rsqrtf(smem[0] / cols + epsilon);
            if (tid == 0) {
                rstd[row] = inv;
            }
            for (int col = tid; col < cols; col += blockDim.x) {
                float h = (xr[col] - mean) * inv;
                hr[col] = h;
                yr[col] = h * gain[col] + bias[col];
            }
        }

        extern "C" __global__ void layer_norm_forward_bf16_row(
            const __nv_bfloat16* x,
            const __nv_bfloat16* gain,
            const __nv_bfloat16* bias,
            __nv_bfloat16* y,
            float* x_hat,
            float* rstd,
            const int rows,
            const int cols,
            const float epsilon
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const __nv_bfloat16* xr = x + ((long long)row) * cols;
            __nv_bfloat16* yr = y + ((long long)row) * cols;
            float* hr = x_hat + ((long long)row) * cols;

            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += __bfloat162float(xr[col]);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float mean = smem[0] / cols;
            __syncthreads();
            float var_sum = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                float centered = __bfloat162float(xr[col]) - mean;
                var_sum += centered * centered;
            }
            smem[tid] = var_sum;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float inv = rsqrtf(smem[0] / cols + epsilon);
            if (tid == 0) {
                rstd[row] = inv;
            }
            for (int col = tid; col < cols; col += blockDim.x) {
                float h = (__bfloat162float(xr[col]) - mean) * inv;
                hr[col] = h;
                float out = h * __bfloat162float(gain[col]) + __bfloat162float(bias[col]);
                yr[col] = __float2bfloat16(out);
            }
        }

        extern "C" __global__ void layer_norm_backward_x_row(
            const float* grad,
            const float* x_hat,
            const float* rstd,
            const float* gain,
            float* dx,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            float* sum1_s = smem;
            float* sum2_s = smem + blockDim.x;
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const float* gr = grad + ((long long)row) * cols;
            const float* hr = x_hat + ((long long)row) * cols;
            float* dxr = dx + ((long long)row) * cols;

            float sum1 = 0.0f;
            float sum2 = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                float dx_hat = gr[col] * gain[col];
                sum1 += dx_hat;
                sum2 += dx_hat * hr[col];
            }
            sum1_s[tid] = sum1;
            sum2_s[tid] = sum2;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sum1_s[tid] += sum1_s[tid + stride];
                    sum2_s[tid] += sum2_s[tid + stride];
                }
                __syncthreads();
            }

            float inv = rstd[row];
            float row_sum1 = sum1_s[0];
            float row_sum2 = sum2_s[0];
            float scale = inv / cols;
            for (int col = tid; col < cols; col += blockDim.x) {
                float dx_hat = gr[col] * gain[col];
                dxr[col] = scale * (cols * dx_hat - row_sum1 - hr[col] * row_sum2);
            }
        }

        extern "C" __global__ void layer_norm_backward_x_bf16_row(
            const __nv_bfloat16* grad,
            const float* x_hat,
            const float* rstd,
            const __nv_bfloat16* gain,
            __nv_bfloat16* dx,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            float* sum1_s = smem;
            float* sum2_s = smem + blockDim.x;
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const __nv_bfloat16* gr = grad + ((long long)row) * cols;
            const float* hr = x_hat + ((long long)row) * cols;
            __nv_bfloat16* dxr = dx + ((long long)row) * cols;

            float sum1 = 0.0f;
            float sum2 = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                float dx_hat = __bfloat162float(gr[col]) * __bfloat162float(gain[col]);
                sum1 += dx_hat;
                sum2 += dx_hat * hr[col];
            }
            sum1_s[tid] = sum1;
            sum2_s[tid] = sum2;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sum1_s[tid] += sum1_s[tid + stride];
                    sum2_s[tid] += sum2_s[tid + stride];
                }
                __syncthreads();
            }

            float inv = rstd[row];
            float row_sum1 = sum1_s[0];
            float row_sum2 = sum2_s[0];
            float scale = inv / cols;
            for (int col = tid; col < cols; col += blockDim.x) {
                float dx_hat = __bfloat162float(gr[col]) * __bfloat162float(gain[col]);
                float out = scale * (cols * dx_hat - row_sum1 - hr[col] * row_sum2);
                dxr[col] = __float2bfloat16(out);
            }
        }

        extern "C" __global__ void layer_norm_backward_param_col(
            const float* grad,
            const float* x_hat,
            float* d_gain,
            float* d_bias,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            float* gain_s = smem;
            float* bias_s = smem + blockDim.x;
            int col = blockIdx.x;
            int tid = threadIdx.x;
            if (col >= cols) return;

            float gain_sum = 0.0f;
            float bias_sum = 0.0f;
            for (int row = tid; row < rows; row += blockDim.x) {
                long long idx = ((long long)row) * cols + col;
                float g = grad[idx];
                gain_sum += g * x_hat[idx];
                bias_sum += g;
            }
            gain_s[tid] = gain_sum;
            bias_s[tid] = bias_sum;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    gain_s[tid] += gain_s[tid + stride];
                    bias_s[tid] += bias_s[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                d_gain[col] = gain_s[0];
                d_bias[col] = bias_s[0];
            }
        }

        extern "C" __global__ void layer_norm_backward_param_bf16_col(
            const __nv_bfloat16* grad,
            const float* x_hat,
            float* d_gain,
            float* d_bias,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            float* gain_s = smem;
            float* bias_s = smem + blockDim.x;
            int col = blockIdx.x;
            int tid = threadIdx.x;
            if (col >= cols) return;

            float gain_sum = 0.0f;
            float bias_sum = 0.0f;
            for (int row = tid; row < rows; row += blockDim.x) {
                long long idx = ((long long)row) * cols + col;
                float g = __bfloat162float(grad[idx]);
                gain_sum += g * x_hat[idx];
                bias_sum += g;
            }
            gain_s[tid] = gain_sum;
            bias_s[tid] = bias_sum;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    gain_s[tid] += gain_s[tid + stride];
                    bias_s[tid] += bias_s[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                d_gain[col] = gain_s[0];
                d_bias[col] = bias_s[0];
            }
        }

        // Coalesced first-stage reduction for bf16 LayerNorm parameter grads.
        // Each block reduces a 256-row tile for 8 adjacent columns. That keeps
        // reads coalesced across columns, unlike the one-column strided kernel.
        extern "C" __global__ void layer_norm_backward_param_bf16_partial_8col(
            const __nv_bfloat16* grad,
            const float* x_hat,
            float* partial_gain,
            float* partial_bias,
            const int rows,
            const int cols,
            const int tiles
        ) {
            __shared__ float gain_s[256];
            __shared__ float bias_s[256];
            int tid = threadIdx.x;
            int lane_col = tid & 7;
            int lane_row = tid >> 3;
            int col = blockIdx.x * 8 + lane_col;
            int tile = blockIdx.y;
            int row_start = tile * 256;
            int row_end = min(row_start + 256, rows);

            float gain_sum = 0.0f;
            float bias_sum = 0.0f;
            if (col < cols) {
                for (int row = row_start + lane_row; row < row_end; row += 32) {
                    long long idx = ((long long)row) * cols + col;
                    float g = __bfloat162float(grad[idx]);
                    gain_sum += g * x_hat[idx];
                    bias_sum += g;
                }
            }
            gain_s[tid] = gain_sum;
            bias_s[tid] = bias_sum;
            __syncthreads();
            for (int offset = 16; offset > 0; offset >>= 1) {
                if (lane_row < offset) {
                    int other = ((lane_row + offset) << 3) + lane_col;
                    gain_s[tid] += gain_s[other];
                    bias_s[tid] += bias_s[other];
                }
                __syncthreads();
            }
            if (lane_row == 0 && col < cols) {
                long long out_idx = ((long long)tile) * cols + col;
                partial_gain[out_idx] = gain_s[tid];
                partial_bias[out_idx] = bias_s[tid];
            }
        }

        extern "C" __global__ void layer_norm_backward_param_bf16_finalize(
            const float* partial_gain,
            const float* partial_bias,
            float* d_gain,
            float* d_bias,
            const int cols,
            const int tiles
        ) {
            extern __shared__ float smem[];
            float* gain_s = smem;
            float* bias_s = smem + blockDim.x;
            int col = blockIdx.x;
            int tid = threadIdx.x;
            float gain_sum = 0.0f;
            float bias_sum = 0.0f;
            for (int tile = tid; tile < tiles; tile += blockDim.x) {
                long long idx = ((long long)tile) * cols + col;
                gain_sum += partial_gain[idx];
                bias_sum += partial_bias[idx];
            }
            gain_s[tid] = gain_sum;
            bias_s[tid] = bias_sum;
            __syncthreads();
            for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    gain_s[tid] += gain_s[tid + offset];
                    bias_s[tid] += bias_s[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                d_gain[col] = gain_s[0];
                d_bias[col] = bias_s[0];
            }
        }
        """,
        options=("--std=c++11",),
    )
    return (
        module.get_function("layer_norm_forward_row"),
        module.get_function("layer_norm_backward_x_row"),
        module.get_function("layer_norm_backward_param_col"),
        module.get_function("layer_norm_forward_bf16_row"),
        module.get_function("layer_norm_backward_x_bf16_row"),
        module.get_function("layer_norm_backward_param_bf16_col"),
        module.get_function("layer_norm_backward_param_bf16_partial_8col"),
        module.get_function("layer_norm_backward_param_bf16_finalize"),
    )


@lru_cache(maxsize=1)
def _cupy_softmax_row_kernels() -> Tuple[Any, Any]:
    module = xp.RawModule(
        code=r"""
        extern "C" __global__ void softmax_forward_row(
            const float* x,
            float* y,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const float* xr = x + ((long long)row) * cols;
            float* yr = y + ((long long)row) * cols;

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < cols; col += blockDim.x) {
                max_val = fmaxf(max_val, xr[col]);
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += expf(xr[col] - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float denom = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                yr[col] = expf(xr[col] - max_val) / denom;
            }
        }

        extern "C" __global__ void softmax_backward_row(
            const float* probs,
            const float* grad,
            float* out,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const float* pr = probs + ((long long)row) * cols;
            const float* gr = grad + ((long long)row) * cols;
            float* orow = out + ((long long)row) * cols;

            float sum_term = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_term += pr[col] * gr[col];
            }
            smem[tid] = sum_term;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            sum_term = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                orow[col] = pr[col] * (gr[col] - sum_term);
            }
        }
        """,
        options=("--std=c++11",),
    )
    return (
        module.get_function("softmax_forward_row"),
        module.get_function("softmax_backward_row"),
    )


def _is_cupy_float32_contiguous(x: Array) -> bool:
    return bool(IS_CUPY and x.dtype == xp.float32 and x.flags.c_contiguous)


def _cupy_softmax_forward(x: Array) -> Optional[Array]:
    if not _is_cupy_float32_contiguous(x) or x.ndim < 1 or x.shape[-1] < 1:
        return None

    cols = int(x.shape[-1])
    rows = int(x.size // cols)
    out = xp.empty_like(x)
    threads = 256
    forward_kernel, _ = _cupy_softmax_row_kernels()
    forward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (x, out, rows, cols),
        shared_mem=threads * 4,
    )
    return out


def _cupy_softmax_backward(probs: Array, grad: Array) -> Optional[Array]:
    if (
        not _is_cupy_float32_contiguous(probs)
        or not _is_cupy_float32_contiguous(grad)
        or probs.shape != grad.shape
        or probs.ndim < 1
        or probs.shape[-1] < 1
    ):
        return None

    cols = int(probs.shape[-1])
    rows = int(probs.size // cols)
    out = xp.empty_like(probs)
    threads = 256
    _, backward_kernel = _cupy_softmax_row_kernels()
    backward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (probs, grad, out, rows, cols),
        shared_mem=threads * 4,
    )
    return out


@lru_cache(maxsize=1)
def _cupy_causal_softmax_row_kernels() -> Tuple[Any, Any]:
    module = xp.RawModule(
        code=r"""
        extern "C" __global__ void causal_softmax_forward_row(
            const float* x,
            float* y,
            const int rows,
            const int seq_q,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            int query_pos = row % seq_q;
            int visible_cols = query_pos + 1;
            if (visible_cols > cols) visible_cols = cols;

            const float* xr = x + ((long long)row) * cols;
            float* yr = y + ((long long)row) * cols;

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < visible_cols; col += blockDim.x) {
                max_val = fmaxf(max_val, xr[col]);
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < visible_cols; col += blockDim.x) {
                sum_val += expf(xr[col] - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float denom = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                yr[col] = col < visible_cols ? expf(xr[col] - max_val) / denom : 0.0f;
            }
        }

        extern "C" __global__ void causal_softmax_backward_row(
            const float* probs,
            const float* grad,
            float* out,
            const int rows,
            const int seq_q,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            int query_pos = row % seq_q;
            int visible_cols = query_pos + 1;
            if (visible_cols > cols) visible_cols = cols;

            const float* pr = probs + ((long long)row) * cols;
            const float* gr = grad + ((long long)row) * cols;
            float* orow = out + ((long long)row) * cols;

            float sum_term = 0.0f;
            for (int col = tid; col < visible_cols; col += blockDim.x) {
                sum_term += pr[col] * gr[col];
            }
            smem[tid] = sum_term;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            sum_term = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                orow[col] = col < visible_cols ? pr[col] * (gr[col] - sum_term) : 0.0f;
            }
        }
        """,
        options=("--std=c++11",),
    )
    return (
        module.get_function("causal_softmax_forward_row"),
        module.get_function("causal_softmax_backward_row"),
    )


def _cupy_causal_softmax_forward(x: Array) -> Optional[Array]:
    if not _is_cupy_float32_contiguous(x) or x.ndim < 2 or x.shape[-1] < 1:
        return None

    seq_q = int(x.shape[-2])
    cols = int(x.shape[-1])
    rows = int(x.size // cols)
    out = xp.empty_like(x)
    threads = 256
    forward_kernel, _ = _cupy_causal_softmax_row_kernels()
    forward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (x, out, rows, seq_q, cols),
        shared_mem=threads * 4,
    )
    return out


def _cupy_causal_softmax_backward(probs: Array, grad: Array) -> Optional[Array]:
    if (
        not _is_cupy_float32_contiguous(probs)
        or not _is_cupy_float32_contiguous(grad)
        or probs.shape != grad.shape
        or probs.ndim < 2
        or probs.shape[-1] < 1
    ):
        return None

    seq_q = int(probs.shape[-2])
    cols = int(probs.shape[-1])
    rows = int(probs.size // cols)
    out = xp.empty_like(probs)
    threads = 256
    _, backward_kernel = _cupy_causal_softmax_row_kernels()
    backward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (probs, grad, out, rows, seq_q, cols),
        shared_mem=threads * 4,
    )
    return out


@lru_cache(maxsize=1)
def _cupy_cross_entropy_kernels() -> Tuple[Any, Any, Any, Any, Any, Any]:
    module = xp.RawModule(
        code=r"""
        #include <cuda_bf16.h>

        extern "C" __global__ void cross_entropy_forward_row(
            const float* logits,
            const long long* targets,
            const float* weights,
            float* probs,
            float* losses,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const float* xr = logits + ((long long)row) * cols;
            float* pr = probs + ((long long)row) * cols;
            int target = (int)targets[row];
            float weight = weights[row];

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < cols; col += blockDim.x) {
                max_val = fmaxf(max_val, xr[col]);
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += expf(xr[col] - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float denom = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                pr[col] = expf(xr[col] - max_val) / denom;
            }
            if (tid == 0) {
                losses[row] = -weight * (xr[target] - max_val - logf(denom));
            }
        }

        extern "C" __global__ void cross_entropy_forward_bf16_row(
            const __nv_bfloat16* logits,
            const long long* targets,
            const float* weights,
            float* probs,
            float* losses,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const __nv_bfloat16* xr = logits + ((long long)row) * cols;
            float* pr = probs + ((long long)row) * cols;
            int target = (int)targets[row];
            float weight = weights[row];

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < cols; col += blockDim.x) {
                max_val = fmaxf(max_val, __bfloat162float(xr[col]));
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += expf(__bfloat162float(xr[col]) - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float denom = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                pr[col] = expf(__bfloat162float(xr[col]) - max_val) / denom;
            }
            if (tid == 0) {
                float target_logit = __bfloat162float(xr[target]);
                losses[row] = -weight * (target_logit - max_val - logf(denom));
            }
        }

        extern "C" __global__ void cross_entropy_forward_bf16_loss_row(
            const __nv_bfloat16* logits,
            const long long* targets,
            const float* weights,
            float* losses,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const __nv_bfloat16* xr = logits + ((long long)row) * cols;
            int target = (int)targets[row];
            float weight = weights[row];

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < cols; col += blockDim.x) {
                max_val = fmaxf(max_val, __bfloat162float(xr[col]));
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += expf(__bfloat162float(xr[col]) - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float target_logit = __bfloat162float(xr[target]);
                losses[row] = -weight * (target_logit - max_val - logf(smem[0]));
            }
        }

        extern "C" __global__ void cross_entropy_backward_row(
            const float* probs,
            const long long* targets,
            const float* weights,
            const float* scale,
            float* out,
            const int rows,
            const int cols
        ) {
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const float* pr = probs + ((long long)row) * cols;
            float* orow = out + ((long long)row) * cols;
            int target = (int)targets[row];
            float row_scale = weights[row] * scale[0];

            for (int col = tid; col < cols; col += blockDim.x) {
                float val = pr[col] - (col == target ? 1.0f : 0.0f);
                orow[col] = val * row_scale;
            }
        }

        extern "C" __global__ void cross_entropy_backward_bf16_logits_row(
            const __nv_bfloat16* logits,
            const long long* targets,
            const float* weights,
            const float* scale,
            float* out,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const __nv_bfloat16* xr = logits + ((long long)row) * cols;
            float* orow = out + ((long long)row) * cols;
            int target = (int)targets[row];
            float row_scale = weights[row] * scale[0];

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < cols; col += blockDim.x) {
                max_val = fmaxf(max_val, __bfloat162float(xr[col]));
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += expf(__bfloat162float(xr[col]) - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float denom = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                float prob = expf(__bfloat162float(xr[col]) - max_val) / denom;
                float val = prob - (col == target ? 1.0f : 0.0f);
                orow[col] = val * row_scale;
            }
        }

        extern "C" __global__ void cross_entropy_backward_bf16_logits_to_bf16_row(
            const __nv_bfloat16* logits,
            const long long* targets,
            const float* weights,
            const float* scale,
            __nv_bfloat16* out,
            const int rows,
            const int cols
        ) {
            extern __shared__ float smem[];
            int row = blockIdx.x + blockIdx.y * gridDim.x;
            int tid = threadIdx.x;
            if (row >= rows) return;

            const __nv_bfloat16* xr = logits + ((long long)row) * cols;
            __nv_bfloat16* orow = out + ((long long)row) * cols;
            int target = (int)targets[row];
            float row_scale = weights[row] * scale[0];

            float max_val = -3.4028234663852886e38F;
            for (int col = tid; col < cols; col += blockDim.x) {
                max_val = fmaxf(max_val, __bfloat162float(xr[col]));
            }
            smem[tid] = max_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            max_val = smem[0];
            __syncthreads();
            float sum_val = 0.0f;
            for (int col = tid; col < cols; col += blockDim.x) {
                sum_val += expf(__bfloat162float(xr[col]) - max_val);
            }
            smem[tid] = sum_val;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float denom = smem[0];
            for (int col = tid; col < cols; col += blockDim.x) {
                float prob = expf(__bfloat162float(xr[col]) - max_val) / denom;
                float val = prob - (col == target ? 1.0f : 0.0f);
                orow[col] = __float2bfloat16(val * row_scale);
            }
        }

        """,
        options=("--std=c++11",),
    )
    return (
        module.get_function("cross_entropy_forward_row"),
        module.get_function("cross_entropy_forward_bf16_row"),
        module.get_function("cross_entropy_backward_row"),
        module.get_function("cross_entropy_forward_bf16_loss_row"),
        module.get_function("cross_entropy_backward_bf16_logits_row"),
        module.get_function("cross_entropy_backward_bf16_logits_to_bf16_row"),
    )


def _cupy_cross_entropy_forward(
    logits: Array,
    targets: Array,
    weights: Array,
) -> Optional[Tuple[Array, Array]]:
    is_float32 = _is_cupy_float32_contiguous(logits)
    is_bfloat16 = bool(
        IS_CUPY
        and hasattr(xp, "bfloat16")
        and logits.dtype == xp.bfloat16
        and logits.flags.c_contiguous
    )
    if (
        not (is_float32 or is_bfloat16)
        or logits.ndim != 2
        or targets.dtype != xp.int64
        or weights.dtype != xp.float32
        or targets.ndim != 1
        or weights.ndim != 1
        or targets.shape[0] != logits.shape[0]
        or weights.shape[0] != logits.shape[0]
    ):
        return None

    rows = int(logits.shape[0])
    cols = int(logits.shape[1])
    probs = xp.empty(logits.shape, dtype=xp.float32)
    losses = xp.empty((rows,), dtype=xp.float32)
    threads = 1024
    (
        forward_float32_kernel,
        forward_bf16_kernel,
        _,
        _,
        _,
        _,
    ) = _cupy_cross_entropy_kernels()
    forward_kernel = forward_float32_kernel if is_float32 else forward_bf16_kernel
    forward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (logits, targets, weights, probs, losses, rows, cols),
        shared_mem=threads * 4,
    )
    return probs, losses


def _cupy_cross_entropy_bf16_forward_loss(
    logits: Array,
    targets: Array,
    weights: Array,
) -> Optional[Array]:
    if (
        not IS_CUPY
        or not hasattr(xp, "bfloat16")
        or logits.dtype != xp.bfloat16
        or not logits.flags.c_contiguous
        or logits.ndim != 2
        or targets.dtype != xp.int64
        or weights.dtype != xp.float32
        or targets.ndim != 1
        or weights.ndim != 1
        or targets.shape[0] != logits.shape[0]
        or weights.shape[0] != logits.shape[0]
    ):
        return None

    rows = int(logits.shape[0])
    cols = int(logits.shape[1])
    losses = xp.empty((rows,), dtype=xp.float32)
    threads = 512
    _, _, _, forward_kernel, _, _ = _cupy_cross_entropy_kernels()
    forward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (logits, targets, weights, losses, rows, cols),
        shared_mem=threads * 4,
    )
    return losses


def _cupy_cross_entropy_backward(
    probs: Array,
    targets: Array,
    weights: Array,
    scale: Array,
) -> Optional[Array]:
    if (
        not _is_cupy_float32_contiguous(probs)
        or probs.ndim != 2
        or targets.dtype != xp.int64
        or weights.dtype != xp.float32
        or targets.ndim != 1
        or weights.ndim != 1
        or targets.shape[0] != probs.shape[0]
        or weights.shape[0] != probs.shape[0]
    ):
        return None

    scale = xp.asarray(scale, dtype=xp.float32).reshape(1)
    rows = int(probs.shape[0])
    cols = int(probs.shape[1])
    out = xp.empty_like(probs)
    threads = 1024
    _, _, backward_kernel, _, _, _ = _cupy_cross_entropy_kernels()
    backward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (probs, targets, weights, scale, out, rows, cols),
    )
    return out


def _cupy_cross_entropy_bf16_backward_from_logits(
    logits: Array,
    targets: Array,
    weights: Array,
    scale: Array,
    out: Optional[Array] = None,
) -> Optional[Array]:
    if (
        not IS_CUPY
        or not hasattr(xp, "bfloat16")
        or logits.dtype != xp.bfloat16
        or not logits.flags.c_contiguous
        or logits.ndim != 2
        or targets.dtype != xp.int64
        or weights.dtype != xp.float32
        or targets.ndim != 1
        or weights.ndim != 1
        or targets.shape[0] != logits.shape[0]
        or weights.shape[0] != logits.shape[0]
    ):
        return None

    scale = xp.asarray(scale, dtype=xp.float32).reshape(1)
    rows = int(logits.shape[0])
    cols = int(logits.shape[1])
    if out is None:
        out = xp.empty(logits.shape, dtype=xp.bfloat16)
    elif (
        out.shape != logits.shape
        or out.dtype != logits.dtype
        or not out.flags.c_contiguous
    ):
        return None
    threads = 512
    _, _, _, _, _, backward_kernel = _cupy_cross_entropy_kernels()
    backward_kernel(
        _cupy_row_grid(rows),
        (threads,),
        (logits, targets, weights, scale, out, rows, cols),
        shared_mem=threads * 4,
    )
    return out


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    r"""
    Applies the log-softmax function.

    Args:
        x (Tensor): The input tensor containing logits.
        dim (int): The dimension along which logprobs are computed.

    Returns:
        Tensor: The tensor with token logprobs.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor(xp.array([2.0, 1.0, 0.1]))
        >>> y = log_softmax(x) # Expected output: logprobs that exponentiate to probabilities summing to 1
    """
    return LogSoftmax.apply(x, dim=dim)


def tanh(x: Tensor) -> Tensor:
    """
    Applies the hyperbolic tangent (tanh) activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the tanh function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 1])
        >>> y = tanh(x) # Expected output: [0, tanh(1)]
    """
    return Tanh.apply(x)


def gelu(x: Tensor) -> Tensor:
    r"""
    Applies the Gaussian Error Linear Unit (GELU) activation function.

    This function uses the approximate formula:
    $$
    0.5 * x * \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \, (x + 0.044715*x^3)\right)\right)
    $$

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The tensor after applying the GELU function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([1.0, -1.0])
        >>> y = gelu(x) # Expected output: approximate GELU values for the inputs
    """
    return Gelu.apply(x)


def scaled_dot_product_attention_mlx_custom(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    is_training: Optional[bool] = None,
    dropout_prob: float = 0.0,
) -> Tensor:
    """
    Custom-implemented MLX kernel causal attention calculation
    """
    dropout_scale_mask = ScaledDotProductAttentionMLXCustom.build_dropout_scale_mask(
        is_training=is_training,
        dropout_prob=dropout_prob,
        query_shape=query.shape,
        key_shape=key.shape,
    )
    ScaledDotProductAttentionMLXCustom.validate(
        query.data,
        key.data,
        value.data,
        dropout_scale_mask=dropout_scale_mask,
    )
    return ScaledDotProductAttentionMLXCustom.apply(
        query,
        key,
        value,
        dropout_scale_mask=dropout_scale_mask,
    )


def scaled_dot_product_attention_cudnn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tensor:
    """
    cuDNN fused causal scaled-dot-product attention for the CuPy backend.

    This path is intentionally narrow: 4D contiguous low-precision Q/K/V tensors
    with identical shape and no dropout. It is used only for structural causal
    self-attention; unsupported cases fall back to the dense Tensor expression.
    """
    ScaledDotProductAttentionCuDNN.validate(query.data, key.data, value.data)
    return ScaledDotProductAttentionCuDNN.apply(query, key, value)


def _cudnn_dtype_for_array(x: Array) -> Optional[Any]:
    try:
        import cudnn  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError:
        return None

    if x.dtype == xp.float16:
        return cudnn.data_type.HALF
    if str(x.dtype) == "bfloat16":
        return cudnn.data_type.BFLOAT16
    return None


def _cudnn_strides(shape: Tuple[int, ...]) -> list[int]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return list(reversed(stride))


def _cudnn_sdpa_bthd_output_stride(
    shape: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    batch_size, num_heads, seq_len, head_dim = shape
    return (
        seq_len * num_heads * head_dim,
        head_dim,
        num_heads * head_dim,
        1,
    )


def _cupy_empty_bhtd_with_bthd_storage(
    shape: Tuple[int, int, int, int],
    dtype: Any,
) -> Array:
    batch_size, num_heads, seq_len, head_dim = shape
    storage = xp.empty((batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    itemsize = int(storage.dtype.itemsize)
    return xp.ndarray(
        shape,
        dtype=dtype,
        memptr=storage.data,
        strides=tuple(
            stride * itemsize for stride in _cudnn_sdpa_bthd_output_stride(shape)
        ),
    )


def _cupy_as_bhtd_with_bthd_storage(
    x: Array,
    shape: Tuple[int, int, int, int],
) -> Array:
    itemsize = int(x.dtype.itemsize)
    output_stride = _cudnn_sdpa_bthd_output_stride(shape)
    if tuple(int(stride // itemsize) for stride in x.strides) == output_stride:
        return x

    out = _cupy_empty_bhtd_with_bthd_storage(shape, x.dtype)
    out[...] = x
    return out


def _cudnn_tensor(
    graph: Any,
    name: str,
    shape: Tuple[int, ...],
    data_type: Any,
    stride: Optional[Tuple[int, ...]] = None,
) -> Any:
    return graph.tensor(
        name=name,
        dim=list(shape),
        stride=list(stride) if stride is not None else _cudnn_strides(shape),
        data_type=data_type,
    )


def _cudnn_mark_output(
    tensor: Any,
    shape: Tuple[int, ...],
    data_type: Any,
    stride: Optional[Tuple[int, ...]] = None,
) -> Any:
    return (
        tensor.set_dim(list(shape))
        .set_stride(list(stride) if stride is not None else _cudnn_strides(shape))
        .set_output(True)
        .set_data_type(data_type)
    )


@lru_cache(maxsize=8)
def _cudnn_sdpa_forward_graph(
    shape: Tuple[int, int, int, int],
    input_stride: Tuple[int, int, int, int],
    output_stride: Tuple[int, int, int, int],
    data_type_name: str,
) -> Tuple[Any, Tuple[Any, Any, Any, Any, Any]]:
    import cudnn  # pyright: ignore[reportMissingImports]

    data_type = getattr(cudnn.data_type, data_type_name)
    batch_size, num_heads, seq_len, head_dim = shape
    graph = cudnn.pygraph(
        io_data_type=data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    query = _cudnn_tensor(graph, "query", shape, data_type, input_stride)
    key = _cudnn_tensor(graph, "key", shape, data_type, input_stride)
    value = _cudnn_tensor(graph, "value", shape, data_type, input_stride)
    output, stats = graph.sdpa(
        query,
        key,
        value,
        attn_scale=float(head_dim) ** -0.5,
        generate_stats=True,
        diagonal_band_right_bound=0,
        compute_data_type=cudnn.data_type.FLOAT,
        name="cudnn_causal_sdpa",
    )
    _cudnn_mark_output(output, shape, data_type, output_stride)
    _cudnn_mark_output(
        stats, (batch_size, num_heads, seq_len, 1), cudnn.data_type.FLOAT
    )
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, (query, key, value, output, stats)


@lru_cache(maxsize=8)
def _cudnn_sdpa_backward_graph(
    shape: Tuple[int, int, int, int],
    input_stride: Tuple[int, int, int, int],
    output_stride: Tuple[int, int, int, int],
    data_type_name: str,
) -> Tuple[Any, Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]]:
    import cudnn  # pyright: ignore[reportMissingImports]

    data_type = getattr(cudnn.data_type, data_type_name)
    batch_size, num_heads, seq_len, head_dim = shape
    stats_shape = (batch_size, num_heads, seq_len, 1)
    graph = cudnn.pygraph(
        io_data_type=data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    query = _cudnn_tensor(graph, "query", shape, data_type, input_stride)
    key = _cudnn_tensor(graph, "key", shape, data_type, input_stride)
    value = _cudnn_tensor(graph, "value", shape, data_type, input_stride)
    output = _cudnn_tensor(graph, "output", shape, data_type, output_stride)
    grad_output = _cudnn_tensor(graph, "grad_output", shape, data_type, output_stride)
    stats = _cudnn_tensor(graph, "stats", stats_shape, cudnn.data_type.FLOAT)
    grad_query, grad_key, grad_value = graph.sdpa_backward(
        query,
        key,
        value,
        output,
        grad_output,
        stats,
        attn_scale=float(head_dim) ** -0.5,
        diagonal_band_right_bound=0,
        compute_data_type=cudnn.data_type.FLOAT,
        name="cudnn_causal_sdpa_backward",
    )
    # grad_{query,key,value} buffers are allocated via xp.empty_like(query/key/value)
    # in the backward, which preserves the (non-contiguous) BTHD-storage stride
    # pattern of Q/K/V. The cuDNN graph must be told to write the gradients with
    # those same strides — otherwise it writes assuming default C-contiguous BHTD
    # layout and downstream reads land on the wrong elements.
    _cudnn_mark_output(grad_query, shape, data_type, input_stride)
    _cudnn_mark_output(grad_key, shape, data_type, input_stride)
    _cudnn_mark_output(grad_value, shape, data_type, input_stride)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return (
        graph,
        (
            query,
            key,
            value,
            output,
            grad_output,
            stats,
            grad_query,
            grad_key,
            grad_value,
        ),
    )


class ScaledDotProductAttentionCuDNN(Function):
    @staticmethod
    def validate(query: Array, key: Array, value: Array) -> None:
        if not IS_CUPY:
            raise RuntimeError("cudnn SDPA requires the CuPy backend")
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("cudnn SDPA requires Q, K, and V to share shape")
        if query.ndim != 4:
            raise ValueError("cudnn SDPA expects 4D [batch, heads, seq, dim] tensors")
        if int(query.shape[-1]) % 8 != 0:
            raise ValueError("cudnn SDPA requires head_dim to be a multiple of 8")
        if query.strides != key.strides or query.strides != value.strides:
            raise ValueError("cudnn SDPA requires Q, K, and V to share strides")
        data_type = _cudnn_dtype_for_array(query)
        if data_type is None:
            raise ValueError("cudnn SDPA requires float16 or bfloat16 Q, K, and V")
        if (
            _cudnn_dtype_for_array(key) != data_type
            or _cudnn_dtype_for_array(value) != data_type
        ):
            raise ValueError("cudnn SDPA requires matching Q, K, and V dtypes")

    @staticmethod
    def _cudnn_data_type_name(x: Array) -> str:
        data_type = _cudnn_dtype_for_array(x)
        if data_type is None:
            raise ValueError("cudnn SDPA requires float16 or bfloat16 tensors")
        return data_type.name

    def forward(self, query: Array, key: Array, value: Array) -> Array:
        self.validate(query, key, value)
        self.data_type_name = self._cudnn_data_type_name(query)
        self.shape: Tuple[int, int, int, int] = (
            int(query.shape[0]),
            int(query.shape[1]),
            int(query.shape[2]),
            int(query.shape[3]),
        )
        itemsize = int(query.dtype.itemsize)
        self.input_stride: Tuple[int, int, int, int] = (
            int(query.strides[0] // itemsize),
            int(query.strides[1] // itemsize),
            int(query.strides[2] // itemsize),
            int(query.strides[3] // itemsize),
        )
        self.output_stride = _cudnn_sdpa_bthd_output_stride(self.shape)

        graph, tensors = _cudnn_sdpa_forward_graph(
            self.shape,
            self.input_stride,
            self.output_stride,
            self.data_type_name,
        )
        query_t, key_t, value_t, output_t, stats_t = tensors
        output = _cupy_empty_bhtd_with_bthd_storage(self.shape, query.dtype)
        stats = xp.empty((*self.shape[:3], 1), dtype=xp.float32)
        workspace = xp.empty((graph.get_workspace_size(),), dtype=xp.uint8)
        graph.execute(
            {
                query_t: query,
                key_t: key,
                value_t: value,
                output_t: output,
                stats_t: stats,
            },
            workspace,
        )
        self.output = output
        self.stats = stats
        return output

    def backward(self, grad: Tensor) -> Tuple[Array, Array, Array]:
        query, key, value = (tensor.data for tensor in self.tensors)
        graph, tensors = _cudnn_sdpa_backward_graph(
            self.shape,
            self.input_stride,
            self.output_stride,
            self.data_type_name,
        )
        (
            query_t,
            key_t,
            value_t,
            output_t,
            grad_output_t,
            stats_t,
            grad_query_t,
            grad_key_t,
            grad_value_t,
        ) = tensors
        grad_query = xp.empty_like(query)
        grad_key = xp.empty_like(key)
        grad_value = xp.empty_like(value)
        grad_output = _cupy_as_bhtd_with_bthd_storage(
            grad.data.astype(query.dtype, copy=False),
            self.shape,
        )
        workspace = xp.empty((graph.get_workspace_size(),), dtype=xp.uint8)
        graph.execute(
            {
                query_t: query,
                key_t: key,
                value_t: value,
                output_t: self.output,
                grad_output_t: grad_output,
                stats_t: self.stats,
                grad_query_t: grad_query,
                grad_key_t: grad_key,
                grad_value_t: grad_value,
            },
            workspace,
        )
        self.output = None
        self.stats = None
        return grad_query, grad_key, grad_value


class Relu(Function):
    r"""
    Rectified Linear Unit (ReLU) activation function.

    The ReLU function is defined as:
        $$
        ReLU(x) = \max(0, x)
        $$

    Note:
        This class is used internally. For applying ReLU, use the `relu` function.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([-3, 0, 3])
        >>> y = Relu.apply(x) # Expected output: [0, 0, 3]
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the ReLU activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The result of applying ReLU to the input.
        """
        self.x = x
        return xp.maximum(x, 0)

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the ReLU activation function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        cupy_grad = _cupy_relu_backward(grad.data, self.x)
        if cupy_grad is not None:
            return cupy_grad
        return grad.data * (self.x > 0)


class LinearAffine(Function):
    def forward(self, x: Array, weight: Array, bias: Array) -> Array:
        if weight.ndim != 2:
            raise ValueError("linear requires a 2D weight matrix")
        if bias.shape != (weight.shape[-1],):
            raise ValueError("linear bias must match the output dimension")
        if x.shape[-1] != weight.shape[0]:
            raise ValueError("linear input dimension must match weight")

        self.x_shape = x.shape
        self.bias_shape = bias.shape

        x_2d = x.reshape(-1, x.shape[-1])
        cols = int(weight.shape[1])
        out_2d = _matmul_autocast(x_2d, weight, bias, with_relu=False)
        out = out_2d.reshape(*x.shape[:-1], cols)
        self.out_shape = out.shape
        return out

    def backward(
        self,
        grad: Tensor,
    ) -> Tuple[Optional[Array], Optional[Array], Optional[Array]]:
        x, weight, bias = self.tensors
        grad_data = Function.unbroadcast(grad.data, self.out_shape)
        grad_2d = grad_data.reshape(-1, grad_data.shape[-1])
        x_2d = x.data.reshape(-1, x.data.shape[-1])
        grad_x = grad_weight = grad_bias = None

        # dX = grad @ W^T. Autocast handles bf16/fp32 mixed precision in one place.
        if x.requires_grad:
            grad_x_2d = _matmul_autocast(grad_2d, xp.swapaxes(weight.data, -1, -2))
            grad_x = grad_x_2d.reshape(self.x_shape)

        # dW = X^T @ grad fused with dbias = column_sum(grad). Pass the weight
        # dtype so the helper can match the forward GEMM precision (bf16 with
        # fp32 accumulator) even when both x and grad arrive as fp32 — without
        # the hint, the dtype-pair policy would pick a slow pure-fp32 GEMM.
        if (
            weight.requires_grad
            and bias.requires_grad
            and grad_data.shape == self.out_shape  # no broadcast on incoming grad
        ):
            grad_weight, grad_bias = _matmul_autocast_dW_bgrad(
                x_2d, grad_2d, param_dtype=weight.data.dtype
            )
        else:
            if weight.requires_grad:
                grad_weight = _matmul_autocast(xp.swapaxes(x_2d, -1, -2), grad_2d)
            if bias.requires_grad:
                grad_bias = Function.unbroadcast(grad_data, self.bias_shape)

        return grad_x, grad_weight, grad_bias


class LinearRelu(Function):
    def forward(self, x: Array, weight: Array, bias: Array) -> Array:
        if weight.ndim != 2:
            raise ValueError("linear_relu requires a 2D weight matrix")
        if bias.shape != (weight.shape[-1],):
            raise ValueError("linear_relu bias must match the output dimension")
        if x.shape[-1] != weight.shape[0]:
            raise ValueError("linear_relu input dimension must match weight")

        self.x_shape = x.shape
        self.bias_shape = bias.shape

        x_2d = x.reshape(-1, x.shape[-1])
        cols = int(weight.shape[1])
        out_2d = _matmul_autocast(x_2d, weight, bias, with_relu=True)
        out = out_2d.reshape(*x.shape[:-1], cols)
        self.out_shape = out.shape
        self.out = out
        return out

    def backward(
        self,
        grad: Tensor,
    ) -> Tuple[Optional[Array], Optional[Array], Optional[Array]]:
        x, weight, bias = self.tensors
        grad_data = Function.unbroadcast(grad.data, self.out_shape)
        grad_act = _cupy_linear_relu_backward(grad_data, self.out)
        if grad_act is None:
            grad_act = xp.where(self.out > 0, grad_data, 0)

        grad_act_2d = grad_act.reshape(-1, grad_act.shape[-1])
        x_2d = x.data.reshape(-1, x.data.shape[-1])
        grad_x = grad_weight = grad_bias = None

        if x.requires_grad:
            grad_x_2d = _matmul_autocast(grad_act_2d, xp.swapaxes(weight.data, -1, -2))
            grad_x = grad_x_2d.reshape(self.x_shape)

        if weight.requires_grad and bias.requires_grad:
            grad_weight, grad_bias = _matmul_autocast_dW_bgrad(
                x_2d, grad_act_2d, param_dtype=weight.data.dtype
            )
        else:
            if weight.requires_grad:
                grad_weight = _matmul_autocast(xp.swapaxes(x_2d, -1, -2), grad_act_2d)
            if bias.requires_grad:
                grad_bias = Function.unbroadcast(grad_act, self.bias_shape)

        return grad_x, grad_weight, grad_bias


class Gelu(Function):
    r"""
    Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = x * P(X \le x) where X ~ N(0, 1)

    This activation function approximates:
    $$
    0.5 * x * \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \, (x + 0.044715*x^3)\right)\right)
    $$

    Paper: https://arxiv.org/abs/1606.08415

    Note:
        Use the `gelu` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0.5, -0.5])
        >>> y = Gelu.apply(x) # Expected output: approximate GELU values
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the GELU activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The output array after applying GELU.
        """
        self.x = x  # Save for backward
        coeff = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + xp.tanh(coeff * (x + 0.044715 * x**3)))

    def backward(self, grad: Array) -> Array:
        r"""
        Computes the backward pass of the GELU activation function.

        The gradient is computed as:
        $$
        \frac{d\,GELU}{dx} = 0.5 \left(1 + tanh(\alpha)\right) + 0.5 \, x \, \left(1 - tanh^2(\alpha)\right) \alpha'
        $$
        where
        $$
        \alpha = \sqrt{\frac{2}{\pi}} (x + 0.044715*x^3)
        $$
        and
        $$
        \alpha' = \sqrt{\frac{2}{\pi}} \left(1 + 3*0.044715*x^2\right)
        $$

        Args:
            grad (xp.ndarray): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        coeff = math.sqrt(2.0 / math.pi)
        alpha = coeff * (self.x + 0.044715 * self.x**3)

        # Compute tanh(alpha)
        tanh_alpha = xp.tanh(alpha)

        # Compute derivative of alpha: alpha'(x)
        alpha_prime = coeff * (1.0 + 3.0 * 0.044715 * self.x**2)

        # Derivative of GELU:
        dgelu_dx = (
            0.5 * (1.0 + tanh_alpha)
            + 0.5 * self.x * (1.0 - tanh_alpha**2) * alpha_prime
        )

        # Chain rule: dL/dx = dL/dy * dy/dx
        return grad * dgelu_dx


class Sigmoid(Function):
    r"""
    Sigmoid activation function.

    The sigmoid function is defined as:
        $$
        sigmoid(x) = \frac{1}{1 + e^{-x}}
        $$

    Note:
        Use the `sigmoid` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 2])
        >>> y = Sigmoid.apply(x) # Expected output: [0.5, ~0.88]
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the sigmoid function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The output after applying the sigmoid function.
        """
        self.out = 1 / (1 + xp.exp(xp.clip(-x, -709, 709)))
        return self.out

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the sigmoid function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * self.out * (1 - self.out)


class Softmax(Function):
    r"""
    Softmax activation function.

    The softmax function is defined as:
        $$
        softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
        $$

    Note:
        Use the `softmax` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> import cupy as np
        >>> x = Tensor(xp.array([1.0, 2.0, 3.0]))
        >>> y = Softmax.apply(x) # Expected output: probabilities that sum to 1
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the softmax activation function.

        Args:
            x (xp.ndarray): Input array of logits.

        Returns:
            xp.ndarray: The softmax probabilities.
        """
        cupy_probs = _cupy_softmax_forward(x)
        if cupy_probs is not None:
            self.probs = cupy_probs
            return self.probs

        exp_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
        self.probs = exp_x / xp.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the softmax activation function.

        This function computes the gradient of the softmax output with respect to the input logits.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input logits.
        """
        # There are two cases for this gradient because each element in the matrix affects
        # every other elements' gradient due to the fact of sum(e^x) in the denominator.
        # Let's denote i, j as the ith and jth elements in the matrix.
        # Case 1: i == j
        # d(softmax(x))/dx_i = softmax(x)_i * (1[i==j] - softmax(x)_i)
        # Case 2: i != j
        # d(softmax(x))/dx_i = -softmax(x)_i * softmax(x)_j

        # dL/dx = y * (dL/dy - sum(dL/dy * y, axis=-1, keepdims=True))
        cupy_grad = _cupy_softmax_backward(self.probs, grad.data)
        if cupy_grad is not None:
            return cupy_grad

        sum_term = xp.sum(grad.data * self.probs, axis=-1, keepdims=True)
        dLdx = self.probs * (grad.data - sum_term)
        return dLdx


class CausalSoftmax(Function):
    def forward(self, x: Array) -> Array:
        cupy_probs = _cupy_causal_softmax_forward(x)
        if cupy_probs is not None:
            self.probs = cupy_probs
            return self.probs

        seq_q = int(x.shape[-2])
        seq_k = int(x.shape[-1])
        mask = xp.triu(xp.ones((seq_q, seq_k), dtype=x.dtype), k=1).reshape(
            (1,) * (x.ndim - 2) + (seq_q, seq_k)
        )
        masked_x = x + (mask * -1e9)
        exp_x = xp.exp(masked_x - xp.max(masked_x, axis=-1, keepdims=True))
        self.probs = exp_x / xp.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad: Tensor) -> Array:
        cupy_grad = _cupy_causal_softmax_backward(self.probs, grad.data)
        if cupy_grad is not None:
            return cupy_grad

        sum_term = xp.sum(grad.data * self.probs, axis=-1, keepdims=True)
        return self.probs * (grad.data - sum_term)


class LayerNormAffine(Function):
    def forward(
        self,
        x: Array,
        gain: Array,
        bias: Array,
        epsilon: float,
    ) -> Array:
        is_bfloat16 = bool(
            IS_CUPY
            and hasattr(xp, "bfloat16")
            and x.dtype == xp.bfloat16
            and gain.dtype == xp.bfloat16
            and bias.dtype == xp.bfloat16
            and x.flags.c_contiguous
        )
        if (
            not (_is_cupy_float32_contiguous(x) or is_bfloat16)
            or not IS_CUPY
            or gain.dtype not in (xp.float32, *LOW_PRECISION_FLOAT_DTYPES)
            or bias.dtype not in (xp.float32, *LOW_PRECISION_FLOAT_DTYPES)
            or not gain.flags.c_contiguous
            or not bias.flags.c_contiguous
            or x.ndim < 1
            or gain.ndim != 1
            or bias.ndim != 1
            or x.shape[-1] != gain.shape[0]
            or gain.shape != bias.shape
        ):
            raise ValueError(
                "LayerNormAffine requires contiguous CuPy float32 or bf16 inputs"
            )

        self.original_shape = x.shape
        self.rows = int(x.size // x.shape[-1])
        self.cols = int(x.shape[-1])
        self.is_bfloat16 = is_bfloat16
        y = xp.empty_like(x)
        self.x_hat = xp.empty_like(x)
        self.rstd = xp.empty((self.rows,), dtype=xp.float32)
        threads = 128
        if self.is_bfloat16:
            self.gain = gain
            self.x_hat = xp.empty(x.shape, dtype=xp.float32)
            _, _, _, forward_kernel, _, _, _, _ = _cupy_layer_norm_kernels()
            kernel_gain = gain
            kernel_bias = bias
        else:
            self.gain = gain if gain.dtype == xp.float32 else gain.astype(xp.float32)
            bias32 = bias if bias.dtype == xp.float32 else bias.astype(xp.float32)
            forward_kernel, _, _, _, _, _, _, _ = _cupy_layer_norm_kernels()
            kernel_gain = self.gain
            kernel_bias = bias32
        forward_kernel(
            _cupy_row_grid(self.rows),
            (threads,),
            (
                x.reshape(self.rows, self.cols),
                kernel_gain,
                kernel_bias,
                y.reshape(self.rows, self.cols),
                self.x_hat.reshape(self.rows, self.cols),
                self.rstd,
                self.rows,
                self.cols,
                float(epsilon),
            ),
            shared_mem=threads * 4,
        )
        return y

    def backward(self, grad: Tensor) -> Tuple[Array, Array, Array]:
        grad_data = grad.data
        if self.is_bfloat16:
            if grad_data.dtype != xp.bfloat16:
                grad_data = grad_data.astype(xp.bfloat16)
        elif grad_data.dtype != xp.float32:
            grad_data = grad_data.astype(xp.float32)
        grad_2d = xp.ascontiguousarray(grad_data.reshape(self.rows, self.cols))
        x_hat_2d = self.x_hat.reshape(self.rows, self.cols)
        dx = xp.empty_like(grad_2d)
        d_gain = xp.empty((self.cols,), dtype=xp.float32)
        d_bias = xp.empty((self.cols,), dtype=xp.float32)
        backward_x_threads = 128
        if self.is_bfloat16:
            (
                _,
                _,
                _,
                _,
                backward_x_kernel,
                _,
                backward_param_partial_kernel,
                backward_param_finalize_kernel,
            ) = _cupy_layer_norm_kernels()
            row_tiles = (self.rows + 255) // 256
            partial_gain = xp.empty((row_tiles, self.cols), dtype=xp.float32)
            partial_bias = xp.empty((row_tiles, self.cols), dtype=xp.float32)
            backward_param_partial_kernel(
                ((self.cols + 7) // 8, row_tiles),
                (256,),
                (
                    grad_2d,
                    x_hat_2d,
                    partial_gain,
                    partial_bias,
                    self.rows,
                    self.cols,
                    row_tiles,
                ),
            )
            backward_param_finalize_kernel(
                (self.cols,),
                (256,),
                (partial_gain, partial_bias, d_gain, d_bias, self.cols, row_tiles),
                shared_mem=256 * 2 * 4,
            )
        else:
            backward_param_threads = 1024
            _, backward_x_kernel, backward_param_kernel, _, _, _, _, _ = (
                _cupy_layer_norm_kernels()
            )
            backward_param_kernel(
                (self.cols,),
                (backward_param_threads,),
                (grad_2d, x_hat_2d, d_gain, d_bias, self.rows, self.cols),
                shared_mem=backward_param_threads * 2 * 4,
            )
        backward_x_kernel(
            _cupy_row_grid(self.rows),
            (backward_x_threads,),
            (grad_2d, x_hat_2d, self.rstd, self.gain, dx, self.rows, self.cols),
            shared_mem=backward_x_threads * 2 * 4,
        )
        return dx.reshape(self.original_shape), d_gain, d_bias


class LogSoftmax(Function):
    r"""
    Log-softmax activation function.

    The log-softmax function is defined as:
        $$
        \log \pi_i = x_i - \log \sum_j e^{x_j}
        $$

    Note:
        Use `log_softmax` when logprobs are needed; it is more stable than
        computing `log(softmax(x))`.
    """

    def forward(self, x: Array, *, dim: int = -1) -> Array:
        r"""
        Computes the forward pass of the log-softmax activation function.

        Args:
            x (xp.ndarray): Input array of logits.
            dim (int): Dimension along which logprobs are computed.

        Returns:
            xp.ndarray: The logprobs.
        """
        r"""
        Shift logits by their maximum along the class/token dimension:
        $$
        m = \max_j x_j,\qquad \text{max\_shifted\_logits}_i = x_i - m
        $$
        This leaves softmax probabilities unchanged because:
        $$
        \frac{e^{x_i}}{\sum_j e^{x_j}}
        =
        \frac{e^{x_i-c}}{\sum_j e^{x_j-c}}
        $$
        while keeping exponentials numerically bounded.
        """
        max_shifted_logits = x - xp.max(x, axis=dim, keepdims=True)

        r"""
        Compute the log of the sum of exponentiated max-shifted logits:
        $$
        \log \operatorname{softmax}(x)_i
        = x_i - \log \sum_j e^{x_j}
        $$
        $$
        = x_i - \log\left(e^m \sum_j e^{x_j-m}\right)
        = (x_i-m) - \log \sum_j e^{x_j-m}
        $$

        Since:

        $\text{max\_shifted\_logits}_i = x_i-m$

        $$
        \log \operatorname{softmax}(x)_i = \text{max\_shifted\_logits}_i - \log \sum_j e^{\text{max\_shifted\_logits}_j}
        $$

        """
        log_sum_exp_max_shifted_logits = xp.log(
            xp.sum(xp.exp(max_shifted_logits), axis=dim, keepdims=True)
        )

        self.dim = dim
        r"""
        Store probabilities for backward:
        $$
        p_i = \operatorname{softmax}(x)_i
        = \exp\left(
            \text{max\_shifted\_logits}_i
            - \log \sum_j e^{\text{max\_shifted\_logits}_j}
        \right)
        $$
        """
        self.probs = xp.exp(max_shifted_logits - log_sum_exp_max_shifted_logits)
        return max_shifted_logits - log_sum_exp_max_shifted_logits

    def backward(self, grad: Tensor) -> Array:
        r"""
        Computes the backward pass of the log-softmax activation function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input logits.
        """
        r"""
        For:
        $$
        y_i = \log \operatorname{softmax}(x)_i = x_i - \log \sum_k e^{x_k}
        $$
        the Jacobian is:
        $$
        \frac{\partial y_i}{\partial x_j}
        = \mathbf{1}[i=j] - p_j
        $$
        Given upstream gradient $g_i = \frac{\partial L}{\partial y_i}$:
        $$
        \frac{\partial L}{\partial x_j}
        = \sum_i g_i(\mathbf{1}[i=j] - p_j)
        = g_j - p_j \sum_i g_i
        $$
        """
        return grad.data - self.probs * xp.sum(
            grad.data,
            axis=self.dim,
            keepdims=True,
        )


class Tanh(Function):
    r"""
    Hyperbolic tangent (tanh) activation function.

    The tanh function is defined as:
        $$
        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
        $$

    Note:
        Use the `tanh` function to apply this activation.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> x = Tensor([0, 1])
        >>> y = Tanh.apply(x) # Expected output: [0, tanh(1)]
    """

    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the tanh activation function.

        Args:
            x (xp.ndarray): Input array.

        Returns:
            xp.ndarray: The output after applying the tanh function.
        """
        # For numerical stability, use the fact that tanh(x) = 2*sigmoid(2x) - 1
        # This avoids computing large exponentials directly
        x = 2 * x
        # Clip x to avoid overflow in exp(-x)
        x = xp.clip(x, -88.72, 88.72)  # ln(max float32) ≈ 88.72
        sigmoid_2x = 1 / (1 + xp.exp(-x))
        self.out = 2 * sigmoid_2x - 1
        return self.out

    def backward(self, grad: Tensor) -> Array:
        """
        Computes the backward pass of the tanh activation function.
        $$
        d(tanh(x))/dx = 1 - tanh(x)^2
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient of the loss with respect to the input.
        """
        return grad.data * (1 - self.out**2)


class ScaledDotProductAttentionMLXCustom(Function):
    _THREADGROUP = (256, 1, 1)

    @staticmethod
    def _causal_mask(seq_len: int) -> Array:
        # Mask contract: 1.0 means forbidden and 0.0 means allowed.
        return xp.triu(xp.ones((seq_len, seq_len), dtype=xp.float32), k=1).reshape(
            1, 1, seq_len, seq_len
        )

    @staticmethod
    def build_dropout_scale_mask(
        *,
        is_training: Optional[bool],
        dropout_prob: float,
        query_shape: Tuple[int, ...],
        key_shape: Tuple[int, ...],
    ) -> Optional[Array]:
        if not is_training or dropout_prob == 0:
            return None

        dropout_shape = (
            int(query_shape[0]),
            int(query_shape[1]),
            int(query_shape[2]),
            int(key_shape[2]),
        )
        keep_prob = float(1 - dropout_prob)
        if keep_prob <= 0:
            return xp.zeros(dropout_shape, dtype=xp.float32)

        keep_mask = xp.random.bernoulli(keep_prob, shape=dropout_shape)
        return xp.array(keep_mask, dtype=xp.float32) / keep_prob

    @classmethod
    def _causal_attention_probs(
        cls,
        query: Array,
        key: Array,
    ) -> Tuple[Array, float]:
        # Recompute the dense causal probabilities in Python so backward stays
        # aligned with the contract without saving a dense forward tensor.
        scale = float(query.shape[-1]) ** -0.5
        seq_len = int(query.shape[-2])

        r"""
        Transpose keys from ``(..., S_k, D)`` to ``(..., D, S_k)`` for ``QK^T``.
        This helper recomputes the causal probability tensor, not the final
        attention output:
        $$
        \text{attention\_scores} = \frac{QK^T}{\sqrt{key\_dim}}
        $$
        $$
        \text{masked\_scores} =
        \text{attention\_scores} + \text{causal\_mask} \cdot (-10^9)
        $$
        $$
        P = \operatorname{softmax}(\text{masked\_scores})
        $$
        """
        attention_scores = (query @ key.transpose(0, 1, 3, 2)) * scale
        masked_scores = attention_scores + cls._causal_mask(seq_len) * -1e9
        exp_scores = xp.exp(
            masked_scores - xp.max(masked_scores, axis=-1, keepdims=True)
        )
        probs = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True)
        return probs, scale

    # Cache the compiled Metal kernels so the causal fast path pays the JIT
    # cost once per specialization instead of once per forward call.
    @staticmethod
    # `maxsize=2` keeps both stable variants cached: with and without dropout.
    @lru_cache(maxsize=2)
    def _kernel(use_dropout_scale_mask: bool) -> Any:
        r"""
        Fused causal attention kernel with optional repo-owned post-softmax
        dropout.

        When `use_dropout_scale_mask` is false, the kernel computes:
        $$
        O_{b,h,q,d}
        =
        \sum_{k=0}^{q} p_{b,h,q,k} V_{b,h,k,d}
        $$

        When `use_dropout_scale_mask` is true, the Python caller has already
        sampled the repo's Bernoulli dropout mask and scaled it by
        ``1 / (1 - p)``, so the kernel computes:
        $$
        O_{b,h,q,d}
        =
        \sum_{k=0}^{q} p_{b,h,q,k} m_{b,h,q,k} V_{b,h,k,d}
        $$

        In both cases one thread computes one output element and recomputes the
        causal score row twice: once to find the stable softmax maximum, then a
        second time to accumulate both the softmax denominator and the output
        numerator without materializing the dense score matrix.
        """
        try:
            import mlx.core.fast as mx_fast  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError("mlx_custom requires the MLX backend") from exc

        # MLX kernel signatures are fixed at compile time, so dropout support
        # needs its own cached specialization with one extra input.
        kernel_name = "scaled_dot_product_attention_mlx_custom"
        input_names = ["query", "key", "value"]
        dropout_setup = ""
        output_init = "float numerator = 0.0f;"
        output_update = (
            "numerator += attention_weight * value[key_row_offset + data_idx];"
        )
        if use_dropout_scale_mask:
            kernel_name = "scaled_dot_product_attention_mlx_custom_dropout"
            input_names.append("dropout_scale_mask")
            dropout_setup = """
                // Row-major offset into the [B, H, S_q, S_k] dropout mask.
                int dropout_row_offset =
                    ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * seq_len;
            """
            output_update = """
                float dropout_scale = dropout_scale_mask[dropout_row_offset + key_idx];
                numerator +=
                    attention_weight * dropout_scale * value[key_row_offset + data_idx];
            """
        scaled_dot_product = """
                    float scaled_dot_product = 0.0f;
                    for (int data_offset = 0; data_offset < head_dim; ++data_offset) {
                        scaled_dot_product +=
                            query[query_row_offset + data_offset]
                            * key[key_row_offset + data_offset];
                    }
                    scaled_dot_product *= scale;
        """

        return mx_fast.metal_kernel(
            name=kernel_name,
            input_names=input_names,
            output_names=["out"],
            source=f"""
                // One thread computes one output element O[b, h, q, d].
                uint flat_idx = thread_position_in_grid.x;
                int head_dim = query_shape[3];
                int seq_len = query_shape[2];
                int num_heads = query_shape[1];

                // Map the flat output index back to (batch, head, query position, feature).
                int data_idx = flat_idx % head_dim;
                int query_idx = (flat_idx / head_dim) % seq_len;
                int head_idx = (flat_idx / (head_dim * seq_len)) % num_heads;
                int batch_idx = flat_idx / (head_dim * seq_len * num_heads);

                // Compute row offsets inside the contiguous [B, H, S, D] layout.
                int head_stride = seq_len * head_dim;
                int batch_stride = num_heads * head_stride;
                int head_sequence_offset =
                    batch_idx * batch_stride + head_idx * head_stride;
                int query_row_offset = head_sequence_offset + query_idx * head_dim;
                // Causal masking allows exactly query_idx + 1 visible keys.
                int causal_keys = query_idx + 1;
                {dropout_setup}

                // Equivalent to 1 / sqrt(D).
                float scale = metal::rsqrt(float(head_dim));

                // Pass 1: find the maximum visible logit for stable softmax.
                float max_score = -INFINITY;
                for (int key_idx = 0; key_idx < causal_keys; ++key_idx) {{
                    int key_row_offset = head_sequence_offset + key_idx * head_dim;
                    {scaled_dot_product}
                    max_score = metal::max(max_score, scaled_dot_product);
                }}

                // Pass 2: accumulate the unnormalized softmax denominator and
                // the output numerator in the same loop.
                float softmax_denominator = 0.0f;
                {output_init}
                for (int key_idx = 0; key_idx < causal_keys; ++key_idx) {{
                    int key_row_offset = head_sequence_offset + key_idx * head_dim;
                    {scaled_dot_product}
                    float attention_weight = metal::exp(scaled_dot_product - max_score);
                    softmax_denominator += attention_weight;
                    {output_update}
                }}

                out[flat_idx] = numerator / softmax_denominator;
            """,
        )

    @staticmethod
    def validate(
        query: Array,
        key: Array,
        value: Array,
        *,
        dropout_scale_mask: Optional[Array] = None,
    ) -> None:
        from autograd.backend import NAME

        if NAME != "mlx":
            raise RuntimeError("mlx_custom requires the MLX backend")

        tensors = (query, key, value)
        if any(len(tensor.shape) != 4 for tensor in tensors):
            raise ValueError(
                "mlx_custom expects query, key, and value to be 4D tensors"
            )
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "mlx_custom requires query, key, and value to share the same shape"
            )
        if any(tensor.dtype != xp.float32 for tensor in tensors):
            raise ValueError("mlx_custom requires float32 query, key, and value")
        if dropout_scale_mask is not None:
            expected_dropout_shape = (
                int(query.shape[0]),
                int(query.shape[1]),
                int(query.shape[2]),
                int(key.shape[2]),
            )
            if dropout_scale_mask.shape != expected_dropout_shape:
                raise ValueError(
                    "mlx_custom dropout mask must match attention probability shape"
                )
            if dropout_scale_mask.dtype != xp.float32:
                raise ValueError("mlx_custom dropout mask must be float32")

    def forward(
        self,
        query: Array,
        key: Array,
        value: Array,
        *,
        dropout_scale_mask: Optional[Array] = None,
    ) -> Array:
        self.validate(
            query,
            key,
            value,
            dropout_scale_mask=dropout_scale_mask,
        )
        self.dropout_scale_mask = dropout_scale_mask

        kernel = self._kernel(dropout_scale_mask is not None)
        inputs = [query, key, value]
        if dropout_scale_mask is not None:
            inputs.append(dropout_scale_mask)
        return kernel(
            inputs=inputs,
            grid=(int(query.size), 1, 1),
            threadgroup=self._THREADGROUP,
            output_shapes=[query.shape],
            output_dtypes=[query.dtype],
        )[0]

    def backward(self, grad: Tensor) -> Tuple[Array, Array, Array]:
        r"""
        The reason we have to define this backward function explicitly is because the kernel's forward function is defined not with our Tensor ops, but with a lower-level kernel ops.
        That's why we don't get the backward implementation for free like other Function classes.

        Backpropagate through the fused causal attention path.

        The no-dropout forward contract is:

        $$
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{key\_dim}}\right) V
        $$

        Train mode keeps the repo's post-softmax dropout contract:
        $$
        \widetilde{P} = P \odot M
        $$
        $$
        \text{Attention}_{\text{dropout}}(Q, K, V) = \widetilde{P}V
        $$
        where $M$ is the already-scaled Bernoulli mask sampled in Python.

        We still recompute the causal softmax probabilities inside backward
        instead of saving dense logits/probabilities from the fused forward
        kernel. We will use these intermediate names:
        $$
        \text{attention\_scores} = \frac{QK^T}{\sqrt{key\_dim}}
        $$
        $$
        P = \operatorname{softmax}(\text{attention\_scores}_{\text{causal}})
        $$
        $$
        \widetilde{P} =
        \begin{cases}
        P & \text{if dropout is disabled} \\
        P \odot M & \text{if dropout is enabled}
        \end{cases}
        $$
        $$
        \text{attention\_output} = \widetilde{P}V
        $$

        Let `loss` be the final scalar training objective, and let the upstream
        gradient be:
        $$
        \text{upstream\_grad} = \frac{\partial \text{loss}}{\partial \text{attention\_output}}
        $$

        1. Derivative of `loss` with respect to `V`:
        $$
        \frac{\partial \text{loss}}{\partial V}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_output}}
        \frac{\partial \text{attention\_output}}{\partial V}
        =
        \widetilde{P}^T \, \text{upstream\_grad}
        $$

        2. Derivative of `loss` with respect to the pre-dropout probabilities:
        $$
        \frac{\partial \text{loss}}{\partial P}
        =
        \left(\text{upstream\_grad} \, V^T\right)
        \odot
        \begin{cases}
        1 & \text{if dropout is disabled} \\
        M & \text{if dropout is enabled}
        \end{cases}
        $$

        3. Derivative of `loss` with respect to `attention_scores`:
        Softmax is applied independently to each query row, so we apply the
        softmax derivative row by row along the key dimension.
        $$
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        =
        \frac{\partial \text{loss}}{\partial P}
        \frac{\partial P}{\partial \text{attention\_scores}}
        =
        P \odot
        \left(
        \frac{\partial \text{loss}}{\partial P}
        -
        \sum_k
        \left(
        \frac{\partial \text{loss}}{\partial P_k}
        P_k
        \right)
        \right)
        $$

        4. Derivative of `loss` with respect to `Q` and `K`, using
        $$
        \text{attention\_scores} = \frac{QK^T}{\sqrt{key\_dim}}
        $$
        $$
        \frac{\partial \text{loss}}{\partial Q}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        \frac{\partial \text{attention\_scores}}{\partial Q}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        K \cdot \frac{1}{\sqrt{key\_dim}}
        $$
        $$
        \frac{\partial \text{loss}}{\partial K}
        =
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        \frac{\partial \text{attention\_scores}}{\partial K}
        =
        \left(
        \frac{\partial \text{loss}}{\partial \text{attention\_scores}}
        \right)^T
        Q \cdot \frac{1}{\sqrt{key\_dim}}
        $$
        """
        query, key, value = (tensor.data for tensor in self.tensors)
        probs, scale = self._causal_attention_probs(query, key)

        upstream_grad = grad.data

        dropout_scale_mask = getattr(self, "dropout_scale_mask", None)
        if dropout_scale_mask is not None:
            # Train mode applies dropout after softmax, so use the scaled mask
            # on the probability tensor before transposing to (..., S_k, S_q).
            grad_value = (probs * dropout_scale_mask).transpose(
                0, 1, 3, 2
            ) @ upstream_grad
        else:
            # Eval mode uses the plain causal probabilities. Transpose from
            # (..., S_q, S_k) to (..., S_k, S_q) before multiplying by dL/dO.
            grad_value = probs.transpose(0, 1, 3, 2) @ upstream_grad

        # By the chain rule, the derivative of loss with respect to the
        # pre-dropout attention weights is:
        # d(loss)/d(attention_weights) =
        # d(loss)/d(effective_attention_weights) * dropout_scale_mask
        # Transpose values from (..., S_k, D) to (..., D, S_k).
        grad_probs = upstream_grad @ value.transpose(0, 1, 3, 2)
        if dropout_scale_mask is not None:
            grad_probs = grad_probs * dropout_scale_mask

        # By the chain rule, the derivative of loss with respect to
        # attention_scores comes from the derivative of the softmax output with
        # respect to its input scores.
        # softmax acts on each query row independently, so this reduction is
        # computed row by row along the key dimension.
        grad_scores = probs * (
            grad_probs - xp.sum(grad_probs * probs, axis=-1, keepdims=True)
        )

        # attention_scores = QK^T / sqrt(key_dim)
        # By the chain rule, propagate d(loss)/d(attention_scores) into Q and K.
        grad_query = (grad_scores @ key) * scale
        # Transpose score gradients from (..., S_q, S_k) to (..., S_k, S_q).
        grad_key = (grad_scores.transpose(0, 1, 3, 2) @ query) * scale
        return grad_query, grad_key, grad_value


###################### Loss Functions #####################
class BinaryCrossEntropy(Function):
    r"""
    Binary Cross Entropy (BCE) Loss.

    This loss function assumes that $y_{pred}$ contains probabilities rather than logits.
    If the input is logits, use :func:`binary_cross_entropy_with_logits`.

    The loss is computed as:
    $$
    BCE = -\left( y_{true} \cdot \log(y_{pred}) + (1 - y_{true}) \cdot \log(1 - y_{pred}) \right)
    $$

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.9, 0.2, 0.1])
        >>> y_true = Tensor([1, 0, 0])
        >>> loss = BinaryCrossEntropy.apply(y_pred, y_true) # Expected output: a small loss value
    """

    def forward(self, y_pred: Array, y_true: Array, **kwargs: Any) -> Array:
        """
        Computes the binary cross entropy loss.

        Args:
            y_pred (xp.ndarray): Predicted probabilities.
            y_true (xp.ndarray): True binary labels.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed binary cross entropy loss.
        """
        y_true = xp.array(y_true, dtype=xp.float32)
        y_pred = xp.array(y_pred, dtype=xp.float32)

        # If labels come in as (batch_size,), explicitly reshaping them to (batch_size, 1) avoids shape mismatch, and certain elementwise operations will broadcast in unintended ways
        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred and y_true must have the same shape")

        self.y_true = y_true
        self.y_pred_prob = xp.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -xp.mean(
            y_true * xp.log(self.y_pred_prob)
            + (1 - y_true) * xp.log(1 - self.y_pred_prob)
        )
        return loss

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
        r"""
        Computes the gradient of the binary cross entropy loss with respect to $y_{pred}$.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = -\left(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}}\right)
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        # Avoid division by zero by clipping probabilities away from 0 and 1
        y_pred_prob = xp.clip(self.y_pred_prob, 1e-7, 1 - 1e-7)
        grad_y_pred = -(
            (self.y_true / y_pred_prob) - ((1 - self.y_true) / (1 - y_pred_prob))
        )
        grad_y_pred /= len(y_pred_prob)
        # Incorporate the upstream gradient
        grad_y_pred *= grad.data

        return grad_y_pred, None  # y_true doesn't need gradient


class BinaryCrossEntropyWithLogits(Function):
    """
    Binary Cross Entropy Loss with logits.

    This implementation is numerically stable for logits input.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([2.0, -1.0, -2.0])  # logits
        >>> y_true = Tensor([1, 0, 0])
        >>> loss = BinaryCrossEntropyWithLogits.apply(y_pred, y_true) # Expected output: a loss value computed using logits
    """

    def forward(self, y_pred: Array, y_true: Array) -> Array:
        """
        Computes the binary cross entropy loss with logits input.

        Args:
            y_pred (xp.ndarray): shape (N, ...) Unbounded real-valued logits.
            y_true (xp.ndarray): True binary labels (0 or 1), same shape as y_pred.

        Returns:
            float: The computed binary cross entropy loss.
        """
        y_true = xp.array(y_true, dtype=xp.float32)
        y_pred = xp.array(y_pred, dtype=xp.float32)

        # If labels come in as (batch_size,), explicitly reshaping them to (batch_size, 1) avoids shape mismatch, and certain elementwise operations will broadcast in unintended ways
        if y_true.ndim == 1 and y_pred.ndim == 1:
            pass
        elif y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape.")

        # compute loss
        # loss_i = max(y_pred, 0) - y_pred * y_true + log(1 + exp(-|y_pred|))
        loss = xp.mean(
            xp.maximum(y_pred, 0.0)
            - y_pred * y_true
            + xp.log1p(xp.exp(-xp.abs(y_pred)))
        )

        self.y_pred = y_pred
        self.y_true = y_true
        return loss

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
        r"""
        Computes the gradient of the binary cross entropy loss with logits with respect to $y_{pred}$.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = sigmoid(y_{pred}) - y_{true}
        = -(\frac{y_{true}}{y_{pred}} - \frac{1-y_{true}}{1-y_{pred}})
        $$
        Where
        $$
        sigmoid = 1 / (1 + exp(-y_{pred}))
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        # Stable sigmoid without NumPy-style masked mutation.
        sig = xp.where(
            self.y_pred >= 0,
            1.0 / (1.0 + xp.exp(-self.y_pred)),
            xp.exp(self.y_pred) / (1.0 + xp.exp(self.y_pred)),
        )

        # 2) Compute dL/dy_pred = sigmoid(y_pred) - y_true divided by batch_size
        grad_y_pred = (sig - self.y_true) / self.y_pred.shape[0]

        # 3) Multiply by upstream gradient
        grad_y_pred *= grad.data

        # we don't need a gradient for y_true (labels)
        return grad_y_pred, None


IGNORE_INDEX = -100


class CrossEntropy(Function):
    """
    Cross-Entropy Loss for multi-dimensional predictions with optional ignored targets and label smoothing.

    This function accepts raw logits (not probabilities) and computes a stable log-softmax internally.

    Examples:
        >>> import cupy as np
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor(xp.array([[2.0, 1.0, 0.1]]))
        >>> y_true = Tensor(xp.array([0]))
        >>> loss = CrossEntropy.apply(y_pred, y_true, ignore_index=-100, label_smoothing=0.1)
    """

    def forward(
        self,
        y_pred: Array,
        y_true: Array,
        ignore_index: int = IGNORE_INDEX,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        destructive_logits_backward: bool = False,
    ) -> Union[Array, float]:
        r"""
        Computes the cross-entropy loss with optional ignored targets and label smoothing.

        Args:
            y_pred (xp.ndarray): Raw logits. Shape can be $(batch\_size, feature\_dim)$ or $(batch\_size, seq\_len, feature\_dim)$.
            y_true (Union[xp.ndarray, Tensor]): True class indices. If $y_{pred}$ is 2D, shape is $(batch\_size,)$; if 3D, shape is $(batch\_size, seq\_len)$.
            ignore_index (int, optional): Target value to ignore in the loss and gradient.
                Defaults to -100.
            label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0. Label smoothing is applied if $label\_smoothing > 0$
            reduction (str, optional): Either ``"mean"`` or ``"sum"``. ``"mean"``
                divides by the total non-ignored target weight; ``"sum"`` returns
                the summed loss over non-ignored targets.
            For label smoothing, we follow the Inception paper notation.
            Here:
            - $x$ is the current training example
            - $y$ is the ground-truth class index for $x$
            - $k$ is a class index
            - $K$ is the total number of classes
            - $\delta_{k,y}$ is the Kronecker delta, equal to $1$ when $k=y$ and $0$ otherwise
            - $u(k)$ is a prior distribution over classes
            $$
            q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \epsilon\,u(k)
            $$
            and for the uniform prior used in the paper,
            $$
            u(k) = \frac{1}{K}
            $$
            so the smoothed target distribution becomes
            $$
            q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}
            $$
            In per-example / per-class notation, where $y_{i,j}$ is the one-hot
            target entry for example $i$ and class $j$, this is
            $$
            T_{i,j} = (1 - \epsilon)\,y_{i,j} + \frac{\epsilon}{K}
            $$
        (Ref: "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567)

        Returns:
            Union[xp.array, float]: The average cross-entropy loss over non-ignored positions.
        """

        y_true = xp.array(y_true, dtype=xp.int64)

        # 1. If 3D logits, flatten them for simpler processing while preserving
        # the original shape for the backward pass.
        self.original_shape = y_pred.shape
        if y_pred.ndim == 3:
            batch_size, seq_len, num_classes = y_pred.shape
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y_true = y_true.reshape(batch_size * seq_len)
        else:
            num_classes = y_pred.shape[-1]  # avoid returning batch_size to save memory

        # 2. Mark which target positions contribute to the loss.
        non_ignored_weights = (y_true != ignore_index).astype(xp.float32)
        safe_y_true = xp.where(non_ignored_weights > 0, y_true, 0)
        if reduction == "mean":
            non_ignored_count = xp.maximum(
                xp.sum(non_ignored_weights),
                xp.array(1.0, dtype=xp.float32),
            )
        elif reduction == "sum":
            non_ignored_count = xp.array(1.0, dtype=xp.float32)
        else:
            raise ValueError(f"Unsupported cross_entropy reduction: {reduction!r}")

        if label_smoothing == 0.0:
            cupy_bf16_losses = _cupy_cross_entropy_bf16_forward_loss(
                y_pred,
                safe_y_true,
                non_ignored_weights,
            )
            if cupy_bf16_losses is not None:
                loss_sum = xp.sum(cupy_bf16_losses)
                if reduction == "mean":
                    loss_val = loss_sum / non_ignored_count
                else:
                    loss_val = loss_sum
                self.logits = y_pred
                self.y_true = safe_y_true
                self.non_ignored_weights = non_ignored_weights
                self.non_ignored_count = non_ignored_count
                self.label_smoothing = label_smoothing
                self.num_classes = num_classes
                self._cupy_fast_path = False
                self._cupy_bf16_remat_path = True
                self._destructive_logits_backward = destructive_logits_backward
                return loss_val

            cupy_ce = _cupy_cross_entropy_forward(
                y_pred,
                safe_y_true,
                non_ignored_weights,
            )
            if cupy_ce is not None:
                probs, losses = cupy_ce
                loss_sum = xp.sum(losses)
                if reduction == "mean":
                    loss_val = loss_sum / non_ignored_count
                else:
                    loss_val = loss_sum
                self.probs = probs
                self.y_true = safe_y_true
                self.non_ignored_weights = non_ignored_weights
                self.non_ignored_count = non_ignored_count
                self.label_smoothing = label_smoothing
                self.num_classes = num_classes
                self._cupy_fast_path = True
                self._destructive_logits_backward = False
                return loss_val

        if y_pred.dtype in LOW_PRECISION_FLOAT_DTYPES:
            y_pred = y_pred.astype(xp.float32)

        r"""
        3. Shift the logits so the softmax-related terms stay numerically stable.

        This is algebraically the same cross-entropy as the standard
        log-softmax formulation; we are only changing how we materialize the
        intermediate tensors.

        Starting from
        $$
        \log \operatorname{softmax}(y_{\text{pred}})_{i,j}
        = y_{\text{pred}, i,j} - \log \sum_k \exp(y_{\text{pred}, i,k})
        $$
        define
        $$
        \operatorname{shifted}_{i,j}
        = y_{\text{pred}, i,j} - \max_k y_{\text{pred}, i,k}
        $$
        Then
        $$
        \exp(\operatorname{shifted}_{i,j})
        = \exp(y_{\text{pred}, i,j} - \max_k y_{\text{pred}, i,k})
        = \frac{\exp(y_{\text{pred}, i,j})}{\exp(\max_k y_{\text{pred}, i,k})}
        $$
        and because softmax is invariant to subtracting the same constant from
        every entry in a row,
        $$
        \log \operatorname{softmax}(y_{\text{pred}})_{i,j}
        = \operatorname{shifted}_{i,j}
        - \log \sum_k \exp(\operatorname{shifted}_{i,k})
        $$
        $$
        \operatorname{softmax}(y_{\text{pred}})_{i,j}
        = \frac{\exp(\operatorname{shifted}_{i,j})}
        {\sum_k \exp(\operatorname{shifted}_{i,k})}
        $$

        The largest value in each shifted row is 0, so the exponentials are
        numerically safe to compute. We store the probabilities
        $p_{i,j} = \operatorname{softmax}(y_{\text{pred}})_{i,j}$ for backward,
        and reconstruct only the specific log-softmax terms needed by the loss,
        rather than materializing the full log-softmax matrix.
        """
        shifted = y_pred - xp.max(y_pred, axis=1, keepdims=True)
        exp_shifted = xp.exp(shifted)
        denom = xp.sum(exp_shifted, axis=1, keepdims=True)
        probs = exp_shifted / denom
        # TODO: If CE memory becomes a bottleneck, consider rematerializing
        # softmax in backward instead of storing this full probability matrix.
        # denom has shape (N, 1); squeeze the singleton dimension so it can be
        # subtracted from the per-row terms below.
        log_denom = xp.log(denom[:, 0])

        r"""
        4. Compute the label-smoothed cross-entropy for each element i.
        $$
        q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}
        $$
        where $\delta_{k,y}$ is 1 for the correct class and 0 otherwise.
        so each row loss is
        $$
        L_i = -\sum_j q'_{i,j}\log p_{i,j}
        $$
        $$
        = -\left((1 - \epsilon)\log p_{i,y_i} + \frac{\epsilon}{K}\sum_j \log p_{i,j}\right)
        $$
        Here $p_{i,j}$ is the model probability for example i and class j.
        """
        # Since log_softmax[i, j] = shifted[i, j] - log_denom[i], we can compute
        # the two terms needed by the smoothed loss directly without
        # materializing the full log_softmax matrix.
        log_p_correct = shifted[xp.arange(len(y_true)), safe_y_true] - log_denom
        mean_log_p = xp.mean(shifted, axis=1) - log_denom
        losses = -(
            (1.0 - label_smoothing) * log_p_correct + label_smoothing * mean_log_p
        )

        # 5) Reduce the loss over non-ignored positions.
        loss_sum = xp.sum(losses * non_ignored_weights)
        if reduction == "mean":
            loss_val = loss_sum / non_ignored_count
        else:
            loss_val = loss_sum

        # 6) Store for backward pass:
        self.probs = probs
        self.y_true = safe_y_true
        self.non_ignored_weights = non_ignored_weights
        self.non_ignored_count = non_ignored_count
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self._cupy_fast_path = False
        self._destructive_logits_backward = False
        return loss_val

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
        r"""
        Computes the backward pass for the cross-entropy loss with label smoothing.

        The gradient with respect to the logits is given by:
        $$
        \frac{\partial L}{\partial logits_{i,j}} = p_{i,j} - T_{i,j}
        $$
        where $p_{i,j}$ is the stored model probability for example $i$ and class
        $j$, and the target distribution $T_{i,j}$ is defined as:
        $$
        T_{i,j} = (1 - \epsilon)\,y_{i,j} + \frac{\epsilon}{K}
        $$
        The gradient is zeroed out for ignored positions and scaled by the upstream gradient divided by the number of non-ignored positions.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to the logits and None for $y_{true}$.
        """
        if getattr(self, "_cupy_bf16_remat_path", False):
            cupy_grad = _cupy_cross_entropy_bf16_backward_from_logits(
                self.logits,
                self.y_true,
                self.non_ignored_weights,
                grad.data / self.non_ignored_count,
                out=(
                    self.logits
                    if getattr(self, "_destructive_logits_backward", False)
                    else None
                ),
            )
            if cupy_grad is not None:
                return cupy_grad.reshape(self.original_shape), None

        if getattr(self, "_cupy_fast_path", False):
            cupy_grad = _cupy_cross_entropy_backward(
                self.probs,
                self.y_true,
                self.non_ignored_weights,
                grad.data / self.non_ignored_count,
            )
            if cupy_grad is not None:
                return cupy_grad.reshape(self.original_shape), None

        # 1. Start from a dense copy of the probabilities.
        # We will transform this buffer into p_{i,j} - T_{i,j} in-place, which
        # avoids building a separate dense target tensor of the same shape.
        grad_out = self.probs * 1.0

        # 2. Expand the non-ignored mask so it can zero out whole rows in the
        # logits-shaped gradient matrix.
        row_mask = xp.expand_dims(self.non_ignored_weights, axis=1)

        r"""
        3. Subtract the label-smoothed target distribution.
        $$
        q'(k \mid x) = (1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}
        $$
        where $\delta_{k,y}$ is the Kronecker delta.

        Since grad_out starts as $p_{i,j}$, subtracting $\frac{\epsilon}{K}$
        from every class and then subtracting $(1 - \epsilon)$ from the correct
        class gives exactly
        $$
        p_{i,j} - \left((1 - \epsilon)\,\delta_{k,y} + \frac{\epsilon}{K}\right)
        = p_{i,j} - T_{i,j}
        $$
        """
        if self.label_smoothing:
            grad_out -= self.label_smoothing / self.num_classes
        grad_out[
            xp.arange(self.y_true.shape[0]),
            self.y_true,
        ] -= 1.0 - self.label_smoothing

        # 4. Zero ignored rows.
        grad_out *= row_mask

        # 5. Apply the upstream gradient and divide by the number of
        # non-ignored positions to match the forward average.
        grad_out *= grad.data / self.non_ignored_count

        # 6. Reshape to original shape if the forward pass flattened 3D logits.
        grad_out = grad_out.reshape(self.original_shape)

        # We do not need grad for y_true
        return grad_out, None


class HingeLoss(Function):
    r"""
    Hinge Loss.

    The hinge loss is defined as:
        $$
        loss = \max(0, 1 - y_{true} \cdot y_{pred})
        $$

    For correctly classified points ($y_{true} \cdot y_{pred} \geq 1$), the loss is 0; otherwise, it is $1 - y_{true} \cdot y_{pred}$. This is because loss functions typically don't go into the negatives so we take the max of 0 and 1 - y_true * y_pred)

    The objective function typically includes a regularization term:
        $$
        \|w\|^2 + C \sum max(0, 1 - y_{true} \cdot y_{pred})
        $$

    where $C$ is a hyperparameter controlling the trade-off between maximizing the margin (through regularization) and minimizing the loss, and $w$ is the weight vector. ($\|w\|^2$ is the regularization term)

    Paper: https://ieeexplore.ieee.org/document/708428

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.8, -0.5, 0.3])
        >>> y_true = Tensor([1, -1, 1])
        >>> loss = HingeLoss.apply(y_pred, y_true, reduction="mean") # Expected output: average hinge loss
    """

    def forward(
        self,
        y_pred: Array,
        y_true: Array,
        reduction: str = "none",
        **kwargs: Any,
    ) -> Union[Array, float]:
        """
        Computes the hinge loss.

        Args:
            y_pred (xp.ndarray): Predicted scores.
            y_true (Union[xp.ndarray, Tensor]): True labels.
            reduction (str, optional): "none", "mean", or "sum". Defaults to "none".
            **kwargs: Additional keyword arguments.

        Returns:
            Union[xp.array, float]: The computed hinge loss.
        """
        y_true = xp.array(y_true, dtype=xp.float32)

        # Reshape y_true to match y_pred if needed
        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)

        self.y_true = y_true
        self.y_pred = y_pred
        self.reduction = reduction

        # hinge loss = max(0, 1 - y_true * y_pred)
        self.margins = 1 - y_true * y_pred
        loss_data = xp.maximum(0, self.margins)

        if reduction == "mean":
            loss_data = xp.mean(loss_data)
        elif reduction == "sum":
            loss_data = xp.sum(loss_data)
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        return loss_data

    def backward(self, grad: Tensor) -> Tuple[Array, None]:
        r"""
        Computes the gradient of the hinge loss with respect to the predictions.

        For each element, the gradient is:
        $$
        \begin{align}
        \frac{\partial loss}{\partial y_{pred}} \\
        &= \frac{d(C * sum(max(0, 1 - y_{true} * y_{pred})))}{dw} \\
        &= C * max(0, 1 - y_{true} * y_{pred}) \\
        &=\begin{cases}
        -y_{true}, & \text{if } y_{true} \cdot y_{pred} < 1 \\
        0, & \text{otherwise}
        \end{cases}
        \end{align}
        $$
        For the gradient of w in the regularization term, $$\frac{d(\frac{1}{2}\|w\|^2)}{dw} = w$$ (we multiple 1/2 because it makes the gradient calculation easier)

        If the reduction is "mean", the gradient is averaged over the number of elements.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tuple[xp.ndarray, None]: A tuple containing the gradient with respect to $y_{pred}$ and None for $y_{true}$.
        """
        grad_y_pred = xp.zeros_like(self.y_pred)

        # Where margin > 0, gradient is -y_true
        margin_violated = self.margins > 0
        grad_y_pred = xp.where(margin_violated, -self.y_true, 0)

        if self.reduction == "mean":
            grad_y_pred /= self.y_pred.size

        # Handle scalar gradient (from mean/sum reduction)
        if grad.data.ndim == 0 or grad.data.size == 1:
            grad_y_pred *= grad.data
        else:
            # For elementwise gradient
            grad_y_pred *= grad.data.reshape(grad_y_pred.shape)

        return grad_y_pred, None


class MeanSquaredLoss(Function):
    r"""
    Mean Squared Error (MSE) Loss.

    The MSE loss is defined as:
        $$
        MSE = \frac{1}{N} \sum (y_{pred} - y_{true})^2
        $$

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([3.0, 5.0])
        >>> y_true = Tensor([2.0, 5.0])
        >>> loss = MeanSquaredLoss.apply(y_pred, y_true) # Expected output: 0.5
    """

    def forward(self, y_pred: Array, y_true: Array, **kwargs: Any) -> Array:
        """
        Computes the Mean Squared Error loss.

        Args:
            y_pred (xp.ndarray): Predicted values.
            y_true (xp.ndarray): True values.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed MSE loss.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return xp.mean((y_pred - y_true) ** 2)

    def backward(self, grad: Tensor) -> Array:
        r"""
        Computes the gradient of the Mean Squared Error loss with respect to the predictions.

        The gradient is given by:
        $$
        \frac{\partial L}{\partial y_{pred}} = \frac{2}{N} (y_{pred} - y_{true})
        $$

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            xp.ndarray: The gradient with respect to y_pred.
        """
        n = self.y_pred.size
        return (2 / n) * (self.y_pred - self.y_true) * grad.data


def binary_cross_entropy(
    y_pred: Tensor, y_true: Union[Tensor, ArrayLike], **kwargs: Any
) -> Tensor:
    """
    Computes the binary cross entropy loss given predicted probabilities.

    This function wraps the :class:`BinaryCrossEntropy` operation.

    Args:
        y_pred (Tensor): Predicted probabilities.
        y_true (Union[Tensor, ArrayLike]): True binary labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed binary cross entropy loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.9, 0.2, 0.1])
        >>> y_true = [1, 0, 0]
        >>> loss = binary_cross_entropy(y_pred, y_true)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropy.apply(y_pred, y_true, **kwargs)


def binary_cross_entropy_with_logits(
    y_pred: Tensor, y_true: Union[Tensor, ArrayLike], **kwargs: Any
) -> Tensor:
    r"""
    Computes the binary cross entropy loss using logits input for improved numerical stability.

    If $y_{pred}$ contains probabilities, use :func:`binary_cross_entropy` instead.

    The loss is computed as:
        $$
        -\left(y_{true} \cdot \log(y_{pred}) + (1 - y_{true}) \cdot \log(1 - y_{pred})\right)
        $$

    Args:
        y_pred (Tensor): Logits.
        y_true (Union[Tensor, ArrayLike]): True binary labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed binary cross entropy loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([2.0, -1.0, -2.0])
        >>> y_true = [1, 0, 0]
        >>> loss = binary_cross_entropy_with_logits(y_pred, y_true)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return BinaryCrossEntropyWithLogits.apply(y_pred, y_true, **kwargs)


def cross_entropy(
    y_pred: Tensor,
    y_true: Union[Tensor, ArrayLike],
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Computes the cross-entropy loss for multi-class classification with logits.

    This function expects $y_{pred}$ to be raw logits and $y_{true}$ to be class indices (not one-hot vectors).

    Args:
        y_pred (Tensor): Raw logits.
        y_true (Union[Tensor, ArrayLike]): True class indices.
        ignore_index (int, optional): Target value to ignore in the loss and gradient.
            Defaults to -100.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0.
        reduction (str, optional): Either ``"mean"`` or ``"sum"``. Defaults to
            ``"mean"``.
    Returns:
        Tensor: The computed cross-entropy loss.

    Examples:
        >>> import cupy as np
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor(xp.array([[2.0, 1.0, 0.1]]))
        >>> y_true = Tensor(xp.array([0]))
        >>> loss = cross_entropy(y_pred, y_true, ignore_index=-100, label_smoothing=0.1)
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return CrossEntropy.apply(
        y_pred,
        y_true,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def cross_entropy_private_logits(
    y_pred: Tensor,
    y_true: Union[Tensor, ArrayLike],
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Cross entropy for call sites where logits are private to the loss.

    On the CuPy bf16 fast path, backward may overwrite the logits buffer with
    the logits gradient. Do not use this if the logits Tensor feeds any other
    backward branch.
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return CrossEntropy.apply(
        y_pred,
        y_true,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
        destructive_logits_backward=True,
    )


def hinge_loss(
    y_pred: Tensor,
    y_true: Union[Tensor, ArrayLike],
    reduction: str = "none",
    **kwargs: Any,
) -> Tensor:
    """
    Computes the hinge loss for binary classification.

    Args:
        y_pred (Tensor): Predicted scores.
        y_true (Union[Tensor, ArrayLike]): True labels.
        reduction (str, optional): Specifies the reduction to apply: "none", "mean", or "sum". Defaults to "none".
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed hinge loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([0.8, -0.5, 0.3])
        >>> y_true = [1, -1, 1]
        >>> loss = hinge_loss(y_pred, y_true, reduction="mean")
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return HingeLoss.apply(y_pred, y_true, reduction=reduction, **kwargs)


def mean_squared_loss(
    y_pred: Tensor, y_true: Union[Tensor, ArrayLike], **kwargs: Any
) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) loss.

    Args:
        y_pred (Tensor): Predicted values.
        y_true (Union[Tensor, ArrayLike]): True values.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed MSE loss.

    Examples:
        >>> from autograd.tensor import Tensor
        >>> y_pred = Tensor([3.0, 5.0])
        >>> y_true = [2.0, 5.0]
        >>> loss = mean_squared_loss(y_pred, y_true) # Expected output: 0.5
    """
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    return MeanSquaredLoss.apply(y_pred, y_true, **kwargs)
