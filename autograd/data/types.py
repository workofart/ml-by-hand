from dataclasses import dataclass

from autograd.backend import Array


@dataclass(frozen=True)
class TokenWindowExample:
    stream: Array  # reference to full token stream
    offset: int  # start position of example
    window_len: int  # number of tokens in the sampled window


@dataclass(frozen=True)
class CausalLMBatch:
    input_ids: Array  # [B, T], int32
    labels: Array  # [B, T], int32
    loss_total_weight: Array  # scalar float32


@dataclass(frozen=True)
class Seq2SeqBatch:
    input_ids: Array  # [B, T], int32
    decoder_input_ids: Array  # [B, T], int32
    labels: Array  # [B, T], int32
    src_mask: Array  # [B, 1, 1, T], float32
    tgt_mask: Array  # [B, 1, 1, T], float32
