from cycler import V
from einops import rearrange, einsum
from math import ceil
import math

import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int

import triton
import triton.language as tl

TRITON_INTERPRET=1
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # tl.debug("Debugging: Q_TILE_SIZE = ", Q_TILE_SIZE)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (Q_TILE_SIZE, D)
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    lse = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # log-sum-exp
    max_per_row = tl.full(
        (Q_TILE_SIZE,), float("-inf"), dtype=tl.float32
    )  # for numerical stability
    max_previous = tl.zeros_like(max_per_row)
    for i in range(0, N_KEYS, K_TILE_SIZE):
        # Load K and V tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # Compute attention scores
        attn_scores = tl.dot(Q_tile, K_tile.T, input_precision='ieee') * scale
        if is_causal:
            # Mask out future positions: for each query row q, mask keys j > q + query_tile_index * Q_TILE_SIZE
            q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE) + i
            mask = q_indices[:, None] >= k_indices[None, :]
            attn_scores = tl.where(mask, attn_scores, float("-inf"))

        # Update max_per_row for numerical stability
        max_previous = max_per_row
        max_per_row = tl.maximum(max_per_row, tl.max(attn_scores, axis=1))

        # Compute attention probabilities
        attn_probs = tl.exp(attn_scores - max_per_row[:, None])

        # Update log-sum-exp scaling before updating max_per_row
        lse = tl.exp(max_previous - max_per_row) * lse + tl.sum(attn_probs, axis=-1)

        # Compute output
        scale_factor = tl.exp(max_previous - max_per_row)
        O_tile *= scale_factor[:, None]
        O_tile += tl.dot(attn_probs.to(V_tile.dtype), V_tile, input_precision='ieee')

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    final_scale_factor = 1.0 / lse
    O_tile *= final_scale_factor[:, None]
    tl.store(O_block_ptr, O_tile, boundary_check=(0, 1))
    lse = tl.log(lse) + max_per_row
    tl.store(L_block_ptr, lse, boundary_check=(0,))

class FlashAttentionFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False) -> Tensor:
        assert (
            torch.cuda.is_available()
        ), "FlashAttentionFunctionTriton only supports CUDA devices."
        assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
        assert (
            Q.dim() == 3 and K.dim() == 3 and V.dim() == 3
        ), "Q, K, V must be 3-dimensional tensors"

        batch_size, seq_length, head_dim = Q.shape
        sqrt_d = math.sqrt(head_dim)
        tile_num = calculate_tile_size(
            device=torch.device("cuda"),
            sequence_length=seq_length,
            head_dim=head_dim,
            minimum_tile_size=16,
        )

        O = torch.empty_like(Q)
        L = torch.empty(
            (batch_size, seq_length), device=Q.device
        )  # log-sum-exp per position

        grid = (ceil(seq_length / tile_num), batch_size)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            seq_length,
            seq_length,
            1.0 / sqrt_d,
            head_dim,
            tile_num,
            tile_num,
            is_causal
        )
        ctx.save_for_backward(L)
        return O


class FlashAttentionFunctionPytorch(torch.autograd.Function):
    # No constructor (__init__) is needed for torch.autograd.Function subclasses.
    # All state should be managed via the context (ctx) in forward/backward.

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, minimum_tile_size=8) -> Tensor:
        """
        Args:
            ctx (torch.autograd.FunctionContext): Context to save information for backward pass
            Q (Tensor): Query tensor of shape (batch_size, seq_length, head_dim)
            K (Tensor): Key tensor of shape (batch_size, seq_length, head_dim)
            V (Tensor): Value tensor of shape (batch_size, seq_length, head_dim)
            is_causal (bool, optional): Whether the attention is causal (i.e., autoregressive). Defaults to False.
        save Tensor: The log-sum-exp tensor L of shape (batch_size, seq_length)
        Returns:
            Tensor: The output tensor O of shape (batch_size, seq_length, head_dim)
        """
        assert (
            Q.dim() == 3 and K.dim() == 3 and V.dim() == 3
        ), "Q, K, V must be 3-dimensional tensors"
        assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
        batch_size, seq_length, head_dim = Q.shape
        sqrt_d = math.sqrt(head_dim)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tile_size = calculate_tile_size(
                device=device,
                sequence_length=seq_length,
                head_dim=head_dim,
                minimum_tile_size=minimum_tile_size,
            )
        else:
            raise NotImplementedError(
                "Only CUDA devices are supported in this implementation."
            )
        O = torch.zeros_like(Q)
        L = torch.zeros(
            (batch_size, seq_length), device=Q.device
        )  # log-sum-exp per position

        n_tiles = ceil(seq_length / tile_size)

        for i in range(n_tiles):
            Q_tile = Q[
                :, i * tile_size : min((i + 1) * tile_size, seq_length)
            ]  # (b, t, d)
            Q_tile = Q_tile.to(device)
            # Initialize accumulators for the output tile and log-sum-exp
            O_tile = torch.zeros_like(Q_tile).to(device)
            max_per_row = torch.full(
                (batch_size, tile_size), float("-inf"), device=Q_tile.device
            )  # save the max logit for numerical stability
            lse = torch.zeros(
                (batch_size, tile_size), device=Q_tile.device
            )  # log-sum-exp

            for j in range(n_tiles):
                K_tile = K[
                    :, j * tile_size : min((j + 1) * tile_size, seq_length)
                ]  # (b, t, d)
                V_tile = V[
                    :, j * tile_size : min((j + 1) * tile_size, seq_length)
                ]  # (b, t, d)
                K_tile = K_tile.to(device)
                V_tile = V_tile.to(device)

                # Compute scaled dot-product attention for the tile pair
                attn_scores = (
                    einsum(Q_tile, K_tile, "... q d, ... k d -> ... q k") / sqrt_d
                )

                # Update max_per_row for numerical stability
                max_previous = max_per_row.clone()
                max_per_row = torch.max(
                    max_per_row, torch.max(attn_scores, dim=-1).values
                )

                attn_probs = torch.exp(attn_scores - max_per_row[:, :, None])

                # Update log-sum-exp scaling before updating max_per_row
                lse = torch.exp(max_previous - max_per_row) * lse + attn_probs.sum(
                    dim=-1
                )

                # Compute O(j) according to the recurrence:
                # O(j)_i = diag(exp(m(j-1)_i - m(j)_i)) * O(j-1)_i + P~(j)_i V(j)
                O_tile = einsum(
                    torch.diag_embed(torch.exp(max_previous - max_per_row)),
                    O_tile,
                    "... q q, ... q d -> ... q d",
                ) + einsum(attn_probs, V_tile, "... q k, ... k d -> ... q d")

            O[:, i * tile_size : min((i + 1) * tile_size, seq_length), :] = einsum(
                torch.diag_embed(1.0 / lse), O_tile, "... q q, ... q d -> ... q d"
            ).to(O.device)
            L[:, i * tile_size : min((i + 1) * tile_size, seq_length)] = (
                torch.log(lse) + max_per_row
            ).to(L.device)

        ctx.save_for_backward(L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (torch.autograd.FunctionContext): Context with saved information from forward pass
            grad_output (Tensor): Gradient tensor from the output of the forward pass
        """
        raise NotImplementedError("Backward pass is not implemented.")
        # Q, K, V = ctx.saved_tensors
        # is_causal = ctx.is_causal

        # # Compute gradients
        # grad_Q, grad_K, grad_V = Flashattention_Autograd_Function_Pytorch.backward_impl(Q, K, V, grad_output, is_causal)

        # return grad_Q, grad_K, grad_V


@staticmethod
def calculate_tile_size(device, sequence_length, head_dim, minimum_tile_size=8):
    return 16
    # torch.cuda.synchronize()
    # mem_info = torch.cuda.mem_get_info(device.index)
    # free_mem_bytes = mem_info[0]
    # # calculate proper tile size based on available memory
    # # assume we can use up to 1/4 of free memory for tiles
    # max_tile_mem = free_mem_bytes // 4
    # # each tile will hold Q_tile, K_tile, V_tile, and output_tile
    # # each of size (tile_size, head_dim) except output_tile which is (tile_size, head_dim)
    # bytes_per_element = 4  # assuming float32
    # tile_size = int((max_tile_mem // (4 * head_dim * bytes_per_element)) ** 0.5)
    # assert tile_size >= minimum_tile_size, "Why your GPU have so little memory???"
    # # make tile_size a multiple of minimum_tile_size for better memory alignment
    # tile_size = ceil(tile_size / minimum_tile_size) * minimum_tile_size
    # # also ensure tile_size does not exceed sequence_length
    # tile_size = min(tile_size, sequence_length)

    # return tile_size
