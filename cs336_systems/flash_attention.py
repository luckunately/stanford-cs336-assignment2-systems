from einops import rearrange, einsum
from math import ceil
import math

import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int

import triton
import triton.language as tl


# @triton.jit
# def flash_fwd_kernel(
#     Q_ptr,
#     K_ptr,
#     V_ptr,
#     O_ptr,
#     L_ptr,
#     stride_qb,
#     stride_qq,
#     stride_qd,
#     stride_kb,
#     stride_kk,
#     stride_kd,
#     stride_vb,
#     stride_vk,
#     stride_vd,
#     stride_ob,
#     stride_oq,
#     stride_od,
#     stride_lb,
#     stride_lq,
#     N_QUERIES,
#     N_KEYS,
#     scale,
#     D: tl.constexpr,
#     Q_TILE_SIZE: tl.constexpr,
#     K_TILE_SIZE: tl.constexpr,
#     is_causal,
# ):
#     # Program indices
#     query_tile_index = tl.program_id(0)
#     batch_index = tl.program_id(1)

#     # Offset each pointer with the corresponding batch index
#     # multiplied with the batch stride for each tensor
#     Q_block_ptr = tl.make_block_ptr(
#         Q_ptr + batch_index * stride_qb,
#         shape=(N_QUERIES, D),
#         strides=(stride_qq, stride_qd),
#         offsets=(query_tile_index * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D),
#         order=(1, 0),
#     )
#     K_block_ptr = tl.make_block_ptr(
#         K_ptr + batch_index * stride_kb,
#         shape=(N_KEYS, D),
#         strides=(stride_kk, stride_kd),
#         offsets=(0, 0),
#         block_shape=(K_TILE_SIZE, D),
#         order=(1, 0),
#     )
#     V_block_ptr = tl.make_block_ptr(
#         V_ptr + batch_index * stride_vb,
#         shape=(N_KEYS, D),
#         strides=(stride_vk, stride_vd),
#         offsets=(0, 0),
#         block_shape=(K_TILE_SIZE, D),
#         order=(1, 0),
#     )
#     O_block_ptr = tl.make_block_ptr(
#         O_ptr + batch_index * stride_ob,
#         shape=(N_QUERIES, D),
#         strides=(stride_oq, stride_od),
#         offsets=(query_tile_index * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D),
#         order=(1, 0),
#     )
#     L_block_ptr = tl.make_block_ptr(
#         L_ptr + batch_index * stride_lb,
#         shape=(N_QUERIES,),
#         strides=(stride_lq,),
#         offsets=(query_tile_index * Q_TILE_SIZE,),
#         block_shape=(Q_TILE_SIZE,),
#         order=(0,),
#     )

#     Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (Q_TILE_SIZE, D)
#     O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
#     lse = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # log-sum-exp
#     max_per_row = tl.full(
#         (Q_TILE_SIZE,), float("-inf"), dtype=tl.float32
#     )  # for numerical stability
#     max_previous = tl.zeros_like(max_per_row)
#     for i in range(0, N_KEYS, K_TILE_SIZE):
#         # Load K and V tiles
#         K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
#         V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

#         # Compute attention scores
#         attn_scores = tl.dot(Q_tile, K_tile.T) * scale
#         if is_causal:
#             # Mask out future positions: for each query row q, mask keys j > q + query_tile_index * Q_TILE_SIZE
#             q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
#             k_indices = tl.arange(0, K_TILE_SIZE) + i
#             mask = q_indices[:, None] >= k_indices[None, :]
#             attn_scores = tl.where(mask, attn_scores, float("-inf"))

#         # Update max_per_row for numerical stability
#         max_previous = max_per_row
#         max_per_row = tl.maximum(max_per_row, tl.max(attn_scores, axis=1))

#         # Compute attention probabilities
#         attn_probs = tl.exp(attn_scores - max_per_row[:, None])

#         # Update log-sum-exp scaling before updating max_per_row
#         lse = tl.exp(max_previous - max_per_row) * lse + tl.sum(attn_probs, axis=1)

#         # Compute output
#         scale_factor = tl.exp(max_previous - max_per_row)
#         O_tile *= scale_factor[:, None]
#         O_tile += tl.dot(attn_probs, V_tile)

#         K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
#         V_block_ptr = V_block_ptr.advance((0, K_TILE_SIZE))

#     final_scale_factor = 1.0 / lse
#     O_tile *= final_scale_factor[:, None]
#     tl.store(O_block_ptr, O_tile, boundary_check=(0, 1))
#     tl.store(L_block_ptr, lse, boundary_check=(0,))

@triton.jit
def _flash_fwd_inner_kernel(
    output, l, m,
    queries, Kt_ptr_base, V_ptr_base,
    stride_kd, stride_kk,
    stride_vk, stride_vd,
    N_KEYS, D,
    scale,
    query_tile_index: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    stage: tl.constexpr
):

    # causal: handles K/V blocks to the left of the diagonal blocks
    if stage == 1:
        lo, hi = 0, query_tile_index * Q_TILE_SIZE
    # causal: handles diagonal blocks
    elif stage == 2:
        lo, hi = query_tile_index * Q_TILE_SIZE, (query_tile_index + 1) * Q_TILE_SIZE
    # non causal
    else:
        lo, hi = 0, N_KEYS

    # making K/V block ptrs in the inner kernek to avoid passing block ptrs between triton kernels (causing compiler error)
    Kt_block_ptr = tl.make_block_ptr(
        Kt_ptr_base,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE_SIZE * (lo // K_TILE_SIZE)))


    V_block_ptr = tl.make_block_ptr(
        V_ptr_base,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE * (lo // K_TILE_SIZE), 0))

    q_offsets = Q_TILE_SIZE * query_tile_index + tl.arange(0, Q_TILE_SIZE)

    for k_index in range(lo // K_TILE_SIZE, tl.cdiv(hi, K_TILE_SIZE)):
        m_prev = m
        keys_t = tl.load(Kt_block_ptr, boundary_check=(0, 1), padding_option="zero") # (D, K_TILE_SIZE)
        values = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        # S = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        S = tl.dot(queries, keys_t) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        if stage == 2:
            k_offsets = k_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_offsets[:, None] < k_offsets[None, :]
            S = tl.where(mask, - 1e6, S)


        m = tl.maximum(tl.max(S, axis=-1), m_prev)
        P = tl.math.exp(S - m[:, None])

        corrector = tl.math.exp(m_prev - m)
        l = corrector * l + tl.sum(P, axis=-1)

        output = corrector[:, None] * output + tl.dot(P.to(values.dtype), values)

        Kt_block_ptr = Kt_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    return output, l, m


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    stage = 3 if is_causal else 1

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

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

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), - float("inf"), dtype=tl.float32)
    queries = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)


    K_ptr_base = K_ptr + batch_index * stride_kb
    V_ptr_base = V_ptr + batch_index * stride_vb

    if stage == 1 or stage == 3:
        output, l, m = _flash_fwd_inner_kernel(
            output, l, m, queries, K_ptr_base, V_ptr_base,
            stride_kd, stride_kk,
            stride_vk, stride_vd,
            N_KEYS, D,
            scale,
            query_tile_index, Q_TILE_SIZE, K_TILE_SIZE, 4 - stage
        )

    if stage == 3:
        output, l, m = _flash_fwd_inner_kernel(
            output, l, m, queries, K_ptr_base, V_ptr_base,
            stride_kd, stride_kk,
            stride_vk, stride_vd,
            N_KEYS, D,
            scale,
            query_tile_index, Q_TILE_SIZE, K_TILE_SIZE, 2
        )


    output /= l[:, None]
    logsum = tl.log(l) + m

    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, logsum.to(L_block_ptr.type.element_ty), boundary_check=(0, ))

class FlashAttentionFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False) -> Tensor:
        # assert (
        #     torch.cuda.is_available()
        # ), "FlashAttentionFunctionTriton only supports CUDA devices."
        # assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
        # assert (
        #     Q.dim() == 3 and K.dim() == 3 and V.dim() == 3
        # ), "Q, K, V must be 3-dimensional tensors"

        # batch_size, seq_length, head_dim = Q.shape
        # sqrt_d = math.sqrt(head_dim)
        # tile_num = calculate_tile_size(
        #     device=torch.device("cuda"),
        #     sequence_length=seq_length,
        #     head_dim=head_dim,
        #     minimum_tile_size=16,
        # )

        # O = torch.empty_like(Q)
        # L = torch.empty(
        #     (batch_size, seq_length), device=Q.device, dtype=Q.dtype    
        # )  # log-sum-exp per position

        # grid = (ceil(seq_length / tile_num), batch_size,)
        # flash_fwd_kernel[grid](
        #     Q,
        #     K,
        #     V,
        #     O,
        #     L,
        #     Q.stride(0),
        #     Q.stride(1),
        #     Q.stride(2),
        #     K.stride(0),
        #     K.stride(1),
        #     K.stride(2),
        #     V.stride(0),
        #     V.stride(1),
        #     V.stride(2),
        #     O.stride(0),
        #     O.stride(1),
        #     O.stride(2),
        #     L.stride(0),
        #     L.stride(1),
        #     seq_length,
        #     seq_length,
        #     1.0 / sqrt_d,
        #     head_dim,
        #     tile_num,
        #     tile_num,
        #     is_causal
        # )
        # ctx.save_for_backward(L)
        # return O

        batch_size, n_queries, d_model = Q.shape
        scale = 1. / d_model ** 0.5
        n_keys = K.size(1)


        O = torch.empty(batch_size, n_queries, d_model, device=Q.device, dtype=Q.dtype)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous Q, K, V"
        assert K.shape[-1] == V.shape[-1] == d_model, "Dimension mismatch"

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        ctx.scale = scale

        flash_fwd_kernel[(math.ceil(n_queries / ctx.Q_TILE_SIZE), batch_size,)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            d_model,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal
        )

        ctx.save_for_backward(O, L, Q, K, V)

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
