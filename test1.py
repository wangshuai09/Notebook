# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from .builder import NPUOpBuilder

try:
    import torch_npu
except ImportError as e:
    pass


def swap_two_rows(x):
    # [..., [x1, x2, x3, x4, ...]] --> [..., [-x2, x1, -x4, x3, ...]]
    x1 = x[..., ::2].clone()
    x2 = x[..., 1::2]

    x[..., ::2] = -x2
    x[..., 1::2] = x1
    return x


class InferenceContext:
    _workspace = None

    _seed = 42
    _curr_offset = 0
    _stream = 0
    _free_memory_size = 0
    _num_tokens = 1
    _attention_unfused_workspace_offset = 0
    _workSpaceSize = 0
    _workSpaceSize = 0
    _workspace = 0

    workSpaceSize = 0
    k_caches = []
    v_caches = []

    @staticmethod
    def reset_tokens(initial_tokens=1):
        InferenceContext._num_tokens = initial_tokens

    @staticmethod
    def current_tokens():
        return InferenceContext._num_tokens

    @staticmethod
    def GetWorkSpace():
        return InferenceContext._workspace

    @staticmethod
    def GetMaxTokenLength():
        return InferenceContext._max_seq_len


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class NPUInference:
    @staticmethod
    def layer_norm(inputs, gamma, beta, epsilon):
        norm, mean, variance = torch.native_layer_norm(inputs, [inputs.shape[-1]], gamma, beta, eps=epsilon)
        return norm

    @staticmethod
    def _qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        inp_norm = torch.nn.functional.layer_norm(inputs, (inputs.shape[2],), gamma, beta, eps)
        tmp = torch.matmul(inp_norm, weight.t())
        if add_bias:
            tmp += bias
        output = [tmp, inp_norm]
        return output

    @staticmethod
    def qkv_gemm_fp16(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return NPUInference._qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    @staticmethod
    def qkv_gemm_bf16(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return NPUInference._qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    @staticmethod
    def qkv_gemm_fp32(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return NPUInference._qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    @classmethod
    def _bias_add_transform_0213(cls, output, k_cache, v_cache, vals, bias,
                                 hidden_dim, seq_length, seq_offset, heads,
                                 num_kv,  # num_kv > 0 ? num_kv : heads,
                                 rotary_dim,
                                 rotate_half,
                                 rotate_every_two,
                                 rope_theta):
        # q,k,v
        # q shape: [bsz, seq, heads, head_dim]
        # k shape: [bsz, seq, num_kv, head_dim]
        # v shape: [bsz, seq, num_kv * head_dim]
        bsz, _, _ = vals.shape
        q = vals[..., :hidden_dim].reshape(bsz, seq_length, heads, -1)
        k = vals[..., hidden_dim: hidden_dim + num_kv * (hidden_dim // heads)].reshape(bsz, seq_length, num_kv, -1)
        v = vals[..., hidden_dim + num_kv * (hidden_dim // heads):]

        # rope 位置编码, npu
        if rotary_dim > 0 and rotate_every_two:
            # sin, cos may use cache
            seq_id = torch.arange(0, seq_length).to("npu")
            inv_freq = torch.arange(0, rotary_dim, 2) / rotary_dim
            inv_freq = inv_freq.to("npu")
            inv_freq = 1.0 / torch.pow(rope_theta, inv_freq)
            inv_freq = torch.outer(seq_id, inv_freq)
            sin = inv_freq.sin()
            cos = inv_freq.cos()
            # shape: [bsz=1, seq_len, heads=1, rotary_dim], 相邻两行相同
            sin = sin.view(-1, seq_length, 1, rotary_dim // 2).repeat_interleave(2, dim=-1)
            cos = cos.view(-1, seq_length, 1, rotary_dim // 2).repeat_interleave(2, dim=-1)

            # 只在 rotary_dim 范围内计算
            q_pos, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_pos, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

            q_pos = q_pos * cos + swap_two_rows(q_pos) * sin
            q = torch.cat([q_pos, q_pass], dim=-1)
            k_pos = k_pos * cos + swap_two_rows(k_pos) * sin
            k = torch.cat([k_pos, k_pass], dim=-1)

        # 结果，v 不变
        output = q.reshape(bsz, seq_length, heads, -1).transpose(1, 2).contiguous()  # [b, n, s, d]
        k_cache = k.reshape(bsz, seq_length, heads, -1).contiguous()  # [b, s, n, d]
        v_cache = v.reshape(bsz, seq_length, heads, -1).contiguous()  # [b, s, n, d]
        print("result:", output.shape, k_cache.shape, v_cache.shape)
        return output, k_cache, v_cache

    @staticmethod
    def _softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads,
                         num_kv, norm_factor, triangular_masking, local_attention, window_size, no_masking,
                         layer_id, num_layers, alibi, rope_theta):

        # bsz, seq_len, k = query_key_value.size()
        # k = k // (heads + 2 * (num_kv if num_kv > 0 else heads))
        # hidden_dim = heads * k
        #
        # is_promt = seq_len > 1
        # if is_promt:
        #     InferenceContext.reset_tokens(seq_len)
        #
        # soft_len = InferenceContext.current_tokens()
        # workspace = InferenceContext.GetWorkSpace()
        # seq_offset = 0 if is_promt else soft_len - 1
        #
        # #
        # output = torch.empty((bsz, seq_len, heads * k), dtype=torch.float16, device="npu")
        # k_cache = torch.empty((bsz, seq_len, (num_kv if num_kv > 0 else heads) * k), dtype=torch.float16, device="npu")
        # v_cache = torch.empty((bsz, seq_len, (num_kv if num_kv > 0 else heads) * k), dtype=torch.float16, device="npu")
        # q, k, v = NPUInference._bias_add_transform_0213(output=output,
        #                                                 k_cache=k_cache,
        #                                                 v_cache=v_cache,
        #                                                 vals=query_key_value,
        #                                                 bias=None,
        #                                                 hidden_dim=hidden_dim,
        #                                                 seq_length=seq_len,
        #                                                 seq_offset=seq_offset,
        #                                                 heads=heads,
        #                                                 num_kv=num_kv if num_kv > 0 else heads,
        #                                                 # num_kv > 0 ? num_kv : heads,
        #                                                 rotary_dim=rotary_dim,
        #                                                 rotate_half=rotate_half,
        #                                                 rotate_every_two=rotate_every_two,
        #                                                 rope_theta=rope_theta)

        q, k, v = split_tensor_along_last_dim(query_key_value, 3)
        bsz, seq_len, H = query_key_value.size()
        n = heads

        q = q.reshape(bsz, seq_len, heads, -1).transpose(1, 2).contiguous()  # [b, n, s, d]
        k = k.reshape(bsz, seq_len, heads, -1).contiguous()  # [b, s, n, d]
        v = v.reshape(bsz, seq_len, heads, -1).contiguous()  # [b, s, n, d]

        torch.npu.synchronize()
        # print('111111111111', q.shape, k.shape, v.shape, attn_mask.shape, query_key_value.shape)
        # output = torch_npu.npu_fusion_attention(
        #     q, k, v, n, "BSH",
        #     pse=None,
        #     padding_mask=None,
        #     atten_mask=attn_mask.bool(),
        #     scale=norm_factor,
        #     pre_tockens=65536,
        #     next_tockens=65536,
        #     keep_prob=1,
        #     inner_precise=0
        # )[0]
        # [b, s, H] --> [b, s, n, d] --> [b, n, s, d]
        # print('111111111111', q.shape, k.shape, v.shape)
        # k = k.view(k.size(0), k.size(1), heads, -1).transpose(1, 2).contiguous()
        # v = v.view(v.size(0), v.size(1), heads, -1).transpose(1, 2).contiguous()
        # print('222222222222', output.shape, k.shape, v.shape)

        q = q.reshape(bsz, seq_len, heads, -1).transpose(1, 2).contiguous()  # [b, n, s, d]
        k = k.reshape(bsz, seq_len, heads, -1).contiguous()  # [b, s, n, d]
        v = v.reshape(bsz, seq_len, heads, -1).contiguous()  # [b, s, n, d]

        # [b, n, s, d] --> [b * n, s, d]
        query_layer = q.reshape(bsz * n, seq_len, -1).contiguous()

        print('111111111111', q.shape, k.shape, v.shape, attn_mask.shape, query_key_value.shape)

        if layer_id < len(InferenceContext.k_caches) and InferenceContext.k_caches[layer_id] is not None:
            print('22222222222222', InferenceContext.k_caches[layer_id].shape)
            k = torch.cat([InferenceContext.k_caches[layer_id]], dim=1)
            v = torch.cat([InferenceContext.v_caches[layer_id]], dim=1)
            import pdb
            pdb.set_trace()

        # [b, s, n, d] --> [b, n, s, d] --> [b * n, s, d]
        print('33333333333333', q.shape, k.shape, v.shape, attn_mask.shape)
        prev_seq_len = k.size(1)
        key_layer = k.transpose(1, 2).reshape(bsz * n, prev_seq_len, -1).contiguous()
        value_layer = v.transpose(1, 2).reshape(bsz * n, prev_seq_len, -1).contiguous()

        # [b * n, s, d] * [b * n, d, s] --> [b * n, s, s]
        out = torch.bmm(query_layer, key_layer.transpose(1, 2))
        out *= norm_factor

        if attn_mask is not None:
            out = out.view(bsz, n, seq_len, -1).contiguous()
            print('4444444444444', out.shape, attn_mask.shape)
            out = out * attn_mask
            out = torch.max(out, torch.tensor(torch.finfo(out.dtype).min, device=out.device))
            out = out.view(bsz * n, seq_len, -1).contiguous()

        dtype = out.dtype
        out = torch.nn.Softmax(dim=-1)(out.float()).to(dtype)

        # [b * n, s, s] * [b * n, s, d] --> [b * n, s, d]
        out = torch.bmm(out, value_layer)

        # [b * n, s, d] --> [b, n, s, d] --> [b, s, n, d] --> [b, s, H]
        output = out.reshape(bsz, n, seq_len, -1).permute(0, 2, 1, 3).reshape(bsz, seq_len, -1).contiguous()

        # [b * n, s, d] --> [b, n, s, d] --> [b, s, n, d]
        key_layer = key_layer.reshape(bsz, n, seq_len, -1).permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.reshape(bsz, n, seq_len, -1).permute(0, 2, 1, 3).contiguous()

        if layer_id < len(InferenceContext.k_caches):
            InferenceContext.k_caches[layer_id] = key_layer
            InferenceContext.v_caches[layer_id] = value_layer
        else:
            InferenceContext.k_caches.append(key_layer)
            InferenceContext.v_caches.append(value_layer)

        return output, key_layer, value_layer

    @staticmethod
    def softmax_context_fp16(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads,
                             num_kv, norm_factor, triangular_masking, local_attention, window_size, no_masking,
                             layer_id, num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two,
                                             heads, num_kv, norm_factor, triangular_masking, local_attention,
                                             window_size, no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def softmax_context_bf16(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads,
                             num_kv, norm_factor, triangular_masking, local_attention, window_size, no_masking,
                             layer_id, num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two,
                                             heads, num_kv, norm_factor, triangular_masking, local_attention,
                                             window_size, no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def softmax_context_fp32(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads,
                             num_kv, norm_factor, triangular_masking, local_attention, window_size, no_masking,
                             layer_id, num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two,
                                             heads, num_kv, norm_factor, triangular_masking, local_attention,
                                             window_size, no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def _vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return torch.matmul(input, weight.t())

    @staticmethod
    def vector_matmul_fp16(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return NPUInference._vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def vector_matmul_bf16(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return NPUInference._vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def vector_matmul_fp32(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return NPUInference._vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def _mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                  mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        residual_add = torch.nn.functional.layer_norm(input + residual + input_bias, (input.shape[2],), gamma, beta,
                                                      eps)
        tmp = torch.matmul(residual_add, weight_interm.t())
        tmp = torch.nn.functional.gelu(tmp + bias)
        output = torch.matmul(tmp, weight_out.t())
        return output, residual_add

    @staticmethod
    def mlp_gemm_fp16(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        return NPUInference._mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                                      transpose)

    @staticmethod
    def mlp_gemm_bf16(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        return NPUInference._mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                                      transpose)

    @staticmethod
    def mlp_gemm_fp32(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        return NPUInference._mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                                      transpose)

    @staticmethod
    def _residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                           mlp_after_attn, add_bias, pre_layer_norm):
        if pre_layer_norm:
            tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                   final_bias.float()) / mp_size + hidden_state.float()
        else:
            tmp = residual.float() + hidden_state.float() + final_bias.float()

        input_dtype = hidden_state.dtype
        residual = tmp.to(input_dtype)
        return residual

    @staticmethod
    def residual_add_bias_fp16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return NPUInference._residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                               mp_size, mlp_after_attn, add_bias, pre_layer_norm)

    @staticmethod
    def residual_add_bias_bf16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return NPUInference._residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                               mp_size, mlp_after_attn, add_bias, pre_layer_norm)

    @staticmethod
    def residual_add_bias_fp32(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return NPUInference._residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                               mp_size, mlp_after_attn, add_bias, pre_layer_norm)


class InferenceBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []

    def load(self, verbose=True):
        return NPUInference
