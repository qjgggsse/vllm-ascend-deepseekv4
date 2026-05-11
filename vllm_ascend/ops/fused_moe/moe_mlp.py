# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from dataclasses import dataclass

import torch
import torch_npu
from torch.nn.functional import pad
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import ensure_mxfp8_moe_available
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpStageOutputs
from vllm_ascend.utils import dispose_tensor, enable_custom_op, get_weight_prefetch_method


@dataclass(frozen=True, slots=True)
class _UnquantMlpStage1Output:
    gate_up_out: torch.Tensor
    stage_input: torch.Tensor


def unquant_apply_mlp_stage1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
) -> _UnquantMlpStage1Output:
    if need_trans:
        w1 = w1.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    if activation == "swigluoai":
        num_experts, _, hidden_size = w1.shape
        swiglu_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out.view(-1, hidden_size))
    else:
        swiglu_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        swiglu_out *= topk_scales

    return _UnquantMlpStage1Output(gate_up_out=swiglu_out, stage_input=hidden_states)


def unquant_apply_mlp_stage2(
    stage1_output: _UnquantMlpStage1Output,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w2_bias: torch.Tensor = None,
    group_list_type: int = 1,
    need_trans: bool = True,
) -> torch.Tensor:
    if need_trans:
        w2 = w2.transpose(1, 2)

    return torch_npu.npu_grouped_matmul(
        x=[stage1_output.gate_up_out],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]


def build_mlp_stage_outputs(*, mlp_compute_input: MoEMlpComputeInput) -> MoEMlpStageOutputs:
    if not mlp_compute_input.quant.is_quant:
        stage1_output = unquant_apply_mlp_stage1(
            hidden_states=mlp_compute_input.hidden_states,
            w1=mlp_compute_input.weights.w1,
            group_list=mlp_compute_input.group_list,
            w1_bias=mlp_compute_input.weights.w1_bias,
            activation=mlp_compute_input.activation,
            group_list_type=mlp_compute_input.group_list_type,
            topk_scales=mlp_compute_input.topk_scales,
            need_trans=mlp_compute_input.need_trans,
        )
        gmm2_output = unquant_apply_mlp_stage2(
            stage1_output=stage1_output,
            w2=mlp_compute_input.weights.w2,
            group_list=mlp_compute_input.group_list,
            w2_bias=mlp_compute_input.weights.w2_bias,
            group_list_type=mlp_compute_input.group_list_type,
            need_trans=mlp_compute_input.need_trans,
        )
        return MoEMlpStageOutputs(
            stage_input=stage1_output.stage_input,
            gmm1_output=stage1_output.gate_up_out,
            swiglu_output=stage1_output.gate_up_out,
            gmm2_output=gmm2_output,
        )

    final_output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)
    return MoEMlpStageOutputs(
        stage_input=mlp_compute_input.hidden_states,
        gmm1_output=mlp_compute_input.hidden_states,
        swiglu_output=mlp_compute_input.hidden_states,
        gmm2_output=final_output,
    )


def apply_mlp_stage1(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    return build_mlp_stage_outputs(mlp_compute_input=mlp_compute_input).swiglu_output


def apply_mlp_stage2(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    return build_mlp_stage_outputs(mlp_compute_input=mlp_compute_input).gmm2_output


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    use_mxfp_quant: bool = False,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool = True,
) -> torch.Tensor:
    input_hidden_dtype = hidden_states.dtype
    use_gmm_swiglu_quant_fusion = use_mxfp_quant or (fusion and not dynamic_eplb)

    if use_mxfp_quant:
        ensure_mxfp8_moe_available("MXFP MoE MLP path")

        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if w1_offset is not None or w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = (
            DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if use_mxfp_quant else dynamic_scale
        )
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(hidden_states)
    is_mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
            )
        elif use_gmm_swiglu_quant_fusion:
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=None,
                use_mxfp_quant=use_mxfp_quant,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=None,
            fallback_output_dtype=w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype,
        )
    elif w1_offset is not None:
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
        dispose_tensor(unquantized_hidden_states)
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                bias=bias1,
            )
        elif use_gmm_swiglu_quant_fusion:
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=bias1,
                use_mxfp_quant=use_mxfp_quant,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if isinstance(w1_scale, list) and w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1 if isinstance(w1, list) else [w1],
                bias=bias1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0] if isinstance(w1_scale, list) else w1_scale,
                activation_scale=pertoken_scale,
                bias=bias1[0] if isinstance(bias1, list) else bias1,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=bias2,
            fallback_output_dtype=_output_dtype,
        )

    return hidden_states


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
) -> torch.Tensor:
    stage1_output = unquant_apply_mlp_stage1(
        hidden_states=hidden_states,
        w1=w1,
        group_list=group_list,
        w1_bias=w1_bias,
        activation=activation,
        group_list_type=group_list_type,
        topk_scales=topk_scales,
        need_trans=need_trans,
    )
    return unquant_apply_mlp_stage2(
        stage1_output=stage1_output,
        w2=w2,
        group_list=group_list,
        w2_bias=w2_bias,
        group_list_type=group_list_type,
        need_trans=need_trans,
    )


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    dynamic_scale = mlp_compute_input.dynamic_scale
    topk_scales = mlp_compute_input.topk_scales
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_bias = mlp_compute_input.weights.w1_bias
    w2_bias = mlp_compute_input.weights.w2_bias
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    w1_scale_bias = mlp_compute_input.weights.w1_scale_bias
    w2_scale_bias = mlp_compute_input.weights.w2_scale_bias
    w1_offset = mlp_compute_input.weights.w1_offset
    w2_offset = mlp_compute_input.weights.w2_offset
    activation = mlp_compute_input.activation
    need_trans = mlp_compute_input.need_trans
    dynamic_eplb = mlp_compute_input.dynamic_eplb
    fusion = mlp_compute_input.fusion

    if not mlp_compute_input.quant.is_quant:
        return unquant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            activation=activation,
            group_list=group_list,
            group_list_type=group_list_type,
            topk_scales=topk_scales,
            need_trans=need_trans,
        )

    assert w1_scale is not None and w2_scale is not None
    act_quant_type = torch.float8_e4m3fn
    weight_quant_type = torch.float8_e4m3fn
    scale_type = None
    per_token_scale_type = None
    use_bf16 = hidden_states.dtype == torch.bfloat16
    use_mxfp_quant = mlp_compute_input.quant.is_mxfp

    if use_mxfp_quant:
        mxfp = mlp_compute_input.quant.mxfp
        assert mxfp is not None, "mlp_compute_input.quant.mxfp is required when quant_type is MXFP8."
        act_quant_type = mxfp.act_quant_type or act_quant_type
        weight_quant_type = mxfp.weight_quant_type or weight_quant_type
        scale_type = mxfp.scale_dtype
        per_token_scale_type = mxfp.per_token_scale_dtype
        use_bf16 = mxfp.use_bf16

    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb,
        use_mxfp_quant=use_mxfp_quant,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
    )
