#
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

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Callable

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.utils import select_experts

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.eplb.adaptor.vllm_adaptor import init_eplb_config
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    FusedExpertsResult,
    setup_moe_comm_method,
)
from vllm_ascend.ops.fused_moe.runner import AscendMoERunner
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_sp, get_compressed_expert_map
from vllm_ascend.worker.model_runner_v1 import tensor_model_parallel_all_reduce
from vllm_ascend.worker.shared_expert_utils import (
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    allow_shared_part1_evt: torch.npu.Event | None = None
    allow_shared_part2_evt: torch.npu.Event | None = None


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    allow_shared_part1: torch.npu.Event | None = field(default=None)
    allow_shared_part2: torch.npu.Event | None = field(default=None)

    def shared_part1_anchor(self) -> torch.npu.Event | None:
        return self.allow_shared_part1 or self.before_dispatch

    def shared_part2_anchor(self) -> torch.npu.Event | None:
        return self.allow_shared_part2 or self.before_combine

    def shared_part1_anchor_name(self) -> str:
        return "allow_shared_part1" if self.allow_shared_part1 is not None else "before_dispatch"

    def shared_part2_anchor_name(self) -> str:
        return "allow_shared_part2" if self.allow_shared_part2 is not None else "before_combine"


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe=None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        else:
            layer.w13_weight.data = layer.w13_weight.data
            layer.w2_weight.data = layer.w2_weight.data

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> FusedExpertsResult:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts,
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids = topk_ids
            topk_weights = topk_weights

        topk_weights = topk_weights.to(x.dtype)
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), global_num_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2 = [layer.w2_weight]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
        else:
            w1 = layer.w13_weight
            w1_scale = None
            w2 = layer.w2_weight
            w2_scale = None

        fused_experts_input = vllm_ascend.ops.fused_moe.moe_runtime_args.build_fused_experts_input(
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            w1_bias=layer.w13_bias if self.moe.has_bias else None,
            w2_bias=layer.w2_bias if self.moe.has_bias else None,
            quant_type=QuantType.NONE,
            dynamic_eplb=self.dynamic_eplb,
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        return moe_comm_method.fused_experts(fused_experts_input=fused_experts_input)


class AscendFusedMoE(FusedMoE):
    gate_stream = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_expert_map = None
        self._expert_map = None
        self.log2phy = None
        self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config) if self.quant_config is None else self.quant_config.get_quant_method(self, self.layer_name)
        assert self.quant_method is not None

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        ascend_config = get_ascend_config()
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
            AscendFusedMoE.gate_stream = torch.npu.Stream()

        eplb_config = ascend_config.eplb_config
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            eplb_config, self.moe_instance_id, self.moe_config
        )
        self.global_num_experts = self.num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        self.local_num_experts = self.global_num_experts // self.ep_size
        if self._expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global number of experts: %s/%s. Experts local to global index map: %s.",
                self.ep_rank,
                self.ep_size,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel
        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()
        self.runner = self._init_runner()

    def _init_runner(self):
        return AscendMoERunner(
            layer=self,
            moe_config=self.moe_config,
            router=self.router,
            routed_input_transform=self._routed_input_transform,
            gate=self.gate,
            shared_experts=self.shared_experts,
            quant_method=self.quant_method,
            reduce_results=self.reduce_results,
            enable_dbo=self.vllm_config.parallel_config.enable_dbo,
        )

    def _get_quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)
        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)
        return quant_type

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        self.ensure_moe_quant_config_init()
        return self.runner.forward(hidden_states, router_logits)

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False):
        assert self.quant_method is not None
        forward_context = get_forward_context()
        if self.enable_npugraph_ex_static_kernel:
            moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
            forward_context.moe_layer_index = moe_layer_index

        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        if self.multistream_overlap_gate:
            assert AscendFusedMoE.gate_stream is not None
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            AscendFusedMoE.gate_stream.wait_stream(torch.npu.current_stream())
            with npu_stream_switch(AscendFusedMoE.gate_stream, enabled=self.multistream_overlap_gate):
                assert fc3_context.shared_experts is not None
                shared_out = fc3_context.shared_experts(hidden_states)
                moe_comm_type = _EXTRA_CTX.moe_comm_type
                if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2} and not shared_expert_dp_enabled():
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
                set_flash_common3_context(shared_out=shared_out)

                topk_weights, topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,
                    use_grouped_topk=self.use_grouped_topk,
                    renormalize=self.renormalize,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    global_num_experts=self.global_num_experts,
                )
                if isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                    topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_weights, True, True)
                    topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_ids, True, True)
                set_flash_common3_context(topk_weights=topk_weights, topk_ids=topk_ids)

        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )
        hidden_states = prepare_output.hidden_states
        router_logits = prepare_output.router_logits
        mc2_mask = prepare_output.mc2_mask
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape
        pertoken_scale = prepare_output.pertoken_scale

        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(AscendFusedMoE.gate_stream)

        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self._expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )

        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
                allow_shared_part1_evt=fused_experts_results.allow_shared_part1_evt,
                allow_shared_part2_evt=fused_experts_results.allow_shared_part2_evt,
            )
        return routed_out


class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):
    def __init__(
        self,
        shared_experts: torch.nn.Module,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        routed_input_transform: torch.nn.Module | None = None,
        **kwargs,
    ):
        AscendFusedMoE.__init__(self, **kwargs)
        self._routed_input_transform = routed_input_transform
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        ascend_config = get_ascend_config()
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert and self._shared_experts is not None
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate and self._shared_experts is not None
        if enable_sp():
            logger.info_once("Sequence parallelism is enabled, shared experts are replicated for best performance.")

        self._gate = gate
        self.runner = self._init_runner()

        if self.multistream_overlap_shared_expert:
            original_process_weights = self.quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self._validate_shared_expert_consistency()
                return result

            self.quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore
        if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
            gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def _validate_shared_expert_consistency(self):
        test_input = torch.rand(10, self.hidden_size, device="npu", dtype=self.moe_config.in_dtype) * 2 - 1
        integrated_out = self._shared_experts(test_input)
        part1_out = self._shared_experts_part1(test_input)
        split_out = self._shared_experts_part2(test_input, part1_out)
        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error("SharedFusedMoE shared experts split computation does not match the integrated computation.")
            logger.error(f"Max absolute difference: {diff.max().item()}")
            logger.error("Integrated output - sum: %s, norm: %s", integrated_out.sum().item(), integrated_out.norm().item())
            logger.error("Split output - sum: %s, norm: %s", split_out.sum().item(), split_out.norm().item())
            raise ValueError("SharedFusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once("SharedFusedMoE shared experts split computation matches the integrated computation.")

    @property
    def gate(self) -> torch.nn.Module | None:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return False

    @property
    def use_dp_chunking(self) -> bool:
        return False

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self._shared_experts is None:
            fused_out = AscendFusedMoE.forward(self, hidden_states=hidden_states, router_logits=router_logits)
            return None, fused_out
        shared_out, fused_out = AscendFusedMoE.forward(self, hidden_states=hidden_states, router_logits=router_logits)
        return shared_out, fused_out

    def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        if self._shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
            torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
            maybe_wait_event(fused_moe_evts.shared_part1_anchor())
            part1_out = self._shared_experts_part1(hidden_states)
            maybe_wait_event(fused_moe_evts.shared_part2_anchor())
            shared_out = self._shared_experts_part2(hidden_states, part1_out)

        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2} and not shared_expert_dp_enabled():
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self.multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self._shared_experts)

        before_routed_experts = torch.npu.current_stream().record_event()
        fused_moe_results = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out

        if self._shared_experts is None:
            return routed_out

        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            shared_out = fc3_context.shared_out
        else:
            shared_out = self._forward_shared_experts(
                hidden_states,
                FusedMoEEvents(
                    before_routed_experts=before_routed_experts,
                    before_dispatch=fused_moe_results.before_dispatch_evt,
                    before_combine=fused_moe_results.before_combine_evt,
                    allow_shared_part1=fused_moe_results.allow_shared_part1_evt,
                    allow_shared_part2=fused_moe_results.allow_shared_part2_evt,
                ),
            )

        return shared_out, routed_out
