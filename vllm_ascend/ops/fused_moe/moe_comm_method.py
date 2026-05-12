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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import build_mlp_stage_outputs, unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoERoutingParams
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

MICRO_BATCH_STAGE_NAMES = (
    "routing_topk",
    "cast_preprocess",
    "dispatch",
    "gmm1",
    "swiglu",
    "gmm2_downproj",
    "combine",
    "finalize_merge",
)


@dataclass
class MicroBatchEvents:
    routing_topk_done_evt: torch.npu.Event | None = None
    cast_preprocess_done_evt: torch.npu.Event | None = None
    dispatch_done_evt: torch.npu.Event | None = None
    gmm1_done_evt: torch.npu.Event | None = None
    swiglu_done_evt: torch.npu.Event | None = None
    gmm2_downproj_done_evt: torch.npu.Event | None = None
    combine_done_evt: torch.npu.Event | None = None


@dataclass
class MicroBatchPlan:
    enabled: bool
    batch_size: int
    min_tokens: int
    total_tokens: int
    split_sizes: tuple[int, int]
    mode: str


@dataclass
class MicroBatchChunk:
    batch_idx: int
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    mc2_mask: torch.Tensor | None
    pertoken_scale: torch.Tensor | None


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None


_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}


def _record_stage_event(stream: torch.npu.Stream | None = None) -> torch.npu.Event:
    current_stream = stream if stream is not None else torch.npu.current_stream()
    return current_stream.record_event()


def _wait_for_stage_event(event: torch.npu.Event | None, event_name: str, *, batch_idx: int, stage: str) -> None:
    if event is None:
        return
    _maybe_log_micro_batch_stage(batch_idx, stage, waits=[event_name], records=[])
    torch.npu.current_stream().wait_event(event)


def _maybe_log_micro_batch_plan(plan: MicroBatchPlan) -> None:
    if not envs_ascend.VLLM_ASCEND_MOE_PREFILL_MICROBATCH_DEBUG:
        return
    logger.info(
        "[MB-PLAN] enabled=%s total_tokens=%s split=%s min_tokens=%s mode=%s stages=%s",
        int(plan.enabled),
        plan.total_tokens,
        plan.split_sizes,
        plan.min_tokens,
        plan.mode,
        "->".join(MICRO_BATCH_STAGE_NAMES),
    )


def _maybe_log_micro_batch_stage(
    batch_idx: int,
    stage: str,
    *,
    waits: list[str] | None = None,
    records: list[str] | None = None,
    stream_name: str = "current",
) -> None:
    if not envs_ascend.VLLM_ASCEND_MOE_PREFILL_MICROBATCH_DEBUG:
        return
    logger.info(
        "[MB-STAGE] batch%s:%s stream=%s wait=%s record=%s",
        batch_idx,
        stage,
        stream_name,
        waits or [],
        records or [],
    )


def _resolve_micro_batch_mode() -> str:
    configured_mode = envs_ascend.VLLM_ASCEND_MOE_PREFILL_MICROBATCH_MODE
    if configured_mode:
        if configured_mode not in {"off", "auto", "conservative"}:
            logger.warning(
                "Invalid VLLM_ASCEND_MOE_PREFILL_MICROBATCH_MODE=%s, fallback to auto",
                configured_mode,
            )
            return "auto"
        return configured_mode
    if envs_ascend.VLLM_ASCEND_ENABLE_MOE_PREFILL_MICROBATCH_OVERLAP:
        return "auto"
    return "off"


def build_micro_batch_plan(num_tokens: int) -> MicroBatchPlan:
    mode = _resolve_micro_batch_mode()
    min_tokens = envs_ascend.VLLM_ASCEND_MOE_PREFILL_MICROBATCH_MIN_TOKENS
    if mode == "off" or (mode == "auto" and num_tokens < min_tokens):
        return MicroBatchPlan(
            enabled=False,
            batch_size=1,
            min_tokens=min_tokens,
            total_tokens=num_tokens,
            split_sizes=(num_tokens, 0),
            mode="legacy",
        )

    batch0 = (num_tokens + 1) // 2
    batch1 = num_tokens - batch0
    plan = MicroBatchPlan(
        enabled=True,
        batch_size=2,
        min_tokens=min_tokens,
        total_tokens=num_tokens,
        split_sizes=(batch0, batch1),
        mode="conservative",
    )
    _maybe_log_micro_batch_plan(plan)
    return plan


def _split_optional_tensor(
    value: torch.Tensor | None,
    plan: MicroBatchPlan,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if value is None:
        return None, None
    if not plan.enabled or value.dim() == 0:
        return value, None
    batch0_tokens, batch1_tokens = plan.split_sizes
    first = value[:batch0_tokens]
    second = value[batch0_tokens : batch0_tokens + batch1_tokens] if batch1_tokens > 0 else None
    return first, second


def build_micro_batch_fused_inputs(
    fused_experts_input: MoEFusedExpertsInput,
    plan: MicroBatchPlan,
) -> list[MoEFusedExpertsInput]:
    if not plan.enabled:
        return [fused_experts_input]

    batch0_tokens, batch1_tokens = plan.split_sizes
    topk_weights0, topk_weights1 = _split_optional_tensor(fused_experts_input.topk_weights, plan)
    topk_ids0, topk_ids1 = _split_optional_tensor(fused_experts_input.topk_ids, plan)
    mc2_mask0, mc2_mask1 = _split_optional_tensor(fused_experts_input.routing.mc2_mask, plan)
    pertoken_scale0, pertoken_scale1 = _split_optional_tensor(fused_experts_input.routing.pertoken_scale, plan)

    outputs = [
        MoEFusedExpertsInput(
            hidden_states=fused_experts_input.hidden_states[:batch0_tokens],
            topk_weights=topk_weights0,
            topk_ids=topk_ids0,
            weights=fused_experts_input.weights,
            routing=MoERoutingParams(
                expert_map=fused_experts_input.routing.expert_map,
                global_redundant_expert_num=fused_experts_input.routing.global_redundant_expert_num,
                mc2_mask=mc2_mask0,
                apply_router_weight_on_input=fused_experts_input.routing.apply_router_weight_on_input,
                log2phy=fused_experts_input.routing.log2phy,
                pertoken_scale=pertoken_scale0,
            ),
            quant=fused_experts_input.quant,
            activation=fused_experts_input.activation,
            need_trans=fused_experts_input.need_trans,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
        )
    ]
    if batch1_tokens > 0:
        outputs.append(
            MoEFusedExpertsInput(
                hidden_states=fused_experts_input.hidden_states[batch0_tokens : batch0_tokens + batch1_tokens],
                topk_weights=topk_weights1,
                topk_ids=topk_ids1,
                weights=fused_experts_input.weights,
                routing=MoERoutingParams(
                    expert_map=fused_experts_input.routing.expert_map,
                    global_redundant_expert_num=fused_experts_input.routing.global_redundant_expert_num,
                    mc2_mask=mc2_mask1,
                    apply_router_weight_on_input=fused_experts_input.routing.apply_router_weight_on_input,
                    log2phy=fused_experts_input.routing.log2phy,
                    pertoken_scale=pertoken_scale1,
                ),
                quant=fused_experts_input.quant,
                activation=fused_experts_input.activation,
                need_trans=fused_experts_input.need_trans,
                dynamic_eplb=fused_experts_input.dynamic_eplb,
            )
        )
    return outputs


def merge_micro_batch_outputs(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
    return _MoECommMethods.get(moe_comm_type)


def setup_moe_comm_method(moe_config):
    _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
    _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
    _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
    _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config
        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert fused_experts_input.hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int8]
        plan = build_micro_batch_plan(fused_experts_input.hidden_states.shape[0])
        fused_inputs = build_micro_batch_fused_inputs(fused_experts_input, plan)

        if len(fused_inputs) == 1:
            return self._run_stage_pipeline(fused_inputs[0], batch_idx=0)

        results: list[FusedExpertsResult] = []
        batch_events = [MicroBatchEvents(), MicroBatchEvents()]
        for batch_idx, chunk_input in enumerate(fused_inputs):
            result = self._run_stage_pipeline(
                chunk_input,
                batch_idx=batch_idx,
                previous_events=batch_events[batch_idx - 1] if batch_idx > 0 else None,
                current_events=batch_events[batch_idx],
            )
            results.append(result)

        return FusedExpertsResult(
            routed_out=merge_micro_batch_outputs([result.routed_out for result in results]),
            before_dispatch_evt=results[0].before_dispatch_evt,
            before_combine_evt=results[-1].before_combine_evt,
            group_list_type=results[-1].group_list_type,
            expert_tokens=results[-1].expert_tokens,
        )

    def _run_stage_pipeline(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        *,
        batch_idx: int,
        previous_events: MicroBatchEvents | None = None,
        current_events: MicroBatchEvents | None = None,
    ) -> FusedExpertsResult:
        if current_events is None:
            current_events = MicroBatchEvents()

        if previous_events is not None:
            _wait_for_stage_event(
                previous_events.dispatch_done_evt,
                "batch0.dispatch_done_evt",
                batch_idx=batch_idx,
                stage="routing_topk",
            )
        _maybe_log_micro_batch_stage(batch_idx, "routing_topk", waits=[], records=["routing_topk_done_evt"])
        current_events.routing_topk_done_evt = _record_stage_event()

        _wait_for_stage_event(
            current_events.routing_topk_done_evt,
            "routing_topk_done_evt",
            batch_idx=batch_idx,
            stage="cast_preprocess",
        )
        _maybe_log_micro_batch_stage(
            batch_idx,
            "cast_preprocess",
            waits=[],
            records=["cast_preprocess_done_evt"],
        )
        current_events.cast_preprocess_done_evt = _record_stage_event()

        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        _wait_for_stage_event(
            current_events.cast_preprocess_done_evt,
            "cast_preprocess_done_evt",
            batch_idx=batch_idx,
            stage="dispatch",
        )
        before_dispatch_evt = _record_stage_event()
        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        _maybe_log_micro_batch_stage(
            batch_idx,
            "dispatch",
            waits=[],
            records=["dispatch_done_evt"],
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)
        current_events.dispatch_done_evt = _record_stage_event()

        _wait_for_stage_event(
            current_events.dispatch_done_evt,
            "dispatch_done_evt",
            batch_idx=batch_idx,
            stage="gmm1",
        )
        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )
        mlp_stage_outputs = build_mlp_stage_outputs(mlp_compute_input=mlp_compute_input)
        _maybe_log_micro_batch_stage(batch_idx, "gmm1", waits=[], records=["gmm1_done_evt"])
        current_events.gmm1_done_evt = _record_stage_event()

        _wait_for_stage_event(
            current_events.gmm1_done_evt,
            "gmm1_done_evt",
            batch_idx=batch_idx,
            stage="swiglu",
        )
        _maybe_log_micro_batch_stage(batch_idx, "swiglu", waits=[], records=["swiglu_done_evt"])
        current_events.swiglu_done_evt = _record_stage_event()

        _wait_for_stage_event(
            current_events.swiglu_done_evt,
            "swiglu_done_evt",
            batch_idx=batch_idx,
            stage="gmm2_downproj",
        )
        if previous_events is not None:
            _wait_for_stage_event(
                previous_events.gmm2_downproj_done_evt,
                "batch0.gmm2_downproj_done_evt",
                batch_idx=batch_idx,
                stage="gmm2_downproj",
            )
        _maybe_log_micro_batch_stage(
            batch_idx,
            "gmm2_downproj",
            waits=[],
            records=["gmm2_downproj_done_evt"],
        )
        current_events.gmm2_downproj_done_evt = _record_stage_event()

        _wait_for_stage_event(
            current_events.gmm2_downproj_done_evt,
            "gmm2_downproj_done_evt",
            batch_idx=batch_idx,
            stage="combine",
        )
        before_combine_evt = _record_stage_event()
        _maybe_log_micro_batch_stage(
            batch_idx,
            "combine",
            waits=[],
            records=["combine_done_evt"],
        )
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_stage_outputs.gmm2_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )
        current_events.combine_done_evt = _record_stage_event()

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    def __init__(self, moe_config):
        super().__init__(moe_config)
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
            "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."
        )
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            out = torch.empty_like(fused_experts_input.hidden_states)
            torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                x=fused_experts_input.hidden_states,
                weight1=fused_experts_input.weights.w1,
                weight2=fused_experts_input.weights.w2,
                expert_idx=topk_ids,
                scale1=fused_experts_input.weights.w1_scale,
                scale2=fused_experts_input.weights.w2_scale,
                probs=fused_experts_input.topk_weights.to(torch.float32),
                group=self.token_dispatcher.moe_all_to_all_group_name,
                max_output_size=65536,
                out=out,
                expert_token_nums=self.expert_token_nums,
            )
            expert_tokens = self.expert_token_nums
        elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
            assert fused_experts_input.routing.expert_map is not None, "expert_map cannot be None."
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                x=fused_experts_input.hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=fused_experts_input.weights.w1,
                gmm1_permuted_weight_scale=fused_experts_input.weights.w1_scale,
                gmm2_weight=fused_experts_input.weights.w2,
                gmm2_weight_scale=fused_experts_input.weights.w2_scale,
                expert_smooth_scales=None,
                expert_scales=fused_experts_input.topk_weights.to(torch.float32),
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                ep_rank_size=self.token_dispatcher.ep_world_size,
                ep_rank_id=self.token_dispatcher.ep_rank_id,
                moe_expert_num=self.moe_config.num_experts,
                global_bs=self.token_dispatcher.global_bs,
            )
        else:
            raise ValueError(f"Wrong value of {envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2=}")
        return FusedExpertsResult(routed_out=out, expert_tokens=expert_tokens)
