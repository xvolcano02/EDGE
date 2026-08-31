# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector
        self.best_checkpoint_score = float("-inf")

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        core_algos.validate_edge_distillation_config(config)
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_trajectory_generations(self, batch, reward_extra_infos_dict, dump_path, filename_prefix=""):
        """Dump complete per-task trajectories (init -> last step) as JSONL.

        Each line in the output file is ONE trajectory, grouping all of its
        per-step prompts/responses/rewards ordered by ``step_idx``. This is
        needed for agentic envs (e.g. ALFWorld) where the live training prompt
        only keeps a truncated ``history_length`` window; to recover the full
        trajectory we must re-aggregate the per-step rows that ``rollout_loop``
        produced.

        ``filename_prefix`` lets callers disambiguate train vs. validation dumps
        when ``rollout_data_dir`` and ``validation_data_dir`` point to the same
        directory (e.g. ``filename_prefix='val_'`` -> ``val_{global_steps}.jsonl``).

        Requires ``traj_uid`` and ``step_idx`` to be present in
        ``batch.non_tensor_batch`` (added by ``vanilla_multi_turn_loop``);
        otherwise falls back to per-step dumping.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{filename_prefix}{self.global_steps}.jsonl")

        non_tensor = batch.non_tensor_batch
        if "traj_uid" not in non_tensor or "step_idx" not in non_tensor:
            # Legacy per-step fallback. Honor filename_prefix so this is safe
            # even when rollout/validation dirs point to the same folder.
            n_fallback = len(batch.batch["prompts"])
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [self.global_steps] * n_fallback,
            }
            for k, v in (reward_extra_infos_dict or {}).items():
                if len(v) == n_fallback:
                    base_data[k] = v
            with open(filename, "w") as f:
                for i in range(n_fallback):
                    entry = {k: v[i] for k, v in base_data.items()}
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            print(f"Dumped per-step generations (fallback) to {filename}")
            return

        n = len(batch.batch["prompts"])
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        token_scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

        traj_uids = non_tensor["traj_uid"]
        step_indices = non_tensor["step_idx"]

        step_rewards = non_tensor.get("rewards", None)
        active_masks_col = non_tensor.get("active_masks", None)
        is_action_valid_col = non_tensor.get("is_action_valid", None)
        uid_col = non_tensor.get("uid", None)
        data_source_col = non_tensor.get("data_source", None)
        episode_rewards_col = non_tensor.get("episode_rewards", None)
        episode_lengths_col = non_tensor.get("episode_lengths", None)
        tool_callings_col = non_tensor.get("tool_callings", None)

        extra_cols = {}
        for k, v in (reward_extra_infos_dict or {}).items():
            if len(v) == n:
                extra_cols[k] = v

        def _to_py(val):
            if hasattr(val, "item"):
                try:
                    return val.item()
                except Exception:
                    pass
            return val

        trajectories: dict = {}
        seen: set = set()
        # adjust_batch(mode='copy') may duplicate rows to satisfy dp divisor;
        # dedupe on (traj_uid, step_idx) so each step appears once.
        for i in range(n):
            traj_uid = str(traj_uids[i])
            step_idx = int(step_indices[i])
            key = (traj_uid, step_idx)
            if key in seen:
                continue
            seen.add(key)

            step_entry = {
                "step_idx": step_idx,
                "input": inputs[i],
                "output": outputs[i],
                "token_level_score": token_scores[i],
            }
            for metadata_key in (
                "experience_ids",
                "experience_query_hash",
                "experience_task_type",
                "experience_retrieval_mode",
                "task_description",
                "current_observation",
                "admissible_actions",
                "current_step",
                "experience_injected",
                "anchor_obs",
            ):
                if metadata_key in non_tensor:
                    step_entry[metadata_key] = _to_py(non_tensor[metadata_key][i])
            if step_rewards is not None:
                step_entry["env_reward"] = _to_py(step_rewards[i])
            if active_masks_col is not None:
                step_entry["active_mask"] = bool(active_masks_col[i])
            if is_action_valid_col is not None:
                step_entry["is_action_valid"] = bool(is_action_valid_col[i])
            for k, v in extra_cols.items():
                step_entry[k] = _to_py(v[i])

            traj = trajectories.setdefault(
                traj_uid,
                {"traj_uid": traj_uid, "global_step": self.global_steps, "steps": []},
            )
            if uid_col is not None and "uid" not in traj:
                traj["uid"] = str(uid_col[i])
            if data_source_col is not None and "data_source" not in traj:
                traj["data_source"] = str(data_source_col[i])
            if episode_rewards_col is not None and "episode_reward" not in traj:
                traj["episode_reward"] = _to_py(episode_rewards_col[i])
            if episode_lengths_col is not None and "episode_length" not in traj:
                traj["episode_length"] = _to_py(episode_lengths_col[i])
            if tool_callings_col is not None and "tool_callings" not in traj:
                traj["tool_callings"] = _to_py(tool_callings_col[i])

            traj["steps"].append(step_entry)

        with open(filename, "w") as f:
            for traj in trajectories.values():
                traj["steps"].sort(key=lambda s: s["step_idx"])
                f.write(json.dumps(traj, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped {len(trajectories)} trajectories to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        episode_length_list = []
        response_length_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # Accumulate per-validation-batch DataProto with token_level_scores attached,
        # so that if validation_data_dir is set we can dump the full agentic trajectories
        # once validation finishes.
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        val_dump_batches = [] if val_data_dir else None

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            if val_dump_batches is not None:
                # attach token_level_scores for this validation batch and stash it
                # so _dump_trajectory_generations can regroup rows by trajectory.
                test_batch.batch["token_level_scores"] = reward_tensor
                val_dump_batches.append(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            episode_length_list.append(test_output_gen_batch.non_tensor_batch['episode_lengths'])
            response_mask = test_output_gen_batch.batch["attention_mask"][
                :, -test_output_gen_batch.batch["responses"].shape[-1]:
            ]
            response_length_list.append(response_mask.sum(-1).cpu().numpy())
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if self._is_success_rate_key(k):
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Dump complete per-task validation trajectories (init -> last step) if configured.
        if val_dump_batches:
            try:
                full_val_batch = DataProto.concat(val_dump_batches)
            except Exception as e:
                print(f"Warning: failed to concat validation batches for dump ({e}); "
                      f"dumping each val batch separately instead.")
                full_val_batch = None

            if full_val_batch is not None:
                self._dump_trajectory_generations(
                    batch=full_val_batch,
                    reward_extra_infos_dict={},
                    dump_path=val_data_dir,
                    filename_prefix="val_",
                )
            else:
                for idx, b in enumerate(val_dump_batches):
                    sub_dir = os.path.join(val_data_dir, f"batch_{idx}")
                    self._dump_trajectory_generations(
                        batch=b,
                        reward_extra_infos_dict={},
                        dump_path=sub_dir,
                        filename_prefix="val_",
                    )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        episode_lengths = np.concatenate(episode_length_list, axis=0)
        response_lengths = np.concatenate(response_length_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        unique_episode_lengths = episode_lengths[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        # Aggregate per-trajectory statistics so each task contributes once.
        data_source_step = defaultdict(list)
        data_source_trajectory_response_length = defaultdict(list)
        traj_response_length = defaultdict(float)
        for traj_uid, response_length in zip(traj_uids, response_lengths):
            traj_response_length[traj_uid] += float(response_length)

        for i, traj_uid in enumerate(unique_traj_uid):
            data_source = unique_data_sources[i]
            data_source_step[data_source].append(float(unique_episode_lengths[i]))
            data_source_trajectory_response_length[data_source].append(traj_response_length[traj_uid])

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for data_source, step_lengths in data_source_step.items():
            metric_dict[f'val/{data_source}/step'] = np.mean(step_lengths)

        for data_source, trajectory_response_lengths in data_source_trajectory_response_length.items():
            metric_dict[f'val/{data_source}/trajectory_response_length'] = np.mean(trajectory_response_lengths)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        # Experience evolution is intentionally not driven by validation/test
        # trajectories.  Dynamic experience updates happen from training rollouts in
        # fit(), avoiding validation-set exposure.

        return metric_dict

    def _is_experience_evolution_enabled(self) -> bool:
        """
        Return whether experience evolution is enabled.

        Preferred flag:
            env.experience_memory.enable_experience_evolution

        Backward-compatible alias:
            env.experience_memory.enable_dynamic_update
        """
        update_config = self.config.env.get('experience_memory', {})
        enabled = bool(update_config.get('enable_experience_evolution', False))
        if not enabled:
            enabled = bool(update_config.get('enable_dynamic_update', False))

        if not enabled:
            return False
        return True

    def _is_dynamic_experience_management_enabled(self) -> bool:
        """Whether training-rollout utility management is explicitly enabled."""
        update_config = self.config.env.get("experience_memory", {}) or {}
        return bool(update_config.get("enable_dynamic_management", False))

    def _is_edge_distillation_enabled(self) -> bool:
        return core_algos.is_edge_distillation_rollout_enabled(
            self.config,
            is_train=True,
            global_step=self.global_steps,
        )

    @staticmethod
    def _normalise_metadata_bool(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return bool(value.item()) if value.shape == () else bool(value.astype(bool).any())
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else False
        return bool(value)

    @staticmethod
    def _normalise_anchor_key(anchor: Any) -> str:
        if isinstance(anchor, np.ndarray):
            if anchor.shape == ():
                return str(anchor.item())
            return json.dumps(anchor.tolist(), ensure_ascii=False, sort_keys=True, default=str)
        if isinstance(anchor, (list, tuple)):
            return json.dumps(list(anchor), ensure_ascii=False, sort_keys=True, default=str)
        if isinstance(anchor, dict):
            return json.dumps(anchor, ensure_ascii=False, sort_keys=True, default=str)
        return str(anchor)

    @staticmethod
    def _normalise_experience_ids(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple, set)):
            ids = []
            for item in value:
                ids.extend(RayPPOTrainer._normalise_experience_ids(item))
            return ids
        return [str(value)] if str(value) else []

    @staticmethod
    def _safe_float_array(values: Any, length: int, default: float = 0.0) -> np.ndarray:
        if values is None:
            return np.full(length, default, dtype=np.float32)
        try:
            return np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            return np.full(length, default, dtype=np.float32)

    def _row_score_vector(self, batch: DataProto) -> tuple[np.ndarray, str]:
        """Score rows for same-anchor contrasts.

        Preference order follows the V2 plan:
        trajectory_success > episode_rewards > per-turn rewards.
        """
        non_tensor = batch.non_tensor_batch
        batch_size = len(batch)
        if "trajectory_success" in non_tensor:
            return self._safe_float_array(non_tensor["trajectory_success"], batch_size), "trajectory_success"
        if "episode_rewards" in non_tensor:
            return self._safe_float_array(non_tensor["episode_rewards"], batch_size), "episode_rewards"
        if "rewards" in non_tensor:
            return self._safe_float_array(non_tensor["rewards"], batch_size), "rewards"
        return np.zeros(batch_size, dtype=np.float32), "none"

    def _same_anchor_groups(self, batch: DataProto) -> dict[tuple[str, str], dict[bool, list[int]]]:
        non_tensor = batch.non_tensor_batch
        batch_size = len(batch)
        groups = defaultdict(lambda: {True: [], False: []})
        for i in range(batch_size):
            uid = str(non_tensor["uid"][i])
            anchor = self._normalise_anchor_key(non_tensor["anchor_obs"][i])
            injected = self._normalise_metadata_bool(non_tensor["experience_injected"][i])
            groups[(uid, anchor)][injected].append(i)
        return groups

    def _compute_edge_distillation_weights(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute current-batch EDGE weights from exact same-anchor contrasts."""
        device = batch.batch["responses"].device
        batch_size = len(batch)
        weights_np = np.zeros(batch_size, dtype=np.float32)

        required_keys = {"uid", "anchor_obs", "experience_injected"}
        missing = required_keys - set(batch.non_tensor_batch.keys())
        if missing:
            batch.batch["edge_distill_weight"] = torch.zeros(batch_size, dtype=torch.float32, device=device)
            return batch, {
                "edge/positive_group_fraction": 0.0,
                "edge/distill_weight_mean": 0.0,
                "edge/distill_weight_max": 0.0,
                "edge/active_row_fraction": 0.0,
                "edge/missing_metadata": 1.0,
            }

        edge_cfg = self.config.actor_rollout_ref.actor.get("edge_distillation", {})
        require_positive_gain = bool(edge_cfg.get("require_positive_gain", True))
        target = edge_cfg.get("target", "w_o_experience_only")
        scores, score_source = self._row_score_vector(batch)
        groups = self._same_anchor_groups(batch)

        positive_groups = 0
        paired_groups = 0
        for by_injection in groups.values():
            exp_rows = by_injection[True]
            no_exp_rows = by_injection[False]
            if not exp_rows or not no_exp_rows:
                continue
            paired_groups += 1
            exp_score = float(np.max(scores[exp_rows]))
            no_exp_score = float(np.max(scores[no_exp_rows]))

            if require_positive_gain:
                group_weight = 1.0 if exp_score > 0.0 and no_exp_score <= 0.0 else 0.0
            else:
                group_weight = max(0.0, exp_score - no_exp_score)
            if group_weight <= 0:
                continue
            positive_groups += 1

            if target == "w_o_experience_only":
                target_rows = [idx for idx in no_exp_rows if scores[idx] <= 0.0]
            elif target == "all":
                target_rows = exp_rows + no_exp_rows
            else:
                target_rows = no_exp_rows
            for idx in target_rows:
                weights_np[idx] = group_weight

        weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
        batch.batch["edge_distill_weight"] = weights
        metrics = {
            "edge/positive_group_fraction": positive_groups / max(len(groups), 1),
            "edge/paired_group_fraction": paired_groups / max(len(groups), 1),
            "edge/distill_weight_mean": float(weights_np.mean()) if batch_size > 0 else 0.0,
            "edge/distill_weight_max": float(weights_np.max()) if batch_size > 0 else 0.0,
            "edge/active_row_fraction": float(np.mean(weights_np > 0)) if batch_size > 0 else 0.0,
            "edge/missing_metadata": 0.0,
            f"edge/score_source/{score_source}": 1.0,
        }
        return batch, metrics

    def _update_experiences_from_training_rollouts(self, batch: DataProto):
        """Update the V2 experience bank from exact same-anchor contrasts."""
        update_config = self.config.env.experience_memory

        update_frequency = int(update_config.get('update_frequency', 1))
        if update_frequency > 1 and self.global_steps % update_frequency != 0:
            print(
                f"[ExperienceUpdate] Skip at step {self.global_steps}; "
                f"update_frequency={update_frequency}"
            )
            return

        retrieval_memory = self._get_experience_bank()
        if retrieval_memory is None:
            print("[ExperienceUpdate] No retrieval_memory found in envs")
            return

        gain_metrics, contrast_examples = self._update_experience_gain_from_batch(
            batch=batch,
            retrieval_memory=retrieval_memory,
        )
        print(f"[ExperienceUpdate] gain metrics: {gain_metrics}")

        success_rate = self._collect_training_success_rates(batch)
        if success_rate:
            print(f"[ExperienceUpdate] success rates: {success_rate}")
        else:
            print("[ExperienceUpdate] No success-rate metadata found")

        self._update_experiences_from_contrasts(
            contrast_examples=contrast_examples,
            success_rate=success_rate,
            retrieval_memory=retrieval_memory,
        )

        save_dir = self.config.trainer.get('default_local_dir', './outputs')
        save_path = os.path.join(save_dir, f'updated_experiences_step{self.global_steps}.json')
        retrieval_memory.save_experiences(save_path)
        print(f"[ExperienceUpdate] Saved updated experience bank to {save_path}")
        canonical_path = update_config.get('experiences_json_path')
        if canonical_path and update_config.get('save_to_canonical_path', True):
            retrieval_memory.save_experiences(canonical_path)
            print(f"[ExperienceUpdate] Saved canonical experience bank to {canonical_path}")
        self._sync_experiences_to_retrieval_server(retrieval_memory, path=save_path)

    def _get_experience_bank(self):
        if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory'):
            return self.envs.retrieval_memory
        return None

    def _sync_experiences_to_retrieval_server(self, retrieval_memory, path: Optional[str] = None) -> None:
        """Push current experiences to the remote retrieval service after dynamic updates."""
        som_cfg = self.config.env.get("experience_memory", {}) or {}
        if not som_cfg.get("experience_retrieval_service_url"):
            return
        if retrieval_memory is None:
            return
        experiences = getattr(retrieval_memory, "experiences", None)
        experience_count = 0
        if isinstance(experiences, dict):
            experience_count = len(experiences.get("task_experiences", [])) + len(experiences.get("step_experiences", []))
        if experience_count == 0 and not path:
            return
        sync_mode = str(som_cfg.get("experience_retrieval_reload_mode", "inline")).lower().strip()
        try:
            if hasattr(retrieval_memory, "reload_remote_experiences"):
                if sync_mode == "path" and path:
                    ok = retrieval_memory.reload_remote_experiences(path=path)
                else:
                    ok = retrieval_memory.reload_remote_experiences(experiences=experiences)
                if ok:
                    print("[ExperienceRetrieval] Synced updated experiences to remote retrieval service")
                else:
                    print("[ExperienceRetrieval] Remote retrieval sync skipped or failed")
        except Exception as exc:
            print(f"[ExperienceRetrieval] Warning: failed to sync remote retrieval service: {exc}")

    def _update_experience_gain_from_batch(self, batch: DataProto, retrieval_memory) -> tuple[dict, list]:
        non_tensor = batch.non_tensor_batch
        required_keys = {"uid", "anchor_obs", "experience_injected", "experience_ids"}
        missing = required_keys - set(non_tensor.keys())
        if missing:
            return {
                "experience_gain/missing_metadata": 1.0,
                "experience_gain/positive_updates": 0,
                "experience_gain/negative_updates": 0,
                "experience_gain/paired_groups": 0,
            }, []

        scores, score_source = self._row_score_vector(batch)
        groups = self._same_anchor_groups(batch)
        positive_updates = 0
        negative_updates = 0
        paired_groups = 0
        contrast_examples = []

        for (uid, anchor_key), by_injection in groups.items():
            exp_rows = by_injection[True]
            no_exp_rows = by_injection[False]
            if not exp_rows or not no_exp_rows:
                continue
            paired_groups += 1
            exp_best = max(exp_rows, key=lambda idx: float(scores[idx]))
            no_exp_best = max(no_exp_rows, key=lambda idx: float(scores[idx]))
            exp_score = float(scores[exp_best])
            no_exp_score = float(scores[no_exp_best])

            exp_ids = self._normalise_experience_ids(non_tensor["experience_ids"][exp_best])
            if exp_score > no_exp_score:
                traj_uid_for_gain = str(non_tensor["traj_uid"][exp_best]) if "traj_uid" in non_tensor else uid
                positive_updates += retrieval_memory.record_experience_gain(
                    exp_ids,
                    positive=True,
                    traj_uid=traj_uid_for_gain,
                    global_step=self.global_steps,
                )
            elif exp_score < no_exp_score:
                traj_uid_for_gain = str(non_tensor["traj_uid"][exp_best]) if "traj_uid" in non_tensor else uid
                negative_updates += retrieval_memory.record_experience_gain(
                    exp_ids,
                    positive=False,
                    traj_uid=traj_uid_for_gain,
                    global_step=self.global_steps,
                )
                if exp_score <= 0.0 and no_exp_score > 0.0:
                    contrast_examples.append(
                        self._build_same_anchor_contrast_example(
                            batch=batch,
                            exp_idx=exp_best,
                            no_exp_idx=no_exp_best,
                            exp_score=exp_score,
                            no_exp_score=no_exp_score,
                            anchor_key=anchor_key,
                            contrast_type="experience_hurt",
                        )
                    )

        metrics = {
            "experience_gain/missing_metadata": 0.0,
            "experience_gain/positive_updates": positive_updates,
            "experience_gain/negative_updates": negative_updates,
            "experience_gain/paired_groups": paired_groups,
            f"experience_gain/score_source/{score_source}": 1.0,
        }
        return metrics, contrast_examples

    def _dynamic_management_config(self) -> dict:
        update_config = self.config.env.get("experience_memory", {}) or {}
        management = update_config.get("management", {}) or {}
        # OmegaConf DictConfig is intentionally converted only at this shallow
        # boundary; the rest of the lifecycle works with ordinary mappings.
        return dict(management)

    def _dynamic_management_interval(self) -> int:
        update_config = self.config.env.get("experience_memory", {}) or {}
        management = self._dynamic_management_config()
        interval = management.get("interval_steps", None)
        if interval is None:
            interval = update_config.get("update_frequency", 1)
        try:
            return max(1, int(interval))
        except (TypeError, ValueError):
            return 1

    def _collect_dynamic_management_credits(self, batch: DataProto) -> tuple[dict, dict, list, list]:
        """Build uid-level A/B gains and per-experience grouped credits.

        ``enable_exp_rollout`` already generates the two arms.  This method
        only reads its metadata; it never creates, reshuffles, or extends an
        A/B mask.  Scores are first collapsed to complete ``traj_uid``
        trajectories and then averaged inside each uid group.
        """
        non_tensor = batch.non_tensor_batch
        required = {"uid", "traj_uid", "experience_injected", "experience_ids"}
        missing = sorted(required - set(non_tensor.keys()))
        if missing:
            return (
                {},
                {
                    "experience_management/missing_metadata": 1.0,
                    "experience_management/paired_groups": 0.0,
                },
                [],
                [],
            )

        scores, score_source = self._row_score_vector(batch)
        trajectories: dict[tuple[str, str], dict] = {}
        for row_idx in range(len(batch)):
            uid = self._normalise_anchor_key(non_tensor["uid"][row_idx])
            traj_uid = self._normalise_anchor_key(non_tensor["traj_uid"][row_idx])
            key = (uid, traj_uid)
            record = trajectories.setdefault(
                key,
                {
                    "uid": uid,
                    "traj_uid": traj_uid,
                    "injected": False,
                    "rows": [],
                    "experience_ids": set(),
                },
            )
            injected = self._normalise_metadata_bool(non_tensor["experience_injected"][row_idx])
            record["injected"] = record["injected"] or injected
            record["rows"].append(row_idx)
            if injected:
                record["experience_ids"].update(
                    self._normalise_experience_ids(non_tensor["experience_ids"][row_idx])
                )

        groups: dict[str, dict[bool, list[dict]]] = defaultdict(lambda: {True: [], False: []})
        for record in trajectories.values():
            row_scores = [float(scores[row_idx]) for row_idx in record["rows"]]
            # trajectory_success and episode_rewards are trajectory-level
            # metadata commonly repeated on every action row.  Per-turn
            # rewards are the documented final fallback and must be summed to
            # obtain a trajectory score.
            if score_source == "rewards":
                record["score"] = float(sum(row_scores))
            else:
                record["score"] = float(np.mean(row_scores)) if row_scores else 0.0
            record["experience_ids"] = sorted(record["experience_ids"])
            groups[record["uid"]][bool(record["injected"])].append(record)

        credits: dict[str, dict] = {}
        group_audits: list[dict] = []
        contrasts: list[dict] = []
        paired_groups = 0
        groups_with_experience = 0
        for uid, by_injection in groups.items():
            exp_trajectories = by_injection[True]
            baseline_trajectories = by_injection[False]
            if not exp_trajectories or not baseline_trajectories:
                continue
            paired_groups += 1
            exp_ids = sorted({exp_id for item in exp_trajectories for exp_id in item["experience_ids"]})
            exp_mean = float(np.mean([item["score"] for item in exp_trajectories]))
            baseline_mean = float(np.mean([item["score"] for item in baseline_trajectories]))
            gain = exp_mean - baseline_mean
            group_audit = {
                "uid": uid,
                "with_experience_trajectory_count": len(exp_trajectories),
                "without_experience_trajectory_count": len(baseline_trajectories),
                "with_experience_mean_score": exp_mean,
                "without_experience_mean_score": baseline_mean,
                "gain": gain,
                "experience_ids": exp_ids,
            }
            group_audits.append(group_audit)
            if not exp_ids:
                continue
            groups_with_experience += 1
            # A group contributes one credit per experience (deduplicated over
            # all injected trajectories).  Exposure remains the count of
            # actual injected trajectories containing that experience.
            for exp_id in exp_ids:
                payload = credits.setdefault(exp_id, {"credits": [], "exposures": 0})
                payload["credits"].append(gain)
                payload["exposures"] += sum(
                    1 for trajectory in exp_trajectories if exp_id in trajectory["experience_ids"]
                )

            worst_exp = min(exp_trajectories, key=lambda item: item["score"])
            best_baseline = max(baseline_trajectories, key=lambda item: item["score"])
            if worst_exp["score"] <= 0.0 and best_baseline["score"] > 0.0:
                contrasts.append(
                    self._build_same_anchor_contrast_example(
                        batch=batch,
                        exp_idx=worst_exp["rows"][-1],
                        no_exp_idx=best_baseline["rows"][-1],
                        exp_score=float(worst_exp["score"]),
                        no_exp_score=float(best_baseline["score"]),
                        anchor_key=uid,
                        contrast_type="group_experience_hurt",
                    )
                )

        metrics = {
            "experience_management/missing_metadata": 0.0,
            "experience_management/uid_groups": float(len(groups)),
            "experience_management/paired_groups": float(paired_groups),
            "experience_management/paired_groups_with_experience": float(groups_with_experience),
            "experience_management/credited_experiences": float(len(credits)),
            f"experience_management/score_source/{score_source}": 1.0,
        }
        return credits, metrics, group_audits, contrasts

    def _write_dynamic_management_audit(
        self,
        *,
        management_record: dict,
        retrieval_memory=None,
        evictions: Optional[dict] = None,
        save_bank: bool = False,
    ) -> Optional[str]:
        """Persist lifecycle audit files without letting I/O stop training."""
        save_dir = self.config.trainer.get("default_local_dir", "./outputs")
        try:
            os.makedirs(save_dir, exist_ok=True)
            step = int(self.global_steps)
            management_path = os.path.join(save_dir, f"experience_management_step{step}.json")
            with open(management_path, "w") as handle:
                json.dump(management_record, handle, indent=2, ensure_ascii=False, default=self._to_jsonable)
            bank_path = None
            if save_bank and retrieval_memory is not None:
                bank_path = os.path.join(save_dir, f"experience_bank_step{step}.json")
                retrieval_memory.save_experiences(bank_path)
                evicted_path = os.path.join(save_dir, f"evicted_experiences_step{step}.json")
                with open(evicted_path, "w") as handle:
                    json.dump((evictions or {}).get("removed", []), handle, indent=2, ensure_ascii=False, default=self._to_jsonable)
            return bank_path
        except Exception as exc:
            print(f"[ExperienceManagement] Warning: failed to write audit snapshot: {exc}")
            return None

    def _run_dynamic_experience_management(self, batch: DataProto) -> dict:
        """Run the training-period experience lifecycle at a fixed interval."""
        metrics = {"experience_management/enabled": 1.0}
        update_config = self.config.env.get("experience_memory", {}) or {}
        management = self._dynamic_management_config()
        interval = self._dynamic_management_interval()
        metrics["experience_management/interval_steps"] = float(interval)

        if not bool(update_config.get("enable_exp_rollout", False)):
            metrics["experience_management/skipped_exp_rollout_disabled"] = 1.0
            return metrics
        if int(self.global_steps) % interval != 0:
            metrics["experience_management/skipped_interval"] = 1.0
            return metrics

        retrieval_memory = self._get_experience_bank()
        if retrieval_memory is None:
            metrics["experience_management/skipped_no_bank"] = 1.0
            return metrics

        credits, collection_metrics, group_audits, contrasts = self._collect_dynamic_management_credits(batch)
        metrics.update(collection_metrics)
        if collection_metrics.get("experience_management/missing_metadata", 0.0):
            metrics["experience_management/skipped_missing_metadata"] = 1.0
            self._write_dynamic_management_audit(
                management_record={
                    "global_step": int(self.global_steps),
                    "skip_reason": "missing_ab_metadata",
                    "metrics": metrics,
                    "group_gains": group_audits,
                },
            )
            return metrics
        if collection_metrics.get("experience_management/paired_groups", 0.0) <= 0:
            metrics["experience_management/skipped_no_ab_arms"] = 1.0
            self._write_dynamic_management_audit(
                management_record={
                    "global_step": int(self.global_steps),
                    "skip_reason": "no_paired_ab_arms",
                    "metrics": metrics,
                    "group_gains": group_audits,
                },
            )
            return metrics
        updates = []
        utility_skip_reason = None
        if credits:
            updates = retrieval_memory.update_utilities_from_group_credits(
                credits,
                global_step=int(self.global_steps),
                beta_task=management.get("utility_ema_beta_task", 0.1),
                beta_step=management.get("utility_ema_beta_step", 0.1),
                management_interval_steps=interval,
            )
        else:
            # A paired A/B batch without retrieved experience has no valid
            # credit, but it is still a real management window: capacity
            # eviction, snapshots, and remote synchronization must proceed.
            utility_skip_reason = "no_experience_ids_in_paired_groups"
            metrics["experience_management/skipped_utility_no_experience_ids"] = 1.0
        negative_window_resets = retrieval_memory.reset_negative_windows_for_uncredited(
            list(credits.keys()),
            global_step=int(self.global_steps),
        )
        metrics["experience_management/utility_updates"] = float(len(updates))
        metrics["experience_management/negative_window_resets"] = float(negative_window_resets)

        # Evolution uses the same uid-level A/B evidence but does not call the
        # legacy +/-1 updater, avoiding a second update of the same utility.
        if credits and self._is_experience_evolution_enabled():
            success_rate = self._collect_training_success_rates(batch)
            self._update_experiences_from_contrasts(
                contrast_examples=contrasts,
                success_rate=success_rate,
                retrieval_memory=retrieval_memory,
            )
            metrics["experience_management/evolution_contrasts"] = float(len(contrasts))

        evictions = {"removed": [], "warnings": []}
        if bool(management.get("eviction_enabled", True)):
            evictions = retrieval_memory.evict_excess_experiences(
                current_step=int(self.global_steps),
                max_task_experiences=management.get("eviction_max_task_experiences", 200),
                max_step_experiences=management.get("eviction_max_step_experiences", 200),
                protect_recent_steps=management.get("eviction_protect_recent_steps", 20),
                score_c=management.get("eviction_score_c", 1.0),
                min_exposures=management.get("eviction_min_exposures", 5),
                utility_threshold=management.get("eviction_utility_threshold", -0.25),
                negative_windows=management.get("eviction_negative_windows", 2),
            )
        metrics["experience_management/evicted"] = float(len(evictions.get("removed", [])))
        metrics["experience_management/eviction_warnings"] = float(len(evictions.get("warnings", [])))

        management_record = {
            "global_step": int(self.global_steps),
            "interval_steps": interval,
            "skip_reason": utility_skip_reason,
            "metrics": metrics,
            "group_gains": group_audits,
            "credits": credits,
            "utility_updates": updates,
            "evictions": evictions,
        }
        bank_path = self._write_dynamic_management_audit(
            management_record=management_record,
            retrieval_memory=retrieval_memory,
            evictions=evictions,
            save_bank=True,
        )
        canonical_path = update_config.get("experiences_json_path")
        if canonical_path and bool(management.get("save_to_canonical_path", False)):
            try:
                retrieval_memory.save_experiences(canonical_path)
                metrics["experience_management/saved_canonical"] = 1.0
            except Exception as exc:
                print(f"[ExperienceManagement] Warning: failed to save canonical bank: {exc}")
                metrics["experience_management/canonical_save_failed"] = 1.0
        self._sync_experiences_to_retrieval_server(retrieval_memory, path=bank_path)
        return metrics

    def _build_same_anchor_contrast_example(
        self,
        *,
        batch: DataProto,
        exp_idx: int,
        no_exp_idx: int,
        exp_score: float,
        no_exp_score: float,
        anchor_key: str,
        contrast_type: str = "same_anchor_contrast",
    ) -> dict:
        non_tensor = batch.non_tensor_batch
        outputs = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
        rewards = non_tensor.get("rewards", None)
        task = non_tensor.get("task_description", [None] * len(batch))[exp_idx]
        anchor_obs = non_tensor.get("current_observation", [None] * len(batch))[exp_idx]
        if anchor_obs is None:
            anchor_obs = non_tensor.get("anchor_obs", [None] * len(batch))[exp_idx]
        task_type_col = non_tensor.get("experience_task_type", None)
        task_type = task_type_col[exp_idx] if task_type_col is not None else self._detect_task_type_from_input(str(task or anchor_obs))
        admissible = non_tensor.get("admissible_actions", [None] * len(batch))[exp_idx]

        return {
            "task": str(task or ""),
            "task_type": str(task_type or "unknown"),
            "anchor_obs": str(anchor_obs),
            "anchor_key": anchor_key,
            "contrast_type": contrast_type,
            "admissible_actions": admissible,
            "paired_rollout": {
                "with_experience": {
                    "action": outputs[exp_idx].strip()[:1000],
                    "score": exp_score,
                    "reward": None if rewards is None else self._to_jsonable(rewards[exp_idx]),
                    "experience_ids": self._normalise_experience_ids(non_tensor["experience_ids"][exp_idx]),
                    "query_hash": None if "experience_query_hash" not in non_tensor else non_tensor["experience_query_hash"][exp_idx],
                },
                "without_experience": {
                    "action": outputs[no_exp_idx].strip()[:1000],
                    "score": no_exp_score,
                    "reward": None if rewards is None else self._to_jsonable(rewards[no_exp_idx]),
                },
            },
            "trajectory_success": exp_score,
            "episode_reward": exp_score,
            "trajectory": [
                {
                    "step_idx": int(non_tensor.get("step_idx", np.array([0] * len(batch)))[exp_idx]),
                    "observation": str(anchor_obs)[:2000],
                    "action": outputs[exp_idx].strip()[:1000],
                },
                {
                    "step_idx": int(non_tensor.get("step_idx", np.array([0] * len(batch)))[no_exp_idx]),
                    "observation": str(anchor_obs)[:2000],
                    "action": outputs[no_exp_idx].strip()[:1000],
                },
            ],
        }

    @staticmethod
    def _to_jsonable(value: Any):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _update_experiences_from_contrasts(
        self,
        *,
        contrast_examples: list,
        success_rate: dict,
        retrieval_memory,
    ):
        update_config = self.config.env.experience_memory
        threshold = update_config.get('update_threshold', 0.5)

        needs_update = bool(contrast_examples)
        if success_rate:
            needs_update = needs_update and any(float(rate) < threshold for rate in success_rate.values())

        if not needs_update:
            print("[ExperienceUpdate] No useful same-anchor contrast selected for LLM update")
            return

        max_examples = int(update_config.get('max_failed_trajectories', 10))
        contrast_examples = contrast_examples[:max_examples]

        if not hasattr(self, 'experience_updater'):
            from agent_system.memory.experience_updater import ExperienceUpdater
            self.experience_updater = ExperienceUpdater(
                api_key=update_config.get('experience_update_api_key', None),
                base_url=update_config.get('experience_update_base_url', None),
                model=update_config.get('experience_update_model', None),
                max_new_experiences_per_update=update_config.get('max_new_experiences', 3),
                max_completion_tokens=update_config.get('max_completion_tokens', 2048),
                token_limit_param=update_config.get('token_limit_param', 'max_tokens'),
            )

        print(
            f"[ExperienceUpdate] Analyzing {len(contrast_examples)} "
            f"same-anchor contrasts with {self.experience_updater.model}..."
        )
        new_experiences = self.experience_updater.analyze_failures(
            failed_trajectories=contrast_examples,
            current_experiences=retrieval_memory.experiences,
        )

        if new_experiences:
            for exp in new_experiences:
                exp.setdefault("evidence", {})
                exp["evidence"].setdefault("last_updated_step", self.global_steps)
            added = retrieval_memory.upsert_experiences(new_experiences)
            print(f"[ExperienceUpdate] Upserted {added} generated experiences")
        else:
            print("[ExperienceUpdate] No new experiences generated")

    def _collect_training_success_rates(self, batch: DataProto) -> dict:
        """Collect per-step training success rates from rollout metadata."""
        non_tensor = batch.non_tensor_batch
        success_rate = {}

        for key, values in non_tensor.items():
            if not self._is_success_rate_key(key):
                continue
            try:
                arr = np.asarray(values, dtype=np.float32)
                if arr.size > 0:
                    success_rate[key] = float(np.mean(arr))
            except (TypeError, ValueError):
                continue

        if success_rate or 'trajectory_success' not in non_tensor:
            return success_rate

        traj_uids = non_tensor.get('traj_uid', None)
        traj_success = np.asarray(non_tensor['trajectory_success'], dtype=np.float32)
        if traj_uids is None:
            success_rate['success_rate'] = float(np.mean(traj_success))
            return success_rate

        per_traj_success = {}
        for traj_uid, success in zip(traj_uids, traj_success):
            per_traj_success.setdefault(str(traj_uid), float(success))
        if per_traj_success:
            success_rate['success_rate'] = float(np.mean(list(per_traj_success.values())))

        return success_rate

    @staticmethod
    def _is_success_rate_key(key: str) -> bool:
        key = str(key)
        return key == 'success_rate' or key.endswith('_success_rate')

    def _detect_task_type_from_input(self, inp: str) -> str:
        """从输入中检测任务类型"""
        inp_lower = inp.lower()
        if 'look at' in inp_lower and ('lamp' in inp_lower or 'light' in inp_lower):
            return 'look_at_obj_in_light'
        elif 'clean' in inp_lower:
            return 'pick_clean_then_place_in_recep'
        elif 'heat' in inp_lower:
            return 'pick_heat_then_place_in_recep'
        elif 'cool' in inp_lower:
            return 'pick_cool_then_place_in_recep'
        elif 'two' in inp_lower and ('place' in inp_lower or 'put' in inp_lower):
            return 'pick_two_obj_and_place'
        elif 'examine' in inp_lower:
            return 'examine'
        else:
            return 'pick_and_place'

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self, checkpoint_dir_name=None, update_latest_tracker=True, track_for_retention=True):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        checkpoint_dir_name = checkpoint_dir_name or f"global_step_{self.global_steps}"
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, checkpoint_dir_name)

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, checkpoint_dir_name, "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep, track_for_retention=track_for_retention)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, checkpoint_dir_name, "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep, track_for_retention=track_for_retention)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if update_latest_tracker:
            local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(self.global_steps))

    def _load_best_checkpoint_score(self):
        metadata_path = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint_info.json")
        if not os.path.exists(metadata_path):
            return
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.best_checkpoint_score = float(metadata.get("score", float("-inf")))
        except Exception as err:
            print(f"Warning: failed to load best checkpoint metadata from {metadata_path}: {err}")

    def _maybe_save_best_checkpoint(self, val_metrics):
        if not self.config.trainer.get("save_best_checkpoint", False):
            return

        best_metric = self.config.trainer.get("best_checkpoint_metric", "val/success_rate")
        if best_metric not in val_metrics:
            print(f"Warning: best checkpoint metric {best_metric} not found in validation metrics; skip saving best checkpoint.")
            return

        metric_value = float(val_metrics[best_metric])
        if metric_value <= self.best_checkpoint_score:
            return

        previous_best = self.best_checkpoint_score
        self.best_checkpoint_score = metric_value
        print(f"New best checkpoint on {best_metric}: {metric_value} (previous: {previous_best}).")
        self._save_checkpoint(checkpoint_dir_name="best_checkpoint", update_latest_tracker=False, track_for_retention=False)

        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        metadata_path = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint_info.json")
        with open(metadata_path, "w") as f:
            json.dump({"metric": best_metric, "score": metric_value, "global_step": self.global_steps}, f, indent=2)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_best_checkpoint_score()
        if self.config.env.get("use_experience_memory", False):
            retrieval_memory = getattr(self.envs, "retrieval_memory", None)
            self._sync_experiences_to_retrieval_server(retrieval_memory)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            self._maybe_save_best_checkpoint(val_metrics)
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch.meta_info["global_step"] = self.global_steps
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self._is_dynamic_experience_management_enabled():
                        # This must run before adjust_batch/balance_batch while
                        # rollout uid/traj_uid and exp-rollout A/B metadata are
                        # still intact.  Validation never invokes this hook.
                        with _timer("experience_management", timing_raw):
                            metrics.update(self._run_dynamic_experience_management(batch))
                    elif self._is_experience_evolution_enabled():
                        with _timer("experience_evolution", timing_raw):
                            self._update_experiences_from_training_rollouts(batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        if self._is_edge_distillation_enabled():
                            batch, edge_metrics = self._compute_edge_distillation_weights(batch)
                            metrics.update(edge_metrics)

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            self._dump_trajectory_generations(
                                batch=batch,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            self._maybe_save_best_checkpoint(val_metrics)
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
