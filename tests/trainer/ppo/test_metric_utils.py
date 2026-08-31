# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Tests for the metric utilities in verl.trainer.ppo.metric_utils.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    bootstrap_metric,
    calc_maj_val,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)

from verl.utils.metric import (
    reduce_metrics,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor, _build_padded_response_token_selection_ids


class TestReduceMetrics(unittest.TestCase):
    """Tests for the reduce_metrics function."""

    def test_reduce_metrics_basic(self):
        """Test that reduce_metrics correctly computes means."""
        metrics = {
            "loss": [1.0, 2.0, 3.0],
            "accuracy": [0.0, 0.5, 1.0],
        }
        result = reduce_metrics(metrics)
        
        self.assertEqual(result["loss"], 2.0)
        self.assertEqual(result["accuracy"], 0.5)
    
    def test_reduce_metrics_empty(self):
        """Test that reduce_metrics handles empty lists."""
        metrics = {
            "empty": [],
        }
        result = reduce_metrics(metrics)
        
        self.assertTrue(np.isnan(result["empty"]))
    
    def test_reduce_metrics_single_value(self):
        """Test that reduce_metrics works with single values."""
        metrics = {
            "single": [5.0],
        }
        result = reduce_metrics(metrics)
        
        self.assertEqual(result["single"], 5.0)


class TestEdgeDistillation(unittest.TestCase):
    """Tests for EDGE gain-gated distillation helpers."""

    def test_edge_weights_positive_gain_only_no_experience_rows(self):
        uids = np.array(["g0", "g0", "g0", "g0"], dtype=object)
        traj_uids = np.array(["s0", "s1", "n0", "n1"], dtype=object)
        experience_injected = np.array([True, True, False, False])
        trajectory_success = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        weights, metrics = core_algos.compute_edge_distillation_weights_np(
            uids, traj_uids, experience_injected, trajectory_success
        )

        np.testing.assert_allclose(weights, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
        self.assertEqual(metrics["edge/positive_group_fraction"], 1.0)

    def test_edge_weights_non_positive_gain_zero(self):
        uids = np.array(["g0", "g0", "g0", "g0"], dtype=object)
        traj_uids = np.array(["s0", "s1", "n0", "n1"], dtype=object)
        experience_injected = np.array([True, True, False, False])
        trajectory_success = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        weights, metrics = core_algos.compute_edge_distillation_weights_np(
            uids, traj_uids, experience_injected, trajectory_success
        )

        np.testing.assert_allclose(weights, np.zeros(4, dtype=np.float32))
        self.assertEqual(metrics["edge/positive_group_fraction"], 0.0)

    def test_edge_weights_dedupe_trajectory_rows(self):
        uids = np.array(["g0", "g0", "g0", "g0", "g0"], dtype=object)
        traj_uids = np.array(["s0", "s0", "n0", "n0", "n1"], dtype=object)
        experience_injected = np.array([True, True, False, False, False])
        trajectory_success = np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        weights, _ = core_algos.compute_edge_distillation_weights_np(
            uids, traj_uids, experience_injected, trajectory_success
        )

        np.testing.assert_allclose(weights, np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32))

    def test_edge_distillation_loss_zero_weight(self):
        student = torch.tensor([[0.2, 0.3]], requires_grad=True)
        teacher = torch.tensor([[0.4, 0.1]])
        mask = torch.ones_like(student)
        weights = torch.zeros(1)

        loss, _ = core_algos.compute_edge_distillation_loss(student, teacher, mask, weights)

        self.assertEqual(loss.item(), 0.0)

    def test_edge_distillation_loss_uses_active_mask_denominator(self):
        student = torch.ones((2, 2), requires_grad=True)
        teacher = torch.zeros_like(student)
        mask = torch.ones_like(student)
        weights = torch.tensor([1.0, 0.0])

        loss, metrics = core_algos.compute_edge_distillation_loss(student, teacher, mask, weights)

        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertEqual(metrics["edge/actor_active_row_fraction"], 0.5)

    def test_edge_distillation_loss_weighted_direction(self):
        student = torch.tensor([[0.2, 0.3]], requires_grad=True)
        teacher = torch.tensor([[0.4, 0.1]])
        mask = torch.ones_like(student)
        weights = torch.ones(1)

        loss, _ = core_algos.compute_edge_distillation_loss(student, teacher, mask, weights)
        loss.backward()

        self.assertLess(student.grad[0, 0].item(), 0.0)
        self.assertGreater(student.grad[0, 1].item(), 0.0)

    def test_teacher_topk_loss_matches_restricted_reverse_kl(self):
        student = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, 0.0, 1.0]]])
        mask = torch.ones((1, 1))
        weights = torch.ones(1)

        loss, metrics = core_algos.compute_edge_teacher_topk_distillation_loss(student, teacher, mask, weights)

        expected = torch.sum(
            torch.softmax(student.detach(), dim=-1)
            * (torch.log_softmax(student.detach(), dim=-1) - torch.log_softmax(teacher, dim=-1))
        )
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertGreater(metrics["edge/teacher_topk_kl"], 0.0)
        self.assertEqual(metrics["edge/teacher_topk_size"], 3.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.any(student.grad != 0))

    def test_teacher_topk_loss_detaches_teacher_logits(self):
        student = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, 0.0, 1.0]]], requires_grad=True)

        loss, _ = core_algos.compute_edge_teacher_topk_distillation_loss(
            student, teacher, torch.ones((1, 1)), torch.ones(1)
        )
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.any(student.grad != 0))
        self.assertIsNone(teacher.grad)

    def test_teacher_topk_full_vocab_matches_reverse_kl(self):
        # K=vocab is the untruncated categorical reverse KL.
        student = torch.tensor([[[0.2, -0.3, 1.5, 0.7]]], requires_grad=True)
        teacher = torch.tensor([[[1.0, 0.4, -0.2, 0.8]]])
        loss, _ = core_algos.compute_edge_teacher_topk_distillation_loss(
            student, teacher, torch.ones((1, 1)), torch.ones(1)
        )
        student_log_probs = torch.log_softmax(student.detach(), dim=-1)
        teacher_log_probs = torch.log_softmax(teacher, dim=-1)
        full_reverse_kl = torch.sum(student_log_probs.exp() * (student_log_probs - teacher_log_probs))

        self.assertTrue(torch.allclose(loss.detach(), full_reverse_kl))

    def test_teacher_topk_loss_all_aggregation_modes(self):
        student = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.5, -0.5], [0.2, 0.8]],
            ],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[-0.5, 0.5], [0.9, 0.1]],
            ]
        )
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        weights = torch.ones(2)
        student_log_probs = torch.log_softmax(student.detach(), dim=-1)
        teacher_log_probs = torch.log_softmax(teacher, dim=-1)
        token_kl = torch.sum(student_log_probs.exp() * (student_log_probs - teacher_log_probs), dim=-1)
        expected = {
            "token-mean": (token_kl * mask).sum() / mask.sum(),
            "seq-mean-token-sum": (token_kl * mask).sum(dim=-1).mean(),
            "seq-mean-token-mean": ((token_kl * mask).sum(dim=-1) / mask.sum(dim=-1)).mean(),
            "seq-mean-token-sum-norm": (token_kl * mask).sum() / mask.shape[-1],
        }

        for loss_agg_mode, expected_loss in expected.items():
            loss, _ = core_algos.compute_edge_teacher_topk_distillation_loss(
                student, teacher, mask, weights, loss_agg_mode=loss_agg_mode
            )
            self.assertTrue(torch.allclose(loss.detach(), expected_loss), msg=loss_agg_mode)

    def test_teacher_topk_loss_uses_active_mask_denominator(self):
        student = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        )
        mask = torch.ones((2, 1))
        weights = torch.tensor([1.0, 0.0])

        loss, metrics = core_algos.compute_edge_teacher_topk_distillation_loss(student, teacher, mask, weights)
        expected = torch.sum(
            torch.softmax(student.detach()[0, 0], dim=-1)
            * (torch.log_softmax(student.detach()[0, 0], dim=-1) - torch.log_softmax(teacher[0, 0], dim=-1))
        )
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertEqual(metrics["edge/actor_active_row_fraction"], 0.5)

    def test_teacher_topk_loss_zero_weight(self):
        student = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
        teacher = torch.tensor([[[1.0, 0.0]]])
        loss, metrics = core_algos.compute_edge_teacher_topk_distillation_loss(
            student, teacher, torch.ones((1, 1)), torch.zeros(1)
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["edge/teacher_topk_kl"], 0.0)
        loss.backward()
        self.assertTrue(torch.all(student.grad == 0))

    def test_teacher_topk_config_validation(self):
        config = {
            "actor_rollout_ref": {
                "model": {"use_fused_kernels": False},
                "actor": {
                    "strategy": "fsdp",
                    "ulysses_sequence_parallel_size": 1,
                    "edge_distillation": {"token_selection": "teacher_topk", "top_k": 2},
                },
            }
        }
        core_algos.validate_edge_distillation_config(config)

        config["actor_rollout_ref"]["actor"]["edge_distillation"]["top_k"] = 1
        with self.assertRaisesRegex(ValueError, "top_k"):
            core_algos.validate_edge_distillation_config(config)

        config["actor_rollout_ref"]["actor"]["edge_distillation"]["top_k"] = 2
        config["actor_rollout_ref"]["model"]["use_fused_kernels"] = True
        with self.assertRaisesRegex(ValueError, "fused"):
            core_algos.validate_edge_distillation_config(config)

        config["actor_rollout_ref"]["model"]["use_fused_kernels"] = False
        config["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = 2
        with self.assertRaisesRegex(ValueError, "ulysses"):
            core_algos.validate_edge_distillation_config(config)

        config["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = 1
        config["actor_rollout_ref"]["actor"]["strategy"] = "megatron"
        with self.assertRaisesRegex(ValueError, "FSDP"):
            core_algos.validate_edge_distillation_config(config)

        config["actor_rollout_ref"]["actor"]["strategy"] = "fsdp"
        config["actor_rollout_ref"]["actor"]["edge_distillation"]["token_selection"] = "invalid"
        with self.assertRaisesRegex(ValueError, "token_selection"):
            core_algos.validate_edge_distillation_config(config)

    def test_teacher_topk_forward_clamps_to_vocab(self):
        class ToyOutput:
            def __init__(self, logits):
                self.logits = logits

        class ToyActor(torch.nn.Module):
            def forward(self, input_ids, **_kwargs):
                vocab_size = 5
                logits = torch.arange(vocab_size, dtype=torch.float32, device=input_ids.device)
                logits = logits.view(1, 1, vocab_size).repeat(input_ids.size(0), input_ids.size(1), 1)
                return ToyOutput(logits)

        actor = DataParallelPPOActor(
            config=OmegaConf.create(
                {
                    "use_remove_padding": False,
                    "use_fused_kernels": False,
                    "ulysses_sequence_parallel_size": 1,
                    "use_torch_compile": False,
                }
            ),
            actor_module=ToyActor(),
        )
        micro_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]]),
            "responses": torch.tensor([[3]]),
        }

        _, log_probs, topk_logits, topk_ids = actor._forward_micro_batch(
            micro_batch,
            temperature=1.0,
            calculate_log_probs=False,
            token_selection_top_k=20,
        )
        self.assertIsNone(log_probs)
        self.assertEqual(topk_logits.shape, (1, 1, 5))
        self.assertEqual(topk_ids.shape, (1, 1, 5))
        torch.testing.assert_close(topk_ids[0, 0], torch.tensor([4, 3, 2, 1, 0]))

    def test_teacher_topk_forward_gathers_given_teacher_ids(self):
        class ToyOutput:
            def __init__(self, logits):
                self.logits = logits

        class ToyActor(torch.nn.Module):
            def forward(self, input_ids, **_kwargs):
                logits = torch.arange(5, dtype=torch.float32, device=input_ids.device)
                return ToyOutput(logits.view(1, 1, 5).repeat(input_ids.size(0), input_ids.size(1), 1))

        actor = DataParallelPPOActor(
            config=OmegaConf.create(
                {
                    "use_remove_padding": False,
                    "use_fused_kernels": False,
                    "ulysses_sequence_parallel_size": 1,
                    "use_torch_compile": False,
                }
            ),
            actor_module=ToyActor(),
        )
        micro_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]]),
            "responses": torch.tensor([[3]]),
        }
        teacher_ids = torch.tensor([[[1, 4, 0]]])
        _, _, selected_logits, selected_ids = actor._forward_micro_batch(
            micro_batch,
            temperature=1.0,
            calculate_log_probs=False,
            token_selection_top_k=3,
            token_selection_ids=teacher_ids,
        )

        torch.testing.assert_close(selected_ids, teacher_ids)
        torch.testing.assert_close(selected_logits, torch.tensor([[[1.0, 4.0, 0.0]]]))

        with self.assertRaisesRegex(ValueError, "shape"):
            actor._forward_micro_batch(
                micro_batch,
                temperature=1.0,
                token_selection_top_k=3,
                token_selection_ids=torch.tensor([[1, 4, 0]]),
            )

    def test_remove_padding_candidate_ids_use_response_predictor_positions(self):
        candidate_ids = torch.tensor(
            [
                [[10, 11], [20, 21]],
                [[30, 31], [40, 41]],
            ]
        )
        padded_ids = _build_padded_response_token_selection_ids(
            candidate_ids, batch_size=2, seqlen=6, response_length=2
        )

        self.assertEqual(padded_ids.shape, (2, 6, 2))
        torch.testing.assert_close(padded_ids[:, 3:5], candidate_ids)
        torch.testing.assert_close(padded_ids[:, :3], torch.zeros((2, 3, 2), dtype=torch.long))
        torch.testing.assert_close(padded_ids[:, 5:], torch.zeros((2, 1, 2), dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "shape"):
            _build_padded_response_token_selection_ids(torch.ones((2, 2), dtype=torch.long), 2, 6, 2)

    def test_edge_distillation_rollout_predicate_requires_exp_rollout(self):
        config = {
            "actor_rollout_ref": {"actor": {"edge_distillation": {"enable": True}}},
            "env": {
                "use_experience_memory": True,
                "experience_memory": {"enable_exp_rollout": True},
                "rollout": {"n": 2},
            },
        }

        self.assertTrue(core_algos.is_edge_distillation_rollout_enabled(config, is_train=True))

        config["env"]["experience_memory"]["enable_exp_rollout"] = False
        self.assertFalse(core_algos.is_edge_distillation_rollout_enabled(config, is_train=True))


class TestComputeDataMetrics(unittest.TestCase):
    """Tests for the compute_data_metrics function."""
    
    def setUp(self):
        """Set up common test data."""
        # Create a mock DataProto object
        self.batch = MagicMock()
        self.batch.batch = {
            "token_level_scores": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "token_level_rewards": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "returns": torch.tensor([[1.1, 1.2], [1.3, 1.4]]),
            "responses": torch.zeros((2, 2)),  # 2 samples, 2 tokens each
            "attention_mask": torch.tensor([
                [1, 1, 1, 1],  # 2 prompt tokens, 2 response tokens
                [1, 1, 1, 1],
            ]),
            "values": torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
        }
        self.batch.non_tensor_batch = {
            "traj_uid": np.array(["traj_0", "traj_1"], dtype=object),
            "episode_rewards": np.array([1.0, 2.0], dtype=np.float32),
            "episode_lengths": np.array([3.0, 4.0], dtype=np.float32),
            "tool_callings": np.array([0.0, 1.0], dtype=np.float32),
        }
    
    def test_compute_data_metrics_with_critic(self):
        """Test compute_data_metrics with critic enabled."""
        metrics = compute_data_metrics(self.batch, use_critic=True)
        
        # Check that all expected metrics are present
        self.assertIn("critic/score/mean", metrics)
        self.assertIn("critic/rewards/mean", metrics)
        self.assertIn("critic/advantages/mean", metrics)
        self.assertIn("critic/returns/mean", metrics)
        self.assertIn("critic/values/mean", metrics)
        self.assertIn("critic/vf_explained_var", metrics)
        self.assertIn("response_length/mean", metrics)
        self.assertIn("prompt_length/mean", metrics)
        
        # Check some specific values
        self.assertAlmostEqual(metrics["critic/score/mean"], 5.0)  # Sum of token_level_scores
        self.assertAlmostEqual(metrics["critic/rewards/mean"], 2.5)  # Sum of token_level_rewards
    
    def test_compute_data_metrics_without_critic(self):
        """Test compute_data_metrics with critic disabled."""
        metrics = compute_data_metrics(self.batch, use_critic=False)
        
        # Check that critic-specific metrics are not present
        self.assertNotIn("critic/values/mean", metrics)
        self.assertNotIn("critic/vf_explained_var", metrics)
        
        # Check that other metrics are still present
        self.assertIn("critic/score/mean", metrics)
        self.assertIn("critic/rewards/mean", metrics)
        self.assertIn("response_length/mean", metrics)

    def test_compute_data_metrics_with_exp_rollout_metrics(self):
        """Test exp rollout success metrics for experience and no-experience trajectories."""
        self.batch.batch = {
            "token_level_scores": torch.ones((4, 2)),
            "token_level_rewards": torch.ones((4, 2)),
            "advantages": torch.ones((4, 2)),
            "returns": torch.ones((4, 2)),
            "responses": torch.zeros((4, 2)),
            "attention_mask": torch.ones((4, 4), dtype=torch.long),
        }
        self.batch.non_tensor_batch = {
            "traj_uid": np.array(["traj_0", "traj_1", "traj_2", "traj_3"], dtype=object),
            "episode_rewards": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "episode_lengths": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            "tool_callings": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "experience_injected": np.array([True, True, False, False]),
            "trajectory_success": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

        metrics = compute_data_metrics(self.batch, use_critic=False)

        self.assertAlmostEqual(metrics["exp_rollout/experience_mean_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["exp_rollout/w_o_experience_mean_success_rate"], 0.0)
        self.assertAlmostEqual(metrics["exp_rollout/experience_gain_diff_mean"], 0.5)
        self.assertAlmostEqual(metrics["exp_rollout/experience_trajectory_response_length"], 2.0)
        self.assertAlmostEqual(metrics["exp_rollout/w_o_experience_trajectory_response_length"], 2.0)

    def test_compute_data_metrics_without_exp_rollout_metrics(self):
        """Test exp rollout metrics are omitted when rollout marker is absent."""
        metrics = compute_data_metrics(self.batch, use_critic=False)

        self.assertNotIn("exp_rollout/experience_mean_success_rate", metrics)
        self.assertNotIn("exp_rollout/w_o_experience_mean_success_rate", metrics)
        self.assertNotIn("exp_rollout/experience_gain_diff_mean", metrics)
        self.assertNotIn("exp_rollout/experience_trajectory_response_length", metrics)
        self.assertNotIn("exp_rollout/w_o_experience_trajectory_response_length", metrics)


class TestGRPOExpRolloutGrouping(unittest.TestCase):
    """Tests for GRPO grouping used by exp rollout."""

    def test_grpo_advantage_uses_split_group_index(self):
        """Test split group ids center experience and no-experience rewards separately."""
        token_level_rewards = torch.tensor([[1.0], [3.0], [10.0], [14.0]])
        response_mask = torch.ones((4, 1))
        traj_index = np.array(["traj_0", "traj_1", "traj_2", "traj_3"], dtype=object)
        split_index = np.array(["task:experience", "task:experience", "task:w_o_experience", "task:w_o_experience"], dtype=object)

        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=split_index,
            traj_index=traj_index,
            norm_adv_by_std_in_grpo=False,
        )

        expected = torch.tensor([[-1.0], [1.0], [-2.0], [2.0]])
        torch.testing.assert_close(advantages, expected)


class TestComputeTimingMetrics(unittest.TestCase):
    """Tests for the compute_timing_metrics function."""
    
    def setUp(self):
        """Set up common test data."""
        # Create a mock DataProto object
        self.batch = MagicMock()
        self.batch.batch = {
            "responses": torch.zeros((2, 3)),  # 2 samples, 3 response tokens each
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1],  # 3 prompt tokens, 3 response tokens
                [1, 1, 1, 1, 1, 1],
            ]),
        }
        
        # Mock the _compute_response_info function to return known values
        self.response_info = {
            "prompt_length": torch.tensor([3.0, 3.0]),
            "response_length": torch.tensor([3.0, 3.0]),
            "response_mask": torch.ones((2, 3)),
        }
    
    @patch("verl.trainer.ppo.metric_utils._compute_response_info")
    def test_compute_timing_metrics(self, mock_compute_response_info):
        """Test compute_timing_metrics with various timing data."""
        mock_compute_response_info.return_value = self.response_info
        
        timing_raw = {
            "gen": 0.5,  # 500ms
            "ref": 0.3,  # 300ms
            "values": 0.2,  # 200ms
        }
        
        metrics = compute_timing_metrics(self.batch, timing_raw)
        
        # Check raw timing metrics
        self.assertEqual(metrics["timing_s/gen"], 0.5)
        self.assertEqual(metrics["timing_s/ref"], 0.3)
        self.assertEqual(metrics["timing_s/values"], 0.2)
        
        # Check per-token timing metrics
        # gen uses only response tokens (6 tokens)
        self.assertAlmostEqual(metrics["timing_per_token_ms/gen"], 0.5 * 1000 / 6, places=5)
        
        # ref and values use all tokens (12 tokens)
        self.assertAlmostEqual(metrics["timing_per_token_ms/ref"], 0.3 * 1000 / 12, places=5)
        self.assertAlmostEqual(metrics["timing_per_token_ms/values"], 0.2 * 1000 / 12, places=5)


class TestComputeThroughputMetrics(unittest.TestCase):
    """Tests for the compute_throughout_metrics function."""
    
    def setUp(self):
        """Set up common test data."""
        # Create a mock DataProto object
        self.batch = MagicMock()
        self.batch.meta_info = {
            "global_token_num": [100, 200, 300],  # 600 tokens total
        }
    
    def test_compute_throughout_metrics(self):
        """Test compute_throughout_metrics with various timing data."""
        timing_raw = {
            "step": 2.0,  # 2 seconds per step
        }
        
        # Test with 1 GPU
        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=1)
        
        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        self.assertEqual(metrics["perf/throughput"], 600 / 2.0)  # 300 tokens/sec
        
        # Test with 2 GPUs
        metrics = compute_throughout_metrics(self.batch, timing_raw, n_gpus=2)
        
        self.assertEqual(metrics["perf/total_num_tokens"], 600)
        self.assertEqual(metrics["perf/time_per_step"], 2.0)
        self.assertEqual(metrics["perf/throughput"], 600 / (2.0 * 2))  # 150 tokens/sec/GPU


class TestBootstrapMetric(unittest.TestCase):
    """Tests for the bootstrap_metric function."""
    
    def test_bootstrap_metric_basic(self):
        """Test bootstrap_metric with simple data and functions."""
        data = [1, 2, 3, 4, 5]
        reduce_fns = [np.mean, np.max]
        
        # Use a fixed seed for reproducibility
        result = bootstrap_metric(data, subset_size=3, reduce_fns=reduce_fns, n_bootstrap=100, seed=42)
        
        # Check that we get two results (one for each reduce_fn)
        self.assertEqual(len(result), 2)
        
        # Each result should be a tuple of (mean, std)
        mean_result, max_result = result
        self.assertEqual(len(mean_result), 2)
        self.assertEqual(len(max_result), 2)
        
        # The mean of means should be close to the true mean (3.0)
        self.assertAlmostEqual(mean_result[0], 3.0, delta=0.3)
        
        # The mean of maxes should be close to the expected value for samples of size 3
        # For samples of size 3 from [1,2,3,4,5], the expected max is around 4.0-4.5
        self.assertGreater(max_result[0], 3.5)
        self.assertLess(max_result[0], 5.0)
    
    def test_bootstrap_metric_empty(self):
        """Test bootstrap_metric with empty data."""
        with self.assertRaises(ValueError):
            bootstrap_metric([], subset_size=1, reduce_fns=[np.mean])


class TestCalcMajVal(unittest.TestCase):
    """Tests for the calc_maj_val function."""
    
    def test_calc_maj_val_basic(self):
        """Test calc_maj_val with simple data."""
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "A", "val": 0.7},
        ]
        
        result = calc_maj_val(data, vote_key="pred", val_key="val")
        
        # "A" is the majority vote, so we should get the first "val" for "A"
        self.assertEqual(result, 0.9)
    
    def test_calc_maj_val_tie(self):
        """Test calc_maj_val with tied votes."""
        data = [
            {"pred": "A", "val": 0.9},
            {"pred": "B", "val": 0.8},
            {"pred": "B", "val": 0.7},
            {"pred": "A", "val": 0.6},
        ]
        
        # In case of a tie, the first key in sorted order wins
        # This depends on Python's dict implementation, but for this test
        # we just verify that one of the valid values is returned
        result = calc_maj_val(data, vote_key="pred", val_key="val")
        
        self.assertTrue(result in [0.9, 0.8])


class TestProcessValidationMetrics(unittest.TestCase):
    """Tests for the process_validation_metrics function."""
    
    def test_process_validation_metrics_basic(self):
        """Test process_validation_metrics with simple data."""
        data_sources = ["source1", "source1", "source2"]
        sample_inputs = ["prompt1", "prompt1", "prompt2"]
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
        }
        
        result = process_validation_metrics(
            data_sources, sample_inputs, infos_dict, seed=42
        )
        
        # Check the structure of the result
        self.assertIn("source1", result)
        self.assertIn("source2", result)
        
        # Check that source1 has metrics for score
        self.assertIn("score", result["source1"])
        
        # Check that mean@2 is present for source1/score
        self.assertIn("mean@2", result["source1"]["score"])
        
        # Check the value of mean@2 for source1/score
        self.assertAlmostEqual(result["source1"]["score"]["mean@2"], 0.85)
    
    def test_process_validation_metrics_with_pred(self):
        """Test process_validation_metrics with prediction data."""
        data_sources = ["source1", "source1", "source1"]
        sample_inputs = ["prompt1", "prompt1", "prompt1"]
        infos_dict = {
            "score": [0.8, 0.9, 0.7],
            "pred": ["A", "B", "A"],
        }
        
        result = process_validation_metrics(
            data_sources, sample_inputs, infos_dict, seed=42
        )
        
        # Check that majority voting metrics are present
        self.assertIn("maj@2/mean", result["source1"]["score"])
        
        # For bootstrap with n=2, the majority vote could be either A or B
        # depending on the random sampling, so we don't check the exact value


if __name__ == "__main__":
    unittest.main()
