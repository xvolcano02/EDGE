# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List, Tuple, Dict, Union, Any, Optional
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def _experience_memory_kwargs(som_cfg):
    load_initial = som_cfg.get('load_initial_experiences', True)
    experiences_path = som_cfg.get('experiences_json_path')
    if not experiences_path or (isinstance(experiences_path, str) and not experiences_path.strip()):
        load_initial = False
        experiences_path = None
    elif not load_initial:
        experiences_path = None

    mgmt_cfg = (som_cfg.get("management") or {}) if som_cfg.get("enable_dynamic_management", False) else {}
    return {
        "experiences_json_path": experiences_path,
        "retrieval_mode": som_cfg.get('retrieval_mode', 'template'),
        "embedding_model_path": som_cfg.get('embedding_model_path', None),
        "task_specific_top_k": som_cfg.get('task_specific_top_k', None),
        "device": som_cfg.get('device', None),
        "experience_retrieval_service_url": som_cfg.get('experience_retrieval_service_url', None),
        "num_gpus": som_cfg.get('num_gpus', 1),
        "experience_text_for_retrieval": som_cfg.get('experience_text_for_retrieval', 'full'),
        "load_initial_experiences": load_initial,
        "similarity_threshold": som_cfg.get('similarity_threshold', None),
        "experience_retrieval_timeout": som_cfg.get('experience_retrieval_timeout', 60),
        "experience_generation_mode": som_cfg.get('experience_generation_mode', 'task_step'),
        "retrieval_top_2k": mgmt_cfg.get('retrieval_top_2k', som_cfg.get('retrieval_top_2k', None)),
        "retrieval_alpha": mgmt_cfg.get('retrieval_alpha', som_cfg.get('retrieval_alpha', None)),
        "retrieval_ucb_c": mgmt_cfg.get('retrieval_ucb_c', som_cfg.get('retrieval_ucb_c', 0.5)),
        "eviction_enabled": mgmt_cfg.get('eviction_enabled', som_cfg.get('eviction_enabled', False)),
    }


def _normalise_experience_generation_mode(mem_config) -> str:
    mode = (mem_config.get('experience_generation_mode') or 'task_step').lower().strip()
    if mode not in ("task_only", "step_only", "task_step"):
        mode = "task_step"
    return mode


def _empty_retrieved_memory(query_text: str = "") -> Dict[str, Any]:
    return {
        "task_experiences": [],
        "step_experiences": [],
        "experiences": [],
        "experience_ids": [],
        "query_text": query_text,
    }


def _combine_retrieved_memory(task_part: Dict[str, Any], step_part: Dict[str, Any]) -> Dict[str, Any]:
    task_experiences = list((task_part or {}).get("task_experiences", []))
    step_experiences = list((step_part or {}).get("step_experiences", []))
    combined = task_experiences + step_experiences
    experience_ids = []
    seen = set()
    for experience in combined:
        sid = experience.get("experience_id")
        if sid and sid not in seen:
            seen.add(sid)
            experience_ids.append(sid)
    return {
        "task_experiences": task_experiences,
        "step_experiences": step_experiences,
        "experiences": combined,
        "experience_ids": experience_ids,
        "query_text": (step_part or {}).get("query_text") or (task_part or {}).get("query_text", ""),
        "query_hash": (step_part or {}).get("query_hash") or (task_part or {}).get("query_hash"),
        "task_type": (step_part or {}).get("task_type") or (task_part or {}).get("task_type"),
        "retrieval_mode": (step_part or {}).get("retrieval_mode") or (task_part or {}).get("retrieval_mode"),
    }


def _empty_experience_metadata(batch_size: int) -> Dict[str, List[Any]]:
    """Metadata attached to observations for later experience-gain credit."""
    return {
        "experience_ids": [[] for _ in range(batch_size)],
        "experience_query_hash": [None for _ in range(batch_size)],
        "experience_task_type": [None for _ in range(batch_size)],
        "experience_retrieval_mode": [None for _ in range(batch_size)],
        "task_description": [None for _ in range(batch_size)],
        "current_observation": [None for _ in range(batch_size)],
        "admissible_actions": [[] for _ in range(batch_size)],
        "current_step": [None for _ in range(batch_size)],
    }


def _observation_with_experience_metadata(
    *,
    text: List[str],
    image,
    anchor: List[str],
    metadata: Optional[Dict[str, List[Any]]],
) -> Dict[str, Any]:
    """Keep prompt provenance aligned with the prompt that was actually built."""
    return {
        "text": text,
        "image": image,
        "anchor": anchor,
        **(metadata or _empty_experience_metadata(len(text))),
    }


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        # Add retrieval memory or experience memory if configured
        if config.env.get('use_experience_memory', False):
            from agent_system.memory import ExperienceMemory
            som_cfg = config.env.experience_memory
            self.retrieval_memory = ExperienceMemory(
                **_experience_memory_kwargs(som_cfg),
            )
            self.retrieved_memories = None
            print(f"[SearchEnvironmentManager] Experience memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                experiences_json_path=config.env.retrieval_memory.get('experiences_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[SearchEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None
            self.retrieved_memories = None

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        self.kwargs = kwargs
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))
        if self.retrieval_memory is not None:
            # Determine which config to use
            if self.config.env.get('use_experience_memory', False):
                mem_config = self.config.env.experience_memory
                mode = _normalise_experience_generation_mode(mem_config)
                top_k_task = mem_config.get('top_k_task') or mem_config.get('task_specific_top_k') or mem_config.get('top_k', 10)
                if mode == "step_only":
                    self.retrieved_memories = [_empty_retrieved_memory(task) for task in self.tasks]
                elif hasattr(self.retrieval_memory, "retrieve_task_experiences_batch"):
                    self.retrieved_memories = self.retrieval_memory.retrieve_task_experiences_batch(
                        self.tasks,
                        top_k=top_k_task,
                    )
                else:
                    self.retrieved_memories = [
                        self.retrieval_memory.retrieve(
                            task_description=task,
                            top_k=top_k_task,
                            similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                            max_tokens=mem_config.get('max_tokens', 2000),
                            include_examples=mem_config.get('include_examples', False)
                        )
                        for task in self.tasks
                    ]
            else:
                mem_config = self.config.env.retrieval_memory
                if hasattr(self.retrieval_memory, "retrieve_batch"):
                    self.retrieved_memories = self.retrieval_memory.retrieve_batch(
                        self.tasks,
                        top_k=mem_config.get('top_k', 10),
                        similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                        max_tokens=mem_config.get('max_tokens', 2000),
                        include_examples=mem_config.get('include_examples', False),
                    )
                else:
                    self.retrieved_memories = [
                        self.retrieval_memory.retrieve(
                            task_description=task,
                            top_k=mem_config.get('top_k', 10),
                            similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                            max_tokens=mem_config.get('max_tokens', 2000),
                            include_examples=mem_config.get('include_examples', False)
                        )
                        for task in self.tasks
                    ]

        observations = _observation_with_experience_metadata(
            text=self.build_text_obs(obs, init=True),
            image=None,
            anchor=obs.copy(),
            metadata=getattr(self, "_latest_experience_metadata", None),
        )
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        if (
            self.retrieval_memory is not None
            and self.config.env.get('use_experience_memory', False)
            and self.tasks
        ):
            mem_config = self.config.env.experience_memory
            mode = _normalise_experience_generation_mode(mem_config)
            top_k_step = mem_config.get('top_k_step') or mem_config.get('top_k', 10)
            queries = [f"{self.tasks[i]}\n\nCurrent observation: {next_obs[i]}" for i in range(len(next_obs))]
            if mode == "task_only":
                if self.retrieved_memories is None or len(self.retrieved_memories) != len(self.tasks):
                    self.retrieved_memories = [_empty_retrieved_memory(self.tasks[i]) for i in range(len(self.tasks))]
            elif hasattr(self.retrieval_memory, "retrieve_step_experiences_batch"):
                step_res = self.retrieval_memory.retrieve_step_experiences_batch(queries, top_k=top_k_step)
                if mode == "step_only":
                    self.retrieved_memories = [
                        _combine_retrieved_memory(_empty_retrieved_memory(), step_res[i])
                        for i in range(len(step_res))
                    ]
                else:
                    prev = self.retrieved_memories if self.retrieved_memories is not None and len(self.retrieved_memories) == len(step_res) else []
                    self.retrieved_memories = [
                        _combine_retrieved_memory(prev[i] if i < len(prev) else _empty_retrieved_memory(), step_res[i])
                        for i in range(len(step_res))
                    ]

        next_observations = _observation_with_experience_metadata(
            text=self.build_text_obs(next_obs),
            image=None,
            anchor=next_obs.copy(),
            metadata=getattr(self, "_latest_experience_metadata", None),
        )
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []
        experience_metadata = _empty_experience_metadata(len(text_obs))

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            uses_experience_memory = self.config.env.get('use_experience_memory', False)
            use_retrieval = (
                self.retrieval_memory is not None
                and self.retrieved_memories is not None
                and self._should_inject_experience(i, init)
                and (uses_experience_memory or not init)
            )
            retrieved_memory = None
            if use_retrieval:
                if i < len(self.retrieved_memories):
                    retrieved_memory = self.retrieved_memories[i]
                else:
                    retrieved_memory = _empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "")
                memory_context = self.retrieval_memory.format_for_prompt(retrieved_memory)
                if uses_experience_memory:
                    experience_metadata["experience_ids"][i] = list(retrieved_memory.get("experience_ids", []))
                    experience_metadata["experience_query_hash"][i] = retrieved_memory.get("query_hash")
                    experience_metadata["experience_task_type"][i] = retrieved_memory.get("task_type")
                    experience_metadata["experience_retrieval_mode"][i] = retrieved_memory.get("retrieval_mode")

            experience_metadata["task_description"][i] = self.tasks[i] if i < len(self.tasks) else ""
            experience_metadata["current_observation"][i] = text_obs[i]
            experience_metadata["admissible_actions"][i] = [
                "<search> query </search>",
                "<answer> answer </answer>",
            ]
            experience_metadata["current_step"][i] = len(self.memory[i]) + 1

            if (init or self.config.env.history_length <= 0) and use_retrieval:
                obs_i = SEARCH_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                )
            elif init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            elif use_retrieval:
                obs_i = SEARCH_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    memory_context=memory_ctx[i],
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        self._latest_experience_metadata = experience_metadata
        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        # Add retrieval memory or experience memory if configured
        if config.env.get('use_experience_memory', False):
            from agent_system.memory import ExperienceMemory
            som_cfg = config.env.experience_memory
            self.retrieval_memory = ExperienceMemory(
                **_experience_memory_kwargs(som_cfg),
            )
            self.retrieved_memories = None
            print(f"[AlfWorldEnvironmentManager] Experience memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                experiences_json_path=config.env.retrieval_memory.get('experiences_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[AlfWorldEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)

    @staticmethod
    def _empty_experience_metadata(batch_size: int) -> Dict[str, List[Any]]:
        return {
            "experience_ids": [[] for _ in range(batch_size)],
            "experience_query_hash": [None for _ in range(batch_size)],
            "experience_task_type": [None for _ in range(batch_size)],
            "experience_retrieval_mode": [None for _ in range(batch_size)],
            "task_description": [None for _ in range(batch_size)],
            "current_observation": [None for _ in range(batch_size)],
            "admissible_actions": [[] for _ in range(batch_size)],
            "current_step": [None for _ in range(batch_size)],
        }

    def _observation_with_experience_metadata(
        self,
        *,
        text: List[str],
        image,
        anchor: List[str],
    ) -> Dict[str, Any]:
        metadata = getattr(self, "_latest_experience_metadata", None)
        if metadata is None:
            metadata = self._empty_experience_metadata(len(text))
        return {
            "text": text,
            "image": image,
            "anchor": anchor,
            **metadata,
        }

    def _use_state_aware_experience_bank(self) -> bool:
        return bool(self.config.env.get('use_experience_memory', False))
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        self._retrieve_task_experiences_on_reset(batch_size=len(text_obs))

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return self._observation_with_experience_metadata(
            text=full_text_obs,
            image=image_obs,
            anchor=text_obs,
        ), infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs
        self._retrieve_step_experiences_for_observations(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = self._observation_with_experience_metadata(
            text=full_text_obs,
            image=image_obs,
            anchor=text_obs,
        )
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _retrieve_task_experiences_on_reset(self, batch_size: int) -> None:
        if self.retrieval_memory is None or not self._use_state_aware_experience_bank():
            return
        mem_config = self.config.env.experience_memory
        mode = _normalise_experience_generation_mode(mem_config)
        top_k_task = mem_config.get("top_k_task") or mem_config.get("task_specific_top_k") or mem_config.get("top_k", 6)
        if mode == "step_only" or not self.tasks:
            self.retrieved_memories = [_empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "") for i in range(batch_size)]
            return
        if hasattr(self.retrieval_memory, "retrieve_task_experiences_batch"):
            self.retrieved_memories = self.retrieval_memory.retrieve_task_experiences_batch(
                self.tasks,
                top_k=top_k_task,
            )
        else:
            self.retrieved_memories = [
                self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=top_k_task,
                )
                for task in self.tasks
            ]

    def _retrieve_step_experiences_for_observations(self, text_obs: List[str]) -> None:
        if self.retrieval_memory is None or not self._use_state_aware_experience_bank():
            return
        mem_config = self.config.env.experience_memory
        mode = _normalise_experience_generation_mode(mem_config)
        top_k_step = mem_config.get("top_k_step") or mem_config.get("top_k", 6)
        if mode == "task_only" or not self.tasks or not text_obs:
            if self.retrieved_memories is None or len(self.retrieved_memories) != len(text_obs):
                self.retrieved_memories = [_empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "") for i in range(len(text_obs))]
            return

        queries = [
            f"{self.tasks[i]}\n\nCurrent observation: {text_obs[i]}"
            for i in range(min(len(text_obs), len(self.tasks)))
        ]
        if not hasattr(self.retrieval_memory, "retrieve_step_experiences_batch"):
            self.retrieved_memories = [
                self.retrieval_memory.retrieve(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    top_k=top_k_step,
                )
                for i in range(len(queries))
            ]
            return

        step_res = self.retrieval_memory.retrieve_step_experiences_batch(queries, top_k=top_k_step)
        if mode == "step_only":
            self.retrieved_memories = [
                _combine_retrieved_memory(_empty_retrieved_memory(), step_res[i])
                for i in range(len(step_res))
            ]
            return

        prev = self.retrieved_memories if self.retrieved_memories is not None and len(self.retrieved_memories) == len(step_res) else []
        self.retrieved_memories = [
            _combine_retrieved_memory(prev[i] if i < len(prev) else _empty_retrieved_memory(), step_res[i])
            for i in range(len(step_res))
        ]
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        experience_metadata = self._empty_experience_metadata(len(text_obs))
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            use_retrieval = (
                self._use_state_aware_experience_bank()
                and self.retrieval_memory is not None
                and self._should_inject_experience(i, init)
            )
            action_history = ""
            history_len = 0
            if not init and self.config.env.history_length > 0:
                action_history = memory_contexts[i]
                history_len = valid_lens[i]

            retrieved_memory = None
            if use_retrieval:
                if self.retrieved_memories is not None and i < len(self.retrieved_memories):
                    retrieved_memory = self.retrieved_memories[i]
                else:
                    retrieved_memory = _empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "")
                memory_context = self.retrieval_memory.format_for_prompt(retrieved_memory)
                experience_metadata["experience_ids"][i] = list(retrieved_memory.get("experience_ids", []))
                experience_metadata["experience_query_hash"][i] = retrieved_memory.get("query_hash")
                experience_metadata["experience_task_type"][i] = retrieved_memory.get("task_type")
                experience_metadata["experience_retrieval_mode"][i] = retrieved_memory.get("retrieval_mode")

            experience_metadata["task_description"][i] = self.tasks[i]
            experience_metadata["current_observation"][i] = text_obs[i]
            experience_metadata["admissible_actions"][i] = [s for s in admissible_actions[i] if s != 'help']
            experience_metadata["current_step"][i] = len(self.memory[i]) + 1

            if (init or self.config.env.history_length <= 0) and use_retrieval:
                obs = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            elif init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            elif use_retrieval:
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=history_len,
                    action_history=action_history,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=history_len,
                    action_history=action_history,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        self._latest_experience_metadata = experience_metadata
        return postprocess_text_obs

    def build_edge_teacher_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """Build state-experience-conditioned prompts for EDGE teacher scoring."""
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            action_history = ""
            if not init and self.config.env.history_length > 0:
                action_history = memory_contexts[i]

            if self._use_state_aware_experience_bank() and self.retrieval_memory is not None:
                if self.retrieved_memories is not None and i < len(self.retrieved_memories):
                    retrieved_memory = self.retrieved_memories[i]
                else:
                    retrieved_memory = _empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "")
                memory_context = self.retrieval_memory.format_for_prompt(retrieved_memory)
            else:
                memory_context = "No state-relevant experience found for this turn."

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=0,
                    history_length=0,
                    action_history="",
                    current_step=1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=action_history,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def save_episode_trajectories(self, batch_data_list, infos_list):
        """
        Save successful/failed trajectories from completed episodes to memory pool.

        Args:
            batch_idx: Index of the batch
            total_batch_list: List of batch data containing trajectories
            infos: List of info dicts containing episode metadata
        """
        if self.retrieval_memory is None:
            return

        save_dir = self.config.env.retrieval_memory.get('save_dir', None)
        if save_dir is None:
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'new_memories.json')

        # Iterate through each environment
        for env_idx in range(len(self.tasks)):
            # Check if episode is done
            # We'll save trajectories when episodes complete
            # This will be called from the trainer after validation/training episodes
            pass  # Actual saving logic will be called from trainer


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        # Experience memory (same interface as AlfWorldEnvironmentManager)
        if config.env.get('use_experience_memory', False):
            from agent_system.memory import ExperienceMemory
            som_cfg = config.env.experience_memory
            self.retrieval_memory = ExperienceMemory(
                **_experience_memory_kwargs(som_cfg),
            )
            self.retrieved_memories = None
            print(f"[WebshopEnvironmentManager] Experience memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        self.memory.reset(batch_size=len(infos))

        self._retrieve_task_experiences_on_reset(batch_size=len(obs))
        # The first WebShop page is already a stateful decision point.  Retrieve
        # step experiences before building the initial prompt so their use can be
        # credited by the same experience-gain path as later turns.
        self._retrieve_step_experiences_for_observations(obs)

        observations = _observation_with_experience_metadata(
            text=self.build_text_obs(obs, infos, init=True),
            image=None,
            anchor=obs.copy(),
            metadata=getattr(self, "_latest_experience_metadata", None),
        )
        self.pre_text_obs = obs
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs
        self._retrieve_step_experiences_for_observations(next_obs)

        next_observations = _observation_with_experience_metadata(
            text=self.build_text_obs(next_obs, infos),
            image=None,
            anchor=next_obs.copy(),
            metadata=getattr(self, "_latest_experience_metadata", None),
        )
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _retrieve_task_experiences_on_reset(self, batch_size: int) -> None:
        if self.retrieval_memory is None or not self.config.env.get('use_experience_memory', False):
            return
        mem_cfg = self.config.env.experience_memory
        mode = _normalise_experience_generation_mode(mem_cfg)
        top_k_task = mem_cfg.get('top_k_task') or mem_cfg.get('task_specific_top_k') or mem_cfg.get('top_k', 6)
        if mode == "step_only" or not self.tasks:
            self.retrieved_memories = [_empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "") for i in range(batch_size)]
            return
        if hasattr(self.retrieval_memory, "retrieve_task_experiences_batch"):
            self.retrieved_memories = self.retrieval_memory.retrieve_task_experiences_batch(self.tasks, top_k=top_k_task)
        else:
            self.retrieved_memories = [
                self.retrieval_memory.retrieve(task_description=task, top_k=top_k_task)
                for task in self.tasks
            ]

    def _retrieve_step_experiences_for_observations(self, text_obs: List[str]) -> None:
        if self.retrieval_memory is None or not self.config.env.get('use_experience_memory', False):
            return
        mem_cfg = self.config.env.experience_memory
        mode = _normalise_experience_generation_mode(mem_cfg)
        top_k_step = mem_cfg.get('top_k_step') or mem_cfg.get('top_k', 6)
        if mode == "task_only" or not self.tasks or not text_obs:
            if self.retrieved_memories is None or len(self.retrieved_memories) != len(text_obs):
                self.retrieved_memories = [_empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "") for i in range(len(text_obs))]
            return
        queries = [
            f"{self.tasks[i]}\n\nCurrent observation: {text_obs[i]}"
            for i in range(min(len(text_obs), len(self.tasks)))
        ]
        if not hasattr(self.retrieval_memory, "retrieve_step_experiences_batch"):
            self.retrieved_memories = [
                self.retrieval_memory.retrieve(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    top_k=top_k_step,
                )
                for i in range(len(queries))
            ]
            return
        step_res = self.retrieval_memory.retrieve_step_experiences_batch(queries, top_k=top_k_step)
        if mode == "step_only":
            self.retrieved_memories = [
                _combine_retrieved_memory(_empty_retrieved_memory(), step_res[i])
                for i in range(len(step_res))
            ]
            return
        prev = self.retrieved_memories if self.retrieved_memories is not None and len(self.retrieved_memories) == len(step_res) else []
        self.retrieved_memories = [
            _combine_retrieved_memory(prev[i] if i < len(prev) else _empty_retrieved_memory(), step_res[i])
            for i in range(len(step_res))
        ]

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        experience_metadata = _empty_experience_metadata(len(text_obs))
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)
            use_retrieval = (
                self.config.env.get('use_experience_memory', False)
                and self.retrieval_memory is not None
                and self.retrieved_memories is not None
                and self._should_inject_experience(i, init)
            )
            retrieved_memory = None
            if use_retrieval:
                if i < len(self.retrieved_memories):
                    retrieved_memory = self.retrieved_memories[i]
                else:
                    retrieved_memory = _empty_retrieved_memory(self.tasks[i] if i < len(self.tasks) else "")
                memory_context = self.retrieval_memory.format_for_prompt(retrieved_memory)
                experience_metadata["experience_ids"][i] = list(retrieved_memory.get("experience_ids", []))
                experience_metadata["experience_query_hash"][i] = retrieved_memory.get("query_hash")
                experience_metadata["experience_task_type"][i] = retrieved_memory.get("task_type")
                experience_metadata["experience_retrieval_mode"][i] = retrieved_memory.get("retrieval_mode")

            experience_metadata["task_description"][i] = self.tasks[i] if i < len(self.tasks) else ""
            experience_metadata["current_observation"][i] = text_obs[i]
            experience_metadata["admissible_actions"][i] = list(available_actions)
            experience_metadata["current_step"][i] = len(self.memory[i]) + 1

            if (init or self.config.env.history_length <= 0) and use_retrieval:
                obs = WEBSHOP_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                )
            elif init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            elif use_retrieval:
                obs = WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            if len(obs) > 13000:
                print(f"Warning len(obs)={len(obs)} is too long")
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                # The fallback prompt does not contain the retrieved memory,
                # so it must not receive experience credit later.
                experience_metadata["experience_ids"][i] = []
                experience_metadata["experience_query_hash"][i] = None
                experience_metadata["experience_task_type"][i] = None
                experience_metadata["experience_retrieval_mode"][i] = None

            postprocess_text_obs.append(obs)

        self._latest_experience_metadata = experience_metadata
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
