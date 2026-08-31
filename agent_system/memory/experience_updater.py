"""
LLM-based experience updater for ALFWorld experience banks.

The updater consumes contrasted rollout snippets and emits structured
state-aware experiences instead of generic experiences.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ExperienceUpdater:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_new_experiences_per_update: int = 3,
        max_completion_tokens: int = 2048,
        token_limit_param: str = "max_tokens",
    ):
        api_key = api_key or os.environ.get("EXPERIENCE_UPDATE_API_KEY")
        base_url = base_url or os.environ.get("EXPERIENCE_UPDATE_BASE_URL")
        model = model or os.environ.get("EXPERIENCE_UPDATE_MODEL")

        if not api_key or not base_url or not model:
            raise EnvironmentError(
                "ExperienceUpdater requires api_key, base_url, and model. Provide them "
                "via constructor args or EXPERIENCE_UPDATE_API_KEY, "
                "EXPERIENCE_UPDATE_BASE_URL, and EXPERIENCE_UPDATE_MODEL environment variables."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_new_experiences_per_update = max_new_experiences_per_update
        self.token_limit_param = token_limit_param
        self.update_history = []

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_experiences: Dict,
    ) -> List[Dict]:
        if not failed_trajectories:
            return []

        next_exp_idx = self._next_exp_index(current_experiences)
        prompt = self._build_analysis_prompt(failed_trajectories, current_experiences, next_exp_idx)

        try:
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                self.token_limit_param: self.max_completion_tokens,
            }
            response = self.client.chat.completions.create(**request_kwargs)
            raw_experiences = self._parse_experiences_response(response.choices[0].message.content)
            reassigned = self._reassign_exp_ids(raw_experiences, next_exp_idx)
            self.update_history.append(
                {
                    "num_failures_analyzed": len(failed_trajectories),
                    "num_experiences_generated": len(reassigned),
                    "experience_ids": [s.get("experience_id") for s in reassigned],
                }
            )
            return reassigned[: self.max_new_experiences_per_update]
        except Exception as e:
            print(f"[ExperienceUpdater] Error calling {self.model}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iter_existing_experiences(current_experiences: Dict):
        """Yield experiences from legacy and V3 banks without double-counting them.

        The initial state-aware bank stored items under ``experiences``.  V3
        stores them in ``task_experiences`` and ``step_experiences`` instead, with a
        legacy ``experience_id`` retained on step experiences.  Experience evolution has
        to understand both forms because an updated V3 bank is passed straight
        back to this updater on the next training iteration.
        """
        if not isinstance(current_experiences, dict):
            return

        seen = set()
        for pool_name in ("experiences", "task_experiences", "step_experiences"):
            pool = current_experiences.get(pool_name, []) or []
            if not isinstance(pool, list):
                continue
            for experience in pool:
                if not isinstance(experience, dict):
                    continue
                # A migration/adapter may expose the same object in two pools.
                # Deduplicate by its stable identifier and fall back to object
                # identity for identifier-free entries.
                identifier = str(experience.get("experience_id") or "")
                key = (identifier, pool_name) if not identifier else (identifier,)
                if key in seen:
                    continue
                seen.add(key)
                yield pool_name, experience

    @classmethod
    def _existing_experience_ids(cls, current_experiences: Dict) -> set[str]:
        """Return all experience IDs present in any supported bank schema."""
        ids = set()
        for _, experience in cls._iter_existing_experiences(current_experiences):
            for field in ("experience_id",):
                value = experience.get(field)
                if value is not None:
                    value = str(value)
                    if value:
                        ids.add(value)
        return ids

    def _next_exp_index(self, current_experiences: Dict) -> int:
        max_idx = 0
        pattern = re.compile(r"^exp_(\d+)$")
        for experience_id in self._existing_experience_ids(current_experiences):
            m = pattern.match(experience_id)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def _reassign_exp_ids(self, experiences: List[Dict], start_idx: int) -> List[Dict]:
        reassigned = []
        for i, exp in enumerate(experiences):
            updated = dict(exp)
            updated["experience_id"] = f"exp_{start_idx + i:06d}"
            reassigned.append(updated)
        return reassigned

    def _build_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_experiences: Dict,
        next_exp_idx: int,
    ) -> str:
        examples = []
        for i, traj in enumerate(failed_trajectories[:10]):
            examples.append(self._format_failure_example(i + 1, traj))

        existing = []
        for pool_name, experience in self._iter_existing_experiences(current_experiences):
            trigger = experience.get("trigger") or experience.get("when_to_apply") or experience.get("title") or ""
            recommended_action = experience.get("recommended_action") or experience.get("principle") or ""
            identifier = experience.get("experience_id") or "unknown-id"
            existing.append(
                f"[{pool_name}; {identifier}; {experience.get('task_type', 'unknown')}] "
                f"{trigger} -> {recommended_action}"
            )
        example_ids = ", ".join(
            f'"exp_{next_exp_idx + j:06d}"' for j in range(self.max_new_experiences_per_update)
        )

        return f"""You are updating an ALFWorld state-aware experience bank.

Analyze the contrasted rollout snippets below and propose NEW or revised experiences that help the agent act better in the same environment state.
These contrasts are selected when retrieved experience appears harmful: the rollout WITH experience failed, while the same-anchor rollout WITHOUT experience succeeded.

Requirements:
- Each experience must be state-aware, not a generic tip.
- Focus on what the successful no-experience rollout did, and what the retrieved experience may have caused the agent to do incorrectly.
- Prefer compact trigger language that can be matched at retrieval time.
- Use actionable wording tied to admissible actions and visible state cues.
- Return only JSON.

FAILED / CONTRASTED TRAJECTORIES:
{''.join(examples)}

EXISTING EXPERIENCES / EXPERIENCE TRIGGERS / ACTIONS:
{existing}

Generate 1-{self.max_new_experiences_per_update} new experiences.
Each item must contain:
- experience_id
- task_type
- trigger
- recommended_action
- avoid_action
- expected_effect
- state_cues (list[str])
- retrieval_text
- evidence {{
    "positive_gain_count": 0,
    "negative_gain_count": 0,
    "source_traj_uids": [],
    "last_updated_step": null
  }}

Use example ids: {example_ids}

Return ONLY a JSON array of experience objects.
"""

    def _format_failure_example(self, idx: int, traj: Dict) -> str:
        task = traj.get("task", "")
        task_type = traj.get("task_type", "unknown")
        anchor = traj.get("anchor_obs", traj.get("anchor", ""))
        contrast_type = traj.get("contrast_type", "same_anchor_contrast")
        success = traj.get("trajectory_success", traj.get("episode_reward", None))
        admissible_actions = traj.get("admissible_actions", [])
        same_anchor_pair = traj.get("paired_rollout", None)

        lines = [
            f"\nExample {idx}:",
            f"Task: {task}",
            f"Task Type: {task_type}",
            f"Contrast Type: {contrast_type}",
            f"Anchor Observation: {anchor}",
            f"Admissible Actions: {admissible_actions}",
        ]
        if same_anchor_pair:
            lines.append("Paired Contrast:")
            with_exp = same_anchor_pair.get("with_experience", {}) or {}
            without_exp = same_anchor_pair.get("without_experience", {}) or {}
            lines.append(
                "  With experience: "
                f"action={with_exp.get('action', '')}; "
                f"score={with_exp.get('score', '')}; "
                f"reward={with_exp.get('reward', '')}; "
                f"experience_ids={with_exp.get('experience_ids', [])}; "
                f"query_hash={with_exp.get('query_hash', '')}"
            )
            lines.append(
                "  Without experience: "
                f"action={without_exp.get('action', '')}; "
                f"score={without_exp.get('score', '')}; "
                f"reward={without_exp.get('reward', '')}"
            )
        lines.append(f"Outcome: {success}")
        lines.append("Trajectory:")
        for step in traj.get("trajectory", [])[:3] + traj.get("trajectory", [])[-5:]:
            step_idx = step.get("step_idx", "unknown")
            observation = step.get("observation", "")
            action = step.get("action", "")
            lines.append(f"  Step {step_idx}: obs={observation[:300]} action={action[:300]}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Parsing                                                             #
    # ------------------------------------------------------------------ #

    def _parse_experiences_response(self, response: str) -> List[Dict]:
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                items = json.loads(response[json_start:json_end])
                parsed = []
                for item in items:
                    if all(
                        key in item
                        for key in (
                            "experience_id",
                            "task_type",
                            "trigger",
                            "recommended_action",
                            "avoid_action",
                            "expected_effect",
                            "state_cues",
                            "retrieval_text",
                        )
                    ):
                        item.setdefault("evidence", {})
                        parsed.append(item)
                return parsed
        except json.JSONDecodeError as e:
            print(f"[ExperienceUpdater] JSON parse error: {e}")
        return []

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {"total_updates": 0, "total_experiences_generated": 0}
        return {
            "total_updates": len(self.update_history),
            "total_experiences_generated": sum(
                h["num_experiences_generated"] for h in self.update_history
            ),
            "all_experience_ids": [
                sid for h in self.update_history for sid in h["experience_ids"]
            ],
        }
