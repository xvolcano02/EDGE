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

"""
Experience bank with batch retrieval and remote embedding service support.

Legacy ``experiences`` banks are migrated on load into the new
``task_experiences`` / ``step_experiences`` structure so the retrieval layer can align with
the independent embedding service deployment.
"""

import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .base import BaseMemory
from .task_extraction import extract_short_task_for_retrieval

DEFAULT_UTILITY = 0.0
DEFAULT_RETRIEVAL_COUNT = 0
DEFAULT_LAST_RETRIEVAL_STEP = 0
DEFAULT_CREATED_AT_STEP = 0

# Runtime lifecycle fields must never be reset when an LLM produces semantic
# content that fingerprints to an experience already in the bank.
_EXPERIENCE_LIFECYCLE_FIELDS = frozenset(
    {
        "experience_id",
        "utility",
        "retrieval_count",
        "last_retrieval_step",
        "created_at_step",
        "evidence",
    }
)

_TASK_QUERY_HINTS = {
    "pick_and_place": "pick up the target object and place it in the destination receptacle",
    "pick_two_obj_and_place": "pick up two target objects and place them in the destination receptacle",
    "look_at_obj_in_light": "look at the object under a lamp or another light source",
    "pick_heat_then_place_in_recep": "heat the target object and place it in the receptacle",
    "pick_cool_then_place_in_recep": "cool the target object and place it in the receptacle",
    "pick_clean_then_place_in_recep": "clean the target object and place it in the receptacle",
    "examine": "examine the object carefully",
    "general": "general ALFWorld interaction and action parsing",
    "unknown": "general ALFWorld interaction and action parsing",
    "all": "general ALFWorld interaction and action parsing",
    "apparel": "shop for clothing with the requested material, size, color, fit, and price",
    "footwear": "shop for shoes or other footwear with the requested size, color, use, and price",
    "home_decor": "shop for a home item with the requested dimensions, material, pattern, and color",
    "electronics": "shop for electronics with the requested model compatibility, connector, and capacity",
    "accessories": "shop for an accessory with the requested category, material, dimensions, and style",
    "beauty_health": "shop for a beauty or health product with the requested formulation, use, and pack size",
    "other": "shop for a product satisfying all requested attributes, variants, and budget",
    "direct_retrieval": "answer a direct who what when or where fact about a named entity",
    "entity_attribute_lookup": "look up a named entity's occupation, nationality, birthplace, genre, or language",
    "multi_hop_reasoning": "answer a multi-hop question by resolving an intermediate entity before the final relation",
    "comparison": "compare two entities using the same requested attribute",
}


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_normalise_text(v) for v in value)
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(v) for v in value]
    return [str(value)]


def _template_terms(value: Any) -> set[str]:
    text = _normalise_text(value)
    terms = re.findall(r"[a-z0-9_]+", text)
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "then",
        "there",
        "this",
        "to",
        "with",
        "you",
        "your",
        "task",
        "task_type",
        "step",
        "observation",
        "admissible",
        "actions",
        "admissible_actions",
        "recent_history",
        "current",
    }
    return {term for term in terms if term not in stopwords}


class ExperienceMemory(BaseMemory):
    """
    Experience bank with task and step pools.

    Retrieval mode:
    - ``template``: keyword overlap with task/step texts.
    - ``embedding``: SentenceTransformer embeddings, optionally via remote HTTP.

    The constructor accepts legacy parameters from the old EDGE implementation
    so existing call sites keep working.
    """

    _http_session_pool: Dict[str, Any] = {}

    def __init__(
        self,
        experiences_json_path: Optional[str] = None,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        device: Optional[str] = None,
        experience_retrieval_service_url: Optional[Union[str, List[str]]] = None,
        num_gpus: int = 1,
        experience_text_for_retrieval: str = "full",
        load_initial_experiences: bool = True,
        similarity_threshold: Optional[float] = None,
        experience_retrieval_timeout: int = 60,
        experience_generation_mode: str = "task_step",
        retrieval_top_2k: Optional[int] = None,
        retrieval_alpha: Optional[float] = None,
        retrieval_ucb_c: float = 0.5,
        eviction_enabled: bool = False,
    ):
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )

        self.load_initial_experiences = bool(load_initial_experiences)
        if self.load_initial_experiences:
            if not experiences_json_path or not os.path.exists(experiences_json_path):
                raise FileNotFoundError(f"Experiences file not found: {experiences_json_path}")
            with open(experiences_json_path, "r") as f:
                loaded = json.load(f)
            self.experiences = self._coerce_bank_schema(loaded)
        else:
            self.experiences = {
                "version": 3,
                "metadata": {"source": "empty"},
                "task_experiences": [],
                "step_experiences": [],
            }

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k
        self.device = device
        self._num_gpus = max(1, int(num_gpus)) if not getattr(num_gpus, "__iter__", None) else 1

        raw_url = experience_retrieval_service_url
        if raw_url is None:
            self._retrieval_service_urls = None
        elif isinstance(raw_url, str):
            u = raw_url.strip()
            self._retrieval_service_urls = [u] if u else None
        else:
            self._retrieval_service_urls = [str(u).strip() for u in raw_url if (u or "").strip()]
            if not self._retrieval_service_urls:
                self._retrieval_service_urls = None

        if experience_text_for_retrieval not in ("full", "when_to_apply", "principle", "retrieval_obs"):
            raise ValueError(
                "experience_text_for_retrieval must be 'full', 'when_to_apply', 'principle', or 'retrieval_obs'"
            )
        self._experience_text_for_retrieval = experience_text_for_retrieval
        self.similarity_threshold = similarity_threshold
        self._retrieval_timeout = max(1, int(experience_retrieval_timeout))
        self.experience_generation_mode = (experience_generation_mode or "task_step").lower().strip()
        if self.experience_generation_mode not in ("task_only", "step_only", "task_step"):
            self.experience_generation_mode = "task_step"
        self._retrieval_top_2k = retrieval_top_2k
        self._retrieval_alpha = retrieval_alpha
        self._retrieval_ucb_c = float(retrieval_ucb_c)
        self._eviction_enabled = bool(eviction_enabled)

        self._embedding_model = None
        self._embedding_models: Optional[List[Any]] = None
        self._task_experience_embeddings_cache: Optional[Dict[str, Any]] = None
        self._step_experience_embeddings_cache: Optional[Dict[str, Any]] = None
        self._query_cache: Dict[str, Dict[str, Any]] = {}

        self._normalize_all_experience_meta()

        print(
            f"[ExperienceMemory] Loaded experiences: {len(self.experiences.get('task_experiences', []))} task_experiences, "
            f"{len(self.experiences.get('step_experiences', []))} step_experiences | retrieval_mode={retrieval_mode}"
            + (f" | remote={len(self._retrieval_service_urls)} server(s)" if self._retrieval_service_urls else "")
            + (f" | num_gpus={self._num_gpus}" if self._num_gpus > 1 else "")
        )

    # ------------------------------------------------------------------ #
    # Schema helpers                                                      #
    # ------------------------------------------------------------------ #

    def _coerce_bank_schema(self, loaded: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(loaded, dict):
            raise ValueError("Experience bank must be a JSON object")

        if "task_experiences" in loaded or "step_experiences" in loaded:
            bank = {
                "version": int(loaded.get("version", 3) or 3),
                "metadata": deepcopy(loaded.get("metadata") or {}),
                "task_experiences": deepcopy(loaded.get("task_experiences") or []),
                "step_experiences": deepcopy(loaded.get("step_experiences") or []),
            }
            return bank

        if "experiences" in loaded:
            return self._legacy_bank_to_new_schema(loaded)

        raise ValueError(
            "Experience bank must contain either `task_experiences`/`step_experiences` or legacy `experiences`."
        )

    def _legacy_bank_to_new_schema(self, loaded: Dict[str, Any]) -> Dict[str, Any]:
        experiences = list(loaded.get("experiences") or [])
        step_experiences = [
            self._legacy_experience_to_step_experience(exp, idx)
            for idx, exp in enumerate(experiences, start=1)
            if isinstance(exp, dict)
        ]
        task_experiences = self._synthesise_task_experiences(step_experiences)
        metadata = deepcopy(loaded.get("metadata") or {})
        metadata.update(
            {
                "migration_source": "legacy_experiences",
                "legacy_version": int(loaded.get("version", 2) or 2),
                "task_experience_count": len(task_experiences),
                "step_experience_count": len(step_experiences),
            }
        )
        return {
            "version": 3,
            "metadata": metadata,
            "task_experiences": task_experiences,
            "step_experiences": step_experiences,
        }

    def _canonical_task_query(self, task_type: str) -> str:
        hint = _TASK_QUERY_HINTS.get(task_type, task_type.replace("_", " ").strip())
        return hint or "general ALFWorld interaction"

    def _legacy_experience_to_step_experience(
        self,
        experience: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        exp_id = str(experience.get("experience_id") or f"exp_{index:06d}")
        task_type = str(experience.get("task_type") or "unknown")
        trigger = str(experience.get("trigger") or "").strip()
        recommended_action = str(experience.get("recommended_action") or "").strip()
        avoid_action = str(experience.get("avoid_action") or "").strip()
        expected_effect = str(experience.get("expected_effect") or "").strip()
        state_cues = [str(c).strip() for c in experience.get("state_cues") or [] if str(c).strip()]
        evidence = deepcopy(experience.get("evidence") or {})
        utility = float(evidence.get("positive_gain_count", 0)) - float(evidence.get("negative_gain_count", 0))
        retrieval_obs = str(experience.get("retrieval_text") or "").strip()
        if not retrieval_obs:
            retrieval_obs = " ".join(
                part
                for part in [
                    self._canonical_task_query(task_type),
                    trigger,
                    recommended_action,
                    avoid_action,
                    expected_effect,
                    " ".join(state_cues),
                ]
                if part
            )
        title = trigger or recommended_action or self._canonical_task_query(task_type)
        principle = ". ".join(
            part
            for part in [recommended_action, f"Avoid {avoid_action}" if avoid_action else "", expected_effect]
            if part
        )
        when_to_apply = ". ".join(
            part
            for part in [f"Task type: {task_type}", trigger, " ".join(state_cues)]
            if part
        )
        return {
            "experience_id": exp_id,
            "task_type": task_type,
            "title": title,
            "principle": principle,
            "when_to_apply": when_to_apply,
            "retrieval_obs": retrieval_obs,
            "trigger": trigger,
            "recommended_action": recommended_action,
            "avoid_action": avoid_action,
            "expected_effect": expected_effect,
            "state_cues": state_cues,
            "utility": utility,
            "retrieval_count": int(evidence.get("retrieval_count", 0) or 0),
            "last_retrieval_step": evidence.get("last_updated_step", DEFAULT_LAST_RETRIEVAL_STEP),
            "created_at_step": int(evidence.get("last_updated_step") or DEFAULT_CREATED_AT_STEP),
            "evidence": {
                "positive_gain_count": int(evidence.get("positive_gain_count", 0) or 0),
                "negative_gain_count": int(evidence.get("negative_gain_count", 0) or 0),
                "source_traj_uids": _as_list(evidence.get("source_traj_uids", [])),
                "last_updated_step": evidence.get("last_updated_step"),
            },
        }

    def _synthesise_task_experiences(self, step_experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not step_experiences:
            return []

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for experience in step_experiences:
            task_type = str(experience.get("task_type") or "unknown")
            groups.setdefault(task_type, []).append(experience)

        task_experiences: List[Dict[str, Any]] = []
        for idx, (task_type, items) in enumerate(sorted(groups.items()), start=1):
            ordered = sorted(
                items,
                key=lambda s: (
                    -float(s.get("utility", 0.0)),
                    -float(s.get("retrieval_count", 0)),
                    str(s.get("experience_id", "")),
                ),
            )
            representative = ordered[0] if ordered else {}
            top_principles = [s.get("principle", "") for s in ordered[:3] if s.get("principle")]
            if top_principles:
                principle = " ".join(top_principles)
            else:
                principle = representative.get("principle") or f"Handle {task_type.replace('_', ' ')} tasks correctly."
            when_to_apply = f"When the task is {task_type.replace('_', ' ')}."
            task_experiences.append(
                {
                    "experience_id": f"task_{idx:06d}",
                    "task_type": task_type,
                    "title": f"{task_type.replace('_', ' ')} strategy",
                    "principle": principle,
                    "when_to_apply": when_to_apply,
                    "retrieval_obs": self._canonical_task_query(task_type),
                    "utility": float(representative.get("utility", 0.0)),
                    "retrieval_count": int(representative.get("retrieval_count", 0)),
                    "last_retrieval_step": int(representative.get("last_retrieval_step", 0) or 0),
                    "created_at_step": int(representative.get("created_at_step", 0) or 0),
                }
            )
        return task_experiences

    def _normalize_experience_meta(self, experience: Dict[str, Any]) -> None:
        if experience.get("utility") is None:
            evidence = experience.get("evidence") or {}
            if evidence:
                experience["utility"] = float(evidence.get("positive_gain_count", 0)) - float(
                    evidence.get("negative_gain_count", 0)
                )
            else:
                experience["utility"] = DEFAULT_UTILITY
        if experience.get("retrieval_count") is None:
            experience["retrieval_count"] = DEFAULT_RETRIEVAL_COUNT
        if experience.get("last_retrieval_step") is None:
            experience["last_retrieval_step"] = DEFAULT_LAST_RETRIEVAL_STEP
        if experience.get("created_at_step") is None:
            experience["created_at_step"] = DEFAULT_CREATED_AT_STEP

    def _normalize_all_experience_meta(self) -> None:
        for experience in self.experiences.get("task_experiences", []):
            self._normalize_experience_meta(experience)
        for experience in self.experiences.get("step_experiences", []):
            self._normalize_experience_meta(experience)

    def _experience_identifier(self, experience: Dict[str, Any]) -> str:
        return str(experience.get("experience_id") or "")

    def _next_task_experience_id(self) -> str:
        max_idx = 0
        pattern = re.compile(r"^task_(\d+)$")
        for experience in self.experiences.get("task_experiences", []):
            match = pattern.match(str(experience.get("experience_id", "")))
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return f"task_{max_idx + 1:06d}"

    def _next_step_experience_id(self) -> str:
        max_idx = 0
        pattern = re.compile(r"^step_(\d+)$")
        for experience in self.experiences.get("step_experiences", []):
            match = pattern.match(str(experience.get("experience_id", "")))
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return f"step_{max_idx + 1:06d}"

    def _experience_content_fingerprint(self, experience: Dict[str, Any]) -> str:
        return "\x00".join(
            [
                _normalise_text(experience.get("retrieval_obs")),
                _normalise_text(experience.get("title")),
                _normalise_text(experience.get("principle")),
                _normalise_text(experience.get("when_to_apply")),
                _normalise_text(experience.get("task_type")),
            ]
        )

    def _pool_content_fingerprints(self, pool_name: str) -> set:
        return {
            self._experience_content_fingerprint(experience)
            for experience in self.experiences.get(pool_name, [])
            if isinstance(experience, dict)
        }

    def _get_all_experience_ids(self) -> set:
        ids = set()
        for pool_name in ("task_experiences", "step_experiences"):
            for experience in self.experiences.get(pool_name, []):
                sid = self._experience_identifier(experience)
                if sid:
                    ids.add(sid)
        return ids

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                   #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding retrieval. "
                "Install with: pip install sentence-transformers"
            ) from exc

        target_device = self.device
        if not target_device:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(target_device).startswith("cuda") and not torch.cuda.is_available():
            target_device = "cpu"
            print("[ExperienceMemory] CUDA not available, using CPU for embedding model.")

        if self._num_gpus > 1 and torch.cuda.is_available():
            if self._embedding_models is None:
                n = min(self._num_gpus, torch.cuda.device_count())
                print(f"[ExperienceMemory] Loading {n} embedding models on cuda:0..{n-1}")
                self._embedding_models = [
                    SentenceTransformer(self.embedding_model_path, device=f"cuda:{i}") for i in range(n)
                ]
                self._embedding_model = self._embedding_models[0]
                print(f"[ExperienceMemory] {n} embedding models ready.")
            return self._embedding_models[0]

        if self._embedding_model is None:
            print(f"[ExperienceMemory] Loading embedding model: {self.embedding_model_path} on {target_device}")
            self._embedding_model = SentenceTransformer(self.embedding_model_path, device=target_device)
            print(f"[ExperienceMemory] Embedding model ready on {target_device}.")
        return self._embedding_model

    def _encode_texts(self, texts: List[str], normalize_embeddings: bool = True):
        import numpy as np

        if not texts:
            return np.array([]).reshape(0, 0)

        # Initialise before choosing the execution path.  Without this call,
        # the first (usually full-bank) encoding request would create multiple
        # models but still run entirely on cuda:0; only later requests would
        # use the other devices.
        model = self._get_embedding_model()
        if self._embedding_models and len(self._embedding_models) > 1:
            n_models = len(self._embedding_models)
            chunk_size = (len(texts) + n_models - 1) // n_models
            chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
            while len(chunks) < n_models:
                chunks.append([])
            chunks = chunks[:n_models]

            def _encode_one(model, batch_texts):
                if not batch_texts:
                    return None
                return model.encode(
                    batch_texts,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

            outputs: List[Optional[Any]] = [None] * n_models
            with ThreadPoolExecutor(max_workers=n_models) as executor:
                futures = {
                    executor.submit(_encode_one, self._embedding_models[i], chunks[i]): i
                    for i in range(n_models)
                    if chunks[i]
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    outputs[idx] = future.result()

            arrays = [arr for arr in outputs if arr is not None and getattr(arr, "size", 0) > 0]
            if not arrays:
                return np.array([]).reshape(0, 0)
            merged = np.concatenate(arrays, axis=0)
            if hasattr(merged, "shape") and len(merged.shape) == 1:
                merged = np.expand_dims(merged, axis=0)
            return merged

        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if hasattr(embeddings, "shape") and len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return embeddings

    def _compute_pool_embeddings(self, pool: str) -> Dict[str, Any]:
        import numpy as np

        items = self.experiences.get(pool, [])
        attr = "_task_experience_embeddings_cache" if pool == "task_experiences" else "_step_experience_embeddings_cache"
        existing = getattr(self, attr, None)
        if not items:
            if existing is None:
                cache = {"items": [], "texts": [], "embeddings": np.array([]).reshape(0, 0)}
                setattr(self, attr, cache)
                return cache
            return existing

        cached_items = existing.get("items", []) if existing else []
        texts = [self._experience_text_for_embedding(experience) for experience in items]
        if existing is not None and len(cached_items) == len(items):
            ids_now = tuple(self._experience_identifier(s) or f"_{i}" for i, s in enumerate(items))
            ids_cached = tuple(self._experience_identifier(s) or f"_{i}" for i, s in enumerate(cached_items))
            cached_texts = existing.get("texts")
            if cached_texts is None:
                cached_texts = [self._experience_text_for_embedding(experience) for experience in cached_items]
            if ids_now == ids_cached and cached_texts == texts:
                return existing

        embeddings = self._encode_texts(texts, normalize_embeddings=True)
        cache = {"items": list(items), "texts": texts, "embeddings": embeddings}
        setattr(self, attr, cache)
        return cache

    def _experience_text_for_embedding(self, experience: Dict[str, Any], mode: Optional[str] = None) -> str:
        retrieval_obs = (experience.get("retrieval_obs") or "").strip()
        if retrieval_obs:
            return retrieval_obs
        use = (mode or self._experience_text_for_retrieval)
        if use == "when_to_apply":
            return (experience.get("when_to_apply") or "").strip()
        if use == "principle":
            return (experience.get("principle") or "").strip()
        parts = []
        for field in ("title", "principle", "when_to_apply"):
            value = (experience.get(field) or "").strip()
            if value:
                parts.append(value)
        task_type = (experience.get("task_type") or "").strip()
        if task_type:
            parts.insert(0, task_type)
        return ". ".join(parts)

    # ------------------------------------------------------------------ #
    # Retrieval scoring helpers                                           #
    # ------------------------------------------------------------------ #

    def _apply_simutil_ucb(
        self,
        pool_name: str,
        indices_2k: List[int],
        sims_1d,
        top_k: int,
    ) -> List[int]:
        import numpy as np

        items = self.experiences.get(pool_name, [])
        alpha = self._retrieval_alpha
        c = self._retrieval_ucb_c
        if alpha is None or not indices_2k:
            return list(indices_2k[:top_k])
        N = sum(int(s.get("retrieval_count", 0)) for s in items)
        log_N = math.log(max(2, 1 + N)) if N >= 0 else math.log(2)
        scored: List[Tuple[float, int]] = []
        for idx in indices_2k:
            sim = float(sims_1d[idx])
            sim_norm = (sim + 1.0) / 2.0 if sim >= -1 else 0.0
            experience = items[idx] if 0 <= idx < len(items) else {}
            self._normalize_experience_meta(experience)
            u = float(experience.get("utility", 0))
            n = int(experience.get("retrieval_count", 0))
            denom = 1 + n
            exploration_bonus = c * (math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0)
            score = alpha * sim_norm + (1.0 - alpha) * (u + exploration_bonus)
            if math.isnan(score) or math.isinf(score):
                score = float(u)
            scored.append((score, idx))
        scored.sort(key=lambda x: -x[0])
        return [idx for _, idx in scored[:top_k]]

    def _get_experience_ranking_meta(self, pool_name: str, idx: int, sim_val: float) -> Dict[str, Any]:
        items = self.experiences.get(pool_name, [])
        experience = items[idx] if 0 <= idx < len(items) else {}
        self._normalize_experience_meta(experience)
        u = float(experience.get("utility", 0))
        sim_norm = (float(sim_val) + 1.0) / 2.0 if float(sim_val) >= -1 else 0.0
        alpha = self._retrieval_alpha
        c = self._retrieval_ucb_c
        if alpha is None:
            return {"utility": u, "ucb": 0.0, "retrieval_score": sim_norm}
        n = int(experience.get("retrieval_count", 0))
        N = sum(int(s.get("retrieval_count", 0)) for s in items)
        log_N = math.log(1 + N) if N > 0 else 0.0
        denom = 1 + n
        exploration_bonus = c * (math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0)
        score = alpha * sim_norm + (1.0 - alpha) * (u + exploration_bonus)
        return {"utility": u, "ucb": exploration_bonus, "retrieval_score": score}

    def _get_experience_ranking_meta_unknown_experience(self, pool_name: str, sim_val: float) -> Dict[str, Any]:
        items = self.experiences.get(pool_name, [])
        u = 0.0
        sim_norm = (float(sim_val) + 1.0) / 2.0 if float(sim_val) >= -1 else 0.0
        alpha = self._retrieval_alpha
        c = self._retrieval_ucb_c
        if alpha is None:
            return {"utility": u, "ucb": 0.0, "retrieval_score": sim_norm}
        N = sum(int(s.get("retrieval_count", 0)) for s in items)
        log_N = math.log(1 + N) if N > 0 else 0.0
        exploration_bonus = c * (math.sqrt(log_N) if log_N > 0 else 0.0)
        score = alpha * sim_norm + (1.0 - alpha) * (u + exploration_bonus)
        return {"utility": u, "ucb": exploration_bonus, "retrieval_score": score}

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def _detect_task_type(self, task_description: str) -> str:
        goal = (task_description or "").lower()
        env = str((self.experiences.get("metadata") or {}).get("env") or "").lower()

        if env == "webshop":
            category_keywords = {
                "apparel": (
                    "shirt", "dress", "t-shirt", "polo", "pants", "jeans", "jacket", "coat",
                    "sweater", "blouse", "skirt", "shorts", "hoodie", "cardigan", "garment",
                ),
                "footwear": (
                    "shoe", "boot", "sandal", "sneaker", "slipper", "loafer", "heel", "footwear",
                ),
                "home_decor": (
                    "pillow", "curtain", "rug", "mat", "blanket", "bedding", "towel", "lamp",
                    "furniture", "cushion", "sheet", "tablecloth", "vase",
                ),
                "electronics": (
                    "phone", "laptop", "tablet", "computer", "headphone", "earphone", "speaker",
                    "charger", "cable", "mouse", "keyboard", "monitor", "camera", "smartwatch",
                ),
                "accessories": (
                    "bag", "wallet", "belt", "hat", "cap", "scarf", "glove", "jewelry",
                    "necklace", "bracelet", "ring", "earring", "sunglasses", "purse", "backpack",
                ),
                "beauty_health": (
                    "makeup", "cosmetic", "skincare", "lotion", "cream", "shampoo", "conditioner",
                    "perfume", "soap", "body wash", "lipstick", "mascara", "serum", "moisturizer",
                ),
            }
            for task_type, keywords in category_keywords.items():
                if any(
                    re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?:s|es)?(?![a-z0-9])", goal)
                    for keyword in keywords
                ):
                    return task_type
            return "other"

        if env == "search":
            comparison_cues = (
                "both", "which is", "which was", "which of", "same", "common", "older", "younger",
                "earlier", "later", "larger", "smaller", "more than", "less than", "compare",
            )
            if any(cue in goal for cue in comparison_cues):
                return "comparison"
            multi_hop_cues = (
                "whose", "who is the", "who was the", "the author of", "the director of",
                "the birthplace of", "the capital of", "that was", "which was", "where the",
            )
            if any(cue in goal for cue in multi_hop_cues):
                return "multi_hop_reasoning"
            attribute_cues = (
                "occupation", "nationality", "birthplace", "place of birth", "genre", "language",
                "profession", "citizenship",
            )
            if any(cue in goal for cue in attribute_cues):
                return "entity_attribute_lookup"
            return "direct_retrieval"

        if "look at" in goal and "under" in goal:
            return "look_at_obj_in_light"
        if "clean" in goal:
            return "pick_clean_then_place_in_recep"
        if "heat" in goal:
            return "pick_heat_then_place_in_recep"
        if "cool" in goal:
            return "pick_cool_then_place_in_recep"
        if "two" in goal and ("place" in goal or "put" in goal):
            return "pick_two_obj_and_place"
        if "examine" in goal:
            return "examine"
        return "pick_and_place"

    def _query_hash(self, query_text: str) -> str:
        return hashlib.sha1(_normalise_text(query_text).encode("utf-8")).hexdigest()

    def _template_retrieve_pool(
        self,
        pool_name: str,
        query_text: str,
        task_type: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_terms = _template_terms(query_text)
        items = self.experiences.get(pool_name, [])
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for experience in items:
            experience_text = self._experience_text_for_embedding(experience)
            experience_terms = _template_terms(experience_text)
            overlap = len(query_terms & experience_terms)
            experience_task_type = experience.get("task_type")
            if experience_task_type == task_type:
                task_bonus = 4.0
            elif experience_task_type in ("general", "all", "unknown"):
                task_bonus = 2.0
            else:
                task_bonus = 0.0
            self._normalize_experience_meta(experience)
            utility_bonus = 0.1 * float(experience.get("utility", 0))
            score = overlap + task_bonus + utility_bonus
            scored.append((score, experience))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [deepcopy(experience) for score, experience in scored[:top_k] if score > 0]

    def _embedding_retrieve_pool(
        self,
        pool_name: str,
        query_texts: List[str],
        task_types: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        cache = self._compute_pool_embeddings(pool_name)
        if cache["embeddings"] is None or getattr(cache["embeddings"], "size", 0) == 0:
            return [
                {
                    pool_name: [],
                    "query_text": query,
                    "query_hash": self._query_hash(query),
                    "task_type": task_type,
                    "retrieval_mode": self.retrieval_mode,
                }
                for query, task_type in zip(query_texts, task_types)
            ]

        query_embs = self._encode_texts(query_texts, normalize_embeddings=True)
        if hasattr(query_embs, "shape") and len(query_embs.shape) == 1:
            query_embs = np.expand_dims(query_embs, axis=0)

        sims = cache["embeddings"] @ query_embs.T
        top_2k = self._retrieval_top_2k if self._retrieval_top_2k is not None else max(2 * top_k, top_k + 1)
        top_2k = min(top_2k, len(cache["items"]))

        out = []
        for j, query in enumerate(query_texts):
            s = np.asarray(sims[:, j]).ravel()
            idx_2k = np.lexsort((-np.arange(len(s)), -s))[:top_2k].tolist()
            if self._retrieval_alpha is not None:
                idx_final = self._apply_simutil_ucb(pool_name, idx_2k, s, top_k)
            else:
                idx_final = idx_2k[:top_k]
            experiences = []
            for i in idx_final:
                sk = dict(cache["items"][int(i)])
                sk["similarity"] = float(s[int(i)])
                sk.update(self._get_experience_ranking_meta(pool_name, int(i), float(s[int(i)])))
                if self.similarity_threshold is None or sk["similarity"] >= self.similarity_threshold:
                    experiences.append(sk)
            out.append(
                {
                    pool_name: experiences,
                    "query_text": query,
                    "query_hash": self._query_hash(query),
                    "task_type": task_types[j],
                    "retrieval_mode": self.retrieval_mode,
                }
            )
        return out

    def _pack_combined_result(
        self,
        *,
        task_experiences: Optional[List[Dict[str, Any]]] = None,
        step_experiences: Optional[List[Dict[str, Any]]] = None,
        query_text: str,
        task_type: str,
    ) -> Dict[str, Any]:
        task_experiences = [deepcopy(s) for s in (task_experiences or [])]
        step_experiences = [deepcopy(s) for s in (step_experiences or [])]
        combined = task_experiences + step_experiences
        seen = set()
        experience_ids = []
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
            "query_text": query_text,
            "query_hash": self._query_hash(query_text),
            "task_type": task_type,
            "retrieval_mode": self.retrieval_mode,
        }

    def retrieve_task_experiences_batch(
        self,
        task_descriptions: List[str],
        top_k: int = 6,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pool = "task_experiences"
        task_descriptions = [extract_short_task_for_retrieval(t or "") for t in task_descriptions]
        task_types = [self._detect_task_type(t) for t in task_descriptions]

        if self.retrieval_mode == "embedding" and self._retrieval_service_urls:
            effective_top_k = kwargs.get("task_specific_top_k") or self.task_specific_top_k or top_k
            return self._remote_retrieve_batch(task_descriptions, top_k=effective_top_k, pool=pool)

        if not self.experiences.get(pool):
            return [
                {
                    "task_experiences": [],
                    "step_experiences": [],
                    "experiences": [],
                    "experience_ids": [],
                    "query_text": query,
                    "query_hash": self._query_hash(query),
                    "task_type": self._detect_task_type(query),
                    "retrieval_mode": self.retrieval_mode,
                }
                for query in task_descriptions
            ]

        effective_top_k = kwargs.get("task_specific_top_k") or self.task_specific_top_k or top_k

        if self.retrieval_mode == "embedding":
            raw = self._embedding_retrieve_pool(pool, task_descriptions, task_types, effective_top_k)
            return [
                self._pack_combined_result(
                    task_experiences=item["task_experiences"],
                    step_experiences=[],
                    query_text=item["query_text"],
                    task_type=item["task_type"],
                )
                for item in raw
            ]

        return [
            self._pack_combined_result(
                task_experiences=self._template_retrieve_pool(pool, query, task_type, effective_top_k),
                step_experiences=[],
                query_text=query,
                task_type=task_type,
            )
            for query, task_type in zip(task_descriptions, task_types)
        ]

    def retrieve_step_experiences_batch(
        self,
        query_texts: List[str],
        top_k: int = 6,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pool = "step_experiences"
        query_texts = [query or "" for query in query_texts]
        task_types = [self._detect_task_type(query) for query in query_texts]

        if self.retrieval_mode == "embedding" and self._retrieval_service_urls:
            return self._remote_retrieve_batch(query_texts, top_k=top_k, pool=pool)

        if not self.experiences.get(pool):
            return [
                {
                    "task_experiences": [],
                    "step_experiences": [],
                    "experiences": [],
                    "experience_ids": [],
                    "query_text": query,
                    "query_hash": self._query_hash(query),
                    "task_type": self._detect_task_type(query),
                    "retrieval_mode": self.retrieval_mode,
                }
                for query in query_texts
            ]

        if self.retrieval_mode == "embedding":
            raw = self._embedding_retrieve_pool(pool, query_texts, task_types, top_k)
            return [
                self._pack_combined_result(
                    task_experiences=[],
                    step_experiences=item["step_experiences"],
                    query_text=item["query_text"],
                    task_type=item["task_type"],
                )
                for item in raw
            ]

        return [
            self._pack_combined_result(
                task_experiences=[],
                step_experiences=self._template_retrieve_pool(pool, query, task_type, top_k),
                query_text=query,
                task_type=task_type,
            )
            for query, task_type in zip(query_texts, task_types)
        ]

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 6,
        pool: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if pool not in ("task_experiences", "step_experiences"):
            raise ValueError("pool must be 'task_experiences' or 'step_experiences'")
        if pool == "task_experiences":
            return self.retrieve_task_experiences_batch(queries, top_k=top_k, **kwargs)
        return self.retrieve_step_experiences_batch(queries, top_k=top_k, **kwargs)

    def retrieve(
        self,
        task_description: str,
        current_observation: Optional[str] = None,
        admissible_actions: Optional[Sequence[str]] = None,
        action_history: Optional[Any] = None,
        step_count: Optional[int] = None,
        top_k: int = 6,
        **kwargs,
    ) -> Dict[str, Any]:
        # Compatibility wrapper for the old single-query API.
        task_query = extract_short_task_for_retrieval(task_description or "")
        task_type = self._detect_task_type(task_query)
        query_text = task_query
        task_result: Dict[str, Any] = self._pack_combined_result(
            task_experiences=[],
            step_experiences=[],
            query_text=task_query,
            task_type=task_type,
        )
        step_result: Optional[Dict[str, Any]] = None

        if self.experience_generation_mode in ("task_only", "task_step"):
            task_result = self.retrieve_task_experiences_batch([task_query], top_k=kwargs.get("task_specific_top_k") or top_k)[0]
        if current_observation is not None and self.experience_generation_mode in ("step_only", "task_step"):
            step_query = f"{task_query}\n\nCurrent observation: {current_observation}"
            query_text = step_query
            step_result = self.retrieve_step_experiences_batch([step_query], top_k=top_k)[0]

        if step_result is None:
            return task_result

        return self._pack_combined_result(
            task_experiences=task_result.get("task_experiences", []),
            step_experiences=step_result.get("step_experiences", []),
            query_text=query_text,
            task_type=task_type,
        )

    def _get_http_session(self, base_url: str):
        import requests
        from requests.adapters import HTTPAdapter

        base_url = base_url.rstrip("/")
        if base_url not in self._http_session_pool:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=256, pool_maxsize=256, pool_block=False)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._http_session_pool[base_url] = session
        return self._http_session_pool[base_url]

    def _remote_retrieve_batch(
        self,
        task_descriptions: List[str],
        top_k: int = 6,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        import requests

        timeout = timeout if timeout is not None else self._retrieval_timeout
        urls = self._retrieval_service_urls or []
        n = len(urls)

        def _normalize_url(u: str) -> str:
            u = u.rstrip("/")
            return u if "/retrieve_batch" in u else f"{u}/retrieve_batch"

        def _request_one(url: str, queries: List[str], pool: Optional[str] = None) -> List[Dict[str, Any]]:
            # New servers use candidate_k for semantic candidate retrieval and
            # leave top_k as the final client-facing size.  Older deployments
            # ignore the extra field and safely return their normal top_k.
            candidate_k = self._retrieval_top_2k
            if candidate_k is None:
                candidate_k = top_k
            try:
                candidate_k = max(int(candidate_k), int(top_k))
            except (TypeError, ValueError):
                candidate_k = top_k
            payload = {
                "queries": queries,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "task_specific_top_k": kwargs.get("task_specific_top_k") or self.task_specific_top_k,
                "experience_text_for_retrieval": kwargs.get("experience_text_for_retrieval") or self._experience_text_for_retrieval,
            }
            if pool is not None:
                payload["pool"] = pool
            session = self._get_http_session(url)
            resp = session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if "result" in data:
                return data["result"]
            return data

        if n == 0:
            return []

        size = len(task_descriptions)
        if size == 0:
            return []
        chunk_size = (size + n - 1) // n
        chunks = [task_descriptions[i : i + chunk_size] for i in range(0, size, chunk_size)]
        while len(chunks) < n:
            chunks.append([])
        chunks = chunks[:n]

        results_by_idx: List[Optional[List[Dict[str, Any]]]] = [None] * n
        with ThreadPoolExecutor(max_workers=n) as executor:
            pool = kwargs.get("pool")
            futures = {
                executor.submit(_request_one, _normalize_url(urls[i]), chunks[i], pool): i
                for i in range(n)
                if chunks[i]
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results_by_idx[idx] = fut.result()

        out: List[Dict[str, Any]] = []
        for i in range(n):
            if results_by_idx[i] is not None:
                out.extend(results_by_idx[i])

        pool = kwargs.get("pool")
        if out and pool in ("task_experiences", "step_experiences"):
            pool_key = pool
            items = self.experiences.get(pool_key, [])
            experience_id_to_idx = {}
            for idx, experience in enumerate(items):
                experience_id = self._experience_identifier(experience)
                if experience_id:
                    experience_id_to_idx[experience_id] = idx
            for item in out:
                experiences = list(item.get(pool_key, []) or [])
                if self.similarity_threshold is not None:
                    experiences = [
                        sk for sk in experiences
                        if sk.get("similarity") is not None and float(sk.get("similarity")) >= self.similarity_threshold
                    ]
                for sk in experiences:
                    sid = sk.get("experience_id") or sk.get("experience_id")
                    sim = sk.get("similarity")
                    if sim is None:
                        continue
                    sim_val = float(sim)
                    if sid is not None and sid in experience_id_to_idx:
                        idx = experience_id_to_idx[sid]
                        meta = self._get_experience_ranking_meta(pool_key, idx, sim_val)
                    else:
                        meta = self._get_experience_ranking_meta_unknown_experience(pool_key, sim_val)
                    sk.update(meta)
                # A current server may return a large semantic candidate set;
                # utility/UCB is deliberately applied on the client because
                # this process owns the freshly updated training bank.  When a
                # legacy server returns only top_k candidates, the same code
                # simply reorders that smaller candidate set.
                if self._retrieval_alpha is not None:
                    experiences.sort(
                        key=lambda experience: (
                            -float(experience.get("retrieval_score", float("-inf"))),
                            -float(experience.get("similarity", float("-inf"))),
                            str(experience.get("experience_id") or ""),
                        )
                    )
                item[pool_key] = experiences[:top_k]
                other_pool = "step_experiences" if pool_key == "task_experiences" else "task_experiences"
                item.setdefault(other_pool, [])
                item["experiences"] = list(item.get("task_experiences", [])) + list(item.get("step_experiences", []))
                seen = set()
                experience_ids = []
                for experience in item["experiences"]:
                    sid = experience.get("experience_id")
                    if sid and sid not in seen:
                        seen.add(sid)
                        experience_ids.append(sid)
                item["experience_ids"] = experience_ids
                query = item.get("query_text") or ""
                item.setdefault("query_hash", self._query_hash(query))
                item.setdefault("task_type", self._detect_task_type(query))
                item.setdefault("retrieval_mode", self.retrieval_mode)
        return out

    # ------------------------------------------------------------------ #
    # Public formatting                                                   #
    # ------------------------------------------------------------------ #

    def _format_experience_section(self, title: str, experiences: List[Dict[str, Any]]) -> Optional[str]:
        if not experiences:
            return None
        lines = [title]
        for experience in experiences:
            heading = experience.get("title") or experience.get("principle") or experience.get("when_to_apply") or ""
            principle = experience.get("principle", "")
            when_to_apply = experience.get("when_to_apply", "")
            if heading and principle:
                lines.append(f"- **{heading}**: {principle}")
            elif heading:
                lines.append(f"- **{heading}**")
            elif principle:
                lines.append(f"- {principle}")
            if when_to_apply:
                lines.append(f"  _When to apply: {when_to_apply}_")
        return "\n".join(lines)

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        task_experiences = retrieved_memories.get("task_experiences")
        step_experiences = retrieved_memories.get("step_experiences")
        if task_experiences is None and step_experiences is None:
            # Backward-compatible support for old ``experiences`` payloads.
            experiences = retrieved_memories.get("experiences", [])
            if not experiences:
                return "No relevant experiences found for this task."
            lines = ["## Retrieved Relevant Experience"]
            for exp in experiences:
                trigger = exp.get("trigger", "")
                do = exp.get("recommended_action", "")
                expect = exp.get("expected_effect", "")
                avoid = exp.get("avoid_action", "")
                lines.append(f"- When: {trigger}")
                if do:
                    lines.append(f"  Do: {do}")
                if expect:
                    lines.append(f"  Expect: {expect}")
                if avoid:
                    lines.append(f"  Avoid: {avoid}")
            return "\n".join(lines)

        sections = []
        task_section = self._format_experience_section("### Task-level experience (for this kind of task)", task_experiences or [])
        if task_section:
            sections.append(task_section)
        step_section = self._format_experience_section("### Step-level experience (relevant to current situation)", step_experiences or [])
        if step_section:
            sections.append(step_section)
        return "\n\n".join(sections) if sections else "No relevant experiences found for this task."

    # ------------------------------------------------------------------ #
    # Update / persistence                                                #
    # ------------------------------------------------------------------ #

    def add_experiences(self, new_experiences: List[Dict], category: str = "general") -> int:
        if category == "task":
            return self.upsert_experience_groups(task_experiences=new_experiences)
        return self.upsert_experiences(new_experiences)

    def upsert_experience_groups(
        self,
        *,
        task_experiences: Optional[List[Dict[str, Any]]] = None,
        step_experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        task_experiences = list(task_experiences or [])
        step_experiences = list(step_experiences or [])
        existing = {
            self._experience_content_fingerprint(experience): experience
            for experience in self.experiences.get("task_experiences", []) + self.experiences.get("step_experiences", [])
        }
        added = 0

        def _append_experience(pool_name: str, experience: Dict[str, Any]) -> None:
            nonlocal added
            clean = deepcopy(experience)
            clean.setdefault("task_type", "unknown")
            clean.setdefault("utility", DEFAULT_UTILITY)
            clean.setdefault("retrieval_count", DEFAULT_RETRIEVAL_COUNT)
            clean.setdefault("last_retrieval_step", DEFAULT_LAST_RETRIEVAL_STEP)
            clean.setdefault("created_at_step", DEFAULT_CREATED_AT_STEP)
            if pool_name == "task_experiences":
                clean["experience_id"] = clean.get("experience_id") or self._next_task_experience_id()
            else:
                clean["experience_id"] = clean.get("experience_id") or self._next_step_experience_id()
            fp = self._experience_content_fingerprint(clean)
            if fp in existing:
                target = existing[fp]
                # A duplicate is a semantic refinement, not a new lifecycle.
                # In particular, never replace the stable IDs, accumulated
                # utility/exposure, or creation time of the existing experience.
                target.update(
                    {
                        k: v
                        for k, v in clean.items()
                        if k not in _EXPERIENCE_LIFECYCLE_FIELDS and v not in (None, "", [])
                    }
                )
                if "evidence" in clean:
                    self._merge_evidence(target, clean.get("evidence", {}))
                return
            self.experiences.setdefault(pool_name, []).append(clean)
            existing[fp] = clean
            added += 1

        for experience in task_experiences:
            _append_experience("task_experiences", experience)
        for experience in step_experiences:
            _append_experience("step_experiences", experience)

        if added:
            self._task_experience_embeddings_cache = None
            self._step_experience_embeddings_cache = None
            self._query_cache.clear()
            self._normalize_all_experience_meta()
        return added

    def upsert_experiences(self, experiences: List[Dict[str, Any]]) -> int:
        start_idx = 1
        pattern = re.compile(r"^step_(\d+)$")
        for experience in self.experiences.get("step_experiences", []):
            match = pattern.match(str(experience.get("experience_id", "")))
            if match:
                start_idx = max(start_idx, int(match.group(1)) + 1)
        step_experiences = [
            self._legacy_experience_to_step_experience(exp, start_idx + offset)
            for offset, exp in enumerate(experiences)
            if isinstance(exp, dict)
        ]
        return self.upsert_experience_groups(step_experiences=step_experiences)

    def _merge_evidence(self, experience: Dict[str, Any], evidence: Dict[str, Any]) -> None:
        target = experience.setdefault("evidence", {})
        # Gain counters and source provenance can be supplied by an incoming
        # semantic experience.  Runtime window state belongs to the target and
        # is intentionally retained below.
        for key in ("positive_gain_count", "negative_gain_count", "neutral_gain_count"):
            target[key] = int(target.get(key, 0)) + int(evidence.get(key, 0) or 0)
        source_ids = set(_as_list(target.get("source_traj_uids", [])))
        source_ids.update(_as_list(evidence.get("source_traj_uids", [])))
        target["source_traj_uids"] = sorted(source_ids)
        target.setdefault("utility_update_count", 0)
        target.setdefault("consecutive_negative_windows", 0)
        target.setdefault("last_negative_management_step", None)
        # An LLM re-emitting a duplicate has not performed a utility update.
        # Preserve the target's lifecycle timestamp; only seed it for a bank
        # that did not have one yet.
        if target.get("last_updated_step") is None:
            target["last_updated_step"] = evidence.get("last_updated_step")

    def record_experience_gain(
        self,
        experience_ids: Sequence[str],
        *,
        positive: bool,
        traj_uid: Optional[str] = None,
        global_step: Optional[int] = None,
    ) -> int:
        ids = {str(exp_id) for exp_id in experience_ids if exp_id}
        if not ids:
            return 0

        updated = 0
        for pool_name in ("task_experiences", "step_experiences"):
            for experience in self.experiences.get(pool_name, []):
                sid = self._experience_identifier(experience)
                if sid not in ids:
                    continue
                self._normalize_experience_meta(experience)
                evidence = experience.setdefault("evidence", {})
                pos_key = "positive_gain_count" if positive else "negative_gain_count"
                evidence[pos_key] = int(evidence.get(pos_key, 0)) + 1
                evidence.setdefault("positive_gain_count", 0)
                evidence.setdefault("negative_gain_count", 0)
                if traj_uid:
                    source_ids = set(_as_list(evidence.get("source_traj_uids", [])))
                    source_ids.add(str(traj_uid))
                    evidence["source_traj_uids"] = sorted(source_ids)
                evidence["last_updated_step"] = global_step
                experience["utility"] = float(experience.get("utility", 0.0)) + (1.0 if positive else -1.0)
                experience["retrieval_count"] = int(experience.get("retrieval_count", 0)) + 1
                experience["last_retrieval_step"] = global_step
                updated += 1
        return updated

    def update_utilities_for_trajectory(
        self,
        experience_ids: List[str],
        credit: float,
        global_step: int,
        beta: float,
    ) -> int:
        if not experience_ids or beta <= 0:
            return 0
        seen: set = set()
        updated = 0
        for pool_name in ("task_experiences", "step_experiences"):
            for experience in self.experiences.get(pool_name, []):
                sid = self._experience_identifier(experience)
                if sid not in experience_ids:
                    continue
                if sid in seen:
                    continue
                seen.add(sid)
                self._normalize_experience_meta(experience)
                u = float(experience.get("utility", 0.0))
                experience["utility"] = (1.0 - beta) * u + beta * float(credit)
                experience["retrieval_count"] = int(experience.get("retrieval_count", 0)) + 1
                experience["last_retrieval_step"] = global_step
                updated += 1
        return updated

    def update_utilities_from_group_credits(
        self,
        credits_by_experience: Mapping[str, Mapping[str, Any]],
        *,
        global_step: int,
        beta_task: float = 0.1,
        beta_step: float = 0.1,
        management_interval_steps: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply one EMA utility update per experience from grouped A/B credits.

        ``credits_by_experience`` is keyed by the stable ``experience_id``. A
        caller records each task group only once per experience; this method deliberately does not expand credits
        per rollout row.  ``retrieval_count`` is instead increased by the
        explicitly supplied number of injected trajectory exposures.

        The legacy :meth:`record_experience_gain` and
        :meth:`update_utilities_for_trajectory` APIs remain unchanged.  This
        method is the dynamic-management path and never converts gains to
        +/-1.
        """
        if not credits_by_experience:
            return []

        try:
            beta_task = float(beta_task)
            beta_step = float(beta_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("utility EMA beta must be numeric") from exc
        if not 0.0 <= beta_task <= 1.0 or not 0.0 <= beta_step <= 1.0:
            raise ValueError("utility EMA beta must be in [0, 1]")
        if management_interval_steps is not None:
            try:
                management_interval_steps = max(1, int(management_interval_steps))
            except (TypeError, ValueError) as exc:
                raise ValueError("management interval must be a positive integer") from exc

        updates: List[Dict[str, Any]] = []
        for pool_name, beta in (("task_experiences", beta_task), ("step_experiences", beta_step)):
            for experience in self.experiences.get(pool_name, []):
                if not isinstance(experience, dict):
                    continue
                sid = self._experience_identifier(experience)
                matched_id = sid
                if not matched_id or matched_id not in credits_by_experience:
                    continue
                payload = credits_by_experience[matched_id] or {}
                values = payload.get("credits", []) if isinstance(payload, Mapping) else payload
                try:
                    credits = [float(value) for value in values]
                except (TypeError, ValueError):
                    continue
                credits = [value for value in credits if math.isfinite(value)]
                if not credits:
                    continue
                try:
                    exposures = int(payload.get("exposures", 0) or 0) if isinstance(payload, Mapping) else 0
                except (TypeError, ValueError):
                    exposures = 0
                exposures = max(0, exposures)

                self._normalize_experience_meta(experience)
                old_utility = float(experience.get("utility", 0.0))
                if not math.isfinite(old_utility):
                    old_utility = DEFAULT_UTILITY
                credit = sum(credits) / len(credits)
                new_utility = (1.0 - beta) * old_utility + beta * credit

                evidence = experience.setdefault("evidence", {})
                positive = sum(1 for value in credits if value > 0.0)
                negative = sum(1 for value in credits if value < 0.0)
                neutral = len(credits) - positive - negative
                evidence["positive_gain_count"] = int(evidence.get("positive_gain_count", 0) or 0) + positive
                evidence["negative_gain_count"] = int(evidence.get("negative_gain_count", 0) or 0) + negative
                evidence["neutral_gain_count"] = int(evidence.get("neutral_gain_count", 0) or 0) + neutral
                evidence["utility_update_count"] = int(evidence.get("utility_update_count", 0) or 0) + 1
                if credit < 0.0:
                    previous_negative_step = evidence.get("last_negative_management_step")
                    is_adjacent_negative = management_interval_steps is None
                    if management_interval_steps is not None and previous_negative_step is not None:
                        try:
                            is_adjacent_negative = (
                                int(global_step) - int(previous_negative_step) == management_interval_steps
                            )
                        except (TypeError, ValueError):
                            is_adjacent_negative = False
                    evidence["consecutive_negative_windows"] = (
                        int(evidence.get("consecutive_negative_windows", 0) or 0) + 1
                        if is_adjacent_negative
                        else 1
                    )
                    evidence["last_negative_management_step"] = int(global_step)
                else:
                    evidence["consecutive_negative_windows"] = 0
                    evidence["last_negative_management_step"] = None
                evidence["last_updated_step"] = int(global_step)
                experience["utility"] = new_utility
                # Here retrieval_count has the precise lifecycle meaning of
                # injected trajectory exposure, rather than utility updates.
                experience["retrieval_count"] = int(experience.get("retrieval_count", 0) or 0) + exposures
                experience["last_retrieval_step"] = int(global_step)
                updates.append(
                    {
                        "pool": pool_name,
                        "experience_id": sid,
                        "matched_id": matched_id,
                        "old_utility": old_utility,
                        "utility": new_utility,
                        "mean_group_gain": credit,
                        "group_count": len(credits),
                        "exposures": exposures,
                        "positive_group_count": positive,
                        "negative_group_count": negative,
                        "neutral_group_count": neutral,
                        "consecutive_negative_windows": evidence["consecutive_negative_windows"],
                        "last_negative_management_step": evidence["last_negative_management_step"],
                    }
                )
        if updates:
            # Cached formatted retrieval results contain the old ordering and
            # utility metadata even though embeddings themselves are reusable.
            self._query_cache.clear()
        return updates

    def reset_negative_windows_for_uncredited(
        self,
        observed_experience_ids: Sequence[str],
        *,
        global_step: int,
    ) -> int:
        """Break negative-window continuity for experiences absent this window.

        A management window with no retrieved exposure is evidence that a
        experience did not receive a consecutive negative A/B result.  This must be
        applied before eviction so an old two-window negative streak cannot be
        hard-pruned after an unobserved interval.
        """
        observed = {str(item) for item in observed_experience_ids if item}
        reset = 0
        for pool_name in ("task_experiences", "step_experiences"):
            for experience in self.experiences.get(pool_name, []):
                if not isinstance(experience, dict):
                    continue
                sid = self._experience_identifier(experience)
                if sid in observed:
                    continue
                evidence = experience.get("evidence") or {}
                if int(evidence.get("consecutive_negative_windows", 0) or 0) <= 0:
                    continue
                evidence["consecutive_negative_windows"] = 0
                evidence["last_negative_management_step"] = None
                evidence["last_negative_window_reset_step"] = int(global_step)
                experience["evidence"] = evidence
                reset += 1
        return reset

    def replace_experiences_keep_cache_incremental(self, new_experiences: Dict[str, Any]) -> None:
        import numpy as np

        new_experiences = self._coerce_bank_schema(new_experiences)
        for pool, attr in [("task_experiences", "_task_experience_embeddings_cache"), ("step_experiences", "_step_experience_embeddings_cache")]:
            new_items = new_experiences.get(pool, [])
            existing = getattr(self, attr, None)
            if not new_items:
                setattr(self, attr, None)
                continue
            cached_items = (existing.get("items", []) if existing else []) or []
            if not existing or len(cached_items) == 0 or len(new_items) < len(cached_items):
                setattr(self, attr, None)
                continue
            ids_cached = tuple(self._experience_identifier(s) or f"_{i}" for i, s in enumerate(cached_items))
            ids_new_prefix = tuple(self._experience_identifier(s) or f"_{i}" for i, s in enumerate(new_items[: len(cached_items)]))
            if ids_cached != ids_new_prefix:
                setattr(self, attr, None)
                continue
            cached_texts = existing.get("texts")
            if cached_texts is None:
                cached_texts = [self._experience_text_for_embedding(experience) for experience in cached_items]
            new_prefix_texts = [
                self._experience_text_for_embedding(experience)
                for experience in new_items[: len(cached_items)]
            ]
            if cached_texts != new_prefix_texts:
                setattr(self, attr, None)
                continue
            tail = new_items[len(cached_items) :]
            if not tail:
                setattr(
                    self,
                    attr,
                    {
                        "items": list(new_items),
                        "texts": new_prefix_texts,
                        "embeddings": existing["embeddings"],
                    },
                )
                continue
            texts = [self._experience_text_for_embedding(s) for s in tail]
            tail_embs = self._encode_texts(texts, normalize_embeddings=True)
            if hasattr(tail_embs, "shape") and len(tail_embs.shape) == 1:
                tail_embs = np.expand_dims(tail_embs, axis=0)
            old_embs = existing["embeddings"]
            merged = np.concatenate([old_embs, tail_embs], axis=0)
            setattr(
                self,
                attr,
                {
                    "items": list(new_items),
                    "texts": new_prefix_texts + texts,
                    "embeddings": merged,
                },
            )
        self.experiences = new_experiences
        self._normalize_all_experience_meta()

    def remove_experience(self, experience_id: str) -> bool:
        removed = False
        for pool_name in ("task_experiences", "step_experiences"):
            pool = self.experiences.get(pool_name, [])
            new_pool = []
            for experience in pool:
                sid = self._experience_identifier(experience)
                if sid == experience_id:
                    removed = True
                    continue
                new_pool.append(experience)
            self.experiences[pool_name] = new_pool
            if removed:
                if pool_name == "task_experiences":
                    self._task_experience_embeddings_cache = None
                else:
                    self._step_experience_embeddings_cache = None
        return removed

    def evict_excess_experiences(
        self,
        current_step: int,
        max_task_experiences: Optional[int] = None,
        max_step_experiences: Optional[int] = None,
        protect_recent_steps: int = 0,
        score_c: float = 1.0,
        min_exposures: int = 0,
        utility_threshold: Optional[float] = None,
        negative_windows: int = 0,
    ) -> Dict[str, Any]:
        """Evict stale experiences and, if necessary, trim pools by UCB retention.

        With the new optional negative-evidence arguments, hard-pruning is
        performed before capacity trimming.  Omitting them keeps the previous
        capacity-only behaviour for callers outside dynamic management.
        """
        out: Dict[str, Any] = {
            "current_step": int(current_step),
            "removed": [],
            "task_experiences_before": 0,
            "task_experiences_after": 0,
            "step_experiences_before": 0,
            "step_experiences_after": 0,
            "warnings": [],
        }
        protect_recent_steps = max(0, int(protect_recent_steps))
        try:
            min_exposures = max(0, int(min_exposures))
        except (TypeError, ValueError):
            min_exposures = 0
        try:
            negative_windows = max(0, int(negative_windows))
        except (TypeError, ValueError):
            negative_windows = 0
        if utility_threshold is not None:
            try:
                utility_threshold = float(utility_threshold)
            except (TypeError, ValueError):
                utility_threshold = None
        try:
            score_c = float(score_c)
            if math.isnan(score_c) or math.isinf(score_c):
                score_c = 1.0
        except (TypeError, ValueError):
            score_c = 1.0
        cutoff_created = int(current_step) - protect_recent_steps

        def _evict_pool(pool_name: str, max_size: Optional[int]) -> None:
            pool = self.experiences.get(pool_name, [])
            key_before = f"{pool_name.replace('_experiences', '')}_experiences_before"
            key_after = f"{pool_name.replace('_experiences', '')}_experiences_after"
            out[key_before] = len(pool)
            N = sum(int(s.get("retrieval_count", 0)) for s in pool)
            log_N = math.log(1 + N) if N > 0 else 0.0
            hard_candidates: List[Tuple[float, int, Dict[str, Any], float]] = []
            capacity_candidates: List[Tuple[float, int, Dict[str, Any], float]] = []
            for idx, experience in enumerate(pool):
                if not isinstance(experience, dict):
                    continue
                try:
                    self._normalize_experience_meta(experience)
                    created = int(experience.get("created_at_step", 0) or 0)
                    if created > cutoff_created:
                        continue
                    u = float(experience.get("utility", 0))
                    if math.isnan(u) or math.isinf(u):
                        u = 0.0
                    n = int(experience.get("retrieval_count", 0))
                    denom = 1 + n
                    ucb_bonus = self._retrieval_ucb_c * (
                        math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0
                    )
                    sort_key = u + score_c * ucb_bonus
                    if math.isnan(sort_key) or math.isinf(sort_key):
                        sort_key = u
                    evidence = experience.get("evidence") or {}
                    consecutive_negative = int(evidence.get("consecutive_negative_windows", 0) or 0)
                    if (
                        utility_threshold is not None
                        and n >= min_exposures
                        and u <= utility_threshold
                        and consecutive_negative >= negative_windows
                    ):
                        hard_candidates.append((u, idx, experience, sort_key))
                    else:
                        capacity_candidates.append((sort_key, idx, experience, sort_key))
                except (TypeError, ValueError):
                    continue
            hard_candidates.sort(key=lambda item: (item[0], item[1]))
            picked: List[Tuple[float, int, Dict[str, Any], float, str]] = [
                (utility, idx, experience, retain_score, "hard_negative_utility")
                for utility, idx, experience, retain_score in hard_candidates
            ]
            removed_indices = {idx for _, idx, _, _, _ in picked}

            remaining_count = len(pool) - len(removed_indices)
            if max_size is not None and max_size >= 0 and remaining_count > max_size:
                excess = remaining_count - max_size
                capacity_candidates.sort(key=lambda item: (item[0], item[1]))
                capacity_picked = capacity_candidates[:excess]
                picked.extend(
                    (retain_score, idx, experience, retain_score, "capacity_retain_score")
                    for retain_score, idx, experience, _ in capacity_picked
                )
                removed_indices.update(idx for _, idx, _, _ in capacity_picked)
                if len(capacity_picked) < excess:
                    out["warnings"].append(
                        f"{pool_name}: need_remove={excess} but only {len(capacity_picked)} deletable "
                        f"(protected by recent_steps={protect_recent_steps}); pool may stay above max."
                    )

            for _, idx, experience, retain_score, reason in picked:
                evidence = experience.get("evidence") or {}
                out["removed"].append(
                    {
                        "pool": pool_name,
                        "pool_index": int(idx),
                        "experience_id": experience.get("experience_id"),
                        "utility": float(experience.get("utility", 0)),
                        "retrieval_count": int(experience.get("retrieval_count", 0)),
                        "exposure": int(experience.get("retrieval_count", 0)),
                        "consecutive_negative_windows": int(evidence.get("consecutive_negative_windows", 0) or 0),
                        "created_at_step": int(experience.get("created_at_step", 0) or 0),
                        "retain_score": float(retain_score),
                        "reason": reason,
                    }
                )
            if removed_indices:
                self.experiences[pool_name] = [s for i, s in enumerate(pool) if i not in removed_indices]
                if pool_name == "task_experiences":
                    self._task_experience_embeddings_cache = None
                else:
                    self._step_experience_embeddings_cache = None
                self._query_cache.clear()
            out[key_after] = len(self.experiences[pool_name])

        _evict_pool("task_experiences", max_task_experiences)
        _evict_pool("step_experiences", max_step_experiences)
        return out

    def save_experiences(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.experiences, f, indent=2, ensure_ascii=False)
        print(f"[ExperienceMemory] Saved {len(self)} experiences to {path}")

    def load_experiences(self, path: str) -> bool:
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            self.experiences = self._coerce_bank_schema(loaded)
            self._normalize_all_experience_meta()
            self._task_experience_embeddings_cache = None
            self._step_experience_embeddings_cache = None
            self._query_cache.clear()
            print(f"[ExperienceMemory] Loaded {len(self)} experiences from {path}")
            return True
        except Exception as exc:
            print(f"[ExperienceMemory] Failed to load experiences from {path}: {exc}")
            return False

    def reload_remote_experiences(self, experiences: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> bool:
        if not self._retrieval_service_urls:
            return False
        import requests

        payload: Dict[str, Any] = {}
        if experiences is not None:
            payload["experiences"] = experiences
        elif path is not None:
            payload["path"] = path
        else:
            payload["experiences"] = self.experiences

        ok = True
        for url in self._retrieval_service_urls:
            base = url.rstrip("/")
            if base.endswith("/retrieve_batch"):
                base = base[: -len("/retrieve_batch")]
            if not base.endswith("/reload_experiences"):
                base = f"{base}/reload_experiences"
            try:
                session = self._get_http_session(base)
                resp = session.post(base, json=payload, timeout=self._retrieval_timeout)
                resp.raise_for_status()
            except Exception as exc:
                ok = False
                print(f"[ExperienceMemory] Failed to reload remote experiences at {base}: {exc}")
        return ok

    # ------------------------------------------------------------------ #
    # BaseMemory interface                                                #
    # ------------------------------------------------------------------ #

    def reset(self, batch_size: int):
        pass

    def store(self, record: Dict[str, List[Any]]):
        pass

    def fetch(self, step: int):
        pass

    def __len__(self):
        return len(self.experiences.get("task_experiences", [])) + len(self.experiences.get("step_experiences", []))

    def __getitem__(self, idx: int):
        return self.experiences

    def get_experience_count(self) -> Dict[str, int]:
        return {
            "task_experiences": len(self.experiences.get("task_experiences", [])),
            "step_experiences": len(self.experiences.get("step_experiences", [])),
            "total": len(self),
        }
