#!/usr/bin/env python3
"""HTTP server for remote experience embedding retrieval."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


app = FastAPI()
_experience_memory = None


class ReloadExperiencesRequest(BaseModel):
    path: Optional[str] = None
    experiences: Optional[Dict[str, Any]] = None


class RetrieveBatchRequest(BaseModel):
    queries: List[str]
    top_k: int = 6
    # Optional semantic-candidate size.  Clients can locally re-rank this
    # larger set with the newest utility/UCB values while retaining top_k as
    # their final requested result size.
    candidate_k: Optional[int] = None
    pool: Optional[str] = None
    task_specific_top_k: Optional[int] = None
    experience_text_for_retrieval: Optional[str] = None


@app.post("/retrieve_batch")
def retrieve_batch_endpoint(request: RetrieveBatchRequest):
    global _experience_memory
    if _experience_memory is None:
        raise HTTPException(status_code=500, detail="Experience memory not initialized")
    if request.pool not in ("task_experiences", "step_experiences"):
        raise HTTPException(status_code=400, detail="pool must be 'task_experiences' or 'step_experiences'")

    print(
        f"[ExperienceRetrievalServer] /retrieve_batch pool={request.pool} "
        f"queries={len(request.queries or [])} top_k={request.top_k} "
        f"candidate_k={request.candidate_k}",
        flush=True,
    )
    candidate_k = request.candidate_k if request.candidate_k is not None else request.top_k
    candidate_k = max(int(request.top_k), int(candidate_k))
    result = _experience_memory.retrieve_batch(
        request.queries or [],
        top_k=candidate_k,
        pool=request.pool,
        # retrieve_task_experiences_batch normally lets task_specific_top_k replace
        # top_k.  On this server that would accidentally shrink the semantic
        # candidate set before the client can apply utility/UCB.
        task_specific_top_k=candidate_k if request.pool == "task_experiences" else request.task_specific_top_k,
        experience_text_for_retrieval=request.experience_text_for_retrieval,
    )
    return {"result": result}


@app.post("/reload_experiences")
def reload_experiences_endpoint(request: ReloadExperiencesRequest):
    global _experience_memory
    if _experience_memory is None:
        raise HTTPException(status_code=500, detail="Experience memory not initialized")

    if request.experiences is not None:
        source = "inline"
        experiences = request.experiences
    elif request.path:
        path = Path(request.path)
        if not path.is_absolute():
            path = _repo_root / path
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {path}")
        source = str(path)
        with open(path, "r", encoding="utf-8") as f:
            experiences = json.load(f)
    else:
        raise HTTPException(status_code=400, detail="Provide either 'path' or 'experiences'")

    print(f"[ExperienceRetrievalServer] /reload_experiences source={source}", flush=True)
    if hasattr(_experience_memory, "replace_experiences_keep_cache_incremental"):
        _experience_memory.replace_experiences_keep_cache_incremental(experiences)
    else:
        _experience_memory.experiences = experiences
        _experience_memory._task_experience_embeddings_cache = None
        _experience_memory._step_experience_embeddings_cache = None

    counts = _experience_memory.get_experience_count() if hasattr(_experience_memory, "get_experience_count") else {}
    total = counts.get("total", "?")
    print(f"[ExperienceRetrievalServer] /reload_experiences done total_experiences={total}", flush=True)
    return {"status": "ok", "source": source, "total_experiences": total, **counts}


def main():
    global _experience_memory
    parser = argparse.ArgumentParser(description="Experience retrieval server (embedding mode).")
    parser.add_argument("--experiences_json_path", type=str, default=None)
    parser.add_argument("--no_load_initial_experiences", action="store_true")
    parser.add_argument("--embedding_model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--experience_text_for_retrieval", type=str, default="full")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_initial = not args.no_load_initial_experiences
    if load_initial and not args.experiences_json_path:
        parser.error("--experiences_json_path is required unless --no_load_initial_experiences is set.")

    from agent_system.memory import ExperienceMemory

    _experience_memory = ExperienceMemory(
        experiences_json_path=args.experiences_json_path if load_initial else None,
        retrieval_mode="embedding",
        embedding_model_path=args.embedding_model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        experience_text_for_retrieval=args.experience_text_for_retrieval,
        load_initial_experiences=load_initial,
    )

    if not load_initial:
        print("[ExperienceRetrievalServer] Warming up embedding model (empty experience bank)...", flush=True)
        _experience_memory._get_embedding_model()
        print("[ExperienceRetrievalServer] Embedding model ready.", flush=True)

    print(
        f"[ExperienceRetrievalServer] Listening on {args.host}:{args.port} "
        f"(num_gpus={args.num_gpus})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
