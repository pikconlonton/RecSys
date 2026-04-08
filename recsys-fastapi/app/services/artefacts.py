from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import faiss  # type: ignore
import numpy as np
import torch


@dataclass(frozen=True)
class RecSysArtefacts:
    """In-memory artefacts for inference.

    Loaded once at startup to avoid per-request disk IO.

    Shapes:
      - user_h: [N_user, D]
      - biz_h : [N_biz, D]
    """

    user_h: torch.Tensor
    biz_h: torch.Tensor
    index: faiss.Index
    user2idx: dict
    biz2idx: dict
    idx2biz: dict


def _repo_root_from_here() -> Path:
    # recsys-fastapi/app/services/artefacts.py -> RecSys/recsys-fastapi/app/services
    return Path(__file__).resolve().parents[3]


def get_outputs_dir() -> Path:
    """Resolve outputs directory.

    Priority:
      1) env RECSYS_OUTPUTS_DIR
      2) repo root ../outputs (assuming monorepo layout)
    """

    env = os.getenv("RECSYS_OUTPUTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root_from_here() / "outputs").resolve()


def load_artefacts(outputs_dir: Path | None = None) -> RecSysArtefacts:
    out = outputs_dir or get_outputs_dir()

    user_h = torch.load(out / "user_h.pt", weights_only=True)
    biz_h = torch.load(out / "biz_h.pt", weights_only=True)
    mappings = torch.load(out / "mappings.pt", weights_only=False)
    index = faiss.read_index(str(out / "faiss_biz.index"))

    user2idx = mappings["user2idx"]
    biz2idx = mappings["biz2idx"]
    idx2biz = mappings["idx2biz"]

    # Defensive: ensure embeddings are float32 on CPU.
    user_h = user_h.detach().to("cpu").float()
    biz_h = biz_h.detach().to("cpu").float()

    return RecSysArtefacts(
        user_h=user_h,
        biz_h=biz_h,
        index=index,
        user2idx=user2idx,
        biz2idx=biz2idx,
        idx2biz=idx2biz,
    )


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vec)
    return vec
