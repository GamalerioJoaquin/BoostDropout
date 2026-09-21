"""Run artefacts with reproducibility metadata."""

import json
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values to JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str | Path, payload: Any) -> Path:
    """Persist JSON with stable formatting and create its parent directory."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_path


def _git_value(arguments: list[str], project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_runtime_metadata(project_root: str | Path = ".") -> dict[str, Any]:
    """Collect the code and runtime facts needed to reproduce a run."""
    root = Path(project_root).resolve()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"], root),
            "branch": _git_value(["branch", "--show-current"], root),
            "dirty": _git_value(["status", "--porcelain"], root) != "",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }


class RunArtifactStore:
    """Persist checkpoints, history and metadata in one isolated run directory."""

    def __init__(self, root: str | Path, run_name: str, project_root: str | Path = ".") -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{timestamp}-{run_name}-{uuid.uuid4().hex[:8]}"
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.project_root = Path(project_root).resolve()

    def save_metadata(self, configuration: dict[str, Any], seeds: dict[str, Any]) -> Path:
        """Write run configuration, seed state, Git identity and runtime versions."""
        payload = {
            "run_id": self.run_id,
            "configuration": configuration,
            "seeds": seeds,
            **collect_runtime_metadata(self.project_root),
        }
        return save_json(self.path / "metadata.json", payload)

    def save_history(self, history: dict[str, list[float]]) -> Path:
        """Write thesis-compatible metric history."""
        return save_json(self.path / "history.json", history)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        history: dict[str, list[float]],
        name: str = "last",
    ) -> Path:
        """Write a portable PyTorch checkpoint containing model and optimizer state."""
        path = self.path / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            path,
        )
        return path
