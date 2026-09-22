"""Human-readable, history-backed progress logging for long pipelines."""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Iterator


LOGGER = logging.getLogger("santander.pipeline")


class PipelineProgress:
    """Print pipeline stage status and estimate remaining time from prior runs."""

    def __init__(self, history_path: str | Path, stages: list[str]) -> None:
        self.history_path, self.stages = Path(history_path), stages
        self.history = self._load_history(self.history_path)
        self.completed: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        index = self.stages.index(name) + 1
        estimate = self._remaining_estimate(name)
        suffix = f" | estimated remaining: {self._format(estimate)}" if estimate is not None else " | estimated remaining: unavailable until a completed run"
        LOGGER.info("[%s/%s] START %s%s", index, len(self.stages), name, suffix)
        started = perf_counter()
        try:
            yield
        except Exception:
            LOGGER.exception("[%s/%s] FAILED %s after %s", index, len(self.stages), name, self._format(perf_counter() - started))
            raise
        duration = perf_counter() - started
        self.completed[name] = duration
        LOGGER.info("[%s/%s] DONE %s in %s", index, len(self.stages), name, self._format(duration))

    def save(self) -> None:
        payload = self.history
        payload.setdefault("runs", []).append({"finished_at_utc": datetime.now(UTC).isoformat(), "stages_seconds": self.completed})
        payload["runs"] = payload["runs"][-20:]
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Pipeline timing summary: %s", ", ".join(f"{stage}={self._format(seconds)}" for stage, seconds in self.completed.items()))
        LOGGER.info("Timing history saved: %s", self.history_path)

    def _remaining_estimate(self, current: str) -> float | None:
        start = self.stages.index(current)
        values: list[float] = []
        for stage in self.stages[start:]:
            durations = [run.get("stages_seconds", {}).get(stage) for run in self.history.get("runs", [])]
            durations = [value for value in durations if isinstance(value, (int, float))]
            if not durations:
                return None
            values.append(sum(durations) / len(durations))
        return sum(values)

    @staticmethod
    def _load_history(path: Path) -> dict[str, object]:
        if not path.is_file(): return {"runs": []}
        try: return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError: return {"runs": []}

    @staticmethod
    def _format(seconds: float) -> str:
        rounded = max(0, round(seconds))
        return f"{rounded // 60}m {rounded % 60:02d}s"
