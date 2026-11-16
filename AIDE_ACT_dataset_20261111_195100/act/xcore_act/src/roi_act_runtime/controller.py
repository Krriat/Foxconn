from __future__ import annotations

import copy
import threading
import time
from queue import Queue
from typing import Any, Callable

import numpy as np


PredictionFunction = Callable[[dict[str, Any]], np.ndarray]


class ChunkFusionController:
    """Fuse overlapping, strided ROI-ACT chunks without blocking every tick."""

    def __init__(
        self,
        predict_chunk: PredictionFunction,
        *,
        raw_chunk_size: int,
        action_steps: int,
        action_stride: int,
        replan_interval: int,
        new_prediction_weight: float,
    ) -> None:
        if raw_chunk_size <= 0 or action_steps <= 0:
            raise ValueError("chunk sizes must be positive")
        if action_stride <= 0 or replan_interval <= 0:
            raise ValueError("stride and replan interval must be positive")
        if not 0.0 <= new_prediction_weight <= 1.0:
            raise ValueError("new prediction weight must be in [0,1]")
        required = 1 + (action_steps - 1) * action_stride
        if required > raw_chunk_size:
            raise ValueError(
                f"need {required} raw actions for action_steps={action_steps} "
                f"and stride={action_stride}, but chunk_size={raw_chunk_size}"
            )
        if replan_interval > action_steps:
            raise ValueError("replan_interval cannot exceed action_steps")

        self.predict_chunk = predict_chunk
        self.raw_chunk_size = raw_chunk_size
        self.action_steps = action_steps
        self.action_stride = action_stride
        self.replan_interval = replan_interval
        self.new_weight = new_prediction_weight
        self.old_weight = 1.0 - new_prediction_weight
        self._actions: dict[int, np.ndarray] = {}
        self._step = 0
        self._actions_since_replan = 0
        self._worker: threading.Thread | None = None
        self._worker_result: Queue[tuple[int, np.ndarray | None, BaseException | None, float]] = Queue(
            maxsize=1
        )

    @property
    def step(self) -> int:
        return self._step

    def _predict(self, observation: dict[str, Any]) -> np.ndarray:
        raw = np.asarray(self.predict_chunk(observation), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != self.raw_chunk_size:
            raise RuntimeError(
                f"predict_chunk must return [{self.raw_chunk_size}, action_dim], got {raw.shape}"
            )
        indices = np.arange(0, self.action_steps * self.action_stride, self.action_stride)
        return raw[indices].copy()

    def _merge(self, origin: int, chunk: np.ndarray) -> None:
        fused = 0
        added = 0
        for offset, new_action in enumerate(chunk):
            timestep = origin + offset
            if timestep < self._step:
                continue
            if timestep in self._actions:
                self._actions[timestep] = (
                    self.old_weight * self._actions[timestep]
                    + self.new_weight * new_action
                )
                fused += 1
            else:
                self._actions[timestep] = new_action.copy()
                added += 1
        print(
            f"[roi-act] chunk@{origin} merged: fused={fused}, "
            f"added={added}, queue={len(self._actions)}",
            flush=True,
        )

    def _run_worker(self, origin: int, observation: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            chunk = self._predict(observation)
            self._worker_result.put(
                (origin, chunk, None, time.perf_counter() - started)
            )
        except BaseException as exc:
            self._worker_result.put(
                (origin, None, exc, time.perf_counter() - started)
            )

    def _start_worker(self, origin: int, observation: dict[str, Any]) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(origin, copy.deepcopy(observation)),
            name="roi-act-chunk-fusion",
            daemon=True,
        )
        self._worker.start()
        print(f"[roi-act] look-ahead started at step {origin}", flush=True)

    def _collect_worker(self, *, block: bool) -> bool:
        if self._worker is None:
            return False
        if block:
            self._worker.join()
        elif self._worker.is_alive():
            return False
        self._worker.join()
        origin, chunk, error, elapsed = self._worker_result.get()
        self._worker = None
        if error is not None:
            raise RuntimeError(f"ROI-ACT inference failed at step {origin}") from error
        assert chunk is not None
        print(f"[roi-act] chunk@{origin} ready in {elapsed:.3f}s", flush=True)
        self._merge(origin, chunk)
        return True

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        self._collect_worker(block=False)
        if self._step not in self._actions:
            if self._worker is not None:
                print(
                    f"[roi-act] queue empty at step {self._step}; waiting for look-ahead",
                    flush=True,
                )
                self._collect_worker(block=True)
            else:
                started = time.perf_counter()
                self._merge(self._step, self._predict(copy.deepcopy(observation)))
                print(
                    f"[roi-act] synchronous seed ready in "
                    f"{time.perf_counter() - started:.3f}s",
                    flush=True,
                )
                self._actions_since_replan = 0
        if self._step not in self._actions:
            raise RuntimeError(f"chunk fusion produced no action for step {self._step}")

        action = self._actions.pop(self._step)
        origin = self._step
        self._step += 1
        self._actions_since_replan += 1
        if self._worker is None and self._actions_since_replan >= self.replan_interval:
            self._start_worker(origin, observation)
            self._actions_since_replan = 0
        return action

    def reset(self) -> None:
        if self._worker is not None:
            self._worker.join()
            self._worker = None
        while not self._worker_result.empty():
            self._worker_result.get_nowait()
        self._actions.clear()
        self._step = 0
        self._actions_since_replan = 0

    def close(self) -> None:
        self.reset()
