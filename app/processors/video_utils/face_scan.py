from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FaceScanMode:
    key: str
    label: str
    samples_per_second: float | None
    max_samples: int | None


FACE_SCAN_MODES = {
    "quick": FaceScanMode("quick", "Quick", 2.0, 1200),
    "smart": FaceScanMode("smart", "Smart", 6.0, 3600),
    "thorough": FaceScanMode("thorough", "Every Frame", None, None),
}


def get_face_scan_mode(mode_key: str) -> FaceScanMode:
    return FACE_SCAN_MODES.get(str(mode_key).lower(), FACE_SCAN_MODES["smart"])


def build_face_scan_frame_numbers(
    frame_count: int,
    source_fps: float,
    mode_key: str,
) -> list[int]:
    """Return frame indices spread across the complete clip for a scan mode."""
    frame_count = max(0, int(frame_count))
    if frame_count == 0:
        return []

    mode = get_face_scan_mode(mode_key)
    fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
    if mode.samples_per_second is None:
        stride = 1
    else:
        stride = max(1, int(round(fps / mode.samples_per_second)))

    frames = list(range(0, frame_count, stride))
    last_frame = frame_count - 1
    if frames[-1] != last_frame:
        frames.append(last_frame)

    if mode.max_samples is not None and len(frames) > mode.max_samples:
        if mode.max_samples == 1:
            return [0]
        last_index = len(frames) - 1
        sample_indices = {
            int(round(i * last_index / (mode.max_samples - 1)))
            for i in range(mode.max_samples)
        }
        frames = [frames[index] for index in sorted(sample_indices)]
        if frames[-1] != last_frame:
            frames[-1] = last_frame

    return frames


def scaled_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Match the application's 0-100 cosine similarity scale."""
    a = np.asarray(vector_a, dtype=np.float32).ravel()
    b = np.asarray(vector_b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("-inf")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return float("-inf")
    cosine = float(np.dot(a, b) / denominator)
    if not np.isfinite(cosine):
        return float("-inf")
    return 50.0 + 50.0 * float(np.clip(cosine, -1.0, 1.0))


class UniqueFaceClusterer:
    """Greedy coverage set for face views under the app's match threshold.

    Every accepted result is below the configured threshold for all earlier
    representatives. This is intentional: each result is a view that the
    current target set would otherwise fail to match.
    """

    def __init__(
        self,
        threshold: float,
        existing_representatives: Iterable[tuple[np.ndarray, float]] = (),
        max_results: int = 100,
    ):
        self.threshold = float(threshold)
        self.max_results = max(1, int(max_results))
        self.existing_representatives = [
            (np.asarray(embedding, dtype=np.float32).copy(), float(face_threshold))
            for embedding, face_threshold in existing_representatives
            if self._valid_embedding(embedding)
        ]
        self.clusters: list[dict] = []
        self.limit_reached = False
        self.rejected_unique_count = 0

    @staticmethod
    def _valid_embedding(embedding) -> bool:
        return (
            isinstance(embedding, np.ndarray)
            and embedding.size > 0
            and bool(np.isfinite(embedding).all())
        )

    def consider(
        self,
        embedding: np.ndarray,
        cropped_face: np.ndarray,
        frame_number: int,
        face_area: float,
    ) -> str:
        if not self._valid_embedding(embedding):
            return "invalid"
        if not isinstance(cropped_face, np.ndarray) or cropped_face.size == 0:
            return "invalid"

        for representative, threshold in self.existing_representatives:
            if scaled_cosine_similarity(embedding, representative) >= threshold:
                return "existing"

        best_cluster = None
        best_similarity = float("-inf")
        for cluster in self.clusters:
            similarity = scaled_cosine_similarity(
                embedding, cluster["representative_embedding"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= self.threshold:
            best_cluster["occurrences"] += 1
            best_cluster["last_frame"] = int(frame_number)
            # A near-identical, larger detection makes a clearer review thumbnail
            # without changing the representative that established coverage.
            if best_similarity >= 85.0 and face_area > best_cluster["thumbnail_area"]:
                best_cluster["cropped_face"] = np.ascontiguousarray(cropped_face.copy())
                best_cluster["thumbnail_embedding"] = np.asarray(
                    embedding, dtype=np.float32
                ).copy()
                best_cluster["thumbnail_area"] = float(face_area)
                best_cluster["thumbnail_frame"] = int(frame_number)
            return "matched"

        if len(self.clusters) >= self.max_results:
            self.limit_reached = True
            self.rejected_unique_count += 1
            return "limit"

        frame_number = int(frame_number)
        self.clusters.append(
            {
                "representative_embedding": np.asarray(
                    embedding, dtype=np.float32
                ).copy(),
                "thumbnail_embedding": np.asarray(embedding, dtype=np.float32).copy(),
                "cropped_face": np.ascontiguousarray(cropped_face.copy()),
                "first_frame": frame_number,
                "last_frame": frame_number,
                "thumbnail_frame": frame_number,
                "occurrences": 1,
                "thumbnail_area": float(face_area),
            }
        )
        return "added"

    def results(self, recognition_model: str, source_fps: float) -> list[dict]:
        fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
        return [
            {
                "embedding_store": {
                    str(recognition_model): cluster["thumbnail_embedding"]
                },
                "cropped_face": cluster["cropped_face"],
                "frame_number": cluster["thumbnail_frame"],
                "first_frame": cluster["first_frame"],
                "last_frame": cluster["last_frame"],
                "timestamp_seconds": cluster["thumbnail_frame"] / fps,
                "occurrences": cluster["occurrences"],
            }
            for cluster in self.clusters
        ]
