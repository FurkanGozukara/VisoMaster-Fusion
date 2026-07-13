import unittest

import numpy as np

from app.processors.video_utils.face_scan import (
    UniqueFaceClusterer,
    build_face_scan_frame_numbers,
    scaled_cosine_similarity,
)


class FaceScanTests(unittest.TestCase):
    def test_smart_scan_samples_complete_clip_including_last_frame(self):
        frames = build_face_scan_frame_numbers(
            frame_count=240,
            source_fps=24.0,
            mode_key="smart",
        )

        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], 239)
        self.assertEqual(frames[:3], [0, 4, 8])
        self.assertEqual(len(frames), 61)

    def test_quick_scan_uniformly_caps_long_clips(self):
        frames = build_face_scan_frame_numbers(
            frame_count=1_000_000,
            source_fps=30.0,
            mode_key="quick",
        )

        self.assertEqual(len(frames), 1200)
        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], 999_999)
        self.assertEqual(frames, sorted(set(frames)))

    def test_scaled_cosine_similarity_matches_application_scale(self):
        self.assertEqual(
            scaled_cosine_similarity(np.array([1, 0]), np.array([1, 0])), 100.0
        )
        self.assertEqual(
            scaled_cosine_similarity(np.array([1, 0]), np.array([0, 1])), 50.0
        )
        self.assertEqual(
            scaled_cosine_similarity(np.array([1, 0]), np.array([-1, 0])), 0.0
        )

    def test_clusterer_excludes_existing_and_groups_similar_new_views(self):
        crop = np.full((16, 16, 3), 127, dtype=np.uint8)
        clusterer = UniqueFaceClusterer(
            threshold=85.0,
            existing_representatives=[(np.array([1.0, 0.0]), 90.0)],
        )

        self.assertEqual(
            clusterer.consider(np.array([0.9, 0.1]), crop, 0, 100), "existing"
        )
        self.assertEqual(
            clusterer.consider(np.array([0.0, 1.0]), crop, 10, 100), "added"
        )
        self.assertEqual(
            clusterer.consider(np.array([0.1, 0.9]), crop, 20, 100), "matched"
        )
        self.assertEqual(
            clusterer.consider(np.array([-1.0, 0.0]), crop, 30, 100), "added"
        )

        results = clusterer.results("Inswapper128ArcFace", source_fps=10.0)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["occurrences"], 2)
        self.assertEqual(results[0]["frame_number"], 10)
        self.assertEqual(results[0]["last_frame"], 20)
        self.assertEqual(results[0]["timestamp_seconds"], 1.0)

    def test_clusterer_enforces_result_safety_limit(self):
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        clusterer = UniqueFaceClusterer(threshold=99.9, max_results=1)

        self.assertEqual(clusterer.consider(np.array([1.0, 0.0]), crop, 0, 64), "added")
        self.assertEqual(clusterer.consider(np.array([0.0, 1.0]), crop, 1, 64), "limit")
        self.assertTrue(clusterer.limit_reached)
        self.assertEqual(clusterer.rejected_unique_count, 1)

    def test_clusterer_keeps_source_frame_in_sync_with_clearer_thumbnail(self):
        small_crop = np.zeros((8, 8, 3), dtype=np.uint8)
        large_crop = np.ones((16, 16, 3), dtype=np.uint8)
        clusterer = UniqueFaceClusterer(threshold=85.0)

        clusterer.consider(np.array([1.0, 0.0]), small_crop, 10, 64)
        clusterer.consider(np.array([0.99, 0.01]), large_crop, 20, 256)

        result = clusterer.results("Inswapper128ArcFace", source_fps=10.0)[0]
        self.assertEqual(result["first_frame"], 10)
        self.assertEqual(result["frame_number"], 20)
        self.assertEqual(result["timestamp_seconds"], 2.0)
        self.assertEqual(result["cropped_face"].shape, (16, 16, 3))
        np.testing.assert_allclose(
            result["embedding_store"]["Inswapper128ArcFace"],
            np.array([0.99, 0.01], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
