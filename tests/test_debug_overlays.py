import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot101.data.utilities.debug_overlays import (
    DebugOverlayRenderer, check_output_mode, write_output_mode,
)


class DebugOverlayTests(unittest.TestCase):
    def record(self):
        return {"subtask": 1, "start_frame": 0, "end_frame": 4,
                "pov": {"source": "cluster", "gripper_filter": {"status": "ok", "window": [2, 3], "mask_file": "mask.npz"},
                        "tracks": [[[40, 50]]*5], "visibility": [[True, False, True, True, True]],
                        "previous_stage": {"tracks": [[[65, 50]]*2], "visibility": [[True, True]]}},
                "third_person": {"top": {"source": "gripper_fallback", "tracks": [[[40, 50]]*5],
                                          "visibility": [[True]*5]}}}

    def renderer(self, root, record):
        masks = np.zeros((2, 80, 100), bool)
        masks[:, 55:60, 80:85] = True
        np.savez(root / "mask.npz", packed_masks=np.packbits(masks, axis=-1),
                 shape=masks.shape, frame_indices=[2, 3])
        return DebugOverlayRenderer([record], root)

    def test_masks_points_and_clean_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = self.renderer(Path(tmp), self.record())
            clean = np.zeros((80, 100, 3), np.uint8)
            image = renderer.render(clean, 2, "wrist")
            np.testing.assert_array_equal(image[56, 81], [102, 0, 102])
            np.testing.assert_array_equal(image[50, 40], [0, 255, 0])
            np.testing.assert_array_equal(image[50, 48], [0, 0, 0])  # no heatmap halo
            self.assertFalse(clean.any())
            for frame in [0, 1, 4]:
                np.testing.assert_array_equal(renderer.render(clean, frame, "wrist")[56, 81], [0, 0, 0])
            np.testing.assert_array_equal(renderer.render(clean, 1, "wrist")[50, 40], [0, 0, 0])
            np.testing.assert_array_equal(renderer.render(clean, 0, "wrist")[50, 65], [0, 255, 255])
            np.testing.assert_array_equal(renderer.render(clean, 2, "wrist")[50, 65], [0, 0, 0])

    def test_stationary_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = self.renderer(Path(tmp), self.record())
            for frame in range(5):
                image = renderer.render(np.zeros((80, 100, 3), np.uint8), frame, "top")
                np.testing.assert_array_equal(image[50, 40], [0, 255, 0])
                np.testing.assert_array_equal(image[56, 81], [0, 0, 0])

    def test_mask_extension_into_next_subtask(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.record()
            first["end_frame"] = 2
            second = {"subtask": 2, "start_frame": 3, "end_frame": 4,
                      "pov": {"source": "cluster", "gripper_filter": {"status": "ok", "window": [4, 4]}},
                      "third_person": {}}
            renderer = self.renderer(Path(tmp), first)
            renderer.records.append(second)
            image = renderer.render(np.zeros((80, 100, 3), np.uint8), 3, "wrist")
            np.testing.assert_array_equal(image[56, 81], [102, 0, 102])

    def test_third_person_forward_points_and_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record()
            result = record["third_person"]["top"]
            result.update(source="semantic_object", tracks=[[[40+t, 50] for t in range(5)]],
                          transition_from_previous={"tracks": [[[65, 50]]], "visibility": [[True]]})
            renderer = self.renderer(Path(tmp), record)
            clean = np.zeros((80, 100, 3), np.uint8)
            np.testing.assert_array_equal(renderer.render(clean, 0, "top")[50, 65], [0, 255, 255])
            np.testing.assert_array_equal(renderer.render(clean, 1, "top")[50, 65], [0, 0, 0])
            np.testing.assert_array_equal(renderer.render(clean, 4, "top")[50, 44], [0, 255, 0])

    def test_failed_mask_has_no_fabricated_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record()
            record["pov"]["gripper_filter"] = {"status": "failed", "window": [2, 3]}
            renderer = DebugOverlayRenderer([record], tmp)
            image = renderer.render(np.zeros((80, 100, 3), np.uint8), 2, "wrist")
            np.testing.assert_array_equal(image[56, 81], [0, 0, 0])
            self.assertTrue(image[:30].any())

    def test_resume_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            check_output_mode(root, False)
            with self.assertRaises(ValueError):
                check_output_mode(root, True)
            write_output_mode(root, True)
            check_output_mode(root, True)
            with self.assertRaises(ValueError):
                check_output_mode(root, False)
            self.assertEqual(json.loads((root / "meta/grounding_output.json").read_text())["video_content"],
                             "debug_masks_and_points")


if __name__ == "__main__":
    unittest.main()
