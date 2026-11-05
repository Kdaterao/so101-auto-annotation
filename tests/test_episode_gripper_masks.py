"""Exercise full POV mask sidecar writes without loading GPU models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from robot101.data.utilities.episode_gripper_masks import prepare_episode_masks


class EpisodeMaskSidecarTests(unittest.TestCase):
    def run_episode(self, root, refresh_fps, *, failure=False):
        frames = np.zeros((5, 20, 20, 3), np.uint8)
        molmo = SimpleNamespace(ground=Mock(return_value={'points_xy': [[6, 6]]}))
        def track(clip, points, **kwargs):
            tracks = np.repeat(points[:, None, :], len(clip), axis=1)
            return tracks, np.ones(tracks.shape[:2], bool), None, None
        tapir = SimpleNamespace(track_video=Mock(side_effect=track))
        scheduled = []
        def propagate(clip, real, initialize, refine, validate, **kwargs):
            scheduled.extend(kwargs['refresh_frames'])
            if failure:
                raise RuntimeError('test_mask_failure')
            _, points, labels, _ = initialize(0)
            mask = np.zeros((len(clip), 20, 20), bool)
            mask[:, 5:9, 5:9] = True
            self.assertIsNone(validate(0, mask[0], None, None, points, labels))
            np.testing.assert_array_equal(initialize(4)[1], points)
            return mask, np.ones(len(clip), bool), []
        sam = SimpleNamespace(model_id='fake/sam', propagate_episode=propagate)
        args = SimpleNamespace(sam2_gripper_refresh_fps=refresh_fps, sam2_seed_search_frames=3,
                               sam2_negative_prompts=False, sam2_max_recoveries=5,
                               sam2_recovery_retry_frames=30, sam2_max_area_fraction=.8,
                               sam2_max_growth_ratio=2, sam2_growth_area_fraction=.02)
        result = prepare_episode_masks(frames, np.ones(5, bool), molmo, tapir, sam,
                                       root, 0, {}, args, fps=2)
        self.assertEqual(molmo.ground.call_count, 1)
        self.assertEqual(tapir.track_video.call_count, 1)
        return result, scheduled

    def test_default_and_periodic_reanchor_schedules_are_saved_as_json_lists(self):
        for rate, expected in ((0, []), (1, [0, 2, 4]), (.5, [0, 4])):
            with self.subTest(rate=rate), TemporaryDirectory() as directory:
                root = Path(directory)
                (masks, valid, metadata), scheduled = self.run_episode(root, rate)
                self.assertEqual(scheduled, expected)
                saved = json.loads((root / 'gripper_masks/ep000000_full.json').read_text())
                self.assertEqual(saved['scheduled_tapir_reanchor_frames'], expected)
                self.assertEqual(saved['status'], 'ok')
                self.assertEqual(saved, metadata)
                with np.load(root / saved['mask_file']) as archive:
                    recovered = np.unpackbits(archive['packed_masks'], axis=-1)[..., :20]
                    np.testing.assert_array_equal(recovered, masks)
                    np.testing.assert_array_equal(archive['valid_frames'], valid)

    def test_failed_mask_pass_still_saves_metadata_and_review_record(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (masks, valid, metadata), _ = self.run_episode(root, 0, failure=True)
            self.assertFalse(valid.any())
            self.assertFalse(masks.any())
            self.assertEqual(metadata['status'], 'failed')
            self.assertEqual(metadata['scheduled_tapir_reanchor_frames'], [])
            self.assertTrue((root / 'gripper_masks/ep000000_full.json').exists())
            review = json.loads((root / 'mask_review/ep000000_stage_recovery000000.json').read_text())
            self.assertIn('test_mask_failure', review['reason'])


if __name__ == '__main__':
    unittest.main()
