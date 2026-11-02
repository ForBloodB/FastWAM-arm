import unittest

import numpy as np

from fastwam_b601_adapter import adapt_libero_action_to_b601


class FastWamB601AdapterTests(unittest.TestCase):
    def test_zero_delta_preserves_pose_and_close_intent(self):
        state = [0.30, 0.0, 0.22, 0.0, 0.0, 0.0, 0.0, 0.0]
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        target = adapt_libero_action_to_b601(state, action)

        np.testing.assert_allclose(target["position_m"], state[:3])
        np.testing.assert_allclose(target["axis_angle_rad"], [0.0, 0.0, 0.0])
        self.assertEqual(target["gripper_intent"], "close")

    def test_large_translation_is_scaled_to_executor_limit(self):
        state = [0.30, 0.0, 0.22, 0.0, 0.0, 0.0, 0.0, 0.0]
        action = [-0.302, 0.117, -0.341, 0.0, 0.0, 0.0, 0.734]

        target = adapt_libero_action_to_b601(state, action)

        applied = np.asarray(target["applied_translation_delta_m"])
        self.assertAlmostEqual(float(np.linalg.norm(applied)), 0.04, places=8)
        np.testing.assert_allclose(target["position_m"], np.asarray(state[:3]) + applied)
        self.assertLess(target["translation_scale"], 1.0)
        self.assertEqual(target["gripper_intent"], "open")

    def test_rotation_delta_is_bounded(self):
        state = [0.30, 0.0, 0.22, 0.0, 0.0, 0.0, 0.0, 0.0]
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5]

        target = adapt_libero_action_to_b601(state, action)

        self.assertAlmostEqual(float(np.linalg.norm(target["axis_angle_rad"])), 0.20, places=7)
        self.assertEqual(target["rotation_scale"], 0.5)

    def test_rejects_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            adapt_libero_action_to_b601([0.0] * 8, [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
