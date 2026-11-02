import unittest

from common.safety import JOINTS, JointLimit, SafetyError, interpolate, validate_trajectory


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.limits = {name: JointLimit(-1, 1, 0.2, 0.05) for name in JOINTS}
        self.now = 1_000_000_000
        self.kw = dict(
            sequence=1, previous_sequence=0, issued_unix_ns=self.now,
            deadline_unix_ns=self.now + 200_000_000, duration_ms=80,
            positions={"shoulder_pan": 0.02}, velocities={"shoulder_pan": 0.1},
            current={"shoulder_pan": 0.0}, limits=self.limits,
            allowed_joints={"shoulder_pan"}, now_unix_ns=self.now,
        )

    def test_accepts_small_fresh_target(self):
        validate_trajectory(**{
            **self.kw,
            "positions": {"shoulder_pan": 0.005},
        })

    def test_rejects_expired_command(self):
        with self.assertRaises(SafetyError):
            validate_trajectory(**{**self.kw, "deadline_unix_ns": self.now - 1})

    def test_rejects_replayed_sequence(self):
        with self.assertRaises(SafetyError):
            validate_trajectory(**{**self.kw, "previous_sequence": 1})

    def test_rejects_unapproved_joint(self):
        with self.assertRaises(SafetyError):
            validate_trajectory(**{**self.kw, "allowed_joints": set()})

    def test_rejects_step_and_speed_limits(self):
        with self.assertRaises(SafetyError):
            validate_trajectory(**{**self.kw, "positions": {"shoulder_pan": 0.2}})
        with self.assertRaises(SafetyError):
            validate_trajectory(**{**self.kw, "velocities": {"shoulder_pan": 1.0}})

    def test_rejects_duration_that_would_exceed_velocity_limit(self):
        with self.assertRaises(SafetyError):
            validate_trajectory(**{
                **self.kw,
                "positions": {"shoulder_pan": 0.02},
                "velocities": {"shoulder_pan": 0.1},
                "duration_ms": 100,
            })

    def test_interpolation_is_bounded(self):
        self.assertAlmostEqual(interpolate(0, 1, 0.04, 0.08), 0.5)
        self.assertEqual(interpolate(0, 1, 1, 0.08), 1)


if __name__ == "__main__":
    unittest.main()
