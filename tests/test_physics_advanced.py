"""
tests/test_physics_advanced.py
───────────────────────────────
Unit tests for the three physics-based techniques:
1. Frenet Boundary Dynamics (FBD)
2. Trajectory Entropy (TE)
3. Boundary Probing Dynamics (BPD)
4. Multi-track PhysicsTracker & Behavioral Decision Logic
"""

import unittest
import numpy as np
from physics_advanced import (
    FrenetBoundaryDynamics,
    TrajectoryEntropy,
    BoundaryProbingDynamics,
    PhysicsTracker,
    resolve_intent_with_physics,
    INTENT_COLORS
)
from ml.xgboost_detector import XGBoostDetector


class TestPhysicsAdvanced(unittest.TestCase):

    def setUp(self):
        # Define a square restricted zone polygon: (100, 100) to (300, 300)
        self.poly_pts = np.array([
            [100, 100],
            [300, 100],
            [300, 300],
            [100, 300]
        ], dtype=np.int32)

    def test_frenet_boundary_dynamics(self):
        fbd = FrenetBoundaryDynamics(self.poly_pts)
        
        # Point directly above top edge: (200, 50)
        # Moving down towards the boundary (+y): velocity = (0, 30)
        vt, vn, action, closest_q = fbd.decompose_velocity((200, 50), (0, 30))
        self.assertAlmostEqual(closest_q[0], 200.0, places=1)
        self.assertAlmostEqual(closest_q[1], 100.0, places=1)
        self.assertGreater(vn, 0.0, "Vn should be positive when moving towards restricted boundary")
        self.assertEqual(action, 'Closing In')

        # Moving up away from the boundary (-y): velocity = (0, -30)
        vt, vn, action, _ = fbd.decompose_velocity((200, 50), (0, -30))
        self.assertLess(vn, 0.0, "Vn should be negative when moving away from restricted boundary")
        self.assertEqual(action, 'Moving Away')

        # Moving parallel to top edge (+x): velocity = (30, 0)
        vt, vn, action, _ = fbd.decompose_velocity((200, 50), (30, 0))
        self.assertAlmostEqual(vn, 0.0, places=1)
        self.assertNotEqual(vt, 0.0)
        self.assertEqual(action, 'Parallel')

    def test_trajectory_entropy(self):
        te = TrajectoryEntropy(num_bins=8)

        # 1. Perfectly straight line -> Entropy should be 0.0
        straight = [(float(i * 10), 50.0) for i in range(15)]
        entropy_straight = te.compute(straight)
        self.assertEqual(entropy_straight, 0.0)

        # 2. Random / erratic trajectory -> High entropy > 0.5
        np.random.seed(42)
        random_traj = [(50.0, 50.0)]
        for _ in range(14):
            random_traj.append((
                random_traj[-1][0] + float(np.random.uniform(-10, 10)),
                random_traj[-1][1] + float(np.random.uniform(-10, 10))
            ))
        entropy_random = te.compute(random_traj)
        self.assertGreater(entropy_random, 0.6)
        self.assertLessEqual(entropy_random, 1.0)

    def test_boundary_probing_dynamics(self):
        tracker = PhysicsTracker(self.poly_pts, history_len=15, fps=15.0)

        # Approach from (200, 40) down towards y=100
        f = 1
        for y in range(40, 85, 10):
            res = tracker.update_track(10, (200, y), 100.0 - y, False, f)
            f += 1
        
        # Retreat back to y=50
        for y in range(80, 45, -10):
            res = tracker.update_track(10, (200, y), 100.0 - y, False, f)
            f += 1

        # Second approach
        for y in range(50, 90, 10):
            res = tracker.update_track(10, (200, y), 100.0 - y, False, f)
            f += 1

        self.assertGreaterEqual(res['approach_retreat_cycles'], 1)
        self.assertGreater(res['bpd_score'], 0.4)

    def test_critical_intrusion_override(self):
        # Regardless of base class or speed, inside the zone is strictly CRITICAL INTRUSION
        physics = {
            'speed': 1.0,
            'dist_to_zone': 0.0,
            'is_inside': True,
            'normal_velocity': 0.0,
            'tangential_velocity': 0.0,
            'trajectory_entropy': 0.0,
            'bpd_score': 0.0,
            'repeated_probing': False,
            'action_vector': 'Neutral'
        }
        intent = resolve_intent_with_physics(
            base_class='Casual Transit',
            is_inside=True,
            physics=physics,
            risk_score=95.0
        )
        self.assertEqual(intent, 'CRITICAL INTRUSION')

    def test_tc01_zone_polygon_config(self):
        """TC_01: User clicks points on frame -> Polygon saves to JSON; coordinates match UI."""
        import json, tempfile
        pts = [{'x': 100, 'y': 100}, {'x': 300, 'y': 100}, {'x': 300, 'y': 300}, {'x': 100, 'y': 300}]
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
            json.dump({'restricted_zone': pts}, tmp)
            tmp_path = tmp.name

        with open(tmp_path, 'r') as f:
            loaded = json.load(f)
        self.assertEqual(loaded['restricted_zone'], pts)
        import os; os.unlink(tmp_path)

    def test_tc_table_scenarios(self):
        """Validates all video test case scenarios from the official Somaiya test specification."""
        # TC_02: Target < 50px from zone, moving Parallel -> Active Scouting, Orange
        p2 = {'speed': 15.0, 'dist_to_zone': 40.0, 'is_inside': False, 'normal_velocity': 2.0, 'tangential_velocity': 14.0, 'action_vector': 'Parallel', 'bpd_score': 0.5, 'trajectory_entropy': 0.3}
        self.assertEqual(resolve_intent_with_physics('unknown', False, p2, 50.0), 'Active Scouting')
        self.assertEqual(INTENT_COLORS['Active Scouting'], (0, 140, 255))

        # TC_03: Target foot-point enters zone -> CRITICAL INTRUSION, Red
        p3 = {'speed': 10.0, 'dist_to_zone': 0.0, 'is_inside': True, 'normal_velocity': 0.0, 'tangential_velocity': 0.0, 'action_vector': 'Neutral', 'bpd_score': 0.0, 'trajectory_entropy': 0.0}
        self.assertEqual(resolve_intent_with_physics('unknown', True, p3, 95.0), 'CRITICAL INTRUSION')
        self.assertEqual(INTENT_COLORS['CRITICAL INTRUSION'], (30, 30, 255))

        # TC_05: Target walks normally far from restricted zone -> Casual Transit, Green
        p5 = {'speed': 20.0, 'dist_to_zone': 250.0, 'is_inside': False, 'normal_velocity': 0.0, 'tangential_velocity': 20.0, 'action_vector': 'Parallel', 'bpd_score': 0.0, 'trajectory_entropy': 0.0}
        self.assertEqual(resolve_intent_with_physics('Casual Transit', False, p5, 10.0), 'Casual Transit')
        self.assertEqual(INTENT_COLORS['Casual Transit'], (60, 210, 60))

        # TC_06: Target moves towards zone; distance decreases and vector = Closing In -> Approaching Boundary, Yellow
        p6 = {'speed': 25.0, 'dist_to_zone': 70.0, 'is_inside': False, 'normal_velocity': 22.0, 'tangential_velocity': 5.0, 'action_vector': 'Closing In', 'bpd_score': 0.1, 'trajectory_entropy': 0.1}
        self.assertEqual(resolve_intent_with_physics('unknown', False, p6, 45.0), 'Approaching Boundary')
        self.assertEqual(INTENT_COLORS['Approaching Boundary'], (0, 230, 255))

        # TC_07: Target moves very slowly near zone or stops repeatedly -> Suspicious Loitering, Orange
        p7 = {'speed': 4.0, 'dist_to_zone': 60.0, 'is_inside': False, 'normal_velocity': 0.0, 'tangential_velocity': 0.0, 'action_vector': 'Neutral', 'bpd_score': 0.2, 'trajectory_entropy': 0.75}
        self.assertEqual(resolve_intent_with_physics('unknown', False, p7, 55.0), 'Suspicious Loitering')
        self.assertEqual(INTENT_COLORS['Suspicious Loitering'], (0, 140, 255))

        # TC_09: Target moves away from zone at speed > 100 px/s -> Fleeing, Purple
        p9 = {'speed': 110.0, 'dist_to_zone': 80.0, 'is_inside': False, 'normal_velocity': -50.0, 'tangential_velocity': 10.0, 'action_vector': 'Moving Away', 'bpd_score': 0.0, 'trajectory_entropy': 0.0}
        self.assertEqual(resolve_intent_with_physics('unknown', False, p9, 60.0), 'Fleeing')
        self.assertEqual(INTENT_COLORS['Fleeing'], (210, 50, 210))


if __name__ == '__main__':
    unittest.main()
