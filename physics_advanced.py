"""
physics_advanced.py
───────────────────
Advanced Physics-Based Behavioral Analytics for Restricted Zone Monitoring.

Implements three core techniques:
1. Boundary Probing Dynamics (BPD):
   Detects repeated approaching, retreating, and tangential casing/movement along the restricted boundary.
2. Trajectory Entropy (TE):
   Measures unpredictability and randomness in a person's movement direction via normalized Shannon entropy.
3. Frenet Boundary Dynamics (FBD):
   Calculates tangential (Vt) and normal (Vn) velocities relative to the restricted boundary polygon,
   identifying whether a person is moving parallel, towards, or away from the boundary.

Designed for real-time CCTV inference with independent multi-track support (15-frame history).
"""

import math
import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


class FrenetBoundaryDynamics:
    """
    Computes local coordinate frames (tangent t and inward normal n) along
    a restricted zone boundary polygon and decomposes velocity into:
      - Tangential velocity (Vt): parallel to the perimeter
      - Normal velocity (Vn): perpendicular (positive = towards zone, negative = away)
    """

    def __init__(self, polygon: np.ndarray):
        """
        Args:
            polygon: Polygon vertices as (N, 2) or (N, 1, 2) numpy array.
        """
        self.pts = polygon.reshape(-1, 2).astype(np.float64)
        self.n_pts = len(self.pts)
        if self.n_pts < 3:
            raise ValueError("Restricted zone polygon must have at least 3 vertices.")

        self._precompute_boundary_geometry()

    def _precompute_boundary_geometry(self):
        """Precomputes segment vectors, lengths, tangents, and inward normals for all edges."""
        self.segments = []
        self.lengths = []
        self.tangents = []
        self.normals = []

        poly_contour = self.pts.reshape((-1, 1, 2)).astype(np.int32)

        for i in range(self.n_pts):
            p1 = self.pts[i]
            p2 = self.pts[(i + 1) % self.n_pts]
            seg = p2 - p1
            length = float(np.linalg.norm(seg))
            if length < 1e-6:
                length = 1e-6
                tangent = np.array([1.0, 0.0], dtype=np.float64)
            else:
                tangent = seg / length

            # Normal candidates: (-ty, tx) or (ty, -tx)
            n_cand = np.array([-tangent[1], tangent[0]], dtype=np.float64)

            # Test a small step from segment midpoint into candidate normal direction
            midpoint = (p1 + p2) * 0.5
            test_pt = midpoint + n_cand * 2.0
            test_res = cv2.pointPolygonTest(poly_contour, (float(test_pt[0]), float(test_pt[1])), False)

            # If inside polygon, n_cand is inward normal; otherwise reverse it
            if test_res >= 0:
                inward_normal = n_cand
            else:
                inward_normal = -n_cand

            self.segments.append((p1, p2))
            self.lengths.append(length)
            self.tangents.append(tangent)
            self.normals.append(inward_normal)

        self.poly_contour = poly_contour

    def get_closest_boundary_point(self, pt: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """
        Finds the closest point on the polygon boundary to the query point.

        Returns:
            (closest_pt, tangent, inward_normal, min_dist, segment_idx)
        """
        px, py = float(pt[0]), float(pt[1])
        p = np.array([px, py], dtype=np.float64)

        min_dist = float('inf')
        best_q = p.copy()
        best_tangent = np.array([1.0, 0.0])
        best_normal = np.array([0.0, 1.0])
        best_idx = 0

        for i in range(self.n_pts):
            p1, p2 = self.segments[i]
            seg = p2 - p1
            L2 = self.lengths[i] ** 2
            if L2 < 1e-6:
                t = 0.0
            else:
                t = np.clip(np.dot(p - p1, seg) / L2, 0.0, 1.0)

            q = p1 + t * seg
            dist = float(np.linalg.norm(p - q))
            if dist < min_dist:
                min_dist = dist
                best_q = q
                best_tangent = self.tangents[i]
                best_normal = self.normals[i]
                best_idx = i

        return best_q, best_tangent, best_normal, min_dist, best_idx

    def decompose_velocity(
        self,
        foot_pt: Tuple[float, float],
        velocity_vector: Tuple[float, float]
    ) -> Tuple[float, float, str, np.ndarray]:
        """
        Decomposes 2D velocity (vx, vy) into tangential (Vt) and normal (Vn) components.

        Returns:
            (vt, vn, action_vector, closest_pt)
            - vt: Tangential velocity (along boundary)
            - vn: Normal velocity (> 0 = approaching zone, < 0 = retreating from zone)
            - action_vector: 'Closing In' | 'Moving Away' | 'Parallel' | 'Neutral'
            - closest_pt: closest boundary point for visual HUD linkage
        """
        vx, vy = float(velocity_vector[0]), float(velocity_vector[1])
        v = np.array([vx, vy], dtype=np.float64)
        speed = float(np.linalg.norm(v))

        closest_q, tangent, normal, dist, _ = self.get_closest_boundary_point(foot_pt)

        vt = float(np.dot(v, tangent))
        vn = float(np.dot(v, normal))

        # Dynamic behavioral categorization
        if speed < 6.0:
            action = 'Neutral'
        elif vn > 10.0 and vn > 0.45 * speed:
            action = 'Closing In'
        elif vn < -10.0 and abs(vn) > 0.45 * speed:
            action = 'Moving Away'
        else:
            action = 'Parallel'

        return vt, vn, action, closest_q


class TrajectoryEntropy:
    """
    Computes trajectory unpredictability / directional entropy using normalized Shannon entropy.
    Smooth straight-line transit yields low entropy (~0.0).
    Erratic pacing, wandering, and back-and-forth scouting yield high entropy (~0.7 - 1.0).
    """

    def __init__(self, num_bins: int = 8, min_displacement: float = 1.2):
        self.num_bins = num_bins
        self.min_displacement = min_displacement
        self.max_entropy = math.log2(self.num_bins) if self.num_bins > 1 else 1.0

    def compute(self, position_history: List[Tuple[float, float]]) -> float:
        """
        Computes normalized directional Shannon entropy over recent tracking positions.
        """
        if len(position_history) < 3:
            return 0.0

        # Extract step vectors and headings
        headings = []
        for i in range(1, len(position_history)):
            p0 = position_history[i - 1]
            p1 = position_history[i]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dist = math.hypot(dx, dy)
            if dist >= self.min_displacement:
                angle = math.atan2(dy, dx)  # -pi to +pi
                if angle < 0:
                    angle += 2 * math.pi
                headings.append(angle)

        if len(headings) < 2:
            return 0.0

        # Bin headings into discrete sectors
        bin_size = (2 * math.pi) / self.num_bins
        counts = [0] * self.num_bins
        for h in headings:
            idx = int(h / bin_size) % self.num_bins
            counts[idx] += 1

        total = len(headings)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)

        norm_entropy = float(np.clip(entropy / self.max_entropy, 0.0, 1.0))
        return round(norm_entropy, 3)


class BoundaryProbingDynamics:
    """
    Quantifies perimeter probing behavior:
    1. Repeated approach-retreat oscillations near the restricted boundary.
    2. Tangential perimeter coverage (lateral walking/casing along the fence).
    3. Dwell/proximity persistence within the probing critical threshold.
    """

    def __init__(
        self,
        proximity_threshold: float = 130.0,
        cycle_retreat_threshold: float = -7.0,
        cycle_approach_threshold: float = 7.0
    ):
        self.proximity_threshold = proximity_threshold
        self.cycle_retreat_threshold = cycle_retreat_threshold
        self.cycle_approach_threshold = cycle_approach_threshold

    def evaluate_probing(
        self,
        history: deque,
        current_dist: float,
        current_vn: float,
        current_vt: float
    ) -> Tuple[float, int, bool, bool]:
        """
        Evaluates BPD metrics for a track based on its history and current dynamics.

        Returns:
            (bpd_score, cycle_count, repeated_probing, is_probing)
        """
        if len(history) < 4:
            return 0.0, 0, False, False

        # Count approach-retreat cycles and time near boundary
        near_count = 0
        lateral_distance = 0.0
        vn_history = []
        prev_pos = None

        for item in history:
            d = item.get('distance', 999.0)
            if d < self.proximity_threshold:
                near_count += 1

            vn = item.get('vn', 0.0)
            vn_history.append(vn)

            pos = item.get('position')
            if prev_pos is not None and pos is not None and d < self.proximity_threshold:
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                lateral_distance += math.hypot(dx, dy)
            prev_pos = pos

        # Detect direction reversals in normal velocity while in proximity zone
        # A cycle occurs when Vn flips from significantly positive (approach) to negative (retreat)
        cycle_count = 0
        state = 'neutral'
        for vn in vn_history:
            if vn > self.cycle_approach_threshold:
                if state == 'retreating':
                    cycle_count += 1
                state = 'approaching'
            elif vn < self.cycle_retreat_threshold:
                if state == 'approaching':
                    cycle_count += 1
                state = 'retreating'

        # Proximity weight: fraction of recent frames spent near boundary
        proximity_factor = min(1.0, near_count / min(len(history), 15))

        # Cycle score: 1 cycle = 0.5, 2+ cycles = 1.0
        cycle_factor = min(1.0, cycle_count / 2.0)

        # Flanking/casing factor: moving along perimeter while close
        flanking_factor = min(1.0, lateral_distance / 120.0)

        # Base BPD score calculation
        if near_count == 0:
            bpd_score = 0.0
        else:
            bpd_score = (
                0.45 * cycle_factor +
                0.35 * proximity_factor +
                0.20 * flanking_factor
            )

        # Repeated boundary interaction triggers
        repeated_probing = (cycle_count >= 2) or (cycle_count >= 1 and proximity_factor > 0.6)
        is_probing = repeated_probing or (bpd_score >= 0.40 and current_dist < self.proximity_threshold)

        return round(float(np.clip(bpd_score, 0.0, 1.0)), 3), cycle_count, repeated_probing, is_probing


class PhysicsTracker:
    """
    Unified manager for Boundary Probing Dynamics, Trajectory Entropy, and
    Frenet Boundary Dynamics across multiple simultaneous tracks.
    Retains a 15-frame rolling tracking history per track.
    """

    def __init__(self, polygon: np.ndarray, history_len: int = 15, fps: float = 30.0):
        self.fps = fps
        self.history_len = history_len
        self.fbd = FrenetBoundaryDynamics(polygon)
        self.te = TrajectoryEntropy(num_bins=8, min_displacement=1.2)
        self.bpd = BoundaryProbingDynamics()

        # track_id -> deque of dicts
        self.tracks_history: Dict[int, deque] = {}
        # track_id -> persistent probing stats
        self.probing_cumulative_cycles: Dict[int, int] = {}

    def update_track(
        self,
        track_id: int,
        foot_pt: Tuple[int, int],
        dist_to_zone: float,
        is_inside: bool,
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Updates physics calculations for a single person at current frame.

        Returns comprehensive dictionary of metrics:
          - speed
          - tangential_velocity (Vt)
          - normal_velocity (Vn)
          - action_vector ('Closing In', 'Moving Away', 'Parallel', 'Neutral')
          - trajectory_entropy (TE)
          - bpd_score (BPD)
          - approach_retreat_cycles
          - repeated_probing (visual highlight trigger)
          - is_probing
          - closest_boundary_pt
          - velocity_arrow: (start_pt, end_pt)
          - trail_pts
        """
        if track_id not in self.tracks_history:
            self.tracks_history[track_id] = deque(maxlen=self.history_len)
            self.probing_cumulative_cycles[track_id] = 0

        history = self.tracks_history[track_id]

        # Calculate instantaneous velocity vector
        vx, vy, speed = 0.0, 0.0, 0.0
        if len(history) > 0:
            prev = history[-1]
            dt_frames = max(1, frame_idx - prev['frame'])
            dt_seconds = dt_frames / self.fps
            dx = foot_pt[0] - prev['position'][0]
            dy = foot_pt[1] - prev['position'][1]
            vx = dx / dt_seconds
            vy = dy / dt_seconds
            speed = math.hypot(dx, dy) / dt_seconds

        # 1. Frenet Boundary Dynamics
        vt, vn, action_vector, closest_q = self.fbd.decompose_velocity(foot_pt, (vx, vy))

        # 2. Trajectory Entropy (over last up to 15 frames)
        positions = [item['position'] for item in history] + [foot_pt]
        entropy = self.te.compute(positions)

        # 3. Boundary Probing Dynamics
        bpd_score, cycle_count, repeated_probing, is_probing = self.bpd.evaluate_probing(
            history, dist_to_zone, vn, vt
        )
        if cycle_count > self.probing_cumulative_cycles[track_id]:
            self.probing_cumulative_cycles[track_id] = cycle_count
        else:
            cycle_count = self.probing_cumulative_cycles[track_id]

        if cycle_count >= 2:
            repeated_probing = True
            is_probing = True

        # Append current frame to rolling history
        record = {
            'frame': frame_idx,
            'position': foot_pt,
            'distance': dist_to_zone,
            'is_inside': is_inside,
            'speed': speed,
            'vx': vx,
            'vy': vy,
            'vt': vt,
            'vn': vn,
        }
        history.append(record)

        # Arrow end point for velocity visualization
        arrow_scale = 0.4
        arrow_end = (
            int(foot_pt[0] + np.clip(vx * arrow_scale, -80, 80)),
            int(foot_pt[1] + np.clip(vy * arrow_scale, -80, 80))
        )

        return {
            'track_id': track_id,
            'foot_pt': foot_pt,
            'speed': speed,
            'dist_to_zone': dist_to_zone,
            'is_inside': is_inside,
            'tangential_velocity': vt,
            'normal_velocity': vn,
            'action_vector': action_vector,
            'trajectory_entropy': entropy,
            'bpd_score': bpd_score,
            'approach_retreat_cycles': cycle_count,
            'repeated_probing': repeated_probing,
            'is_probing': is_probing,
            'closest_boundary_pt': (int(closest_q[0]), int(closest_q[1])),
            'velocity_arrow': (foot_pt, arrow_end),
            'trail_pts': [h['position'] for h in history]
        }

    def remove_stale_tracks(self, active_track_ids: List[int]):
        """Clean up tracks that have left the scene."""
        stale = [tid for tid in self.tracks_history if tid not in active_track_ids]
        for tid in stale:
            del self.tracks_history[tid]
            if tid in self.probing_cumulative_cycles:
                del self.probing_cumulative_cycles[tid]


# ─── Intent Color Mapping (BGR format) ─────────────────────────────────────────
INTENT_COLORS = {
    'Casual Transit':       (60,  210, 60),    # Green
    'Approaching Boundary': (0,   230, 255),   # Yellow
    'Active Scouting':      (0,   140, 255),   # Orange
    'Suspicious Loitering': (0,   140, 255),   # Orange
    'Fleeing':              (210, 50,  210),   # Purple
    'CRITICAL INTRUSION':   (30,  30,  255),   # Red
    'Critical Intrusion':   (30,  30,  255),   # Red
}


def resolve_intent_with_physics(
    base_class: str,
    is_inside: bool,
    physics: Dict[str, Any],
    risk_score: float
) -> str:
    """
    Integrates BPD, TE, and FBD metrics into behavioural decision logic as supplementary evidence.
    Preserves all 6 existing intent classes and the Critical Intrusion override.
    """
    # 1. CRITICAL INTRUSION OVERRIDE (Inviolable top priority)
    if is_inside or physics.get('is_inside', False):
        return 'CRITICAL INTRUSION'

    speed = physics.get('speed', 0.0)
    dist = physics.get('dist_to_zone', 999.0)
    vt = physics.get('tangential_velocity', 0.0)
    vn = physics.get('normal_velocity', 0.0)
    te = physics.get('trajectory_entropy', 0.0)
    bpd_score = physics.get('bpd_score', 0.0)
    repeated_probing = physics.get('repeated_probing', False)
    action_vector = physics.get('action_vector', 'Neutral')

    base = base_class.lower() if base_class else ''

    # 2. FLEEING: Rapid retreat away from boundary
    if (vn < -35.0 and speed > 70.0) or (action_vector == 'Moving Away' and speed > 90.0) or ('flee' in base):
        return 'Fleeing'

    # 3. ACTIVE SCOUTING: Repeated approach-retreat (probing), high BPD, or lateral perimeter movement
    if repeated_probing or bpd_score >= 0.45 or ('scout' in base) or (dist < 85.0 and abs(vt) > 18.0 and abs(vn) < 22.0):
        return 'Active Scouting'

    # 4. APPROACHING BOUNDARY: High inward normal velocity towards boundary or close closing-in
    if (vn > 15.0 and dist < 120.0) or (action_vector == 'Closing In' and dist < 70.0) or ('approach' in base) or ('active_intrusion' in base):
        return 'Approaching Boundary'

    # 5. SUSPICIOUS LOITERING: High trajectory entropy (wandering/pacing) or stationary lingering near boundary
    if (te > 0.60 and speed < 16.0 and dist < 150.0) or (speed < 8.0 and dist < 120.0 and action_vector == 'Neutral') or ('loiter' in base):
        return 'Suspicious Loitering'

    # 6. CASUAL TRANSIT (Default baseline flow)
    return 'Casual Transit'


# ─── Live Video HUD Drawing Functions ─────────────────────────────────────────

def draw_corner_rect(frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, corner_len: int = 15):
    """Draws a tactical bounding box with reinforced corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cl = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, thickness + 2)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, thickness + 2)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, thickness + 2)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, thickness + 2)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, thickness + 2)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, thickness + 2)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, thickness + 2)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, thickness + 2)


def draw_hud_person(
    frame: np.ndarray,
    track_id: int,
    bbox: Tuple[int, int, int, int],
    intent: str,
    risk_score: float,
    physics: Dict[str, Any],
    pulse_phase: float = 0.0
):
    """
    Renders live HUD elements for one tracked person:
      - Color-changing bounding box based on 6 intent classes
      - Track ID, predicted intent, and risk score
      - BPD score, trajectory entropy, tangential velocity, normal velocity
      - Movement direction arrow
      - Trajectory comet trail
      - Visual highlight for repeated boundary interaction (probing tether & banner)
    """
    x1, y1, x2, y2 = bbox
    color = INTENT_COLORS.get(intent, (200, 200, 200))
    foot = physics.get('foot_pt', ((x1 + x2) // 2, y2))
    repeated_probing = physics.get('repeated_probing', False)

    # 1. Trajectory Comet Trail (15-frame history)
    trail = physics.get('trail_pts', [])
    if len(trail) > 1:
        pts = np.array(trail, np.int32)
        n = len(pts)
        for i in range(1, n):
            alpha = (i / n)
            line_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), line_color, 2, cv2.LINE_AA)

    # 2. Repeated Boundary Interaction Highlight
    if repeated_probing:
        # Pulsing outer alert border
        expand = int(4 + 3 * math.sin(pulse_phase * 6.0))
        cv2.rectangle(frame, (x1 - expand, y1 - expand), (x2 + expand, y2 + expand), (0, 165, 255), 2)

        # Tether line from foot to closest boundary contact point
        closest_q = physics.get('closest_boundary_pt')
        if closest_q:
            cv2.line(frame, foot, closest_q, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, closest_q, 6, (0, 165, 255), -1)
            cv2.circle(frame, closest_q, 10, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "PROBE CONTACT", (closest_q[0] + 8, closest_q[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

    # 3. Main Bounding Box
    draw_corner_rect(frame, (x1, y1), (x2, y2), color, thickness=2)

    # 4. Movement Direction Arrow
    arrow = physics.get('velocity_arrow')
    if arrow:
        start_pt, end_pt = arrow
        if math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]) > 5:
            # Drop shadow
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 0), 3, tipLength=0.35)
            cv2.arrowedLine(frame, start_pt, end_pt, color, 2, tipLength=0.35)

    # 5. Multi-field Telemetry HUD Card
    bpd_score = physics.get('bpd_score', 0.0)
    cycles = physics.get('approach_retreat_cycles', 0)
    te = physics.get('trajectory_entropy', 0.0)
    vt = physics.get('tangential_velocity', 0.0)
    vn = physics.get('normal_velocity', 0.0)
    action = physics.get('action_vector', 'Neutral')

    # Card dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    card_w = 230
    card_h = 70
    card_x = max(10, min(x1, frame.shape[1] - card_w - 10))
    card_y = y1 - card_h - 10
    if card_y < 50:  # If too close to top bar, flip card below bbox
        card_y = y2 + 10

    # Glassmorphic semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 25), -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)
    # Header badge
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + 20), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Header text: ID | Intent | Risk
    header_text = f"ID:{track_id} {intent}"
    risk_text = f"{risk_score:.0f}%"
    text_color = (0, 0, 0) if (color[0] + color[1] + color[2] > 380) else (255, 255, 255)
    cv2.putText(frame, header_text, (card_x + 5, card_y + 14), font, 0.42, text_color, 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(risk_text, font, 0.42, 1)
    cv2.putText(frame, risk_text, (card_x + card_w - rw - 5, card_y + 14), font, 0.42, text_color, 1, cv2.LINE_AA)

    # Row 1: BPD and TE
    r1_text = f"BPD:{bpd_score:.2f} (Cyc:{cycles})  TE:{te:.2f}"
    cv2.putText(frame, r1_text, (card_x + 6, card_y + 36), font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # Row 2: Vt and Vn
    r2_text = f"Vt:{vt:+.1f}  Vn:{vn:+.1f} [{action}]"
    cv2.putText(frame, r2_text, (card_x + 6, card_y + 54), font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # If repeated probing active, render small animated alert banner
    if repeated_probing:
        alert_w = 170
        alert_x = card_x + (card_w - alert_w) // 2
        alert_y = card_y - 18
        cv2.rectangle(frame, (alert_x, alert_y), (alert_x + alert_w, alert_y + 16), (0, 140, 255), -1)
        cv2.putText(frame, "! PROBING DETECTED !", (alert_x + 8, alert_y + 12), font, 0.38, (0, 0, 0), 1, cv2.LINE_AA)


def draw_restricted_zone(frame: np.ndarray, polygon_pts: np.ndarray, opacity: float = 0.25):
    """Renders semi-transparent restricted zone polygon and boundary border."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_pts], (0, 0, 180))
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)
    # Bright boundary perimeter
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # Label at polygon centroid
    moments = cv2.moments(polygon_pts)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        badge = "RESTRICTED ZONE (ACTIVE PERIMETER)"
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (bw, bh), _ = cv2.getTextSize(badge, font, fs, th)
        bx1 = cx - bw // 2 - 8
        by1 = cy - bh // 2 - 6
        bx2 = cx + bw // 2 + 8
        by2 = cy + bh // 2 + 6
        sub = frame[by1:by2, bx1:bx2]
        if sub.shape[0] > 0 and sub.shape[1] > 0:
            dark_bg = np.zeros_like(sub)
            cv2.addWeighted(dark_bg, 0.7, sub, 0.3, 0, sub)
            frame[by1:by2, bx1:bx2] = sub
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 1)
        cv2.putText(frame, badge, (bx1 + 8, cy + bh // 2 - 2), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_top_telemetry_bar(frame: np.ndarray, fps: float, frame_idx: int, counts_by_intent: Dict[str, int]):
    """Renders the top command & telemetry HUD status bar."""
    h, w = frame.shape[:2]
    bar_h = 42

    overlay = frame[:bar_h, :w].copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 12, 16), -1)
    cv2.addWeighted(overlay, 0.88, frame[:bar_h, :w], 0.12, 0, frame[:bar_h, :w])
    cv2.line(frame, (0, bar_h), (w, bar_h), (50, 50, 60), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Left: Title
    cv2.putText(frame, "RESTRICTED ZONE PERIMETER HUD", (15, 25), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "| BPD + TE + FBD", (295, 25), font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Middle: Performance metrics
    perf_str = f"FPS: {fps:.1f} | Frame: {frame_idx}"
    cv2.putText(frame, perf_str, (460, 25), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Right: Threat counts
    cx = w - 460
    badges = [
        ('Casual', counts_by_intent.get('Casual Transit', 0), (60, 210, 60)),
        ('Approach', counts_by_intent.get('Approaching Boundary', 0), (0, 230, 255)),
        ('Scout/Loiter', counts_by_intent.get('Active Scouting', 0) + counts_by_intent.get('Suspicious Loitering', 0), (0, 140, 255)),
        ('Flee', counts_by_intent.get('Fleeing', 0), (210, 50, 210)),
        ('INTRUSION', counts_by_intent.get('CRITICAL INTRUSION', 0), (30, 30, 255)),
    ]
    for label, count, col in badges:
        text = f"{label}:{count}"
        cv2.rectangle(frame, (cx - 2, 10), (cx + 80, 32), (30, 30, 35), -1)
        cv2.rectangle(frame, (cx - 2, 10), (cx + 80, 32), col, 1)
        cv2.putText(frame, text, (cx + 3, 25), font, 0.38, col, 1, cv2.LINE_AA)
        cx += 90


def draw_hud_legend(frame: np.ndarray):
    """Renders the 6 intent color classes legend in the bottom right corner."""
    h, w = frame.shape[:2]
    legend_w = 460
    legend_h = 32
    lx = w - legend_w - 15
    ly = h - legend_h - 15

    overlay = frame[ly:ly + legend_h, lx:lx + legend_w].copy()
    cv2.rectangle(overlay, (0, 0), (legend_w, legend_h), (12, 12, 16), -1)
    cv2.addWeighted(overlay, 0.85, frame[ly:ly + legend_h, lx:lx + legend_w], 0.15, 0, frame[ly:ly + legend_h, lx:lx + legend_w])
    cv2.rectangle(frame, (lx, ly), (lx + legend_w, ly + legend_h), (50, 50, 60), 1)

    items = [
        ("Casual", (60, 210, 60)),
        ("Approach", (0, 230, 255)),
        ("Scout", (0, 140, 255)),
        ("Loiter", (0, 140, 255)),
        ("Flee", (210, 50, 210)),
        ("Critical", (30, 30, 255)),
    ]
    cur_x = lx + 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, col in items:
        cv2.rectangle(frame, (cur_x, ly + 9), (cur_x + 14, ly + 23), col, -1)
        cv2.putText(frame, name, (cur_x + 18, ly + 21), font, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        cur_x += 74

