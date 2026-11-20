"""
core/physics_advanced.py
────────────────────────
Bridge module re-exporting all physics-based techniques and HUD visualizers.
"""
from physics_advanced import (
    FrenetBoundaryDynamics,
    TrajectoryEntropy,
    BoundaryProbingDynamics,
    PhysicsTracker,
    INTENT_COLORS,
    resolve_intent_with_physics,
    draw_hud_person,
    draw_restricted_zone,
    draw_top_telemetry_bar,
    draw_hud_legend
)

__all__ = [
    'FrenetBoundaryDynamics',
    'TrajectoryEntropy',
    'BoundaryProbingDynamics',
    'PhysicsTracker',
    'INTENT_COLORS',
    'resolve_intent_with_physics',
    'draw_hud_person',
    'draw_restricted_zone',
    'draw_top_telemetry_bar',
    'draw_hud_legend'
]
