from __future__ import annotations

from typing import Mapping

GRID_COLOR = "#E5E7EB"
PANEL_FACE_COLOR = "#F8FAFC"
SPINE_COLOR = "#D1D5DB"

ROBOT_COLOR = "#111827"
HUMAN_COLOR = "#059669"
REFERENCE_COLOR = "#111827"
LEASH_COLOR = "#4B5563"

OBSTACLE_FACE_COLOR = "#B45309"
OBSTACLE_EDGE_COLOR = "#7C2D12"
OBSTACLE_HIT_FACE_COLOR = "#FCA5A5"
OBSTACLE_HIT_EDGE_COLOR = "#DC2626"
WALL_COLOR = "#9CA3AF"
SAFETY_RING_COLOR = "#9CA3AF"
SAFETY_PROJECTION_COLOR = "#111827"
COLLISION_COLOR = "#DC2626"
START_COLOR = "#10B981"
GOAL_COLOR = "#111827"
MARGIN_COLOR = "#16A34A"
WARNING_COLOR = "#DC2626"

RAW_CLOUD_COLOR = "#DC2626"
OBSERVATION_CLOUD_COLOR = "#2563EB"

ALGORITHM_COLORS: Mapping[str, str] = {
    "diffusion": "#B91C1C",
    "raw_diffusion": "#B91C1C",
    "raw_policy": "#B91C1C",
    "diffusion_qp": "#D97706",
    "robot_qp": "#D97706",
    "safe_compliance": "#7C3AED",
    "compliance_no_safety": "#7C3AED",
    "full_time_compliance": "#7C3AED",
    "ours": "#2563EB",
    "compliance_safe": "#2563EB",
    "human_robot_qp": "#2563EB",
    "interaction_aware": "#2563EB",
}

STATE_COLORS: Mapping[str, str] = {
    "guide": "#2563EB",
    "leash": "#EA580C",
    "tether": "#EA580C",
    "unknown": "#9CA3AF",
}

STATE_SPAN_COLORS: Mapping[str, str] = {
    "guide": "#DBEAFE",
    "leash": "#FFEDD5",
    "tether": "#FFEDD5",
    "unknown": "#F3F4F6",
}

DEFAULT_LINE_COLOR = "#6B7280"


def algorithm_color(name: str) -> str:
    return ALGORITHM_COLORS.get(str(name), DEFAULT_LINE_COLOR)


def state_color(name: str) -> str:
    return STATE_COLORS.get(str(name).lower(), DEFAULT_LINE_COLOR)


def state_span_color(name: str) -> str:
    return STATE_SPAN_COLORS.get(str(name).lower(), "#F3F4F6")
