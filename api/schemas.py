"""KITE Phase 3 — Pydantic schemas for the streaming API boundary.

Inbound user data (zones, session params) is validated here. Outbound
per-frame packets are built as plain dicts in streaming.py — validating
thousands of frames per session buys nothing and costs latency.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Zone(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=64)

    @field_validator("points")
    @classmethod
    def finite_coords(cls, v):
        for x, y in v:
            if not (-1e6 < x < 1e6 and -1e6 < y < 1e6):
                raise ValueError("zone coordinates out of range")
        return v


class ZonesUpdate(BaseModel):
    zones: list[Zone] = Field(max_length=16)


class StartStreamResponse(BaseModel):
    session_id: str
    video_fps: float
    frame_count: int
    width: int
    height: int


class SessionStatus(BaseModel):
    session_id: str
    running: bool
    frames_processed: int
