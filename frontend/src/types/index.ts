export type BoundingBox = [number, number, number, number]; // [x1, y1, x2, y2]

export interface Detection {
    class: string;
    confidence: number;
    bbox: BoundingBox;
}

export interface DetectionResponse {
    detections: Detection[];
    image_size: [number, number]; // [width, height]
}

// ---------------------------------------------------------------------------
// KITE — live video intelligence
// ---------------------------------------------------------------------------

export type ThreatLevel = "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface KinematicFeatures {
    n_samples: number;
    duration_s: number;
    straightness: number;
    speed_mean: number;
    speed_std: number;
    heading_change_rate: number;
    accel_mag: number;
    jerk_mag: number;
    hover_score: number;
    turn_sharpness: number;
    periodicity_available: boolean;
    vertical_periodicity_hz: number;
    vertical_periodicity_power: number;
    aspect_oscillation: number;
    speed_mean_norm: number;
    speed_std_norm: number;
}

export interface ThreatAssessment {
    score: number;              // 0..100
    level: ThreatLevel;
    zone_inbound: boolean;
    eta_s: number | null;
}

export interface Track {
    id: number;
    class: string;              // fused (corrected) class
    appearance_class: string;
    appearance_conf: number;
    fused_conf: number;
    reason: string;
    flagged: boolean;
    bbox: BoundingBox;
    trail: [number, number][];
    predicted: [number, number][];
    kinematics: KinematicFeatures | null;
    threat: ThreatAssessment | null;
}

export interface TacticalEvent {
    type: string;
    track_id: number;
    severity: ThreatLevel;
    ts: number;
    eta_s?: number | null;
    conf?: number;
}

export interface Zone {
    name: string;
    points: [number, number][];
}

export interface IntelPacket {
    frame_id: number;
    ts: number;
    fps: number;
    infer_ms: number;
    intel_ms: number;
    image_size: [number, number];
    frame_jpeg_b64: string;
    tracks: Track[];
    events: TacticalEvent[];
}

export interface StartStreamResponse {
    session_id: string;
    video_fps: number;
    frame_count: number;
    width: number;
    height: number;
}
