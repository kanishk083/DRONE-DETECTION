"use client";

import React, { useCallback, useEffect, useRef } from "react";
import { IntelPacket, ThreatLevel, Track, Zone } from "@/types";

const THREAT_COLORS: Record<ThreatLevel, string> = {
    NONE: "#9ca3af",
    LOW: "#22d3ee",
    MEDIUM: "#f59e0b",
    HIGH: "#f97316",
    CRITICAL: "#ef4444",
};
const BIRD_COLOR = "#87ceeb";

interface LiveDetectionCanvasProps {
    packet: IntelPacket;
    zones: Zone[];
    /** points of the zone currently being drawn (image space) */
    draftZone: [number, number][];
    drawingZone: boolean;
    selectedTrackId: number | null;
    onCanvasClick: (x: number, y: number, hitTrackId: number | null) => void;
}

function trackColor(t: Track): string {
    if (t.class === "bird") return BIRD_COLOR;
    return THREAT_COLORS[t.threat?.level ?? "NONE"];
}

export function LiveDetectionCanvas({
    packet,
    zones,
    draftZone,
    drawingZone,
    selectedTrackId,
    onCanvasClick,
}: LiveDetectionCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement | null>(null);

    // scale: overlay coords are in original video space, canvas gets the
    // (possibly downscaled) streamed JPEG size
    const render = useCallback(() => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const sx = canvas.width / packet.image_size[0];
        const sy = canvas.height / packet.image_size[1];
        ctx.drawImage(img, 0, 0);

        // keep-out zones
        for (const z of [...zones]) {
            if (z.points.length < 3) continue;
            ctx.beginPath();
            z.points.forEach(([x, y], i) =>
                i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)
            );
            ctx.closePath();
            ctx.fillStyle = "rgba(239, 68, 68, 0.13)";
            ctx.fill();
            ctx.strokeStyle = "#ef4444";
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.fillStyle = "#ef4444";
            ctx.font = "12px monospace";
            ctx.fillText(z.name.toUpperCase(), z.points[0][0] * sx + 4, z.points[0][1] * sy + 14);
        }

        // draft zone being drawn
        if (draftZone.length > 0) {
            ctx.beginPath();
            draftZone.forEach(([x, y], i) =>
                i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)
            );
            ctx.strokeStyle = "#fbbf24";
            ctx.setLineDash([6, 4]);
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.setLineDash([]);
            for (const [x, y] of draftZone) {
                ctx.beginPath();
                ctx.arc(x * sx, y * sy, 4, 0, Math.PI * 2);
                ctx.fillStyle = "#fbbf24";
                ctx.fill();
            }
        }

        for (const t of packet.tracks) {
            const color = trackColor(t);

            // fading trail
            if (t.trail.length > 1) {
                for (let i = 1; i < t.trail.length; i++) {
                    ctx.beginPath();
                    ctx.moveTo(t.trail[i - 1][0] * sx, t.trail[i - 1][1] * sy);
                    ctx.lineTo(t.trail[i][0] * sx, t.trail[i][1] * sy);
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = (i / t.trail.length) * 0.8;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
            }

            // predicted-path ghost (dotted)
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.55;
            for (let i = 0; i < t.predicted.length; i += 2) {
                const [px, py] = t.predicted[i];
                ctx.beginPath();
                ctx.arc(px * sx, py * sy, 2, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.globalAlpha = 1;

            // bbox
            const [x1, y1, x2, y2] = t.bbox;
            ctx.strokeStyle = color;
            ctx.lineWidth = t.id === selectedTrackId ? 3.5 : 2;
            ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

            // label
            let label = `#${t.id} ${t.class} ${(t.fused_conf * 100).toFixed(0)}%`;
            if (t.threat && t.class === "drone" && t.threat.level !== "NONE") {
                label += ` [${t.threat.level} ${t.threat.score}]`;
            }
            if (t.flagged) label += " !";
            ctx.font = "bold 13px monospace";
            const tw = ctx.measureText(label).width;
            const ly = Math.max(16, y1 * sy - 6);
            ctx.fillStyle = "rgba(5, 5, 5, 0.75)";
            ctx.fillRect(x1 * sx - 2, ly - 13, tw + 8, 17);
            ctx.fillStyle = color;
            ctx.fillText(label, x1 * sx + 2, ly);
        }
    }, [packet, zones, draftZone, selectedTrackId]);

    useEffect(() => {
        const img = new Image();
        img.onload = () => {
            imgRef.current = img;
            render();
        };
        img.src = `data:image/jpeg;base64,${packet.frame_jpeg_b64}`;
    }, [packet, render]);

    useEffect(() => {
        render();
    }, [render]);

    const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        // CSS pixels -> canvas pixels -> original image space
        const cx = ((e.clientX - rect.left) / rect.width) * canvas.width;
        const cy = ((e.clientY - rect.top) / rect.height) * canvas.height;
        const ix = (cx / canvas.width) * packet.image_size[0];
        const iy = (cy / canvas.height) * packet.image_size[1];

        let hit: number | null = null;
        for (const t of packet.tracks) {
            const [x1, y1, x2, y2] = t.bbox;
            if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) {
                hit = t.id;
                break;
            }
        }
        onCanvasClick(ix, iy, hit);
    };

    return (
        <canvas
            ref={canvasRef}
            onClick={handleClick}
            className={`w-full h-auto block ${drawingZone ? "cursor-crosshair" : "cursor-pointer"}`}
        />
    );
}
