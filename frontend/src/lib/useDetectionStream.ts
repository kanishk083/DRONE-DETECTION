"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { IntelPacket, StartStreamResponse, TacticalEvent, Zone } from "@/types";

const API = "http://localhost:8000";
const WS = "ws://localhost:8000";
const MAX_EVENTS = 100;

export type StreamStatus =
    | "idle"
    | "uploading"
    | "streaming"
    | "done"
    | "error";

export interface DetectionStream {
    status: StreamStatus;
    packet: IntelPacket | null;
    events: TacticalEvent[];
    error: string | null;
    videoInfo: StartStreamResponse | null;
    start: (file: File) => Promise<void>;
    stop: () => void;
    sendZones: (zones: Zone[]) => Promise<void>;
}

export function useDetectionStream(): DetectionStream {
    const [status, setStatus] = useState<StreamStatus>("idle");
    const [packet, setPacket] = useState<IntelPacket | null>(null);
    const [events, setEvents] = useState<TacticalEvent[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [videoInfo, setVideoInfo] = useState<StartStreamResponse | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const sessionRef = useRef<string | null>(null);

    const cleanup = useCallback(() => {
        wsRef.current?.close();
        wsRef.current = null;
        const sid = sessionRef.current;
        sessionRef.current = null;
        if (sid) {
            // best-effort: tell the backend to release the session
            fetch(`${API}/stream/${sid}`, { method: "DELETE" }).catch(() => {});
        }
    }, []);

    useEffect(() => cleanup, [cleanup]); // on unmount

    const start = useCallback(
        async (file: File) => {
            cleanup();
            setStatus("uploading");
            setError(null);
            setPacket(null);
            setEvents([]);
            try {
                const form = new FormData();
                form.append("file", file);
                const res = await fetch(`${API}/stream/start`, {
                    method: "POST",
                    body: form,
                });
                if (!res.ok) {
                    const detail = await res.text();
                    throw new Error(`stream start failed (${res.status}): ${detail}`);
                }
                const info: StartStreamResponse = await res.json();
                setVideoInfo(info);
                sessionRef.current = info.session_id;

                const ws = new WebSocket(`${WS}/ws/stream/${info.session_id}`);
                wsRef.current = ws;
                ws.onopen = () => setStatus("streaming");
                ws.onmessage = (msg) => {
                    const data = JSON.parse(msg.data);
                    if (data.done) {
                        setStatus("done");
                        sessionRef.current = null; // backend already cleaned up
                        ws.close();
                        return;
                    }
                    const pkt = data as IntelPacket;
                    setPacket(pkt);
                    if (pkt.events.length) {
                        setEvents((prev) =>
                            [...prev, ...pkt.events].slice(-MAX_EVENTS)
                        );
                    }
                };
                ws.onerror = () => {
                    setError("WebSocket connection failed");
                    setStatus("error");
                };
                ws.onclose = () => {
                    setStatus((s) => (s === "streaming" ? "done" : s));
                };
            } catch (err: unknown) {
                setError(err instanceof Error ? err.message : "stream failed");
                setStatus("error");
                cleanup();
            }
        },
        [cleanup]
    );

    const stop = useCallback(() => {
        cleanup();
        setStatus("idle");
        setPacket(null);
    }, [cleanup]);

    const sendZones = useCallback(async (zones: Zone[]) => {
        const sid = sessionRef.current;
        if (!sid) return;
        await fetch(`${API}/stream/${sid}/zones`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ zones }),
        });
    }, []);

    return { status, packet, events, error, videoInfo, start, stop, sendZones };
}
