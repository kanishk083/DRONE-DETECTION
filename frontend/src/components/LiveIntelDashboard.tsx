"use client";

import React, { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Film, Loader2, Radar, Square } from "lucide-react";

import { LiveDetectionCanvas } from "./LiveDetectionCanvas";
import { ThreatZoneEditor } from "./ThreatZoneEditor";
import { EventFeed } from "./EventFeed";
import { useDetectionStream } from "@/lib/useDetectionStream";
import { ThreatLevel, Track, Zone } from "@/types";

const LEVEL_COLORS: Record<ThreatLevel, string> = {
    NONE: "text-slate-400",
    LOW: "text-cyan-400",
    MEDIUM: "text-amber-400",
    HIGH: "text-orange-400",
    CRITICAL: "text-red-500",
};
const LEVEL_ORDER: ThreatLevel[] = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"];

export function LiveIntelDashboard() {
    const stream = useDetectionStream();
    const fileRef = useRef<HTMLInputElement>(null);

    const [zones, setZones] = useState<Zone[]>([]);
    const [drawing, setDrawing] = useState(false);
    const [draft, setDraft] = useState<[number, number][]>([]);
    const [selectedId, setSelectedId] = useState<number | null>(null);

    const handleCanvasClick = useCallback(
        (x: number, y: number, hitTrackId: number | null) => {
            if (drawing) {
                setDraft((d) => [...d, [x, y]]);
            } else {
                setSelectedId(hitTrackId);
            }
        },
        [drawing]
    );

    const finishZone = useCallback(() => {
        if (draft.length < 3) return;
        const next = [...zones, { name: `zone ${zones.length + 1}`, points: draft }];
        setZones(next);
        setDraft([]);
        setDrawing(false);
        stream.sendZones(next);
    }, [draft, zones, stream]);

    const clearZones = useCallback(() => {
        setZones([]);
        stream.sendZones([]);
    }, [stream]);

    const pkt = stream.packet;
    const tracks: Track[] = pkt?.tracks ?? [];
    const drones = tracks.filter((t) => t.class === "drone");
    const birds = tracks.filter((t) => t.class === "bird");
    const maxThreat = drones.reduce(
        (best, t) =>
            LEVEL_ORDER.indexOf(t.threat?.level ?? "NONE") >
            LEVEL_ORDER.indexOf(best)
                ? (t.threat?.level ?? "NONE")
                : best,
        "NONE" as ThreatLevel
    );
    const maxScore = Math.max(0, ...drones.map((t) => t.threat?.score ?? 0));
    const selected = tracks.find((t) => t.id === selectedId) ?? null;

    // ---- upload state ------------------------------------------------------
    if (!pkt) {
        return (
            <div className="w-full max-w-4xl mx-auto shadow-2xl rounded-2xl bg-[#0a0a0a] border border-slate-800 p-12 text-center">
                <input
                    ref={fileRef}
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) stream.start(f);
                    }}
                />
                {stream.status === "uploading" ? (
                    <div className="flex flex-col items-center gap-4 text-slate-400 font-mono">
                        <Loader2 className="w-10 h-10 animate-spin text-emerald-500" />
                        <p>INITIALIZING INTELLIGENCE PIPELINE...</p>
                    </div>
                ) : (
                    <button
                        onClick={() => fileRef.current?.click()}
                        className="flex flex-col items-center gap-4 mx-auto text-slate-400 hover:text-white transition-colors group"
                    >
                        <Film className="w-12 h-12 text-emerald-500 group-hover:scale-110 transition-transform" />
                        <span className="font-mono text-sm tracking-widest">
                            [ UPLOAD VIDEO FOR LIVE INTELLIGENCE ]
                        </span>
                        <span className="text-xs text-slate-600 max-w-md">
                            Streams the KITE pipeline: tracking, kinematic
                            classification, trajectory prediction, threat zones
                            and tactical events — in real time.
                        </span>
                    </button>
                )}
                {stream.error && (
                    <p className="mt-6 text-sm font-mono text-red-400">
                        {stream.error}
                    </p>
                )}
            </div>
        );
    }

    // ---- live view ---------------------------------------------------------
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start"
        >
            {/* canvas + zone controls */}
            <div className="col-span-1 lg:col-span-2 space-y-3">
                <div className="rounded-xl overflow-hidden shadow-2xl border border-slate-800 bg-black relative">
                    <LiveDetectionCanvas
                        packet={pkt}
                        zones={zones}
                        draftZone={draft}
                        drawingZone={drawing}
                        selectedTrackId={selectedId}
                        onCanvasClick={handleCanvasClick}
                    />
                    <div className="absolute top-3 right-3 flex items-center gap-2 text-[11px] font-mono bg-black/70 rounded px-2.5 py-1.5 border border-slate-800">
                        <Radar className="w-3.5 h-3.5 text-emerald-500 animate-pulse" />
                        <span className="text-emerald-400">
                            {stream.status === "done" ? "STREAM COMPLETE" : "LIVE"}
                        </span>
                        <span className="text-slate-500">
                            {pkt.fps.toFixed(1)} FPS · intel {pkt.intel_ms} ms
                        </span>
                    </div>
                </div>
                <div className="flex items-center justify-between gap-3">
                    <ThreatZoneEditor
                        zones={zones}
                        drawing={drawing}
                        draftPoints={draft.length}
                        onStartDrawing={() => {
                            setDrawing(true);
                            setDraft([]);
                        }}
                        onFinishZone={finishZone}
                        onCancelDrawing={() => {
                            setDrawing(false);
                            setDraft([]);
                        }}
                        onClearZones={clearZones}
                    />
                    <button
                        onClick={stream.stop}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-700 text-xs font-mono text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
                    >
                        <Square className="w-3 h-3" />
                        TERMINATE
                    </button>
                </div>
            </div>

            {/* right rail: telemetry + kinematics + events */}
            <div className="col-span-1 space-y-4">
                <div className="grid grid-cols-2 gap-3">
                    <LiveStat label="UAV TARGETS" value={drones.length}
                        className={drones.length ? "text-red-500" : "text-emerald-500"} />
                    <LiveStat label="BIOLOGICAL" value={birds.length}
                        className="text-sky-300" />
                    <LiveStat label="THREAT LEVEL" value={maxThreat}
                        className={LEVEL_COLORS[maxThreat]} />
                    <LiveStat label="THREAT SCORE" value={maxScore}
                        className={maxScore >= 65 ? "text-red-500" : "text-slate-300"} />
                </div>

                {/* threat gauge */}
                <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
                    <div className="h-2 rounded bg-slate-800 overflow-hidden">
                        <div
                            className={`h-full transition-all duration-300 ${
                                maxScore >= 65 ? "bg-red-500"
                                : maxScore >= 45 ? "bg-orange-400"
                                : maxScore >= 25 ? "bg-amber-400" : "bg-emerald-500"
                            }`}
                            style={{ width: `${maxScore}%` }}
                        />
                    </div>
                </div>

                {/* selected track kinematic readout */}
                <div className="rounded-xl bg-slate-900 border border-slate-800 p-4 font-mono text-xs space-y-1.5">
                    <h3 className="text-slate-500 font-bold tracking-widest mb-2">
                        TRACK ANALYSIS {selected ? `— #${selected.id}` : ""}
                    </h3>
                    {selected ? (
                        <>
                            <Row k="CLASS" v={`${selected.class} ${(selected.fused_conf * 100).toFixed(0)}% (yolo: ${selected.appearance_class} ${(selected.appearance_conf * 100).toFixed(0)}%)`} />
                            <Row k="REASON" v={selected.reason} />
                            {selected.kinematics && (
                                <>
                                    <Row k="STRAIGHTNESS" v={selected.kinematics.straightness.toFixed(2)} />
                                    <Row k="SPEED" v={`${selected.kinematics.speed_mean.toFixed(0)} px/s ± ${selected.kinematics.speed_std.toFixed(0)}`} />
                                    <Row k="WINGBEAT" v={selected.kinematics.periodicity_available
                                        ? `${selected.kinematics.vertical_periodicity_hz.toFixed(1)} Hz (p=${selected.kinematics.vertical_periodicity_power.toFixed(2)})`
                                        : "n/a"} />
                                    <Row k="HOVER" v={selected.kinematics.hover_score.toFixed(2)} />
                                </>
                            )}
                            {selected.threat && (
                                <Row k="THREAT" v={`${selected.threat.level} (${selected.threat.score})${selected.threat.zone_inbound ? " INBOUND" : ""}`} />
                            )}
                        </>
                    ) : (
                        <p className="text-slate-600">click a target on the feed</p>
                    )}
                </div>

                <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
                    <EventFeed events={stream.events} />
                </div>
            </div>
        </motion.div>
    );
}

function LiveStat({ label, value, className }: {
    label: string;
    value: string | number;
    className?: string;
}) {
    return (
        <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
            <p className="text-[10px] font-bold font-mono tracking-widest text-slate-500 mb-1.5">
                {label}
            </p>
            <p className={`text-2xl font-mono font-bold ${className ?? "text-white"}`}>
                {value}
            </p>
        </div>
    );
}

function Row({ k, v }: { k: string; v: string }) {
    return (
        <div className="flex justify-between gap-3">
            <span className="text-slate-500">{k}</span>
            <span className="text-slate-200 text-right truncate">{v}</span>
        </div>
    );
}
