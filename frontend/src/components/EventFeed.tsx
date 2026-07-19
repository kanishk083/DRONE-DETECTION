"use client";

import React, { useEffect, useRef } from "react";
import { TacticalEvent, ThreatLevel } from "@/types";

const SEVERITY_STYLES: Record<ThreatLevel, string> = {
    NONE: "text-slate-500 border-slate-800",
    LOW: "text-cyan-400 border-cyan-900/50",
    MEDIUM: "text-amber-400 border-amber-900/50",
    HIGH: "text-orange-400 border-orange-900/50",
    CRITICAL: "text-red-400 border-red-900/60",
};

interface EventFeedProps {
    events: TacticalEvent[];
}

export function EventFeed({ events }: EventFeedProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [events.length]);

    return (
        <div className="flex flex-col h-full">
            <h3 className="text-xs font-bold font-mono tracking-widest text-slate-500 mb-3">
                EVENT FEED
            </h3>
            <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 max-h-64 min-h-32">
                {events.length === 0 && (
                    <p className="text-xs font-mono text-slate-600">
                        awaiting tactical events...
                    </p>
                )}
                {events.map((ev, i) => (
                    <div
                        key={`${ev.type}-${ev.track_id}-${ev.ts}-${i}`}
                        className={`text-[11px] font-mono px-2.5 py-1.5 rounded border bg-black/40 flex items-center justify-between gap-2 ${SEVERITY_STYLES[ev.severity] ?? SEVERITY_STYLES.LOW}`}
                    >
                        <span className="font-bold">{ev.type}</span>
                        <span className="text-slate-500">
                            #{ev.track_id}
                            {ev.eta_s != null && ` eta ${ev.eta_s}s`}
                            {" · "}
                            {ev.ts.toFixed(1)}s
                        </span>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
}
