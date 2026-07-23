"use client";

import React from "react";
import { Pentagon, Check, Trash2, X } from "lucide-react";
import { Zone } from "@/types";

interface ThreatZoneEditorProps {
    zones: Zone[];
    drawing: boolean;
    draftPoints: number;        // points placed so far in the draft zone
    onStartDrawing: () => void;
    onFinishZone: () => void;
    onCancelDrawing: () => void;
    onClearZones: () => void;
}

export function ThreatZoneEditor({
    zones,
    drawing,
    draftPoints,
    onStartDrawing,
    onFinishZone,
    onCancelDrawing,
    onClearZones,
}: ThreatZoneEditorProps) {
    return (
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            {!drawing ? (
                <>
                    <button
                        onClick={onStartDrawing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-red-900/60 text-red-400 hover:border-red-500 hover:text-red-300 transition-colors"
                    >
                        <Pentagon className="w-3.5 h-3.5" />
                        DRAW KEEP-OUT ZONE
                    </button>
                    {zones.length > 0 && (
                        <button
                            onClick={onClearZones}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white transition-colors"
                        >
                            <Trash2 className="w-3.5 h-3.5" />
                            CLEAR ({zones.length})
                        </button>
                    )}
                </>
            ) : (
                <>
                    <span className="text-amber-400 animate-pulse">
                        CLICK MAP TO PLACE VERTICES ({draftPoints})
                    </span>
                    <button
                        onClick={onFinishZone}
                        disabled={draftPoints < 3}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-emerald-800 text-emerald-400 hover:border-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                        <Check className="w-3.5 h-3.5" />
                        ARM ZONE
                    </button>
                    <button
                        onClick={onCancelDrawing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-700 text-slate-400 hover:border-slate-500 transition-colors"
                    >
                        <X className="w-3.5 h-3.5" />
                        CANCEL
                    </button>
                </>
            )}
        </div>
    );
}
