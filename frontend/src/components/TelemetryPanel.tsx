"use client";

import React from "react";
import { motion } from "framer-motion";
import { Target, Activity, ShieldCheck, LucideIcon } from "lucide-react";
import { Detection } from "@/types";
import { cn } from "@/lib/utils";

interface TelemetryPanelProps {
    detections: Detection[];
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
    detections,
}) => {
    const droneCount = detections.filter((d) => d.class === "drone").length;
    const birdCount = detections.filter((d) => d.class === "bird").length;
    const maxConfidence = detections.length
        ? Math.max(...detections.map((d) => d.confidence))
        : 0;

    // Animation variants for staggering children
    const containerVariants = {
        hidden: { opacity: 0 },
        show: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2, // Small delay after results arrive
            },
        },
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        show: {
            opacity: 1,
            y: 0,
            transition: { type: "spring", stiffness: 300, damping: 24 },
        },
    };

    return (
        <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="show"
            className="flex flex-col gap-4 w-full max-w-sm mx-auto"
        >
            <TelemetryCard
                title="UAV TARGETS"
                value={droneCount}
                icon={Target}
                variants={itemVariants}
                highlight={droneCount > 0 ? "text-red-500" : "text-emerald-500"}
            />

            <TelemetryCard
                title="BIOLOGICAL (BIRDS)"
                value={birdCount}
                icon={Activity}
                variants={itemVariants}
                highlight="text-slate-300"
            />

            <TelemetryCard
                title="MAX CONFIDENCE"
                value={`${(maxConfidence * 100).toFixed(1)}%`}
                icon={ShieldCheck}
                variants={itemVariants}
                highlight={maxConfidence > 0.6 ? "text-emerald-500" : "text-amber-500"}
            />
        </motion.div>
    );
};

interface TelemetryCardProps {
    title: string;
    value: string | number;
    highlight: string;
    icon: LucideIcon;
    variants: any;
}

const TelemetryCard: React.FC<TelemetryCardProps> = ({
    title,
    value,
    highlight,
    icon: Icon,
    variants,
}) => {
    return (
        <motion.div
            variants={variants}
            className="relative overflow-hidden rounded-xl bg-slate-900 border border-slate-800 p-6 flex flex-col justify-between min-h-[140px]"
        >
            <div className="flex items-start justify-between mb-4 gap-2">
                <span className="text-xs font-bold tracking-widest text-slate-500 whitespace-normal leading-tight">
                    {title}
                </span>
                <Icon className={cn("w-5 h-5 shrink-0", highlight)} />
            </div>

            <div className="flex items-baseline gap-2 overflow-hidden w-full">
                <span className={cn("text-3xl md:text-4xl font-mono font-bold tracking-tight text-white truncate w-full")}>
                    {value}
                </span>
            </div>

            {/* Decorative background grid */}
            <div className="absolute inset-0 pointer-events-none opacity-5"
                style={{
                    backgroundImage: `linear-gradient(to right, #808080 1px, transparent 1px), linear-gradient(to bottom, #808080 1px, transparent 1px)`,
                    backgroundSize: '1rem 1rem'
                }}
            />
        </motion.div>
    );
};
