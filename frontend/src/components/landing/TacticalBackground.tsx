"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

export const TacticalBackground = () => {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    // Helper to generate random hex strings for the data cascade
    const generateHexLine = () => {
        const chars = "0123456789ABCDEF";
        return Array.from({ length: 12 })
            .map(() => chars[Math.floor(Math.random() * chars.length)])
            .join(" ");
    };

    return (
        <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">

            {/* 1. Deep Central Gradient Glow */}
            <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-[800px] h-[800px] bg-emerald-900/5 rounded-full blur-[120px] mix-blend-screen opacity-50" />
                <div className="absolute w-[600px] h-[600px] bg-slate-800/10 rounded-full blur-[100px] mix-blend-screen opacity-30" />
            </div>

            {/* 2. Left-side Data Cascade */}
            <div className="absolute left-4 md:left-12 top-0 bottom-0 w-32 md:w-48 overflow-hidden opacity-[0.04] flex flex-col items-start justify-start py-24 font-mono text-[10px] md:text-xs leading-tight tracking-[0.2em] text-emerald-500">
                <div className="flex flex-col gap-2">
                    {/* Header */}
                    <div className="mb-4 text-emerald-400 font-bold border-b border-emerald-500/30 pb-2 w-full">
                        RAW.STREAM_0X9A
                    </div>
                    {/* Scrolling Columns */}
                    <div className="relative h-full w-full overflow-hidden mask-image:linear-gradient(to bottom, transparent, black 10%, black 90%, transparent)">
                        <motion.div
                            animate={{ y: ["0%", "-50%"] }}
                            transition={{
                                repeat: Infinity,
                                ease: "linear",
                                duration: 30, // Slow, purposeful scroll
                            }}
                            className="flex flex-col gap-1 whitespace-nowrap"
                        >
                            {/* Create two identical blocks for seamless looping */}
                            {[...Array(2)].map((_, blockIdx) => (
                                <div key={blockIdx} className="flex flex-col gap-1">
                                    {[...Array(50)].map((_, i) => (
                                        <div key={i} className="flex items-center gap-4">
                                            <span className="opacity-50">L{(i + blockIdx * 50).toString().padStart(4, '0')}</span>
                                            <span>{generateHexLine()}</span>
                                            {/* Occasional pulsing blip */}
                                            {Math.random() > 0.9 && (
                                                <motion.div
                                                    animate={{ opacity: [0, 1, 0] }}
                                                    transition={{ duration: 2, repeat: Infinity, delay: Math.random() * 2 }}
                                                    className="w-1.5 h-1.5 bg-emerald-400 rounded-full shadow-[0_0_8px_rgba(52,211,153,0.8)] shrink-0"
                                                />
                                            )}
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </motion.div>
                    </div>
                </div>
            </div>

            {/* 2b. Right-side Data Cascade (Mirrored) */}
            <div className="absolute right-4 md:right-12 top-0 bottom-0 w-32 md:w-48 overflow-hidden opacity-[0.04] flex flex-col items-end justify-start py-24 font-mono text-[10px] md:text-xs leading-tight tracking-[0.2em] text-emerald-500 text-right">
                <div className="flex flex-col gap-2 w-full items-end">
                    {/* Header */}
                    <div className="mb-4 text-emerald-400 font-bold border-b border-emerald-500/30 pb-2 w-full text-right">
                        SIG.INT_0x4F
                    </div>
                    {/* Scrolling Columns */}
                    <div className="relative h-full w-full overflow-hidden mask-image:linear-gradient(to bottom, transparent, black 10%, black 90%, transparent)">
                        <motion.div
                            animate={{ y: ["0%", "-50%"] }}
                            transition={{
                                repeat: Infinity,
                                ease: "linear",
                                duration: 35, // Slightly offset speed
                            }}
                            className="flex flex-col gap-1 whitespace-nowrap items-end"
                        >
                            {/* Create two identical blocks for seamless looping */}
                            {[...Array(2)].map((_, blockIdx) => (
                                <div key={blockIdx} className="flex flex-col gap-1 items-end">
                                    {[...Array(50)].map((_, i) => (
                                        <div key={i} className="flex items-center gap-4 justify-end">
                                            {/* Occasional pulsing blip (on the left for right side) */}
                                            {Math.random() > 0.9 && (
                                                <motion.div
                                                    animate={{ opacity: [0, 1, 0] }}
                                                    transition={{ duration: 2, repeat: Infinity, delay: Math.random() * 2 }}
                                                    className="w-1.5 h-1.5 bg-emerald-400 rounded-full shadow-[0_0_8px_rgba(52,211,153,0.8)] shrink-0"
                                                />
                                            )}
                                            <span>{generateHexLine()}</span>
                                            <span className="opacity-50">R{(i + blockIdx * 50).toString().padStart(4, '0')}</span>
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </motion.div>
                    </div>
                </div>
            </div>

            {/* 3. Faint LiDAR Grid (CSS Background Pattern) */}
            <div
                className="absolute inset-0 opacity-[0.03] perspective-[1000px]"
                style={{
                    backgroundImage: `linear-gradient(rgba(16, 185, 129, 0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(16, 185, 129, 0.5) 1px, transparent 1px)`,
                    backgroundSize: '4rem 4rem',
                    transform: 'rotateX(60deg) scale(2) translateY(20%) z-index(-1)'
                }}
            />

            {/* Simulated Point Cloud Clusters over the grid */}
            {[...Array(6)].map((_, i) => (
                <motion.div
                    key={`cluster-${i}`}
                    className="absolute w-[100px] h-[100px] border border-emerald-500/10 rounded-full flex items-center justify-center opacity-10"
                    style={{
                        top: `${Math.random() * 80 + 10}%`,
                        left: `${Math.random() * 80 + 10}%`,
                        transform: `scale(${Math.random() * 1.5 + 0.5})`
                    }}
                    animate={{
                        rotate: 360,
                        opacity: [0.05, 0.15, 0.05]
                    }}
                    transition={{
                        rotate: { duration: 40 + Math.random() * 20, repeat: Infinity, ease: "linear" },
                        opacity: { duration: 4 + Math.random() * 4, repeat: Infinity, ease: "easeInOut" }
                    }}
                >
                    <div className="w-[60%] h-[60%] border-t border-r border-emerald-500/20 rounded-full" />
                </motion.div>
            ))}

            {/* 4. Minimal Corners / HUD Markers */}
            <div className="absolute top-24 left-8 opacity-10 flex flex-col gap-1 items-start">
                <div className="w-8 h-[1px] bg-emerald-500" />
                <div className="w-[1px] h-8 bg-emerald-500" />
                <span className="text-[9px] font-mono text-emerald-400 mt-2 tracking-widest">+42.109 / -71.223</span>
            </div>

            <div className="absolute top-24 right-8 opacity-10 flex flex-col gap-1 items-end">
                <div className="w-8 h-[1px] bg-emerald-500" />
                <div className="w-[1px] h-8 bg-emerald-500 absolute right-0 top-0" />
                <span className="text-[9px] font-mono text-emerald-400 mt-10 tracking-widest">SYS.NOMINAL</span>
            </div>

        </div>
    );
};
