"use client";

import React, { useState, useEffect } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { Shield, Crosshair, Cpu, Eye, ChevronDown } from "lucide-react";
import { Dropzone } from "@/components/Dropzone";

export const ScrollOverlay = () => {
    const { scrollYProgress } = useScroll();

    // Opacities for the 4 phases
    const phase1Opacity = useTransform(scrollYProgress, [0, 0.15, 0.25], [1, 1, 0]);
    const phase2Opacity = useTransform(scrollYProgress, [0.2, 0.35, 0.45], [0, 1, 0]);
    const phase3Opacity = useTransform(scrollYProgress, [0.45, 0.6, 0.7], [0, 1, 0]);
    const phase4Opacity = useTransform(scrollYProgress, [0.75, 0.85, 1], [0, 1, 1]);

    // Targeting box animation for Phase 3
    const targetBoxScale = useTransform(scrollYProgress, [0.5, 0.6], [2, 1]);

    return (
        <div className="relative z-10 w-full pointer-events-none">

            {/* 
        This empty div creates the scrollable height.
        400vh gives us plenty of room to scroll through the 4 phases. 
      */}
            <div className="h-[400vh] w-full" />

            {/* --- PHASE 1: AERIAL OPS Reveal --- */}
            <motion.section
                className="fixed inset-0 flex flex-col items-center justify-center pointer-events-none"
                style={{ opacity: phase1Opacity }}
            >
                <div className="text-center mt-64 md:mt-80 px-6">
                    <h1 className="text-5xl md:text-8xl font-black tracking-tighter text-white uppercase drop-shadow-2xl">
                        AERIAL <span className="text-emerald-500">OPS</span>
                    </h1>
                    <p className="mt-4 text-sm md:text-lg tracking-[0.3em] font-semibold text-slate-400 uppercase">
                        Empowering Security & Surveillance
                    </p>
                    <motion.div
                        animate={{ y: [0, 10, 0] }}
                        transition={{ repeat: Infinity, duration: 2 }}
                        className="mt-12 flex justify-center text-slate-500"
                    >
                        <ChevronDown className="w-8 h-8" />
                    </motion.div>
                </div>
            </motion.section>

            {/* --- PHASE 2: Multispectral Sensing --- */}
            <motion.section
                className="fixed inset-0 flex items-center px-6 md:px-24 pointer-events-none"
                style={{ opacity: phase2Opacity }}
            >
                <div className="max-w-lg">
                    <div className="flex items-center gap-3 mb-4 text-sky-400">
                        <Eye className="w-6 h-6" />
                        <span className="font-mono text-sm tracking-widest uppercase">Target Acquisition</span>
                    </div>
                    <h2 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight">
                        Beyond Visual <br /> Line of Sight.
                    </h2>
                    <p className="text-lg text-slate-400 leading-relaxed">
                        Equipped with state-of-the-art optical payloads. Identify and classify aerial threats miles before they enter restricted airspace.
                    </p>

                    {/* Stylized Data Output */}
                    <div className="mt-8 p-4 rounded-lg bg-sky-950/30 border border-sky-900/50 backdrop-blur font-mono text-xs text-sky-300">
                        <p>&gt; RUNNING OPTICAL SCAN...</p>
                        <p>&gt; SENSOR ARRAY: ONLINE</p>
                        <p>&gt; TELEMETRY LINK: STABLE</p>
                        <p className="animate-pulse mt-2">&gt; AWAITING TARGETS_</p>
                    </div>
                </div>
            </motion.section>

            {/* --- PHASE 3: Precise Detection (Targeting Box) --- */}
            <motion.section
                className="fixed inset-0 pointer-events-none flex items-center justify-center md:items-center md:justify-start md:px-24"
                style={{ opacity: phase3Opacity }}
            >
                <div className="max-w-md bg-black/40 backdrop-blur-md p-8 rounded-2xl border border-white/10 relative z-20">
                    <div className="flex items-center gap-3 mb-4 text-emerald-500">
                        <Crosshair className="w-6 h-6" />
                        <span className="font-mono text-sm tracking-widest uppercase">YOLO11n Powered</span>
                    </div>
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                        Zeroing In.
                    </h2>
                    <p className="text-base text-slate-400">
                        Our neural engine instantly differentiates between authorized commercial drones, biological clutter (birds), and unidentified threats.
                    </p>
                </div>

                {/* Framing Box for the Drone model sitting top-right */}
                <motion.div
                    className="absolute right-[5%] md:right-[15%] top-[15%] md:top-[25%] w-64 h-64 md:w-96 md:h-96 border border-emerald-500/50 before:absolute before:-top-2 before:-left-2 before:w-4 before:h-4 before:border-t-2 before:border-l-2 before:border-emerald-500 pb-2 pr-2"
                    style={{ scale: targetBoxScale }}
                >
                    <div className="absolute -bottom-2 -right-2 w-4 h-4 border-b-2 border-r-2 border-emerald-500" />
                    <div className="absolute top-2 right-2 text-[10px] font-mono text-emerald-500">
                        CONF: 99.8%<br />CLASS: VTOL_STEALTH
                    </div>
                </motion.div>
            </motion.section>

            {/* --- PHASE 4: Integrated Dashboard --- */}
            <motion.section
                className="fixed inset-0 flex flex-col justify-end pointer-events-none pb-12 px-6"
                style={{ opacity: phase4Opacity }}
            >
                <div className="w-full max-w-7xl mx-auto pointer-events-auto">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-end">

                        <div className="bg-slate-900/80 backdrop-blur-xl border border-slate-700/50 p-8 rounded-2xl shadow-2xl">
                            <h3 className="flex items-center gap-2 text-white font-bold text-2xl mb-4">
                                <Shield className="text-emerald-500" /> System Access
                            </h3>
                            <p className="text-slate-400 mb-6 font-mono text-sm">
                                Drop telemetry feed to initiate neural analysis block.
                            </p>

                            {/* Replace the interactive segment here with a placeholder or the actual component */}
                            <div className="opacity-50 pointer-events-none hover:opacity-100 hover:pointer-events-auto transition-opacity duration-500">
                                <p className="p-4 bg-slate-800 text-slate-300 text-center rounded border border-dashed border-slate-600 font-mono text-sm">
                                    [ DROPZONE U/I MOVED TO DEDICATED APP DASHBOARD ]<br />
                                    <a href="/app" className="text-emerald-500 hover:underline mt-2 inline-block">LAUNCH SYSTEM &rarr;</a>
                                </p>
                            </div>
                        </div>

                        <div className="flex flex-col gap-4 text-right hidden md:flex pb-8">
                            <h2 className="text-5xl font-black text-white mix-blend-overlay">INTEGRATED</h2>
                            <h2 className="text-5xl font-black text-emerald-500">DEFENSE</h2>
                            <h2 className="text-5xl font-black text-slate-700">GRID.</h2>
                        </div>

                    </div>
                </div>
            </motion.section>

        </div>
    );
};
