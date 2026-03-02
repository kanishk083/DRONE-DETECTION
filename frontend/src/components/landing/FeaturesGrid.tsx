"use client";

import React from "react";
import { Activity, Radio, Eye, LockKeyhole } from "lucide-react";

const features = [
    {
        icon: <Eye className="w-8 h-8 text-sky-400" />,
        title: "Multispectral Tracking",
        description: "Utilize advanced YOLO computer vision to actively monitor and classify both UAV targets and biological airspace clutter in real-time."
    },
    {
        icon: <Activity className="w-8 h-8 text-emerald-500" />,
        title: "Threat Classification",
        description: "Deep learning neural network correctly identifies commercial drone footprints versus standard consumer aviation models with 99.8% precision."
    },
    {
        icon: <LockKeyhole className="w-8 h-8 text-red-500" />,
        title: "Automated Countermeasures",
        description: "Upon hostile classification, systems immediately log the MAC addresses and vector coords, interfacing with local ground-defense arrays."
    },
    {
        icon: <Radio className="w-8 h-8 text-purple-500" />,
        title: "Remote Ops Dashboard",
        description: "Full tactical feed available via our secure web application, delivering live telemetry and visual bounding-box payloads securely over JWT."
    }
];

export const FeaturesGrid = () => {
    return (
        <section className="relative z-20 w-full bg-[#050505] py-32 px-6 border-t border-white/5">
            <div className="max-w-7xl mx-auto">
                <div className="mb-16 md:mb-24">
                    <h2 className="text-sm font-mono tracking-[0.2em] text-emerald-500 mb-4 uppercase">System Capabilities</h2>
                    <h3 className="text-4xl md:text-5xl font-bold text-white max-w-2xl">
                        Unparalleled precision in airspace security networks.
                    </h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {features.map((feature, idx) => (
                        <div
                            key={idx}
                            className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800 hover:bg-slate-800 transition-colors duration-300"
                        >
                            <div className="mb-6">{feature.icon}</div>
                            <h4 className="text-xl font-bold text-white mb-3">{feature.title}</h4>
                            <p className="text-slate-400 text-sm leading-relaxed">
                                {feature.description}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};
