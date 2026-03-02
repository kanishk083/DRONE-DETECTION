"use client";

import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Environment, Float, PresentationControls, useGLTF } from "@react-three/drei";
import { motion } from "framer-motion";
import { TacticalBackground } from "./landing/TacticalBackground";

// The imported DJI drone model
const DjiDrone = () => {
    const { scene } = useGLTF("/models/dji_fpv.glb");
    return (
        <primitive object={scene} scale={1.0} position={[0, -0.2, 0]} />
    );
};

// Preload to avoid popping
useGLTF.preload("/models/dji_fpv.glb");

export const Hero3D = () => {
    return (
        <section id="home" className="relative w-full h-[70vh] md:h-[90vh] bg-[#050505] overflow-hidden flex flex-col items-center justify-center pt-16">

            {/* Extended Abstract UI layer */}
            <TacticalBackground />

            {/* 3D Canvas Layer */}
            <div className="absolute inset-0 z-10 cursor-grab active:cursor-grabbing">
                <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
                    {/* Cinematic Lighting */}
                    <ambientLight intensity={0.2} />
                    <spotLight position={[5, 10, 5]} angle={0.2} penumbra={1} intensity={2} color="#ffffff" castShadow />
                    <spotLight position={[-5, 5, -5]} angle={0.3} penumbra={1} intensity={1} color="#10b981" />
                    <spotLight position={[10, -5, -5]} angle={0.3} penumbra={1} intensity={2} color="#ef4444" />

                    <Suspense fallback={null}>
                        <PresentationControls
                            global
                            rotation={[0, 0, 0]}
                            polar={[-Math.PI / 3, Math.PI / 3]}
                            azimuth={[-Math.PI, Math.PI]}
                        >
                            <Float
                                speed={2}
                                rotationIntensity={0.5}
                                floatIntensity={2}
                                floatingRange={[-0.2, 0.2]}
                            >
                                <DjiDrone />
                            </Float>
                        </PresentationControls>
                        <Environment preset="city" />
                    </Suspense>
                </Canvas>
            </div>

            {/* UI Overlay Layer (Pixella Style) */}
            <div className="absolute z-20 w-full max-w-7xl mx-auto px-6 inset-0 pointer-events-none flex flex-col justify-between py-12">
                <div /> {/* Spacer for top nav */}

                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 1, delay: 0.2 }}
                    className="flex flex-col md:flex-row md:items-end justify-between w-full pointer-events-auto mt-auto"
                >
                    <div>
                        <h1 className="text-4xl md:text-7xl font-extrabold tracking-tight text-white mb-2 uppercase drop-shadow-lg" style={{ WebkitTextStroke: "1px rgba(255,255,255,0.1)" }}>
                            AERIAL OPS
                        </h1>
                        <p className="text-sm md:text-lg tracking-[0.2em] text-emerald-400 uppercase font-bold drop-shadow">
                            Precision Threat Detection
                        </p>
                    </div>

                    <a href="#analyze" className="mt-8 md:mt-0 px-8 py-4 rounded-full bg-white/10 backdrop-blur-md border border-white/20 text-white text-sm font-bold tracking-widest uppercase hover:bg-white hover:text-black transition-all duration-300 shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                        Launch System ↓
                    </a>
                </motion.div>
            </div>

        </section>
    );
};
