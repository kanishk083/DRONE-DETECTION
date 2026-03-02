"use client";

import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Environment, Preload } from "@react-three/drei";
import { DroneModel } from "./DroneModel";

export const Scene = () => {
    return (
        <div className="fixed inset-0 z-0 pointer-events-none bg-[#020202]">
            <Canvas
                camera={{ position: [0, 0, 8], fov: 45 }}
                gl={{ antialias: true, alpha: false }}
                dpr={[1, 2]}
            >
                <color attach="background" args={["#020202"]} />

                {/* Dramatic Cinematic Lighting */}
                <ambientLight intensity={0.1} />

                {/* Main key light */}
                <spotLight
                    position={[10, 10, 10]}
                    angle={0.15}
                    penumbra={1}
                    intensity={2}
                    color="#ffffff"
                    castShadow
                />

                {/* Cold blue rim light */}
                <spotLight
                    position={[-10, 5, -10]}
                    angle={0.3}
                    penumbra={1}
                    intensity={5}
                    color="#0ea5e9"
                />

                {/* Red tech rim light */}
                <spotLight
                    position={[10, -5, -5]}
                    angle={0.3}
                    penumbra={1}
                    intensity={2}
                    color="#ef4444"
                />

                <Suspense fallback={null}>
                    <DroneModel />
                    <Environment preset="city" />
                    <Preload all />
                </Suspense>
            </Canvas>
        </div>
    );
};
