"use client";

import React, { useRef } from "react";
import { useGLTF } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useScroll, useTransform } from "framer-motion";
import * as THREE from "three";

export const DroneModel = () => {
    const group = useRef<THREE.Group>(null);

    // The user will provide this model. Wait for it using Suspense.
    const { scene } = useGLTF("/models/dji_fpv.glb");

    const { scrollYProgress } = useScroll();
    const { viewport } = useThree();

    // Phase 1 (0-25%): Reveal & Center
    // Phase 2 (25-50%): Rotate to show payload
    // Phase 3 (50-75%): Scale down, move top right
    // Phase 4 (75-100%): Park into "About Us" section

    // Responsive scaling based on viewport width
    const baseScale = viewport.width < 5 ? 0.6 : 1.2;

    // --- POSITIONS ---
    const yPos = useTransform(
        scrollYProgress,
        [0, 0.25, 0.5, 0.75, 1],
        [3, 0, 0, 2, -1] // Drops in, stays, moves up slightly, moves down
    );

    const xPos = useTransform(
        scrollYProgress,
        [0, 0.25, 0.5, 0.75, 1],
        [0, 0, 0, viewport.width * 0.25, viewport.width * 0.3] // Center -> Right side
    );

    // --- ROTATIONS ---
    const rotX = useTransform(
        scrollYProgress,
        [0, 0.25, 0.5, 0.75, 1],
        [0.5, 0.1, 0.6, 0.2, 0.1] // Tilts forward to show top, flattens, tilts up to show payload
    );

    const rotY = useTransform(
        scrollYProgress,
        [0, 0.25, 0.5, 0.75, 1],
        [-Math.PI, -Math.PI / 8, Math.PI / 4, Math.PI / 6, Math.PI / 2] // Spins around to side profile
    );

    const rotZ = useTransform(
        scrollYProgress,
        [0, 0.25, 0.5, 0.75, 1],
        [0, 0, -0.2, -0.1, 0] // Slight banking
    );

    // --- SCALES ---
    const dynamicScale = useTransform(
        scrollYProgress,
        [0, 0.25, 0.5, 0.75, 1],
        [baseScale * 0.5, baseScale, baseScale, baseScale * 0.6, baseScale * 0.8]
    );

    useFrame(() => {
        if (!group.current) return;

        // Add subtle ambient hovering regardless of scroll
        const time = performance.now() / 1000;
        const hoverOffset = Math.sin(time * 2) * 0.05;

        // Apply Framer Motion transforms to Three.js Object3D
        group.current.position.set(xPos.get(), yPos.get() + hoverOffset, 0);
        group.current.rotation.set(rotX.get(), rotY.get(), rotZ.get());

        const s = dynamicScale.get();
        group.current.scale.set(s, s, s);
    });

    return (
        <group ref={group} dispose={null}>
            <primitive object={scene} />
        </group>
    );
};

// Preload the model for performance
useGLTF.preload("/models/stealth-drone.glb");
