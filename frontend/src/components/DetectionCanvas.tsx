"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Detection } from "@/types";
import { cn } from "@/lib/utils";

interface DetectionCanvasProps {
    imageUrl: string;
    detections: Detection[];
    originalSize: [number, number]; // [width, height] from API
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
    imageUrl,
    detections,
    originalSize,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [scale, setScale] = useState({ x: 1, y: 1 });
    const [isImageLoaded, setIsImageLoaded] = useState(false);

    // Recalculate scale whenever window resizes or image loads
    useEffect(() => {
        const updateScale = () => {
            if (containerRef.current && isImageLoaded && originalSize[0] > 0) {
                const renderedWidth = containerRef.current.clientWidth;
                const renderedHeight = containerRef.current.clientHeight;

                setScale({
                    x: renderedWidth / originalSize[0],
                    y: renderedHeight / originalSize[1],
                });
            }
        };

        updateScale();
        window.addEventListener("resize", updateScale);
        return () => window.removeEventListener("resize", updateScale);
    }, [isImageLoaded, originalSize]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="relative w-full max-w-5xl mx-auto rounded-2xl overflow-hidden border border-slate-700 bg-slate-900 shadow-2xl"
        >
            {/* 
        We use an img tag to dictate the aspect ratio of the responsive container.
        The containerRef wraps it to measure the EXACT rendered pixels.
      */}
            <div ref={containerRef} className="relative w-full h-auto">
                <img
                    src={imageUrl}
                    alt="Analyzed target"
                    className="w-full h-auto object-contain block"
                    onLoad={() => setIsImageLoaded(true)}
                />

                {/* Bounding Boxes Layer */}
                <AnimatePresence>
                    {isImageLoaded &&
                        detections.map((det, idx) => {
                            // Map YOLO array coords directly to the currently rendered DOM size
                            const [x1, y1, x2, y2] = det.bbox;

                            const scaledX = x1 * scale.x;
                            const scaledY = y1 * scale.y;
                            const scaledWidth = (x2 - x1) * scale.x;
                            const scaledHeight = (y2 - y1) * scale.y;

                            const isDrone = det.class === "drone";
                            const colorClass = isDrone
                                ? "border-red-500 bg-red-500/10 text-red-500"
                                : "border-sky-500 bg-sky-500/10 text-sky-500";
                            const bgHeaderClass = isDrone ? "bg-red-500" : "bg-sky-500";

                            return (
                                <motion.div
                                    key={`bbox-${idx}`}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.9 }}
                                    transition={{
                                        type: "spring",
                                        stiffness: 400,
                                        damping: 25,
                                        delay: idx * 0.1, // Stagger drawing boxes
                                    }}
                                    className={cn(
                                        "absolute border-2 pointer-events-none group",
                                        colorClass
                                    )}
                                    style={{
                                        left: scaledX,
                                        top: scaledY,
                                        width: scaledWidth,
                                        height: scaledHeight,
                                    }}
                                >
                                    {/* Label */}
                                    <motion.div
                                        initial={{ opacity: 0, y: 5 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: idx * 0.1 + 0.2 }}
                                        className={cn(
                                            "absolute -top-7 left-[-2px] px-2 py-1 text-xs font-bold uppercase tracking-wider text-slate-950 flex items-center gap-2 whitespace-nowrap",
                                            bgHeaderClass
                                        )}
                                    >
                                        <span>{det.class}</span>
                                        <span className="opacity-80 font-mono">
                                            {(det.confidence * 100).toFixed(0)}%
                                        </span>
                                    </motion.div>

                                    {/* Corner Accents for high-tech look */}
                                    <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-current" />
                                    <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-current" />
                                </motion.div>
                            );
                        })}
                </AnimatePresence>
            </div>
        </motion.div>
    );
};
