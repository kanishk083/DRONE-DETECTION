"use client";

import React, { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileImage, ShieldAlert, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropzoneProps {
    onFileSelect: (file: File) => void;
    isLoading: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({
    onFileSelect,
    isLoading,
}) => {
    const [isDragActive, setIsDragActive] = useState(false);
    const [errorDetails, setErrorDetails] = useState<string | null>(null);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setIsDragActive(true);
        } else if (e.type === "dragleave") {
            setIsDragActive(false);
        }
    }, []);

    const validateAndProcessFile = (file: File) => {
        setErrorDetails(null);
        if (!file.type.startsWith("image/")) {
            setErrorDetails("Invalid file format. Please upload an image.");
            return;
        }
        // Limit to 10MB
        if (file.size > 10 * 1024 * 1024) {
            setErrorDetails("File too large. Maximum size is 10MB.");
            return;
        }
        onFileSelect(file);
    };

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragActive(false);

            if (isLoading) return;

            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                validateAndProcessFile(e.dataTransfer.files[0]);
            }
        },
        [isLoading, onFileSelect]
    );

    const handleChange = function (e: React.ChangeEvent<HTMLInputElement>) {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            validateAndProcessFile(e.target.files[0]);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto">
            <motion.div
                className={cn(
                    "relative flex flex-col items-center justify-center w-full h-80 rounded-2xl border-2 border-dashed transition-colors duration-300 overflow-hidden group cursor-pointer",
                    isDragActive
                        ? "border-emerald-500 bg-emerald-500/10"
                        : "border-slate-700 bg-slate-900 hover:bg-slate-800 hover:border-slate-500",
                    isLoading && "opacity-60 pointer-events-none"
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                animate={{
                    scale: isDragActive ? 1.02 : 1,
                }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
            >
                <input
                    type="file"
                    accept="image/*"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    onChange={handleChange}
                    disabled={isLoading}
                />

                <AnimatePresence mode="wait">
                    {isLoading ? (
                        <motion.div
                            key="loading"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            className="flex flex-col items-center text-emerald-500"
                        >
                            <Loader2 className="w-16 h-16 animate-spin mb-4" />
                            <p className="text-lg font-medium tracking-wide">
                                ANALYZING PAYLOAD...
                            </p>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center px-6 text-center"
                        >
                            <motion.div
                                animate={{
                                    y: [0, -8, 0],
                                }}
                                transition={{
                                    repeat: Infinity,
                                    duration: 4,
                                    ease: "easeInOut",
                                }}
                                className="mb-6 p-4 rounded-full bg-slate-800/50 shadow-inner group-hover:bg-slate-700/50 transition-colors"
                            >
                                <UploadCloud className="w-12 h-12 text-slate-400 group-hover:text-emerald-400 transition-colors" />
                            </motion.div>
                            <p className="text-xl font-semibold text-slate-200 mb-2">
                                Drop telemetry image here
                            </p>
                            <p className="text-sm text-slate-400">
                                Supports JPG, PNG, AVIF up to 10MB
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Scan line effect when loading */}
                {isLoading && (
                    <motion.div
                        className="absolute top-0 left-0 right-0 h-1 bg-emerald-500 shadow-[0_0_20px_rgba(16,185,129,1)]"
                        animate={{
                            y: ["0%", "32000%"], // Need a large % because container height is 320px
                        }}
                        transition={{
                            repeat: Infinity,
                            duration: 2,
                            ease: "linear",
                        }}
                    />
                )}
            </motion.div>

            <AnimatePresence>
                {errorDetails && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="mt-4 p-4 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-3 text-red-500"
                    >
                        <ShieldAlert className="w-5 h-5 flex-shrink-0" />
                        <p className="text-sm font-medium">{errorDetails}</p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
