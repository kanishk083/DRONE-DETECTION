"use client";

import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

// UI Components
import { Dropzone } from "@/components/Dropzone";
import { DetectionCanvas } from "@/components/DetectionCanvas";
import { TelemetryPanel } from "@/components/TelemetryPanel";
import { Hero3D } from "@/components/Hero3D";
import { Navbar } from "@/components/landing/Navbar";
import { FeaturesGrid } from "@/components/landing/FeaturesGrid";

// Types
import { DetectionResponse } from "@/types";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [results, setResults] = useState<DetectionResponse | null>(null);

  const handleUpload = async (file: File) => {
    setIsLoading(true);

    const previewUrl = URL.createObjectURL(file);
    setImageUrl(previewUrl);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Sending image to FastAPI backend
      const response = await fetch("http://localhost:8000/predict?conf=0.10", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed. Backend returned status: " + response.status);
      }

      const data: DetectionResponse = await response.json();
      setResults(data);

    } catch (err) {
      console.error(err);
      setImageUrl(null);
      alert("Failed to reach classification backend. Ensure FastAPI server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setResults(null);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-slate-200 font-sans selection:bg-emerald-500/30 overflow-x-hidden">

      {/* 1. Global Navigation */}
      <Navbar />

      {/* 2. Interactive 3D Hero */}
      <Hero3D />

      {/* 3. The YOLO Analysis Dashboard */}
      <main id="analyze" className="max-w-7xl mx-auto px-6 py-24 relative z-20 scroll-mt-16">
        <AnimatePresence mode="wait">
          {!results ? (
            <motion.section
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col items-center justify-center pt-8"
            >
              <div className="text-center mb-10 w-full max-w-2xl">
                <h2 className="text-4xl font-extrabold tracking-tight text-white mb-4">
                  Target Acquisition Panel
                </h2>
                <p className="text-lg text-slate-400">
                  Upload visual data to initialize YOLO11n neural network detection.
                </p>
              </div>

              <div className="w-full max-w-4xl shadow-2xl rounded-2xl bg-[#0a0a0a] border border-slate-800 p-8">
                <Dropzone onFileSelect={handleUpload} isLoading={isLoading} />
              </div>
            </motion.section>

          ) : (
            <motion.section
              key="results"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="space-y-8"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold font-mono tracking-widest text-emerald-500 uppercase">
                  Telemetry Report
                </h2>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 rounded border border-slate-700 text-sm font-mono text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
                >
                  [ INITIALIZE NEW SCAN ]
                </button>
              </div>

              {/* Data Visualization Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">

                {/* Left Col: The Annotated Image Canvas */}
                <div className="col-span-1 lg:col-span-2 rounded-xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm relative">
                  {imageUrl && (
                    <DetectionCanvas
                      imageUrl={imageUrl}
                      detections={results.detections}
                      originalSize={results.image_size}
                    />
                  )}
                </div>

                {/* Right Col: Metrics Panel */}
                <div className="col-span-1 border border-slate-800 rounded-xl bg-slate-900/50 backdrop-blur-sm p-6 shadow-2xl">
                  <TelemetryPanel detections={results.detections} />
                </div>
              </div>

            </motion.section>
          )}
        </AnimatePresence>
      </main>

      {/* 4. Features/Context Grid */}
      <div id="features">
        <FeaturesGrid />
      </div>

      <footer className="relative z-20 border-t border-white/5 py-12 px-6 text-center text-sm font-mono text-slate-600 bg-[#050505]">
        <p>AERIAL OPS © {new Date().getFullYear()}. NEURAL DETECTION SYSTEM.</p>
      </footer>
    </div>
  );
}
