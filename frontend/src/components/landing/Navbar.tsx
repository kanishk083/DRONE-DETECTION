"use client";

import React from "react";
import { HardDrive } from "lucide-react";

export const Navbar = () => {
    return (
        <header className="fixed top-0 w-full z-50 border-b border-white/5 bg-black/40 backdrop-blur-md">
            <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                <div className="flex items-center gap-3 text-emerald-500">
                    <HardDrive className="w-6 h-6" />
                    <h1 className="font-mono font-bold text-lg tracking-widest text-white">
                        AERIAL<span className="text-emerald-500">.OPS</span>
                    </h1>
                </div>

                <nav className="hidden md:flex items-center gap-8 text-xs font-mono tracking-widest uppercase text-slate-400">
                    <a href="#home" className="hover:text-emerald-400 transition-colors">Home</a>
                    <a href="#analyze" className="hover:text-emerald-400 transition-colors">Dashboard</a>
                    <a href="#features" className="hover:text-emerald-400 transition-colors">System Capabilities</a>
                </nav>

                <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
                    <span className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        SECURE LINK
                    </span>
                </div>
            </div>
        </header>
    );
};
