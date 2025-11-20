import React, { useRef, useState, useEffect } from "react";
import { inferImage } from "./api";
import CanvasOverlay from "./components/CanvasOverlay";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Camera, Upload, X, Zap, ScanLine, Aperture, Activity, 
  Sliders, Download, History, ChevronRight, Target, RefreshCw, Trash2, 
  Info, User, Cpu, Layers, ArrowRight, Rocket, Menu
} from "lucide-react";

// --- ABOUT COMPONENT (Mobile Optimized) ---
function AboutView({ onClose }) {
  const container = {
    hidden: { opacity: 0, y: 20 },
    show: {
      opacity: 1,
      y: 0,
      transition: { type: "spring", damping: 25, stiffness: 200, staggerChildren: 0.1 }
    },
    exit: { opacity: 0, y: 20 }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <motion.div 
      variants={container}
      initial="hidden"
      animate="show"
      exit="exit"
      className="fixed inset-0 z-[60] bg-[#050911] overflow-y-auto custom-scrollbar"
    >
      <div className="cyber-grid-bg" />
      
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-[#050911]/90 backdrop-blur-md border-b border-white/5 px-6 py-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white leading-tight">Architecture</h2>
          <p className="text-xs text-slate-400">System Design & Credits</p>
        </div>
        <button 
          onClick={onClose}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition active:scale-95"
        >
          <X size={20} className="text-white" />
        </button>
      </div>

      <div className="p-6 pb-24 space-y-6 max-w-2xl mx-auto">
        
        {/* 1. The Core Logic */}
        <motion.div variants={item} className="bg-[#0F131C] border border-white/10 rounded-2xl p-5 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 p-24 bg-indigo-500/5 rounded-full blur-3xl -mr-10 -mt-10" />
          <div className="flex items-center gap-3 mb-4 relative z-10">
            <div className="p-2.5 bg-indigo-500/10 rounded-xl text-indigo-400">
              <Cpu size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Core Logic</h3>
          </div>
          
          <div className="space-y-4 relative z-10">
            <div className="pl-4 border-l-2 border-indigo-500/30">
              <h4 className="text-indigo-300 text-sm font-semibold">Stage 1: The Finder</h4>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                Utilizes <span className="text-white font-mono">YOLOv8</span> to localize objects with high-speed bounding box regression.
              </p>
            </div>
            <div className="pl-4 border-l-2 border-emerald-500/30">
              <h4 className="text-emerald-300 text-sm font-semibold">Stage 2: The Thinker</h4>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                A custom <span className="text-white font-mono">MobileNetV2</span> classifier predicting 14 functional affordances from visual features.
              </p>
            </div>
          </div>
        </motion.div>

        {/* 2. The Architect */}
        <motion.div variants={item} className="bg-[#0F131C] border border-white/10 rounded-2xl p-5 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 p-24 bg-emerald-500/5 rounded-full blur-3xl -mr-10 -mt-10" />
          <div className="flex items-center gap-3 mb-4 relative z-10">
             <div className="p-2.5 bg-emerald-500/10 rounded-xl text-emerald-400">
              <User size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">The Architect</h3>
          </div>

          <div className="space-y-3 relative z-10">
            <div className="p-3 bg-white/5 rounded-xl border border-white/5">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Lead Developer</div>
              <div className="text-base font-bold text-white">{atob("Tml0aGluIE4=")}</div>
            </div>
             <div className="p-3 bg-white/5 rounded-xl border border-white/5 flex items-center gap-3">
                <div className="p-1.5 bg-indigo-500/20 rounded-full text-indigo-400"><Layers size={14}/></div>
                <div className="text-xs text-slate-300">Personal AI Research Initiative</div>
            </div>
          </div>
        </motion.div>

        {/* 3. Roadmap */}
        <motion.div variants={item} className="bg-gradient-to-br from-[#0F131C] to-[#131722] border border-white/10 rounded-2xl p-5 shadow-xl">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 bg-amber-500/10 rounded-xl text-amber-400">
              <Rocket size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Roadmap</h3>
          </div>

          <div className="space-y-3">
            {[
              { title: "Mobile Native PWA", desc: "Offline inference capabilities." },
              { title: "30FPS Stream", desc: "Real-time continuous analysis." },
              { title: "Scene Graphs", desc: "Contextual object relationships." }
            ].map((feat, i) => (
              <div key={i} className="p-3 rounded-xl border border-white/5 bg-black/20">
                <h4 className="font-bold text-slate-200 text-xs mb-1 flex items-center gap-2">
                  <ArrowRight size={12} className="text-amber-500" /> {feat.title}
                </h4>
                <p className="text-[10px] text-slate-500">{feat.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

      </div>
    </motion.div>
  );
}

// --- MAIN APP COMPONENT ---
export default function App() {
  const videoRef = useRef(null);
  const previewRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [status, setStatus] = useState("System Ready");
  
  const [minConfidence, setMinConfidence] = useState(15); 
  const [scanHistory, setScanHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [activeIdx, setActiveIdx] = useState(null);
  const [imgSize, setImgSize] = useState({ width: 1280, height: 720 });
  
  const [currentView, setCurrentView] = useState('scanner'); 

  // --- Camera Control ---

  async function startCamera() {
    setPreviewSrc(null);
    setDetections([]);
    setActiveIdx(null);
    
    try {
      const s = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } } 
      });
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.play();
        }
      }, 100);
      setStreaming(true);
      setStatus("Optical Feed Active");
    } catch (e) {
      setStatus("Error: Camera Blocked");
    }
  }

  function stopCamera() {
    const s = videoRef.current?.srcObject;
    if (s) { 
      s.getTracks().forEach(t => t.stop()); 
      videoRef.current.srcObject = null; 
    }
    setStreaming(false);
  }

  // --- Main Actions ---

  async function captureAndInfer() {
    if (!videoRef.current) return;
    setIsProcessing(true);
    setStatus("Acquiring Target...");
    setActiveIdx(null);

    const w = videoRef.current.videoWidth || 640;
    const h = videoRef.current.videoHeight || 480;
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const ctx = off.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, w, h);
    
    off.toBlob(async (blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      
      setPreviewSrc(url);
      stopCamera(); 
      
      try {
        const json = await inferImage(new File([blob], "capture.jpg", { type: blob.type }));
        handleSuccess(url, json);
      } catch (err) {
        setStatus("Inference Failed");
        setIsProcessing(false);
      }
    }, "image/jpeg", 0.9);
  }

  async function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    stopCamera(); 
    const url = URL.createObjectURL(file);
    setPreviewSrc(url);
    setDetections([]);
    setActiveIdx(null);
    setIsProcessing(true);
    setStatus("Processing File...");
    
    try {
      const json = await inferImage(file);
      handleSuccess(url, json);
    } catch (err) {
      setStatus("Error: " + err.message);
      setIsProcessing(false);
    }
  }

  function handleSuccess(imgUrl, json) {
    const objects = (json.objects || []).map(obj => ({
      x1: obj.bbox?.[0] || obj.box?.[0] || 0,
      y1: obj.bbox?.[1] || obj.box?.[1] || 0,
      x2: obj.bbox?.[2] || obj.box?.[2] || 0,
      y2: obj.bbox?.[3] || obj.box?.[3] || 0,
      score: obj.score ?? 0, 
      label: obj.label ?? obj.cls ?? "Object",
      affordances: obj.affordances ?? {}
    }));

    setPreviewSrc(imgUrl);
    setDetections(objects);
    setIsProcessing(false);
    setStatus(`Analysis Complete: ${objects.length} Targets`);

    const newItem = { 
      id: Date.now(), 
      img: imgUrl, 
      count: objects.length, 
      detections: objects, 
      date: new Date().toLocaleTimeString() 
    };
    setScanHistory(prev => [newItem, ...prev].slice(0, 20)); 
  }

  function loadHistoryItem(item) {
    stopCamera();
    setPreviewSrc(item.img);
    setDetections(item.detections || []); 
    setActiveIdx(null);
    setStatus(`Reviewing Archive: ${item.count} Targets`);
    setShowHistory(false);
  }

  function deleteHistoryItem(e, id) {
    e.stopPropagation(); 
    setScanHistory(prev => prev.filter(item => item.id !== id));
  }

  function downloadJSON() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(detections, null, 2));
    const anchor = document.createElement('a');
    anchor.href = dataStr;
    anchor.download = "affordance_scan.json";
    anchor.click();
  }

  const onImageLoad = (e) => {
    setImgSize({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight
    });
  };

  return (
    <div className="fixed inset-0 bg-[#050911] text-slate-200 font-sans selection:bg-indigo-500/30 overflow-hidden flex flex-col">
      <div className="cyber-grid-bg" /> 

      {/* --- Navbar --- */}
      <nav className="border-b border-white/5 bg-[#050911]/90 backdrop-blur-md z-50 shrink-0">
        <div className="max-w-8xl mx-auto px-4 h-14 flex items-center justify-between">
          <div 
            className="flex items-center gap-2 cursor-pointer" 
            onClick={() => setCurrentView('scanner')}
          >
            <div className="p-1.5 bg-gradient-to-br from-indigo-600 to-violet-700 rounded-lg shadow-lg shadow-indigo-500/20">
              <ScanLine size={18} className="text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white leading-none">Affordance<span className="text-indigo-400">Net</span></h1>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/5">
              <Activity size={12} className={isProcessing ? "text-emerald-400 animate-pulse" : "text-slate-600"} />
              <span className="text-[10px] font-mono text-indigo-200">{status}</span>
            </div>
            
            {/* About Button */}
            <button 
              onClick={() => setCurrentView(currentView === 'about' ? 'scanner' : 'about')}
              className={`p-2 rounded-lg transition ${currentView === 'about' ? 'bg-indigo-500/20 text-indigo-400' : 'hover:bg-white/5 text-slate-400'}`}
              title="About Project"
            >
              <Info size={20} />
            </button>

            {/* History Button */}
            <button 
              onClick={() => setShowHistory(!showHistory)}
              className={`p-2 rounded-lg transition ${showHistory ? 'bg-indigo-500/20 text-indigo-400' : 'hover:bg-white/5 text-slate-400'}`}
              title="Scan History"
            >
              <History size={20} />
            </button>
          </div>
        </div>
      </nav>

      {/* --- Main Content Area --- */}
      <main className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
        
        {/* CONDITIONAL VIEW RENDERING */}
        <AnimatePresence mode="wait">
          
          {currentView === 'about' ? (
            <AboutView key="about" onClose={() => setCurrentView('scanner')} />
          ) : (
            // --- SCANNER VIEW (Original) ---
            <>
            {/* --- LEFT COLUMN: Viewport --- */}
            <div className="flex-1 flex flex-col min-h-0 relative lg:p-6 p-0">
              
              <div className="flex-1 relative bg-[#0F131C] lg:rounded-2xl overflow-hidden border-b lg:border border-white/10 shadow-2xl group">
                
                {/* VIDEO */}
                <video 
                  ref={videoRef} 
                  className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${streaming && !previewSrc ? 'opacity-100' : 'opacity-0'}`} 
                  playsInline 
                />

                {/* RESULT */}
                {previewSrc && (
                  <div className="absolute inset-0 z-10 bg-[#0F131C]">
                    <img 
                      ref={previewRef} 
                      src={previewSrc} 
                      alt="preview" 
                      onLoad={onImageLoad}
                      className="w-full h-full object-contain opacity-90" 
                    />
                    <div className="absolute inset-0">
                      <CanvasOverlay width={imgSize.width} height={imgSize.height} detections={detections} activeIndex={activeIdx} />
                    </div>
                  </div>
                )}

                {/* EMPTY STATE */}
                {!streaming && !previewSrc && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-600 gap-4 p-6 text-center">
                    <div className="p-4 rounded-full bg-white/5 border border-white/5">
                      <Aperture size={32} />
                    </div>
                    <p className="text-sm font-medium">Initialize Sensor Array</p>
                  </div>
                )}

                {/* LOADING */}
                {isProcessing && (
                  <div className="absolute inset-0 z-20 bg-indigo-500/5 backdrop-blur-[2px]">
                     <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                        <div className="w-16 h-16 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
                     </div>
                     <motion.div 
                      initial={{ top: "0%" }} animate={{ top: "100%" }} 
                      transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                      className="absolute left-0 right-0 h-0.5 bg-indigo-400 shadow-[0_0_40px_rgba(99,102,241,1)]"
                    />
                  </div>
                )}

                {/* CONTROL BAR */}
                <div className="absolute bottom-6 left-0 right-0 flex justify-center z-30 px-4 pointer-events-none">
                  <div className="flex items-center gap-3 p-2 bg-[#050911]/90 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl pointer-events-auto">
                    
                    {previewSrc ? (
                      <button 
                        onClick={startCamera} 
                        className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 px-6 py-3 rounded-xl font-semibold text-white transition shadow-lg shadow-emerald-500/20 active:scale-95"
                      >
                        <RefreshCw size={18} /> <span className="hidden sm:inline">Scan Again</span>
                      </button>
                    ) : 
                    !streaming ? (
                      <button onClick={startCamera} className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-6 py-3 rounded-xl font-semibold text-white transition shadow-lg shadow-indigo-500/20 active:scale-95">
                        <Camera size={18} /> Activate
                      </button>
                    ) : (
                       <>
                        <button onClick={captureAndInfer} disabled={isProcessing} className="bg-white text-black w-14 h-14 rounded-full flex items-center justify-center hover:scale-105 active:scale-95 transition shadow-lg disabled:opacity-50">
                          <div className="w-4 h-4 bg-red-600 rounded-full animate-pulse" />
                        </button>
                        <button onClick={stopCamera} className="w-12 h-12 flex items-center justify-center bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-xl font-medium transition border border-red-500/20 active:scale-95">
                          <X size={20} />
                        </button>
                       </>
                    )}

                    <div className="w-px h-8 bg-white/10 mx-1" />
                    
                    <label className="cursor-pointer p-3 hover:bg-white/10 rounded-xl transition text-slate-400 hover:text-white active:bg-white/20">
                      <Upload size={20} />
                      <input type="file" accept="image/*" onChange={onUpload} className="hidden" />
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* --- RIGHT COLUMN: Analysis --- */}
            <div className="w-full lg:w-96 flex flex-col gap-4 p-4 lg:p-6 lg:pl-0 h-[40vh] lg:h-auto border-t lg:border-t-0 border-white/10 bg-[#0B0F17] lg:bg-transparent z-10">
              
              <div className="bg-[#0F131C] rounded-2xl border border-white/10 p-5 flex-1 overflow-hidden flex flex-col shadow-xl">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Zap size={14} className="text-indigo-400" /> Data
                  </h3>
                  {detections.length > 0 && (
                    <button onClick={downloadJSON} className="text-xs flex items-center gap-1 text-slate-400 hover:text-white transition">
                      <Download size={12} /> Save
                    </button>
                  )}
                </div>

                <div className="flex-1 overflow-y-auto pr-1 space-y-3 custom-scrollbar">
                  <AnimatePresence mode="popLayout">
                    {detections.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-600 text-sm opacity-50">
                        <Activity size={24} className="mb-2" />
                        <span className="text-xs">Awaiting Input</span>
                      </div>
                    ) : (
                      detections.map((d, i) => (
                        <motion.div
                          key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                          onClick={() => setActiveIdx(activeIdx === i ? null : i)} 
                          className={`rounded-xl p-3 border cursor-pointer transition-all group ${
                            activeIdx === i 
                              ? "bg-[#181E2E] border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.1)]" 
                              : "bg-[#131722] border-white/5 hover:border-indigo-500/30"
                          }`}
                        >
                          <div className="flex justify-between items-start mb-2">
                            <div className="flex items-center gap-2">
                              {activeIdx === i && <Target size={12} className="text-emerald-400 animate-pulse" />}
                              <span className={`text-sm font-bold capitalize ${activeIdx === i ? "text-emerald-300" : "text-slate-100"}`}>
                                {d.label}
                              </span>
                            </div>
                            <span className="text-[10px] font-mono text-slate-400 bg-black/20 px-1.5 py-0.5 rounded">
                              {Math.round(d.score * 100)}%
                            </span>
                          </div>

                          <div className="space-y-1.5">
                             {d.affordances && Object.keys(d.affordances).length > 0 ? (
                               Object.entries(d.affordances)
                                .filter(([k, v]) => v > (minConfidence / 100)) 
                                .sort((a, b) => b[1] - a[1])
                                .map(([k, v]) => (
                                  <div key={k} className="text-[10px] sm:text-xs">
                                    <div className="flex justify-between mb-0.5">
                                      <span className={v > 0.5 ? "text-slate-200" : "text-slate-400"}>{k}</span>
                                      <span className="font-mono text-slate-500">{Math.round(v * 100)}%</span>
                                    </div>
                                    <div className="h-1 bg-black rounded-full overflow-hidden">
                                      <motion.div 
                                        initial={{ width: 0 }} animate={{ width: `${v * 100}%` }}
                                        transition={{ duration: 0.8 }}
                                        className={`h-full rounded-full ${v > 0.5 ? 'bg-emerald-400' : 'bg-slate-600'}`} 
                                      />
                                    </div>
                                  </div>
                                ))
                             ) : (
                               <div className="text-[10px] text-slate-600 italic p-1 text-center">Low confidence data</div>
                             )}
                          </div>
                        </motion.div>
                      ))
                    )}
                  </AnimatePresence>
                </div>

                <div className="mt-4 pt-4 border-t border-white/5">
                  <div className="flex justify-between text-xs text-slate-400 mb-2">
                    <span className="flex items-center gap-1"><Sliders size={12} /> Threshold</span>
                    <span className="font-mono text-indigo-400">{minConfidence}%</span>
                  </div>
                  <input 
                    type="range" min="0" max="100" value={minConfidence} 
                    onChange={(e) => setMinConfidence(e.target.value)}
                    className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
            </div>
            </>
          )}

        </AnimatePresence>

        {/* --- History DRAWER --- */}
        <AnimatePresence>
          {showHistory && (
            <>
              <motion.div 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                onClick={() => setShowHistory(false)}
                className="absolute inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
              />
              
              <motion.div 
                initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
                transition={{ type: "spring", damping: 25, stiffness: 200 }}
                className="absolute top-0 right-0 bottom-0 w-72 bg-[#050911] border-l border-white/10 z-50 shadow-2xl flex flex-col"
              >
                <div className="p-4 border-b border-white/10 flex items-center justify-between bg-[#050911]">
                   <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                      <History size={14} /> Session History
                   </h4>
                   <button onClick={() => setShowHistory(false)} className="text-slate-500 hover:text-white"><X size={18} /></button>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
                  {scanHistory.length === 0 ? (
                    <div className="text-xs text-slate-700 italic text-center py-10">No recent scans</div>
                  ) : (
                    scanHistory.map((item) => (
                      <div 
                        key={item.id} 
                        onClick={() => loadHistoryItem(item)}
                        className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 transition text-left group cursor-pointer relative border border-transparent hover:border-white/5"
                      >
                        <img src={item.img} alt="scan" className="w-10 h-10 rounded object-cover bg-black border border-white/5" />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium text-slate-300">{item.count} Objects</div>
                          <div className="text-[10px] text-slate-600">{item.date}</div>
                        </div>
                        <button 
                          onClick={(e) => deleteHistoryItem(e, item.id)}
                          className="p-2 text-slate-600 hover:text-red-400 hover:bg-red-500/10 rounded transition"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}