import React, { useRef, useState, useEffect } from "react";
import { inferImage } from "./api";
import CanvasOverlay from "./components/CanvasOverlay";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Camera, Upload, X, Zap, ScanLine, Aperture, Activity, 
  Sliders, Download, History, ChevronRight, Target, RefreshCw, Trash2 
} from "lucide-react";

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
  const [showHistory, setShowHistory] = useState(true);
  const [activeIdx, setActiveIdx] = useState(null);
  const [imgSize, setImgSize] = useState({ width: 1280, height: 720 }); 

  // --- Camera Control ---

  async function startCamera() {
    setPreviewSrc(null);
    setDetections([]);
    setActiveIdx(null);
    
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
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
      
      // --- FIX 1: Set Preview IMMEDIATELY ---
      // This ensures the image is visible while the scanner runs
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

    // previewSrc is already set in captureAndInfer, but we set it again here for uploads
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

  useEffect(() => {
    if (previewRef.current) {
      previewRef.current.width = previewRef.current.clientWidth;
      previewRef.current.height = previewRef.current.clientHeight;
    }
  }, [previewSrc]);

  return (
    <div className="min-h-screen text-slate-200 font-sans selection:bg-indigo-500/30 relative">
      <div className="cyber-grid-bg" /> 

      {/* --- Navbar --- */}
      <nav className="border-b border-white/5 bg-[#050911]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-indigo-600 to-violet-700 rounded-lg shadow-lg shadow-indigo-500/20">
              <ScanLine size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white leading-none">Affordance<span className="text-indigo-400">Net</span></h1>
              <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Pro Workstation</span>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/5">
              <Activity size={14} className={isProcessing ? "text-emerald-400 animate-pulse" : "text-slate-600"} />
              <span className="text-xs font-mono text-indigo-200">{status}</span>
            </div>
            <button 
              onClick={() => setShowHistory(!showHistory)}
              className={`p-2 rounded-lg transition ${showHistory ? 'bg-indigo-500/20 text-indigo-400' : 'hover:bg-white/5 text-slate-400'}`}
            >
              <History size={20} />
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-8xl mx-auto p-6 flex flex-col lg:flex-row gap-6 h-[calc(100vh-80px)]">
        
        {/* --- LEFT COLUMN: Viewport --- */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          
          <div className="flex-1 relative bg-[#0F131C] rounded-2xl overflow-hidden border border-white/10 shadow-2xl group">
            <video ref={videoRef} className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${streaming && !previewSrc ? 'opacity-100' : 'opacity-0'}`} />

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

            {!streaming && !previewSrc && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-600 gap-4">
                <div className="p-4 rounded-full bg-white/5 border border-white/5">
                  <Aperture size={32} />
                </div>
                <p className="text-sm font-medium">Initialize Sensor Array</p>
              </div>
            )}

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

            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 p-2 bg-[#050911]/90 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl z-30">
              
              {previewSrc ? (
                <button 
                  onClick={startCamera} 
                  className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 px-6 py-3 rounded-xl font-semibold text-white transition shadow-lg shadow-emerald-500/20"
                >
                  <RefreshCw size={18} /> Scan Again
                </button>
              ) : 
              !streaming ? (
                <button onClick={startCamera} className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-6 py-3 rounded-xl font-semibold text-white transition shadow-lg shadow-indigo-500/20">
                  <Camera size={18} /> Activate Cam
                </button>
              ) : (
                 <>
                  <button onClick={captureAndInfer} disabled={isProcessing} className="bg-white text-black p-4 rounded-xl hover:scale-105 active:scale-95 transition shadow-lg disabled:opacity-50">
                    <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse" />
                  </button>
                  <button onClick={stopCamera} className="flex items-center gap-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 px-4 py-3 rounded-xl font-medium transition border border-red-500/20">
                    <X size={18} />
                  </button>
                 </>
              )}

              <div className="w-px h-8 bg-white/10 mx-1" />
              
              <label className="cursor-pointer p-3 hover:bg-white/10 rounded-xl transition text-slate-400 hover:text-white">
                <Upload size={20} />
                <input type="file" accept="image/*" onChange={onUpload} className="hidden" />
              </label>
            </div>
          </div>
        </div>

        {/* --- RIGHT COLUMN: Analysis & History --- */}
        <div className="w-full lg:w-96 flex flex-col gap-4">
          
          <div className="bg-[#0F131C] rounded-2xl border border-white/10 p-5 flex-1 overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Zap size={16} className="text-indigo-400" /> Analysis Data
              </h3>
              {detections.length > 0 && (
                <button onClick={downloadJSON} className="text-xs flex items-center gap-1 text-slate-400 hover:text-white transition">
                  <Download size={12} /> Export
                </button>
              )}
            </div>

            <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
              <AnimatePresence mode="popLayout">
                {detections.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-600 text-sm opacity-50">
                    <Activity size={32} className="mb-2" />
                    No Active Signatures
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
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-2">
                          {activeIdx === i && <Target size={14} className="text-emerald-400 animate-pulse" />}
                          <span className={`font-bold capitalize ${activeIdx === i ? "text-emerald-300" : "text-slate-100"}`}>
                            {d.label}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-slate-400 bg-black/20 px-1.5 py-0.5 rounded">
                          {Math.round(d.score * 100)}%
                        </span>
                      </div>

                      <div className="space-y-2">
                         {d.affordances && Object.keys(d.affordances).length > 0 ? (
                           Object.entries(d.affordances)
                            .filter(([k, v]) => v > (minConfidence / 100)) 
                            .sort((a, b) => b[1] - a[1])
                            .map(([k, v]) => (
                              <div key={k} className="text-xs">
                                <div className="flex justify-between mb-1">
                                  <span className={v > 0.5 ? "text-white" : "text-slate-400"}>{k}</span>
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
                           <div className="text-[10px] text-slate-600 italic p-2 text-center">No affordances found above threshold</div>
                         )}
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>

            <div className="mt-4 pt-4 border-t border-white/5">
              <div className="flex justify-between text-xs text-slate-400 mb-2">
                <span className="flex items-center gap-1"><Sliders size={12} /> Confidence Threshold</span>
                <span className="font-mono text-indigo-400">{minConfidence}%</span>
              </div>
              <input 
                type="range" min="0" max="100" value={minConfidence} 
                onChange={(e) => setMinConfidence(e.target.value)}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          {showHistory && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
              className="bg-[#0F131C] rounded-2xl border border-white/10 p-4 max-h-48 overflow-y-auto custom-scrollbar"
            >
              <h4 className="text-xs font-bold text-slate-500 uppercase mb-3 flex items-center gap-2">
                <History size={12} /> Session History
              </h4>
              <div className="space-y-2">
                {scanHistory.length === 0 ? (
                  <div className="text-xs text-slate-700 italic text-center py-4">No recent scans</div>
                ) : (
                  scanHistory.map((item) => (
                    <div 
                      key={item.id} 
                      onClick={() => loadHistoryItem(item)}
                      className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 transition text-left group cursor-pointer relative"
                    >
                      <img src={item.img} alt="scan" className="w-10 h-10 rounded object-cover bg-black border border-white/5 group-hover:border-indigo-500/50 transition" />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium text-slate-300">{item.count} Objects Found</div>
                        <div className="text-[10px] text-slate-600">{item.date}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={(e) => deleteHistoryItem(e, item.id)}
                          className="p-1.5 rounded-md text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition opacity-0 group-hover:opacity-100"
                        >
                          <Trash2 size={14} />
                        </button>
                        <ChevronRight size={12} className="text-slate-700 group-hover:text-white transition" />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </motion.div>
          )}
        </div>
      </main>
    </div>
  );
}