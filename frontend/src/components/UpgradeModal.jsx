import React, { useState } from 'react';
import { upgradeImage } from '../services/api';

export default function UpgradeModal({ creative, isOpen, onClose, onApply }) {
  const [isUpgrading, setIsUpgrading] = useState(false);
  const [upgradedData, setUpgradedData] = useState(null);

  React.useEffect(() => {
    if (isOpen) {
      setUpgradedData(null);
      setIsUpgrading(false);
    }
  }, [isOpen, creative?.id]);

  if (!isOpen || !creative) return null;

  const handleUpgrade = async () => {
    setIsUpgrading(true);
    const result = await upgradeImage(creative.id);
    setUpgradedData(result);
    setIsUpgrading(false);
  };

  return (
    <div className="fixed inset-0 bg-slate-900/90 backdrop-blur-xl z-[100] flex items-center justify-center p-0 md:p-6 animate-in fade-in duration-300">
      <div className="bg-white md:rounded-[3rem] shadow-2xl w-full h-full md:h-auto md:max-w-6xl md:max-h-[90vh] overflow-y-auto flex flex-col md:flex-row relative border border-white/20">
        
        {/* Close Button */}
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 md:top-8 md:right-8 w-10 h-10 md:w-12 md:h-12 bg-slate-100 hover:bg-slate-200 rounded-xl md:rounded-2xl flex items-center justify-center text-slate-500 transition-all z-20"
        >
          <span className="text-xl md:text-2xl">✕</span>
        </button>

        {/* Left: Original Asset */}
        <div className="w-full md:flex-1 bg-slate-50 p-6 md:p-10 flex flex-col justify-center border-b md:border-b-0 md:border-r border-slate-100">
          <div className="mb-6">
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-indigo-500 mb-2 block">Source Asset</span>
            <h3 className="text-2xl md:text-3xl font-black tracking-tighter uppercase italic">{creative.subject}</h3>
          </div>
          <div className="relative group rounded-[1.5rem] md:rounded-[2rem] overflow-hidden shadow-2xl shadow-slate-300 bg-slate-900 flex items-center justify-center">
            <img 
              src={creative.image_url} 
              alt="Original" 
              className="w-full h-auto object-contain aspect-[9/16] max-h-[40vh] md:max-h-[55vh]"
            />
            <div className="absolute inset-0 bg-red-500/10 mix-blend-overlay pointer-events-none"></div>
          </div>
          <div className="mt-8 space-y-4">
            <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-slate-400 border-b border-slate-200 pb-2">
              <span>Performance</span>
              <span className="text-red-500">Poor (Fatigued)</span>
            </div>
            <p className="text-xs md:text-sm text-slate-500 leading-relaxed font-medium">
              "{creative.insights}"
            </p>
          </div>
        </div>

        {/* Right: AI Upgrade Slot */}
        <div className="w-full md:flex-1 p-6 md:p-10 flex flex-col bg-white">
          <div className="mb-6 md:mb-8">
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-500 mb-2 block">PixelForge Diffusion V4</span>
            <h3 className="text-2xl md:text-3xl font-black tracking-tighter uppercase italic">AI Reconstruction</h3>
          </div>

          <div className="flex-1 flex flex-col items-center justify-center border-4 border-dashed border-slate-100 rounded-[1.5rem] md:rounded-[2rem] bg-slate-50/50 p-6 md:p-8 relative min-h-[300px]">
            {!isUpgrading && !upgradedData && (
              <div className="text-center group">
                <div className="text-5xl md:text-7xl mb-6 group-hover:scale-110 transition-transform duration-500">✨</div>
                <h4 className="text-base md:text-lg font-black text-slate-800 uppercase tracking-tighter mb-2">Ready for optimization</h4>
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-8">Estimated compute: 3.2s</p>
                <button 
                  onClick={handleUpgrade}
                  className="bg-indigo-600 hover:bg-emerald-500 text-white font-black px-8 md:px-12 py-4 md:py-5 rounded-xl md:rounded-[1.5rem] shadow-xl shadow-indigo-200 hover:shadow-emerald-200 transition-all active:scale-95 uppercase tracking-widest text-[10px] flex items-center gap-3 mx-auto"
                >
                  Start AI Generation
                </button>
              </div>
            )}

            {isUpgrading && (
              <div className="text-center">
                <div className="w-12 h-12 md:w-20 md:h-20 border-4 md:border-8 border-indigo-100 border-t-indigo-600 rounded-full animate-spin mx-auto mb-6 md:mb-8"></div>
                <h4 className="text-lg md:text-xl font-black text-slate-800 italic animate-pulse tracking-tighter">RECONSTRUCTING ASSET...</h4>
                <p className="text-[9px] text-indigo-400 font-bold uppercase tracking-[0.3em] mt-4">Analyzing performance clusters</p>
              </div>
            )}

            {upgradedData && (
              <div className="animate-in zoom-in duration-700 w-full h-full flex flex-col">
                <div className="relative rounded-[1.5rem] md:rounded-[2rem] overflow-hidden shadow-2xl shadow-emerald-200 mb-6 md:mb-8 border-4 border-emerald-500 bg-slate-900 flex items-center justify-center">
                  <img 
                    src={upgradedData.newImageUrl} 
                    alt="Upgraded" 
                    className="w-full h-auto object-contain aspect-[9/16] max-h-[40vh] md:max-h-[55vh]"
                  />
                  <div className="absolute top-4 left-4 bg-emerald-500 text-white text-[10px] px-3 py-1 rounded-full font-black uppercase tracking-widest shadow-lg">
                    Optimized
                  </div>
                </div>
                <div className="bg-emerald-50/50 p-4 md:p-6 rounded-2xl md:rounded-3xl border border-emerald-100">
                  <h5 className="text-[10px] font-black text-emerald-600 uppercase tracking-widest mb-2 md:mb-3 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span>
                    AI Logic Engine
                  </h5>
                  <p className="text-[11px] md:text-sm text-slate-600 font-medium leading-relaxed italic">
                    "{upgradedData.aiReasoning}"
                  </p>
                </div>
                
                <div className="mt-6 flex flex-col md:flex-row gap-3">
                  <button 
                    onClick={() => onApply(creative.id, upgradedData.newImageUrl)}
                    className="flex-1 bg-emerald-600 hover:bg-indigo-600 text-white font-black py-4 rounded-xl md:rounded-2xl shadow-xl shadow-emerald-100 transition-all active:scale-95 uppercase tracking-widest text-[10px]"
                  >
                    Replace Image
                  </button>
                  <button 
                    onClick={onClose}
                    className="flex-1 md:flex-none px-6 py-4 border border-slate-200 text-slate-400 font-black hover:bg-slate-50 rounded-xl md:rounded-2xl transition-all uppercase tracking-widest text-[10px]"
                  >
                    Keep Original
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
