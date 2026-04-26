import React from 'react';

const CreativeCard = ({ creative, onUpgrade, onChat }) => {
  return (
    <div className={`creative-card group transition-all duration-500 ${creative.fatigued ? 'ring-2 ring-red-500/20' : ''}`}>
      <div className="relative overflow-hidden aspect-[9/16] bg-slate-900 flex items-center justify-center">
        <img 
          src={creative.image_url} 
          alt={creative.advertiser} 
          className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-700"
        />
        {creative.is_upgraded && (
          <div className="absolute top-4 right-4 bg-emerald-500 text-white text-[10px] px-2.5 py-1 rounded-full font-bold uppercase tracking-widest shadow-lg z-10">
            ✨ Upgraded
          </div>
        )}
        {creative.fatigued && (
          <div className="absolute top-4 left-4 flex items-center gap-1.5 bg-red-600/90 backdrop-blur-sm text-white text-[10px] px-2.5 py-1 rounded-full font-bold uppercase tracking-widest shadow-lg">
            <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></span>
            Performance Fatigue
          </div>
        )}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
          <div className="flex justify-between items-end">
            <div>
              <p className="text-[10px] text-white/70 uppercase font-black tracking-widest mb-1">{creative.advertiser}</p>
              <h3 className="text-white font-bold text-lg leading-tight">{creative.subject}</h3>
            </div>
            <div className="text-right">
              <span className="block text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-1">Score: {creative.performance_score}</span>
              <span className="block text-xl font-black text-green-400">{creative.ctr}% <span className="text-[10px] uppercase text-white/50 block">CTR</span></span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="p-5">
        <div className="flex flex-wrap gap-2 mb-4">
          <div className="px-3 py-1 bg-slate-100 rounded-full text-[10px] font-bold text-slate-500 uppercase tracking-tighter">
            {creative.format}
          </div>
          <div className="px-3 py-1 bg-indigo-50 rounded-full text-[10px] font-bold text-indigo-500 uppercase tracking-tighter">
            {creative.theme}
          </div>
          <div className="px-3 py-1 bg-emerald-50 rounded-full text-[10px] font-bold text-emerald-500 uppercase tracking-tighter">
             {creative.hook_type}
          </div>
        </div>

        <div className="bg-slate-50 border-l-4 border-slate-200 p-3 rounded-r-xl text-xs text-slate-600 mb-6 font-medium leading-relaxed italic">
           "{creative.insights}"
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <button
            onClick={() => onChat(creative)}
            className="w-full py-3 bg-white hover:bg-indigo-50 text-indigo-600 border border-indigo-100 rounded-2xl font-black text-[10px] uppercase tracking-widest transition-all shadow-lg shadow-indigo-100/50 hover:-translate-y-0.5"
          >
            Talk with Creative
          </button>
          <button 
            onClick={() => onUpgrade(creative)}
            className="w-full py-3 bg-slate-900 hover:bg-indigo-600 text-white rounded-2xl font-black text-[10px] uppercase tracking-widest transition-all shadow-xl shadow-slate-200 hover:shadow-indigo-200 hover:-translate-y-1 flex items-center justify-center gap-2 group"
          >
            <span className="text-lg group-hover:rotate-12 transition-transform">🪄</span> Upgrade AI
          </button>
        </div>
      </div>
    </div>
  );
};

export default CreativeCard;
