import React, { useState, useEffect } from 'react';
import { fetchCreatives, updateCreativeImage } from '../../services/api';
import CreativeCard from '../../components/CreativeCard';
import UpgradeModal from '../../components/UpgradeModal';
import './Dashboard.css';

function Dashboard() {
  const [creatives, setCreatives] = useState([]);
  const [allCreatives, setAllCreatives] = useState([]); // Store all for the advertiser to derive dynamic filters
  const [loading, setLoading] = useState(true);
  const [advertiser, setAdvertiser] = useState(null);
  const [filters, setFilters] = useState({ 
    format: 'All', 
    theme: 'All',
    hook_type: 'All'
  });
  const [sortOrder, setSortOrder] = useState('score'); // 'default', 'fatigue', 'score', 'ctr'
  const [selectedCreative, setSelectedCreative] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleApplyUpgrade = (id, newUrl) => {
    // 1. Update backend (mock data store)
    updateCreativeImage(id, newUrl);
    // 2. Refresh local state
    setCreatives(prev => prev.map(c => 
      c.id === id ? { ...c, image_url: newUrl, is_upgraded: true } : c
    ));
    setAllCreatives(prev => prev.map(c => 
      c.id === id ? { ...c, image_url: newUrl, is_upgraded: true } : c
    ));
    setIsModalOpen(false);
  };

  useEffect(() => {
    if (advertiser) {
      loadInitialData();
    } else {
      setAllCreatives([]);
      setCreatives([]);
    }
  }, [advertiser]);

  useEffect(() => {
    if (advertiser) {
      applyFilters();
    }
  }, [filters, allCreatives, sortOrder]);

  const loadInitialData = async () => {
    setLoading(true);
    // Fetch all for this advertiser once
    const data = await fetchCreatives({ advertiser });
    setAllCreatives(data);
    setLoading(false);
  };

  const applyFilters = () => {
    let filtered = [...allCreatives];
    if (filters.format !== 'All') {
      filtered = filtered.filter(c => c.format === filters.format);
    }
    if (filters.theme !== 'All') {
      filtered = filtered.filter(c => c.theme === filters.theme);
    }
    if (filters.hook_type !== 'All') {
      filtered = filtered.filter(c => c.hook_type === filters.hook_type);
    }

    // Apply Sorting
    if (sortOrder === 'fatigue') {
      filtered.sort((a, b) => (b.fatigued === a.fatigued) ? 0 : b.fatigued ? 1 : -1);
    } else if (sortOrder === 'score') {
      filtered.sort((a, b) => b.performance_score - a.performance_score);
    } else if (sortOrder === 'ctr') {
      filtered.sort((a, b) => b.ctr - a.ctr);
    }
    
    setCreatives(filtered);
  };

  // Derive dynamic filter options based on the advertiser's available data
  const dynamicFormats = ['All', ...new Set(allCreatives.map(c => c.format).filter(Boolean))];
  const dynamicThemes = ['All', ...new Set(allCreatives.map(c => c.theme).filter(Boolean))];
  const dynamicHooks = ['All', ...new Set(allCreatives.map(c => c.hook_type).filter(Boolean))];

  const resetFilters = () => {
    setFilters({ format: 'All', theme: 'All', hook_type: 'All' });
  };

  const handleUpgrade = (creative) => {
    setSelectedCreative(creative);
    setIsModalOpen(true);
  };

  const advertiserOptions = [
    'PixelForge', 'VantaStyle', 'EarnPath', 'CloudRoute', 'SnapMeal', 'MoonScreen', 
    'NeonArena', 'ThreadMark', 'CoinBridge', 'DriftPass', 'BiteRush', 'FluxStream', 
    'VoxelRealm', 'BoltCart', 'VaultRun', 'HorizonHop', 'FreshDash', 'PulsePlay', 
    'StormByte', 'ZenDrop', 'NovaPay', 'RouteBloom', 'CraveLoop', 'NeonCast', 
    'GlitchWave', 'LuxShelf', 'ClearLedger', 'WayPoint', 'QuickFork', 'BingeLayer', 
    'IronQuest', 'CrispBox', 'SwiftStack', 'NomadLayer', 'PlateSprint', 'StarVault'
  ];

  if (!advertiser) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-950 to-slate-950 overflow-y-auto">
        <div className="max-w-6xl w-full text-center py-12">
          <div className="w-20 h-20 bg-indigo-600 rounded-[2rem] flex items-center justify-center text-white font-black text-4xl shadow-2xl shadow-indigo-500/50 rotate-6 mx-auto mb-10">S</div>
          <h1 className="text-4xl font-black text-white tracking-tighter mb-4 italic uppercase">Smadex Intelligence</h1>
          <p className="text-slate-400 font-bold uppercase tracking-[0.2em] text-xs mb-12">Search or select your advertiser profile</p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {advertiserOptions.map(adv => (
              <button 
                key={adv}
                onClick={() => setAdvertiser(adv)}
                className="p-4 bg-white/5 hover:bg-indigo-600 border border-white/10 rounded-2xl transition-all text-left group"
              >
                <h3 className="text-xs font-black text-white uppercase italic tracking-tighter group-hover:scale-105 transition-transform">{adv}</h3>
              </button>
            ))}
          </div>
          
          <p className="mt-20 text-slate-600 text-[9px] font-bold uppercase tracking-[0.4em]">Enterprise Asset Management System | HackUPC 2026</p>
        </div>
      </div>
    );
  }

  const fatiguedCount = creatives.filter(c => c.fatigued).length;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans w-full pb-20">
      {/* Header */}
      <nav className="bg-white/90 backdrop-blur-xl border-b border-slate-200 px-4 md:px-8 py-3 flex flex-col md:flex-row justify-between items-center sticky top-0 z-50 w-full shadow-sm gap-4">
        <div className="flex items-center gap-3 self-start md:self-center">
          <div className="w-8 h-8 md:w-10 md:h-10 bg-indigo-600 rounded-xl md:rounded-2xl flex items-center justify-center text-white font-black shadow-lg shadow-indigo-100 rotate-3 text-sm md:text-base">S</div>
          <div>
            <h1 className="text-lg md:text-xl font-black tracking-tighter leading-none">Smadex <span className="text-indigo-600">Intelligence</span></h1>
            <p className="text-[9px] md:text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">Logged in as {advertiser}</p>
          </div>
        </div>
        
        <div className="flex flex-wrap md:flex-nowrap justify-center gap-2 md:gap-3 bg-slate-100/50 p-1.5 rounded-2xl md:rounded-3xl shadow-inner border border-slate-200 w-full md:w-auto">
          <button 
            onClick={() => {
              setAdvertiser(null);
              resetFilters();
              setSortOrder('default');
            }}
            className="px-3 md:px-4 py-1.5 md:py-2 rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-red-500 transition-colors"
          >
            Logout
          </button>
          <div className="hidden md:block w-px h-6 bg-slate-300 my-auto"></div>
          <select 
            className="flex-1 md:flex-none px-3 md:px-4 py-1.5 md:py-2 rounded-xl text-[10px] md:text-xs font-black uppercase tracking-widest bg-white border-0 shadow-sm focus:ring-2 focus:ring-indigo-500 outline-none cursor-pointer appearance-none"
            value={filters.format}
            onChange={(e) => setFilters({ ...filters, format: e.target.value })}
          >
            {dynamicFormats.map(opt => (
              <option key={opt} value={opt}>{opt === 'All' ? 'All Formats' : opt}</option>
            ))}
          </select>
          <select 
            className="flex-1 md:flex-none px-3 md:px-4 py-1.5 md:py-2 rounded-xl text-[10px] md:text-xs font-black uppercase tracking-widest bg-white border-0 shadow-sm focus:ring-2 focus:ring-indigo-500 outline-none cursor-pointer appearance-none"
            value={filters.theme}
            onChange={(e) => setFilters({ ...filters, theme: e.target.value })}
          >
            {dynamicThemes.map(opt => (
              <option key={opt} value={opt}>{opt === 'All' ? 'All Themes' : opt}</option>
            ))}
          </select>
          <select 
            className="flex-1 md:flex-none px-3 md:px-4 py-1.5 md:py-2 rounded-xl text-[10px] md:text-xs font-black uppercase tracking-widest bg-white border-0 shadow-sm focus:ring-2 focus:ring-indigo-500 outline-none cursor-pointer appearance-none"
            value={filters.hook_type}
            onChange={(e) => setFilters({ ...filters, hook_type: e.target.value })}
          >
            {dynamicHooks.map(opt => (
              <option key={opt} value={opt}>{opt === 'All' ? 'All Hooks' : opt}</option>
            ))}
          </select>
        </div>
      </nav>

      {/* Main Content */}
      <main className="px-8 mt-12 max-w-7xl mx-auto w-full">
        <header className="mb-12 flex justify-between items-end">
          <div>
            <h2 className="text-4xl font-black mb-3 tracking-tighter text-slate-900 italic uppercase underline decoration-indigo-500 decoration-8 underline-offset-4">Asset Performance</h2>
            <p className="text-slate-500 font-medium">Monitoring Smadex Creative Dataset: <code>creatives_v1.csv</code></p>
          </div>
          <div className="flex gap-4">
            <button 
              onClick={() => setSortOrder(prev => prev === 'fatigue' ? 'score' : prev === 'score' ? 'ctr' : prev === 'ctr' ? 'default' : 'fatigue')}
              className={`px-6 py-3 rounded-3xl border transition-all flex items-center gap-3 shadow-xl ${
                sortOrder !== 'default' ? 'bg-indigo-600 border-indigo-400 text-white shadow-indigo-200' : 'bg-white border-slate-100 text-slate-400 shadow-slate-200/50'
              }`}
            >
              <span className="font-bold text-[10px] uppercase tracking-widest">
                {sortOrder === 'default' ? 'Default Order' : sortOrder === 'fatigue' ? 'Sorting: Fatigue' : sortOrder === 'score' ? 'Sorting: Score' : 'Sorting: CTR'}
              </span>
              <span className={`text-xs transition-transform duration-300 ${sortOrder !== 'default' ? 'rotate-180' : ''}`}>↓</span>
            </button>

            {fatiguedCount > 0 && (
              <div className="px-6 py-3 bg-red-50 border border-red-100 rounded-3xl shadow-xl shadow-red-200/50 flex items-center gap-3 animate-bounce">
                <span className="w-3 h-3 bg-red-500 rounded-full"></span>
                <span className="font-black text-xs uppercase tracking-widest text-red-600">{fatiguedCount} Assets Fatigued</span>
              </div>
            )}
            <div className="px-6 py-3 bg-white rounded-3xl border border-slate-100 shadow-xl shadow-slate-200/50 flex items-center gap-3">
              <span className="w-3 h-3 bg-green-500 rounded-full animate-ping"></span>
              <span className="font-bold text-xs uppercase tracking-widest text-slate-400">Live Simulation</span>
            </div>
          </div>
        </header>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 w-full animate-pulse">
            {[1, 2, 3, 4, 5, 6].map(n => <div key={n} className="h-[450px] bg-slate-200 rounded-[2.5rem]"></div>)}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 w-full">
            {creatives.map(item => (
              <CreativeCard 
                key={item.id} 
                creative={item} 
                onUpgrade={handleUpgrade}
              />
            ))}
          </div>
        )}

        {!loading && creatives.length === 0 && (
          <div className="w-full text-center py-32 bg-white rounded-[3rem] border-4 border-dashed border-slate-100 flex flex-col items-center">
            <div className="text-8xl mb-8 grayscale drop-shadow-2xl">🕵️‍♂️</div>
            <h3 className="text-2xl font-black text-slate-800 uppercase tracking-tighter mb-4">No assets match your current filters</h3>
            <p className="text-slate-400 font-bold uppercase tracking-widest text-xs mb-8">Try adjusting your vertical or theme selection</p>
            <button 
              onClick={resetFilters}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-black px-10 py-4 rounded-2xl shadow-2xl shadow-indigo-200 transition-all active:scale-95 uppercase tracking-widest text-xs"
            >
              Reset Filters
            </button>
          </div>
        )}
      </main>

      <UpgradeModal 
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        creative={selectedCreative}
        onApply={handleApplyUpgrade}
      />
    </div>
  );
}

export default Dashboard;
