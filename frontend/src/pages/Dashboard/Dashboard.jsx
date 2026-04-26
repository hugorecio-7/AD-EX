import React, { useState, useEffect } from 'react';
import { fetchCreatives, updateCreativeImage } from '../../services/api';
import CreativeCard from '../../components/CreativeCard';
import UpgradeModal from '../../components/UpgradeModal';
import CreativeChatModal from '../../components/CreativeChatModal';
import './Dashboard.css';

function Dashboard() {
  const [creatives, setCreatives] = useState([]);
  const [allCreatives, setAllCreatives] = useState([]);
  const [loading, setLoading] = useState(true);
  const [advertiser, setAdvertiser] = useState(null);
  const [filters, setFilters] = useState({ 
    format: 'All', 
    theme: 'All',
    hook_type: 'All'
  });
  const [sortOrder, setSortOrder] = useState('score');
  const [selectedCreative, setSelectedCreative] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedChatCreative, setSelectedChatCreative] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const [chatHistories, setChatHistories] = useState({});

  const handleApplyUpgrade = (newId, newUrl) => {
    // Update the ORIGINAL creative's image (selectedCreative.id), not the new clone id
    const originalId = selectedCreative?.id;
    updateCreativeImage(originalId, newUrl);
    const patch = (c) =>
      c.id === originalId ? { ...c, image_url: newUrl + '?t=' + Date.now(), is_upgraded: true } : c;
    setCreatives(prev => prev.map(patch));
    setAllCreatives(prev => prev.map(patch));
    setIsModalOpen(false);
  };

  // Called from chat when Implement is applied
  const handleApplyImplement = (newId, newUrl) => {
    // Always update original card by current selected chat creative
    const originalId = selectedChatCreative?.id;
    updateCreativeImage(originalId, newUrl);
    const urlWithBust = newUrl + '?t=' + Date.now();
    const patch = (c) =>
      c.id === originalId ? { ...c, image_url: urlWithBust, is_upgraded: true } : c;
    setCreatives(prev => prev.map(patch));
    setAllCreatives(prev => prev.map(patch));
    // Update the chat creative ref so subsequent implements use the new image context
    setSelectedChatCreative(prev => prev ? { ...prev, image_url: urlWithBust } : prev);
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

  const handleOpenChat = (creative) => {
    setSelectedChatCreative(creative);
    setIsChatOpen(true);
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
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6" style={{background: 'radial-gradient(circle at top right, #1e1b4b 0%, #020617 60%)'}}>
        <div className="max-w-5xl w-full text-center py-12">
          <div className="w-20 h-20 bg-indigo-600 rounded-[2rem] flex items-center justify-center text-white font-black text-3xl shadow-2xl shadow-indigo-500/50 rotate-6 mx-auto mb-8">
            AD
          </div>
          <h1 className="text-5xl font-black text-white tracking-tighter mb-2 italic uppercase">AD-EX</h1>
          <p className="text-slate-400 font-bold uppercase tracking-[0.2em] text-xs mb-10">Select your advertiser profile</p>
          
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
          
          <p className="mt-16 text-slate-600 text-[9px] font-bold uppercase tracking-[0.4em]">AD-EX Creative Intelligence · HackUPC 2026</p>
        </div>
      </div>
    );
  }

  const fatiguedCount = creatives.filter(c => c.fatigued).length;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans w-full pb-20">
      {/* Header */}
      <nav className="bg-white/90 backdrop-blur-xl border-b border-slate-200 px-4 md:px-8 py-3 flex flex-col md:flex-row justify-between items-center sticky top-0 z-50 w-full shadow-sm gap-3">
        <div className="flex items-center gap-3 self-start md:self-center">
          <div className="w-8 h-8 bg-indigo-600 rounded-xl flex items-center justify-center text-white font-black shadow-lg shadow-indigo-100 rotate-3 text-[10px] tracking-tighter">AD</div>
          <div>
            <h1 className="text-lg font-black tracking-tighter leading-none">AD-<span className="text-indigo-600">EX</span></h1>
            <p className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">{advertiser}</p>
          </div>
        </div>
        
        <div className="flex flex-wrap justify-center gap-2 bg-slate-100/50 p-1.5 rounded-2xl shadow-inner border border-slate-200 w-full md:w-auto">
          <button 
            onClick={() => { setAdvertiser(null); resetFilters(); setSortOrder('default'); }}
            className="px-3 py-1.5 rounded-xl text-[9px] font-black uppercase tracking-widest text-slate-400 hover:text-red-500 transition-colors"
          >
            Logout
          </button>
          <div className="hidden md:block w-px h-5 bg-slate-300 my-auto"></div>
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
            {creatives.length > 0 && (
              <div className="px-6 py-3 bg-indigo-50 border border-indigo-100 rounded-3xl shadow-xl shadow-indigo-200/50 flex items-center gap-3">
                <span className="w-3 h-3 bg-indigo-500 rounded-full"></span>
                <span className="font-black text-xs uppercase tracking-widest text-indigo-600">
                  Avg Score: {(creatives.reduce((a, c) => a + (c.performance_score || 0), 0) / creatives.length).toFixed(2)}
                </span>
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
                onChat={handleOpenChat}
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

      <CreativeChatModal
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        creative={selectedChatCreative}
        onApplyImplement={handleApplyImplement}
        initialMessages={selectedChatCreative ? (chatHistories[selectedChatCreative.id] || []) : []}
        onHistoryChange={(msgs) => {
          if (selectedChatCreative)
            setChatHistories(prev => ({ ...prev, [selectedChatCreative.id]: msgs }));
        }}
      />
    </div>
  );
}

export default Dashboard;
