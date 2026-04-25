import mockData from '../mocks/data.json';

let sessionData = [...mockData];

export const fetchCreatives = async (filters = { format: 'All', theme: 'All', hook_type: 'All', advertiser: 'All' }) => {
  await new Promise((resolve) => setTimeout(resolve, 500));
  let filtered = [...sessionData];
  if (filters.advertiser && filters.advertiser !== 'All') {
    filtered = filtered.filter(ad => ad.advertiser === filters.advertiser);
  }
  return filtered;
};

export const updateCreativeImage = (id, newImageUrl) => {
  sessionData = sessionData.map(c => 
    c.id === id ? { ...c, image_url: newImageUrl, is_upgraded: true } : c
  );
  return true;
};

export const upgradeImage = async (creative) => {
  console.log('[API] Triggering PixelForge AI Generation via Python for:', creative.id);
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        success: true,
        newImageUrl: creative.image_url, 
        aiReasoning: 'PixelForge AI analyzed the ' + creative.hook_type + ' hook and enhanced the visual assets for ' + creative.theme + ' theme. Optimization complete for ' + creative.format + ' format.'
      });
    }, 2500); 
  });
};