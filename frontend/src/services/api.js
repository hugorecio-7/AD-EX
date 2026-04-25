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

export const upgradeImage = async (creativeId) => {
  console.log('[API] Requesting Backend Evaluation for:', creativeId);
  
  try {
    const response = await fetch(`http://localhost:8000/evaluate/${creativeId}`);
    const data = await response.json();
    
    if (data.status === 'success') {
      return {
        success: true,
        newImageUrl: `/data/assets/creative_${creativeId}.png`, 
        predictedUplift: data.metrics.predicted_uplift,
        performanceScore: data.metrics.performance_score,
        aiReasoning: data.ai_reasoning
      };
    }
  } catch (error) {
    console.error('[API] Backend unreachable, falling back to mock:', error);
  }

  // Fallback for demo stability
  return {
    success: true,
    newImageUrl: `/data/assets/creative_${creativeId}.png`, 
    predictedUplift: "+22.4%",
    performanceScore: 0.85,
    aiReasoning: 'Computed via local fallback engine. Backend connection failed.'
  };
};