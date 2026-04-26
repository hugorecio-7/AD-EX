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
  console.log('[API] Triggering full AI upgrade pipeline for:', creativeId);

  try {
    // POST /api/creatives/{id}/upgrade — runs SAM mask → SD inpainting → LightGBM evaluation
    const response = await fetch(`http://localhost:8000/api/creatives/${creativeId}/upgrade`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Upgrade failed: ${err.detail || response.status}`);
    }

    const data = await response.json();
    console.log('[API] Upgrade result:', data);

    return {
      success: true,
      // Backend returns the new image URL inside metadata or as new_image_url
      newImageUrl: data.new_image_url || `/data/assets/creative_${creativeId}_upgraded.png`,
      predictedUplift: data.metadata?.predicted_uplift ?? '+0.0%',
      performanceScore: data.metadata?.performance_score ?? null,
      aiReasoning: data.metadata?.missing_features_explained ?? 'AI upgrade complete.',
    };

  } catch (error) {
    console.error('[API] Upgrade request failed:', error.message);

    // Hard fail — show error instead of silently returning the original image
    return {
      success: false,
      error: error.message,
      newImageUrl: null,
      predictedUplift: null,
      performanceScore: null,
      aiReasoning: `Upgrade failed: ${error.message}`,
    };
  }
};

// Separate lightweight call just for the score card in the dashboard header
export const evaluateCreative = async (creativeId) => {
  console.log('[API] Evaluating creative score for:', creativeId);
  try {
    const response = await fetch(`http://localhost:8000/evaluate/${creativeId}`);
    const data = await response.json();
    if (data.status === 'success') return data.metrics;
  } catch (e) {
    console.error('[API] Evaluation failed:', e);
  }
  return null;
};

export const askCreativeChat = async (creativeId, message, history = [], language = 'catalan') => {
  try {
    const response = await fetch(`http://localhost:8000/api/creatives/${creativeId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history, language }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `Chat failed: ${response.status}`);
    }

    const data = await response.json();
    return {
      success: true,
      answer: data.answer || 'No response',
      model: data.model,
      language: data.language || language,
    };
  } catch (error) {
    console.error('[API] Creative chat failed:', error.message);
    return {
      success: false,
      error: error.message,
      answer: `No he pogut consultar el bot ara mateix: ${error.message}`,
    };
  }
};