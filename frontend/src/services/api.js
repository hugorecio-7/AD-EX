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

export const upgradeImage = async (creativeId, { numSteps = 3 } = {}) => {
  console.log('[API] Triggering full AI upgrade pipeline for:', creativeId, '| steps:', numSteps);

  try {
    const response = await fetch(`http://localhost:8000/api/creatives/${creativeId}/upgrade`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_steps: numSteps }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Upgrade failed: ${err.detail || response.status}`);
    }

    const data = await response.json();
    console.log('[API] Upgrade result:', data);

    const newImageUrl = data.new_image_url || `/data/assets/creative_${creativeId}_upgraded.png`;
    const newCreativeId = data.creative_id || `${creativeId}_v2`;

    return {
      success: true,
      creativeId: newCreativeId,
      newImageUrl,
      predictedUplift: data.metadata?.predicted_uplift ?? '+0.0%',
      performanceScore: data.metadata?.performance_score ?? null,
      aiReasoning: data.metadata?.missing_features_explained ?? 'AI upgrade complete.',
    };

  } catch (error) {
    console.error('[API] Upgrade request failed:', error.message);
    return {
      success: false,
      error: error.message,
      creativeId: null,
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

export const askCreativeChat = async (creativeId, message, history = [], language = 'catalan', agentic = false) => {
  try {
    const response = await fetch(`http://localhost:8000/api/creatives/${creativeId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history, language, agentic }),
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
      action: data.action || null,
    };
  } catch (error) {
    console.error('[API] Creative chat failed:', error.message);
    return {
      success: false,
      error: error.message,
      answer: `No he pogut consultar el bot ara mateix: ${error.message}`,
      action: null,
    };
  }
};


export const fetchCtrPrediction = async (creativeId, { countries = 'US,ES', os = 'iOS,Android', compareImageUrl = null, seqLen = 30 } = {}) => {
  try {
    const params = new URLSearchParams({ countries, os, seq_len: seqLen });
    if (compareImageUrl) params.set('compare_image_url', compareImageUrl);

    const response = await fetch(`http://localhost:8000/api/creatives/${creativeId}/predict?${params}`);
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `Predict failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('[API] CTR prediction failed:', error.message);
    return null;
  }
};