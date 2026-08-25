/**
 * frontend/js/api.js
 * ──────────────────
 * Complete REST API client communicating with FastAPI backend at /api/v1/
 */

const API_BASE = window.location.origin + '/api/v1';

export const ApiClient = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  },

  async getDemos() {
    const res = await fetch(`${API_BASE}/demos`);
    if (!res.ok) throw new Error('Failed to fetch demo benchmarks');
    return await res.json();
  },

  async ingestDemo(demoName) {
    const res = await fetch(`${API_BASE}/ingest/demo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ demo_name: demoName }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Demo ingestion failed');
    }
    return await res.json();
  },

  async uploadCSV(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/ingest/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'CSV upload failed');
    }
    return await res.json();
  },

  async autoDetectObjectives(sessionId) {
    const res = await fetch(`${API_BASE}/autodetect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error('Auto-detection failed');
    return await res.json();
  },

  async getProfile(sessionId) {
    const res = await fetch(`${API_BASE}/profile/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load dataset profile');
    return await res.json();
  },

  async getReadiness(sessionId) {
    const res = await fetch(`${API_BASE}/readiness/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load readiness audit');
    return await res.json();
  },

  async trainPipeline(sessionId, goal, userTargetCol = null, selectedModels = null) {
    const res = await fetch(`${API_BASE}/pipeline/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        goal: goal,
        user_target_col: userTargetCol || null,
        selected_models: selectedModels || null,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'AutoML training failed');
    }
    return await res.json();
  },

  async predict(sessionId, features) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, features: features }),
    });
    if (!res.ok) throw new Error('Prediction failed');
    return await res.json();
  },

  async simulate(sessionId, overrides) {
    const res = await fetch(`${API_BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, feature_overrides: overrides }),
    });
    if (!res.ok) throw new Error('Simulation failed');
    return await res.json();
  },

  async optimizeRoi(sessionId, benefitTp, costFp, costFn, benefitTn) {
    const res = await fetch(`${API_BASE}/roi-optimizer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        benefit_tp: parseFloat(benefitTp),
        cost_fp: parseFloat(costFp),
        cost_fn: parseFloat(costFn),
        benefit_tn: parseFloat(benefitTn),
      }),
    });
    if (!res.ok) throw new Error('ROI optimization failed');
    return await res.json();
  },

  async getActiveLearningQueue(sessionId) {
    const res = await fetch(`${API_BASE}/active-learning/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load active learning queue');
    return await res.json();
  },

  async getCounterfactual(sessionId, rowIndex, desiredOutcome) {
    const res = await fetch(`${API_BASE}/causal/counterfactual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        row_index: parseInt(rowIndex, 10),
        desired_outcome: desiredOutcome,
      }),
    });
    if (!res.ok) throw new Error('Counterfactual failed');
    return await res.json();
  },

  async getUplift(sessionId, treatmentCol) {
    const res = await fetch(`${API_BASE}/causal/uplift`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        treatment_column: treatmentCol,
      }),
    });
    if (!res.ok) throw new Error('Uplift estimation failed');
    return await res.json();
  },

  async getAutoencoder(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/autoencoder/${sessionId}`);
    if (!res.ok) throw new Error('Autoencoder analysis failed');
    return await res.json();
  },

  async simulateBandits(sessionId, nSteps, alpha) {
    const res = await fetch(`${API_BASE}/adaptive/bandits`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        n_steps: parseInt(nSteps, 10),
        alpha: parseFloat(alpha),
      }),
    });
    if (!res.ok) throw new Error('Bandit simulation failed');
    return await res.json();
  },

  async getOnlineLearning(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/online-learning/${sessionId}`);
    if (!res.ok) throw new Error('Online learning simulation failed');
    return await res.json();
  },

  async forecastTimeSeries(sessionId, dateCol, horizon) {
    const res = await fetch(`${API_BASE}/timeseries/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        date_col: dateCol,
        forecast_horizon: parseInt(horizon, 10),
      }),
    });
    if (!res.ok) throw new Error('Forecasting failed');
    return await res.json();
  },

  async generateSynthetic(sessionId, nSamples, applyDp, epsilon) {
    const res = await fetch(`${API_BASE}/synthetic/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        n_samples: parseInt(nSamples, 10),
        apply_dp_noise: Boolean(applyDp),
        epsilon: parseFloat(epsilon),
      }),
    });
    if (!res.ok) throw new Error('Synthetic generation failed');
    return await res.json();
  },

  async getNlpAnalysis(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/nlp/${sessionId}`);
    if (!res.ok) throw new Error('NLP analysis failed');
    return await res.json();
  },

  async getGraphIntelligence(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/graph/${sessionId}`);
    if (!res.ok) throw new Error('Graph analysis failed');
    return await res.json();
  },

  async getDataContract(sessionId) {
    const res = await fetch(`${API_BASE}/governance/contract/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load data contract');
    return await res.json();
  },

  async getGdprAudit(sessionId) {
    const res = await fetch(`${API_BASE}/governance/gdpr/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load GDPR audit');
    return await res.json();
  },

  async getDriftMonitor(sessionId) {
    const res = await fetch(`${API_BASE}/governance/drift/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load drift monitor');
    return await res.json();
  },

  async getAllArtifacts(sessionId) {
    const res = await fetch(`${API_BASE}/artifacts/${sessionId}`);
    if (!res.ok) throw new Error('Failed to load artifacts');
    return await res.json();
  },

  async chat(sessionId, query) {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, query: query }),
    });
    if (!res.ok) throw new Error('Chat failed');
    return await res.json();
  },

  getExportUrl(sessionId, formatType) {
    return `${API_BASE}/export/${sessionId}/${formatType}`;
  },
};
