/**
 * frontend/js/api.js
 * ──────────────────
 * Complete REST API client communicating with FastAPI backend at /api/v1/
 */

const API_BASE = window.location.origin + '/api/v1';

async function parseErrorResponse(res, fallbackMessage) {
  try {
    const data = await res.json();
    if (data && data.detail) {
      if (Array.isArray(data.detail)) {
        return data.detail.map(d => d.msg || (typeof d === 'string' ? d : JSON.stringify(d))).join(', ');
      }
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    }
    return fallbackMessage;
  } catch {
    return `${fallbackMessage} (HTTP ${res.status}: ${res.statusText || 'Error'})`;
  }
}

export const ApiClient = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Health check failed'));
    return await res.json();
  },

  async getDemos() {
    const res = await fetch(`${API_BASE}/demos`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to fetch demo benchmarks'));
    return await res.json();
  },

  async ingestDemo(demoName) {
    const res = await fetch(`${API_BASE}/ingest/demo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ demo_name: demoName }),
    });
    if (!res.ok) {
      throw new Error(await parseErrorResponse(res, 'Demo ingestion failed'));
    }
    return await res.json();
  },

  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/ingest/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      throw new Error(await parseErrorResponse(res, 'Dataset upload failed'));
    }
    return await res.json();
  },

  async uploadCSV(file) {
    return this.uploadDataset(file);
  },

  async autoDetectObjectives(sessionId) {
    const res = await fetch(`${API_BASE}/autodetect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Auto-detection failed'));
    return await res.json();
  },

  async getProfile(sessionId) {
    const res = await fetch(`${API_BASE}/profile/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load dataset profile'));
    return await res.json();
  },

  async getReadiness(sessionId) {
    const res = await fetch(`${API_BASE}/readiness/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load readiness audit'));
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
      throw new Error(await parseErrorResponse(res, 'AutoML training failed'));
    }
    return await res.json();
  },

  async predict(sessionId, features) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, features: features }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Prediction failed'));
    return await res.json();
  },

  async simulate(sessionId, overrides) {
    const res = await fetch(`${API_BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, feature_overrides: overrides }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Simulation failed'));
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
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'ROI optimization failed'));
    return await res.json();
  },

  async getActiveLearningQueue(sessionId) {
    const res = await fetch(`${API_BASE}/active-learning/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load active learning queue'));
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
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Counterfactual failed'));
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
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Uplift estimation failed'));
    return await res.json();
  },

  async getAutoencoder(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/autoencoder/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Autoencoder analysis failed'));
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
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Bandit simulation failed'));
    return await res.json();
  },

  async getOnlineLearning(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/online-learning/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Online learning simulation failed'));
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
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Forecasting failed'));
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
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Synthetic generation failed'));
    return await res.json();
  },

  async getNlpAnalysis(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/nlp/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'NLP analysis failed'));
    return await res.json();
  },

  async getGraphIntelligence(sessionId) {
    const res = await fetch(`${API_BASE}/adaptive/graph/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Graph analysis failed'));
    return await res.json();
  },

  async getDataContract(sessionId) {
    const res = await fetch(`${API_BASE}/governance/contract/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load data contract'));
    return await res.json();
  },

  async getGdprAudit(sessionId) {
    const res = await fetch(`${API_BASE}/governance/gdpr/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load GDPR audit'));
    return await res.json();
  },

  async getDriftMonitor(sessionId) {
    const res = await fetch(`${API_BASE}/governance/drift/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load drift monitor'));
    return await res.json();
  },

  async getAllArtifacts(sessionId) {
    const res = await fetch(`${API_BASE}/artifacts/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to load artifacts'));
    return await res.json();
  },

  async chat(sessionId, query) {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, query: query }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Chat failed'));
    return await res.json();
  },

  async getLeakageReport(sessionId) {
    const res = await fetch(`${API_BASE}/governance/leakage/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to fetch leakage report'));
    return await res.json();
  },

  async getCausalGraph(sessionId) {
    const res = await fetch(`${API_BASE}/causal/graph/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to fetch causal graph'));
    return await res.json();
  },

  async simulateIntervention(sessionId, treatmentCol, outcomeCol, interventionValue) {
    const res = await fetch(`${API_BASE}/causal/intervention`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        treatment_col: treatmentCol,
        outcome_col: outcomeCol,
        intervention_value: parseFloat(interventionValue),
      }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Causal intervention simulation failed'));
    return await res.json();
  },

  async optimizeParetoFrontier(sessionId, weights = {}) {
    const res = await fetch(`${API_BASE}/optimization/pareto/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(weights),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Pareto optimization failed'));
    return await res.json();
  },

  async getEdgeModelBundle(sessionId, download = false) {
    const res = await fetch(`${API_BASE}/export/edge-bundle/${sessionId}?download=${download}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to fetch edge bundle'));
    return await res.json();
  },

  async getRecourseSample(sessionId) {
    const res = await fetch(`${API_BASE}/recourse/sample/${sessionId}`);
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Failed to fetch sample record'));
    return await res.json();
  },

  async calculateRecourse(sessionId, rowIndex = 0, desiredClass = 1, targetProbThreshold = 0.55, immutableFeatures = [], customOverrides = null) {
    const res = await fetch(`${API_BASE}/recourse/counterfactual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        row_index: rowIndex,
        desired_class: desiredClass,
        target_probability_threshold: targetProbThreshold,
        immutable_features: immutableFeatures,
        custom_feature_overrides: customOverrides,
      }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Recourse optimization failed'));
    return await res.json();
  },

  async runStressTest(sessionId, customShocks = null, nSimulations = 250) {
    const res = await fetch(`${API_BASE}/simulation/stress-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        custom_shocks: customShocks,
        n_simulations: nSimulations,
      }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Stress simulation failed'));
    return await res.json();
  },

  async getConformalBounds(sessionId, alpha = 0.10) {
    const res = await fetch(`${API_BASE}/uncertainty/conformal-bounds`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        alpha: alpha,
      }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Conformal uncertainty evaluation failed'));
    return await res.json();
  },

  async discoverSymbolicFeatures(sessionId, maxCandidates = 100, topK = 5) {
    const res = await fetch(`${API_BASE}/features/symbolic-discovery/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        max_candidates: maxCandidates,
        top_k: topK,
      }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Symbolic feature discovery failed'));
    return await res.json();
  },

  async runDriftSentinel(sessionId, sampleFraction = 0.35, syntheticShiftStrength = 0.0) {
    const res = await fetch(`${API_BASE}/monitoring/drift-sentinel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        sample_fraction: sampleFraction,
        synthetic_shift_strength: syntheticShiftStrength,
      }),
    });
    if (!res.ok) throw new Error(await parseErrorResponse(res, 'Drift sentinel audit failed'));
    return await res.json();
  },

  getExportUrl(sessionId, formatType) {
    return `${API_BASE}/export/${sessionId}/${formatType}`;
  },
};
