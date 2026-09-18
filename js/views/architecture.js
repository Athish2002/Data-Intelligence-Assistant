/**
 * frontend/js/views/architecture.js
 * ─────────────────────────────────
 * Enterprise Full-Stack System Architecture & Design Blueprint Cockpit.
 * Provides interactive C4 Container Topology Diagrams, End-to-End Dataflow Steppers,
 * UI/UX Pro Max Design System & Component State Matrices, and API Contract / SLA Explorers.
 */

import { state, el, escapeHtml } from '../state.js';

let activeArchView = 'c4_diagram'; // 'c4_diagram' | 'dataflow' | 'design_system' | 'api_sla'
let selectedNodeId = 'api_gw';

// System Node Definitions for C4 Container Topology
const C4_NODES = {
  client_spa: {
    id: 'client_spa',
    tier: 'Presentation Tier',
    icon: 'layout',
    title: 'Single Page Application (SPA)',
    subtitle: 'Vanilla ES Modules & CSS Custom Properties',
    tech: 'ES2022 / HTML5 / Lucide / Plotly.js / Tailwind CSS',
    responsibility: 'Renders reactive workspaces, 60fps micro-interactions, responsive 8pt layout, and hardware-accelerated themes.',
    latencySla: '< 16.6ms (60fps DOM render)',
    memoryFootprint: '~25 MB V8 heap',
    pattern: 'Model-View-Controller / Reactive Store',
    sourceFile: 'frontend/js/app.js',
    endpoints: ['Client-Side State Engine', 'Theme Engine (Cyber Aurora / Obsidian / Tokyo Sand)'],
  },
  client_wasm: {
    id: 'client_wasm',
    tier: 'Presentation Tier',
    icon: 'zap',
    title: 'Edge C99 & WASM Runtime',
    subtitle: 'Client-Side Offline Inference Engine',
    tech: 'WebAssembly (WASM/WAT) & Standalone JavaScript',
    responsibility: 'Executes row-level and batch inference offline directly in the client browser without external server calls.',
    latencySla: '< 0.05ms / row, < 15ms / 1k batch',
    memoryFootprint: '< 500 KB bundle size',
    pattern: 'AST Transpiled Interpreter',
    sourceFile: 'dia/wasm_compiler.py',
    endpoints: ['POST /api/v1/export/edge-bundle/{session_id}'],
  },
  api_gw: {
    id: 'api_gw',
    tier: 'Gateway & Networking Tier',
    icon: 'globe',
    title: 'FastAPI Gateway & Router',
    subtitle: 'Async High-Concurrency Entrypoint',
    tech: 'Python 3.12+ / FastAPI / Uvicorn / Pydantic V2',
    responsibility: 'Performs unified schema validation, CORS security, rate limiting, request routing, and real-time SSE streaming.',
    latencySla: 'p50 < 30ms, p95 < 65ms (overhead < 3ms)',
    memoryFootprint: '~45 MB base process RSS',
    pattern: 'API Gateway / Front Controller',
    sourceFile: 'api/server.py',
    endpoints: ['/api/v1/health', '/api/v1/ingest', '/api/v1/train', '/api/v1/causal/*', '/api/v1/optimization/*'],
  },
  session_store: {
    id: 'session_store',
    tier: 'In-Memory State & Cache Tier',
    icon: 'database',
    title: 'Bounded In-Memory Session Store',
    subtitle: 'Thread-Safe LRU Cache with Auto-GC',
    tech: 'dia/session_manager.py / Python gc / WeakRef',
    responsibility: 'Maintains isolated analytical sessions, dataset matrices, trained models, and triggers automated deallocation.',
    latencySla: '< 1ms memory lookup',
    memoryFootprint: '< 50 MB / active session (max 20 sessions)',
    pattern: 'Bounded Repository / Singleton',
    sourceFile: 'dia/session_manager.py',
    endpoints: ['POST /api/v1/system/gc', 'GET /api/v1/system/metrics'],
  },
  leakage_sleuth: {
    id: 'leakage_sleuth',
    tier: 'Analytical Subsystem',
    icon: 'shield-alert',
    title: 'Target Leakage & Poisoning Sleuth',
    subtitle: 'Information-Theoretic Vulnerability Audit',
    tech: 'Scikit-Learn / Mutual Information / NumPy',
    responsibility: 'Calculates MI ratio I(X; Y) / H(Y), quasi-target Pearson correlations, future timestamps, and flags features for quarantine.',
    latencySla: '< 120ms for 100k data points',
    memoryFootprint: '< 10 MB temporary working array',
    pattern: 'Filter Chain / Strategy',
    sourceFile: 'dia/leakage_detector.py',
    endpoints: ['GET /api/v1/governance/leakage/{session_id}'],
  },
  causal_pc: {
    id: 'causal_pc',
    tier: 'Analytical Subsystem',
    icon: 'git-branch',
    title: 'Observational Causal Discovery',
    subtitle: 'PC Algorithm & Pearl\'s Do-Calculus',
    tech: 'Fisher\'s z-Transform / NetworkX / Graphviz CPDAG',
    responsibility: 'Induces causal DAG skeletons, applies Meek orientation rules, and simulates policy intervention outcomes E[Y | do(X = x)].',
    latencySla: '< 350ms for 15-variable graph',
    memoryFootprint: '< 15 MB adjacency matrix',
    pattern: 'Constraint-Based Graph Inducer',
    sourceFile: 'dia/causal_discovery.py',
    endpoints: ['GET /api/v1/causal/graph/{session_id}', 'POST /api/v1/causal/intervention'],
  },
  automl_cockpit: {
    id: 'automl_cockpit',
    tier: 'Analytical Subsystem',
    icon: 'trophy',
    title: 'AutoML Engine & Ensembles',
    subtitle: '5-Fold CV Model Benchmark & SHAP',
    tech: 'Scikit-Learn / LightGBM / XGBoost / SHAP',
    responsibility: 'Trains 6 diverse model architectures, evaluates ROC/AUC, computes TreeSHAP explanations, and runs Pareto optimization.',
    latencySla: '< 8.5s total training run (10k rows)',
    memoryFootprint: '< 60 MB model ensemble in RAM',
    pattern: 'Strategy Pattern / Model Factory',
    sourceFile: 'dia/model_trainer.py',
    endpoints: ['POST /api/v1/train', 'GET /api/v1/explainability/shap/{session_id}'],
  },
  recourse_engine: {
    id: 'recourse_engine',
    tier: 'Analytical Subsystem',
    icon: 'sliders',
    title: 'Wachter Counterfactual Recourse',
    subtitle: 'Constrained Actionability Optimization',
    tech: 'SciPy Optimize / L-BFGS-B / Median Absolute Deviation',
    responsibility: 'Solves minimum-distance feature shift vectors to flip classification outcomes while respecting immutable variables.',
    latencySla: '< 180ms per individual recourse query',
    memoryFootprint: '< 5 MB gradient state',
    pattern: 'Numerical Constrained Optimizer',
    sourceFile: 'dia/recourse_engine.py',
    endpoints: ['POST /api/v1/optimization/recourse'],
  },
  conformal_sets: {
    id: 'conformal_sets',
    tier: 'Analytical Subsystem',
    icon: 'check-circle',
    title: 'Split Conformal Prediction Sets',
    subtitle: 'Finite-Sample Coverage & Epistemic Bounds',
    tech: 'Conformal Calibration / Mahalanobis Distance',
    responsibility: 'Calculates guaranteed (1 - alpha) prediction sets and decouples aleatoric entropy from epistemic out-of-distribution distance.',
    latencySla: '< 45ms batch calibration',
    memoryFootprint: '< 8 MB non-conformity vector',
    pattern: 'Statistical Calibrator',
    sourceFile: 'dia/conformal_uncertainty.py',
    endpoints: ['POST /api/v1/uncertainty/conformal'],
  },
  governance_dossier: {
    id: 'governance_dossier',
    tier: 'Governance & Output Tier',
    icon: 'file-check',
    title: 'Executive Regulatory Audit Dossier',
    subtitle: 'EU AI Act, Basel & GDPR Compliance Export',
    tech: 'Jinja2 / Great Expectations / Cryptographic Digest',
    responsibility: 'Synthesizes model risk profiles, Great Expectations contracts, GDPR scans, and drift into certified PDF/HTML audit dossiers.',
    latencySla: '< 250ms document assembly',
    memoryFootprint: '< 12 MB rendered buffer',
    pattern: 'Builder / Document Compiler',
    sourceFile: 'dia/report_generator.py',
    endpoints: ['GET /api/v1/export/{session_id}/dossier'],
  },
};

// End-to-End Pipeline Stages
const PIPELINE_STAGES = [
  {
    step: 1,
    name: 'Universal Ingestion & Target Leakage Sleuth',
    files: ['dia/data_loader.py', 'dia/data_sanitizer.py', 'dia/leakage_detector.py'],
    inputs: 'Raw CSV, JSON, Parquet, or SQL connection',
    outputs: 'Sanitized DataFrame, Column Roles Dictionary, Feature Quarantine List',
    formula: 'I(X; Y) / H(Y) \\ge 0.85 \\implies \\text{QUARANTINE_CRITICAL}',
    badge: 'INGESTION & DATA QUALITY',
  },
  {
    step: 2,
    name: 'Symbolic Feature Discovery & Semantic Text Fusion',
    files: ['dia/feature_engineer.py', 'dia/symbolic_features.py', 'dia/multimodal_fusion.py'],
    inputs: 'Sanitized Feature Matrix + Free-Text Columns',
    outputs: 'Non-Linear Invariants (Ratios, Products, Logs) & Dense Semantic Embeddings',
    formula: '\\text{Score}(f) = \\Delta I(f; Y) - \\lambda \\cdot \\text{Complexity}(f)',
    badge: 'FEATURE EXTRACTION',
  },
  {
    step: 3,
    name: 'Observational Causal Discovery & Do-Calculus',
    files: ['dia/causal_discovery.py'],
    inputs: 'Continuous & Discretized Feature Space',
    outputs: 'Completed Partially Directed Acyclic Graph (CPDAG) & Average Treatment Effect (ATE)',
    formula: '\\mathbb{E}[Y \\mid do(X = x)] = \\sum_z \\mathbb{E}[Y \\mid X=x, Z=z] P(Z=z)',
    badge: 'CAUSAL INTELLIGENCE',
  },
  {
    step: 4,
    name: 'AutoML Benchmark, Recourse & Conformal Prediction',
    files: ['dia/model_trainer.py', 'dia/recourse_engine.py', 'dia/conformal_uncertainty.py'],
    inputs: 'Engineered Features, Target Vector, Stratified Splits',
    outputs: 'Champion Ensemble, SHAP Attribution, Wachter Counterfactuals, Coverage Sets',
    formula: '\\min_{x\'} \\text{loss}(f(x\'), y^*) + \\lambda \\sum_{j \\in \\text{mutable}} \\frac{|x_j\' - x_j|}{\\text{MAD}_j}',
    badge: 'AUTONOMOUS ML',
  },
  {
    step: 5,
    name: 'Edge C99 & WASM Transpiler & Regulatory Dossier',
    files: ['dia/wasm_compiler.py', 'dia/report_generator.py', 'dia/drift_sentinel.py'],
    inputs: 'Trained Pipeline Object & Governance Metadata',
    outputs: 'Standalone JavaScript / WebAssembly Bundle (<500KB) & Signed Audit Dossier',
    formula: '\\text{MMD}^2(P, Q) = \\mathbb{E}[k(X,X\')] - 2\\mathbb{E}[k(X,Y)] + \\mathbb{E}[k(Y,Y\')]',
    badge: 'EDGE DEPLOYMENT & GOVERNANCE',
  },
];

// API Endpoints SLA & Contract Matrix
const API_SLA_MATRIX = [
  { method: 'POST', path: '/api/v1/ingest/demo', subsystem: 'Ingestion', slaP50: '18ms', slaP95: '42ms', reqType: '{ dataset_name: string }', resType: 'DatasetMetadataResponse', status: 'Healthy' },
  { method: 'POST', path: '/api/v1/train', subsystem: 'AutoML', slaP50: '3.2s', slaP95: '7.8s', reqType: '{ session_id, goal, target_col }', resType: 'AutoMLResultResponse', status: 'Healthy' },
  { method: 'GET', path: '/api/v1/causal/graph/{id}', subsystem: 'Causal AI', slaP50: '85ms', slaP95: '210ms', reqType: 'Path parameter session_id', resType: 'CausalGraphResponse (nodes, edges)', status: 'Healthy' },
  { method: 'POST', path: '/api/v1/causal/intervention', subsystem: 'Causal AI', slaP50: '45ms', slaP95: '95ms', reqType: '{ session_id, treatment_col, value }', resType: 'DoCalculusResponse (ATE, baseline)', status: 'Healthy' },
  { method: 'POST', path: '/api/v1/optimization/recourse', subsystem: 'Explainability', slaP50: '65ms', slaP95: '140ms', reqType: '{ session_id, row_index, target_class }', resType: 'RecourseResult (shifts, cost)', status: 'Healthy' },
  { method: 'POST', path: '/api/v1/uncertainty/conformal', subsystem: 'Validation', slaP50: '32ms', slaP95: '75ms', reqType: '{ session_id, alpha: 0.1 }', resType: 'ConformalBoundsResponse', status: 'Healthy' },
  { method: 'GET', path: '/api/v1/export/edge-bundle/{id}', subsystem: 'Edge Runtime', slaP50: '28ms', slaP95: '55ms', reqType: 'Path parameter session_id', resType: 'Text/JavaScript (eval bundle)', status: 'Healthy' },
  { method: 'GET', path: '/api/v1/export/{id}/dossier', subsystem: 'Governance', slaP50: '95ms', slaP95: '220ms', reqType: 'Path parameter session_id', resType: 'Text/HTML Audit Dossier', status: 'Healthy' },
  { method: 'GET', path: '/api/v1/system/metrics', subsystem: 'Core Telemetry', slaP50: '4ms', slaP95: '12ms', reqType: 'None', resType: 'SystemMetricsResponse (RAM, CPU, Cache)', status: 'Healthy' },
  { method: 'POST', path: '/api/v1/system/gc', subsystem: 'Core Telemetry', slaP50: '25ms', slaP95: '60ms', reqType: 'None', resType: '{ freed_sessions: int, gc_collected: int }', status: 'Healthy' },
];

export function renderArchitectureCockpit() {
  el.tabContent.innerHTML = `
    <div class="space-y-6 pb-12">
      <!-- 1. Header Banner -->
      <div class="glass-card p-6 relative overflow-hidden">
        <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div class="space-y-2 max-w-3xl">
            <div class="flex items-center space-x-2">
              <span class="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-emerald-500/10 border border-emerald-500/30 text-[#00e575]">
                🏛️ ENTERPRISE SYSTEM ARCHITECTURE &amp; DESIGN BLUEPRINT
              </span>
              <span class="text-xs font-mono px-2.5 py-1 rounded border border-white/10" style="background-color: var(--tag-bg); color: var(--tag-text);">
                C4 Model &bull; HLD &amp; LLD &bull; Design Tokens &bull; API SLAs
              </span>
            </div>
            <h1 class="text-2xl sm:text-3xl font-extrabold tracking-tight" style="color: var(--text-primary);">
              Full-Stack System Architecture &amp; Engineering Blueprint
            </h1>
            <p class="text-sm leading-relaxed" style="color: var(--text-secondary);">
              Interactive C4 Container topologies, asynchronous dataflow lifecycles, Pro Max UI/UX design token matrices, and OpenAPI contract SLA guarantees governing the Data Intelligence Assistant.
            </p>
          </div>

          <!-- Quick Navigation Actions -->
          <div class="flex flex-wrap items-center gap-2 font-mono text-xs">
            <a href="/" class="px-3.5 py-2 rounded-lg border border-white/10 hover:bg-white/5 transition-all text-slate-300 hover:text-white flex items-center space-x-1.5 cursor-pointer no-underline" title="Return to Landing Page">
              <i data-lucide="home" class="w-4 h-4 text-[#00e575]"></i>
              <span>Return to Landing Page</span>
            </a>
            <button onclick="window.openSystemMetrics()" class="px-3.5 py-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 text-[#00e575] font-semibold flex items-center space-x-1.5 cursor-pointer transition-all">
              <i data-lucide="activity" class="w-4 h-4"></i>
              <span>Live System Telemetry HUD</span>
            </button>
          </div>
        </div>

        <!-- Cockpit Mode Tabs -->
        <div class="flex items-center space-x-1 sm:space-x-2 border-b border-white/10 pt-6 font-mono text-xs overflow-x-auto">
          <button onclick="window.switchArchMode('c4_diagram')" id="arch-tab-c4_diagram" class="px-4 py-2 rounded-t-lg font-bold transition-all cursor-pointer border-b-2 ${activeArchView === 'c4_diagram' ? 'border-[#00e575] text-[#00e575] bg-white/5' : 'border-transparent text-slate-400 hover:text-white'}">
            📐 C4 Container Topology
          </button>
          <button onclick="window.switchArchMode('dataflow')" id="arch-tab-dataflow" class="px-4 py-2 rounded-t-lg font-bold transition-all cursor-pointer border-b-2 ${activeArchView === 'dataflow' ? 'border-[#00e575] text-[#00e575] bg-white/5' : 'border-transparent text-slate-400 hover:text-white'}">
            ⚡ 5-Stage Dataflow Pipeline
          </button>
          <button onclick="window.switchArchMode('design_system')" id="arch-tab-design_system" class="px-4 py-2 rounded-t-lg font-bold transition-all cursor-pointer border-b-2 ${activeArchView === 'design_system' ? 'border-[#00e575] text-[#00e575] bg-white/5' : 'border-transparent text-slate-400 hover:text-white'}">
            🎨 Pro Max UI Design System &amp; Tokens
          </button>
          <button onclick="window.switchArchMode('api_sla')" id="arch-tab-api_sla" class="px-4 py-2 rounded-t-lg font-bold transition-all cursor-pointer border-b-2 ${activeArchView === 'api_sla' ? 'border-[#00e575] text-[#00e575] bg-white/5' : 'border-transparent text-slate-400 hover:text-white'}">
            📡 API Contracts &amp; SLA Performance
          </button>
        </div>
      </div>

      <!-- 2. Dynamic View Container -->
      <div id="arch-dynamic-content" class="space-y-6">
        ${renderArchSubView(activeArchView)}
      </div>
    </div>
  `;

  lucide.createIcons();
}

function renderArchSubView(view) {
  switch (view) {
    case 'c4_diagram':
      return renderC4DiagramView();
    case 'dataflow':
      return renderDataflowView();
    case 'design_system':
      return renderDesignSystemView();
    case 'api_sla':
      return renderApiSlaView();
    default:
      return renderC4DiagramView();
  }
}

// ─── 1. C4 Container Topology View ───────────────────────────────────────────

function renderC4DiagramView() {
  const node = C4_NODES[selectedNodeId] || C4_NODES.api_gw;

  return `
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Left: Interactive Topology Grid (8 Cols) -->
      <div class="lg:col-span-7 space-y-4">
        <div class="glass-card p-5">
          <div class="flex items-center justify-between pb-3 border-b border-white/10">
            <div>
              <h3 class="text-sm font-bold font-mono tracking-tight" style="color: var(--text-primary);">Interactive C4 Container Nodes</h3>
              <p class="text-xs" style="color: var(--text-secondary);">Click any node to inspect architectural invariants, memory limits, and data contracts.</p>
            </div>
            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#00e575]/10 text-[#00e575] border border-[#00e575]/30">
              CLICK TO INSPECT
            </span>
          </div>

          <!-- Tier 1: Presentation Tier -->
          <div class="mt-4 space-y-2">
            <div class="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400">1. Client-Side Presentation Tier</div>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              ${renderNodeCard('client_spa')}
              ${renderNodeCard('client_wasm')}
            </div>
          </div>

          <!-- Connecting Arrows -->
          <div class="flex justify-around py-1 text-slate-500 font-mono text-xs">
            <span>&darr; HTTP/3 &amp; WebSocket Telemetry</span>
            <span>&uarr; Edge Standalone JS &darr;</span>
          </div>

          <!-- Tier 2: Gateway & Session Cache -->
          <div class="space-y-2">
            <div class="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400">2. Gateway &amp; In-Memory State Tier</div>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              ${renderNodeCard('api_gw')}
              ${renderNodeCard('session_store')}
            </div>
          </div>

          <!-- Connecting Arrows -->
          <div class="flex justify-around py-1 text-slate-500 font-mono text-xs">
            <span>&darr; Async Task Dispatch</span>
            <span>&darr; Thread-Safe LRU Eviction</span>
          </div>

          <!-- Tier 3: Core Analytical Subsystems -->
          <div class="space-y-2">
            <div class="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400">3. Core Analytical &amp; Governance Subsystems</div>
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              ${renderNodeCard('causal_pc')}
              ${renderNodeCard('automl_cockpit')}
              ${renderNodeCard('recourse_engine')}
              ${renderNodeCard('leakage_sleuth')}
              ${renderNodeCard('conformal_sets')}
              ${renderNodeCard('governance_dossier')}
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Detailed Node Inspector (5 Cols) -->
      <div class="lg:col-span-5">
        <div class="glass-card p-6 sticky top-6 border border-white/10 space-y-5">
          <div class="flex items-start justify-between">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-[#00e575]">
                <i data-lucide="${escapeHtml(node.icon)}" class="w-5 h-5"></i>
              </div>
              <div>
                <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-400">${escapeHtml(node.tier)}</span>
                <h3 class="text-base font-bold font-mono tracking-tight mt-1" style="color: var(--text-primary);">${escapeHtml(node.title)}</h3>
                <p class="text-xs" style="color: var(--text-secondary);">${escapeHtml(node.subtitle)}</p>
              </div>
            </div>
          </div>

          <div class="space-y-3 text-xs">
            <div>
              <span class="font-mono text-slate-400 font-semibold block mb-1">Architecture Pattern:</span>
              <span class="px-2.5 py-1 rounded bg-white/5 border border-white/10 font-mono text-emerald-400">${escapeHtml(node.pattern)}</span>
            </div>

            <div>
              <span class="font-mono text-slate-400 font-semibold block mb-1">Core Responsibility:</span>
              <p class="leading-relaxed p-3 rounded-lg bg-black/20 border border-white/5 text-slate-300">${escapeHtml(node.responsibility)}</p>
            </div>

            <div class="grid grid-cols-2 gap-3 font-mono">
              <div class="p-3 rounded-lg bg-black/20 border border-white/5">
                <span class="text-slate-400 text-[11px] block">Latency SLA Budget</span>
                <span class="text-[#00e575] font-bold text-xs mt-1 block">${escapeHtml(node.latencySla)}</span>
              </div>
              <div class="p-3 rounded-lg bg-black/20 border border-white/5">
                <span class="text-slate-400 text-[11px] block">Memory Footprint</span>
                <span class="text-sky-400 font-bold text-xs mt-1 block">${escapeHtml(node.memoryFootprint)}</span>
              </div>
            </div>

            <div>
              <span class="font-mono text-slate-400 font-semibold block mb-1">Primary Source File:</span>
              <code class="px-2.5 py-1.5 rounded bg-black/30 border border-white/10 font-mono text-[11px] text-amber-300 block overflow-x-auto">${escapeHtml(node.sourceFile)}</code>
            </div>

            <div>
              <span class="font-mono text-slate-400 font-semibold block mb-1">Connected Endpoints &amp; Contracts:</span>
              <ul class="space-y-1 font-mono text-[11px] text-slate-300">
                ${node.endpoints.map(e => `<li class="flex items-center space-x-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#00e575]"></span><span>${escapeHtml(e)}</span></li>`).join('')}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderNodeCard(nodeId) {
  const n = C4_NODES[nodeId];
  if (!n) return '';
  const isSelected = selectedNodeId === nodeId;

  return `
    <div onclick="window.selectArchNode('${escapeHtml(n.id)}')" class="p-3.5 rounded-xl border transition-all cursor-pointer ${isSelected ? 'border-[#00e575] bg-emerald-500/10 shadow-lg shadow-emerald-500/10 scale-[1.02]' : 'border-white/10 bg-black/20 hover:border-white/25 hover:bg-white/5'}">
      <div class="flex items-center space-x-2.5">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center ${isSelected ? 'bg-[#00e575] text-black' : 'bg-white/5 text-[#00e575]'}">
          <i data-lucide="${escapeHtml(n.icon)}" class="w-4 h-4"></i>
        </div>
        <div class="overflow-hidden">
          <h4 class="text-xs font-bold font-mono truncate" style="color: var(--text-primary);">${escapeHtml(n.title)}</h4>
          <span class="text-[10px] font-mono text-slate-400 truncate block">${escapeHtml(n.tech.split('/')[0])}</span>
        </div>
      </div>
    </div>
  `;
}

// ─── 2. 5-Stage Dataflow Pipeline View ───────────────────────────────────────

function renderDataflowView() {
  return `
    <div class="glass-card p-6 space-y-6">
      <div>
        <h3 class="text-base font-bold font-mono tracking-tight" style="color: var(--text-primary);">Asynchronous 5-Stage Analytical Pipeline Lifecycle</h3>
        <p class="text-xs" style="color: var(--text-secondary);">End-to-end data transformation pipeline executing with bounded memory and zero single points of failure.</p>
      </div>

      <div class="space-y-4">
        ${PIPELINE_STAGES.map(s => `
          <div class="p-5 rounded-xl border border-white/10 bg-black/20 hover:border-white/20 transition-all space-y-3">
            <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
              <div class="flex items-center space-x-3">
                <span class="w-7 h-7 rounded-full bg-[#00e575]/20 border border-[#00e575]/50 text-[#00e575] font-mono font-bold text-xs flex items-center justify-center">
                  ${s.step}
                </span>
                <h4 class="text-sm font-bold font-mono" style="color: var(--text-primary);">${escapeHtml(s.name)}</h4>
              </div>
              <span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/10 border border-emerald-500/30 text-[#00e575]">
                ${escapeHtml(s.badge)}
              </span>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
              <div class="p-3 rounded-lg bg-white/5 border border-white/5">
                <span class="text-slate-400 font-mono text-[11px] block mb-1">Inputs</span>
                <span class="font-mono text-slate-200">${escapeHtml(s.inputs)}</span>
              </div>
              <div class="p-3 rounded-lg bg-white/5 border border-white/5">
                <span class="text-slate-400 font-mono text-[11px] block mb-1">Outputs</span>
                <span class="font-mono text-slate-200">${escapeHtml(s.outputs)}</span>
              </div>
              <div class="p-3 rounded-lg bg-white/5 border border-white/5">
                <span class="text-slate-400 font-mono text-[11px] block mb-1">Algorithmic Invariant</span>
                <code class="font-mono text-amber-300 text-[11px]">${escapeHtml(s.formula)}</code>
              </div>
            </div>

            <div class="flex flex-wrap items-center gap-2 pt-1 font-mono text-[11px] text-slate-400">
              <span class="text-slate-500">Source Modules:</span>
              ${s.files.map(f => `<code class="px-2 py-0.5 rounded bg-black/30 border border-white/5 text-emerald-400">${escapeHtml(f)}</code>`).join('')}
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

// ─── 3. UI/UX Pro Max Design System & Component State Matrix ──────────────────

function renderDesignSystemView() {
  return `
    <div class="space-y-6">
      <!-- Design Tokens Overview -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- 8pt Grid -->
        <div class="glass-card p-5 space-y-3">
          <div class="flex items-center space-x-2 text-[#00e575]">
            <i data-lucide="grid" class="w-4 h-4"></i>
            <h4 class="text-xs font-bold font-mono uppercase tracking-wider">8pt Spatial Grid Scale</h4>
          </div>
          <p class="text-xs" style="color: var(--text-secondary);">Mathematical 4pt/8pt increments ensuring visual cadence across viewports.</p>
          <div class="space-y-1.5 font-mono text-xs">
            <div class="flex items-center justify-between p-1.5 rounded bg-white/5"><span>space-1 (4px)</span><span class="w-4 h-2 bg-[#00e575] rounded"></span></div>
            <div class="flex items-center justify-between p-1.5 rounded bg-white/5"><span>space-2 (8px)</span><span class="w-8 h-2 bg-[#00e575] rounded"></span></div>
            <div class="flex items-center justify-between p-1.5 rounded bg-white/5"><span>space-3 (12px)</span><span class="w-12 h-2 bg-[#00e575] rounded"></span></div>
            <div class="flex items-center justify-between p-1.5 rounded bg-white/5"><span>space-4 (16px)</span><span class="w-16 h-2 bg-[#00e575] rounded"></span></div>
            <div class="flex items-center justify-between p-1.5 rounded bg-white/5"><span>space-6 (24px)</span><span class="w-24 h-2 bg-[#00e575] rounded"></span></div>
          </div>
        </div>

        <!-- Typography Hierarchy -->
        <div class="glass-card p-5 space-y-3">
          <div class="flex items-center space-x-2 text-[#00e575]">
            <i data-lucide="type" class="w-4 h-4"></i>
            <h4 class="text-xs font-bold font-mono uppercase tracking-wider">Fluid Typography (Major Third)</h4>
          </div>
          <p class="text-xs" style="color: var(--text-secondary);">Dynamic clamp() formulas adapting between mobile and desktop without layout shifts.</p>
          <div class="space-y-2 text-xs">
            <div><span class="text-lg font-bold block" style="color: var(--text-primary);">Display Title</span><span class="font-mono text-[10px] text-slate-400">clamp(1.5rem, 1.2rem + 1.2vw, 2.5rem)</span></div>
            <div><span class="text-sm font-bold block" style="color: var(--text-primary);">Section Heading</span><span class="font-mono text-[10px] text-slate-400">1.125rem (18px) &bull; Bold</span></div>
            <div><span class="text-xs font-medium block" style="color: var(--text-secondary);">Body Paragraph</span><span class="font-mono text-[10px] text-slate-400">1rem (16px) &bull; line-height 1.55</span></div>
            <div><code class="font-mono text-[11px] text-[#00e575]">Mono Data Telemetry</code><span class="font-mono text-[10px] text-slate-400 block">JetBrains Mono &bull; Tabular Nums</span></div>
          </div>
        </div>

        <!-- Frosted Glass Elevation -->
        <div class="glass-card p-5 space-y-3">
          <div class="flex items-center space-x-2 text-[#00e575]">
            <i data-lucide="layers" class="w-4 h-4"></i>
            <h4 class="text-xs font-bold font-mono uppercase tracking-wider">Frosted Glass Tiers &amp; Bezel</h4>
          </div>
          <p class="text-xs" style="color: var(--text-secondary);">Multi-tier frosted backdrop-blur with inner-bezel highlight layers.</p>
          <div class="space-y-2 font-mono text-xs">
            <div class="p-2.5 rounded-lg border border-white/10 bg-white/5">
              <span class="text-slate-300 font-bold block">Tier 1: Card Bezel</span>
              <span class="text-[10px] text-slate-400 block mt-1">inset 0 1px 0 0 rgba(255,255,255,0.08)</span>
            </div>
            <div class="p-2.5 rounded-lg border border-white/15 bg-white/10">
              <span class="text-slate-300 font-bold block">Tier 2: Modal Backdrop</span>
              <span class="text-[10px] text-slate-400 block mt-1">backdrop-blur-md &bull; z-index 50</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Live Component State Matrix (8 Mandatory States) -->
      <div class="glass-card p-6 space-y-5">
        <div>
          <div class="flex items-center space-x-2">
            <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/10 text-[#00e575] border border-emerald-500/30">WCAG 2.2 AAA COMPLIANT</span>
            <h3 class="text-sm font-bold font-mono tracking-tight" style="color: var(--text-primary);">Mandatory 8-State Interactive Component Matrix</h3>
          </div>
          <p class="text-xs mt-1" style="color: var(--text-secondary);">Interactive preview of the 8 canonical visual states for buttons and form controls.</p>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <!-- 1. Default -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">1. DEFAULT</span>
            <button class="px-3.5 py-1.5 rounded-lg bg-[#00e575] text-black font-bold text-xs w-full cursor-pointer">
              Launch Demo
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">Standard contrast</span>
          </div>

          <!-- 2. Hover -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">2. HOVER</span>
            <button class="px-3.5 py-1.5 rounded-lg bg-[#00ff82] text-black font-bold text-xs w-full shadow-lg shadow-emerald-500/20 -translate-y-0.5 transition-all cursor-pointer">
              Launch Demo
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">+4px elevation</span>
          </div>

          <!-- 3. Active / Pressed -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">3. ACTIVE</span>
            <button class="px-3.5 py-1.5 rounded-lg bg-[#00c966] text-black font-bold text-xs w-full scale-95 transition-transform cursor-pointer">
              Launch Demo
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">scale(0.98) tactile</span>
          </div>

          <!-- 4. Focus Visible -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">4. FOCUS-VISIBLE</span>
            <button class="px-3.5 py-1.5 rounded-lg bg-[#00e575] text-black font-bold text-xs w-full ring-2 ring-white ring-offset-2 ring-offset-black cursor-pointer">
              Launch Demo
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">2px offset ring</span>
          </div>

          <!-- 5. Disabled -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">5. DISABLED</span>
            <button disabled class="px-3.5 py-1.5 rounded-lg bg-[#00e575] text-black font-bold text-xs w-full opacity-45 cursor-not-allowed">
              Launch Demo
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">opacity 45% inert</span>
          </div>

          <!-- 6. Loading -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">6. LOADING</span>
            <button class="px-3.5 py-1.5 rounded-lg bg-[#00e575] text-black font-bold text-xs w-full flex items-center justify-center space-x-1.5">
              <i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i>
              <span>Fitting...</span>
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">Zero layout shift</span>
          </div>

          <!-- 7. Empty State -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">7. EMPTY STATE</span>
            <div class="p-2 rounded bg-white/5 border border-dashed border-white/20 text-[11px] text-slate-400">
              No Data &bull; 1-Click Launch
            </div>
            <span class="text-[10px] font-mono text-slate-500 block">Context onboarding</span>
          </div>

          <!-- 8. Error / Destructive -->
          <div class="p-3.5 rounded-xl border border-white/10 bg-black/20 text-center space-y-2">
            <span class="text-[10px] font-mono font-bold text-slate-400 block">8. ERROR</span>
            <button class="px-3.5 py-1.5 rounded-lg bg-red-500/20 border border-red-500 text-red-400 font-bold text-xs w-full">
              Failed &bull; Retry
            </button>
            <span class="text-[10px] font-mono text-slate-500 block">Actionable retry</span>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ─── 4. API Contracts & SLA Performance Matrix View ──────────────────────────

function renderApiSlaView() {
  return `
    <div class="glass-card p-6 space-y-5">
      <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
        <div>
          <h3 class="text-base font-bold font-mono tracking-tight" style="color: var(--text-primary);">OpenAPI 3.1 &amp; Pydantic V2 Service Contract Matrix</h3>
          <p class="text-xs" style="color: var(--text-secondary);">Production SLA latency targets, typed schemas, and error boundaries across all endpoints.</p>
        </div>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 font-mono text-xs text-emerald-400 flex items-center space-x-1.5 transition-all">
          <i data-lucide="external-link" class="w-3.5 h-3.5"></i>
          <span>Open Interactive Swagger UI</span>
        </a>
      </div>

      <!-- Table Container -->
      <div class="overflow-x-auto border border-white/10 rounded-xl">
        <table class="w-full text-left border-collapse text-xs font-mono">
          <thead>
            <tr class="border-b border-white/10" style="background-color: var(--bg-secondary);">
              <th class="p-3 text-slate-400 font-semibold">METHOD &bull; PATH</th>
              <th class="p-3 text-slate-400 font-semibold">SUBSYSTEM</th>
              <th class="p-3 text-slate-400 font-semibold">SLA (p50 / p95)</th>
              <th class="p-3 text-slate-400 font-semibold">REQUEST PAYLOAD</th>
              <th class="p-3 text-slate-400 font-semibold">RESPONSE PAYLOAD</th>
              <th class="p-3 text-slate-400 font-semibold text-right">STATUS</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-white/5">
            ${API_SLA_MATRIX.map(r => `
              <tr class="hover:bg-white/5 transition-colors">
                <td class="p-3 font-bold">
                  <span class="px-2 py-0.5 rounded text-[10px] mr-1.5 ${r.method === 'POST' ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}">
                    ${escapeHtml(r.method)}
                  </span>
                  <span style="color: var(--text-primary);">${escapeHtml(r.path)}</span>
                </td>
                <td class="p-3 text-slate-400">${escapeHtml(r.subsystem)}</td>
                <td class="p-3">
                  <span class="text-[#00e575] font-bold">${escapeHtml(r.slaP50)}</span>
                  <span class="text-slate-500"> / </span>
                  <span class="text-slate-400">${escapeHtml(r.slaP95)}</span>
                </td>
                <td class="p-3 text-slate-400 truncate max-w-[200px]" title="${escapeHtml(r.reqType)}">${escapeHtml(r.reqType)}</td>
                <td class="p-3 text-slate-300 truncate max-w-[220px]" title="${escapeHtml(r.resType)}">${escapeHtml(r.resType)}</td>
                <td class="p-3 text-right">
                  <span class="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] bg-emerald-500/10 border border-emerald-500/30 text-[#00e575]">
                    <span class="w-1 h-1 rounded-full bg-[#00e575]"></span>
                    <span>${escapeHtml(r.status)}</span>
                  </span>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ─── Global Handler Registrations ───────────────────────────────────────────

window.switchArchMode = function (mode) {
  activeArchView = mode;
  const container = document.getElementById('arch-dynamic-content');
  if (container) {
    container.innerHTML = renderArchSubView(activeArchView);
    lucide.createIcons();
  }

  // Update tab highlights
  ['c4_diagram', 'dataflow', 'design_system', 'api_sla'].forEach(m => {
    const tab = document.getElementById(`arch-tab-${m}`);
    if (tab) {
      if (m === mode) {
        tab.className = 'px-4 py-2 rounded-t-lg font-bold transition-all cursor-pointer border-b-2 border-[#00e575] text-[#00e575] bg-white/5';
      } else {
        tab.className = 'px-4 py-2 rounded-t-lg font-bold transition-all cursor-pointer border-b-2 border-transparent text-slate-400 hover:text-white';
      }
    }
  });
};

window.selectArchNode = function (nodeId) {
  selectedNodeId = nodeId;
  const container = document.getElementById('arch-dynamic-content');
  if (container && activeArchView === 'c4_diagram') {
    container.innerHTML = renderC4DiagramView();
    lucide.createIcons();
  }
};
