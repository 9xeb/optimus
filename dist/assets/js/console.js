
const TASKS_URL = '/tasks', REVIEW_URL = '/review', POLL_INTERVAL = 2000, DEFAULT_TTL = 60;
const CANCEL_URL = '/tasks/cancel';
const SANDBOX_URL = `http://${location.hostname || '127.0.0.1'}:8080/`;
const STREAM_URL = '/tasks/stream';
const STREAM_QUIET_MS = 2000, STREAM_STALL_MS = 15000;
const LABEL_MAX_CHARS = 22, LABEL_CHAR_W = 6.6, LABEL_LINE_H = 13, LABEL_GAP = 16;
const EVENT_LABELS = { 'task-proposal': 'TASK', 'tool-approval': 'ACTION', 'question': 'QUESTION', 'review': 'REVIEW' };
const EVENT_CLASSES = { 'task-proposal': 'event-task', 'tool-approval': 'event-action', 'question': 'event-question', 'review': 'event-review' };

const HOLD_MS = 350, MOVE_SLOP = 8;
const state = {
  tasks: new Map(),
  mode: 'map',
  activeRoot: null,
  expanded: new Set(),
  selected: null,
  anchorId: null,
  sandboxReturnMode: 'map',
  situation: null,
  situationSig: null,
  situationStatus: 'idle',
  reviewDraft: '',
  reviewError: null,
  composerMode: null,
  actionLock: false,
  actionLockTimer: null,
  layout: null,
  signature: null,
  viewport: d3.zoomIdentity,
  fitPending: true,
  layoutChanged: false,
  run: { id: null, pending: false },
  runPrompt: { open: false, taskId: null, tip: '', tipSaved: false, pending: false, error: null },
  mutation: { kind: null },
  form: { open: false, x: 0, y: 0, title: '', deps: [], error: null, pending: false, sourceId: null },
  link: { open: false, taskId: null, query: '', items: [], index: -1, error: null, pending: false },
  deleteConfirm: { open: false, pending: false, error: null },
  gesture: { mode: 'idle', pointerId: null, sourceId: null, edge: null, startX: 0, startY: 0, x: 0, y: 0, targetId: null, targetState: 'empty', holdRAF: 0, edgePath: null, suppressClickUntil: 0 },
  toast: { timer: null },
  stop: { confirming: false, pending: false, error: null },
  stream: {
    status: 'idle',
    task: null,
    lastChunkAt: null,
    chunkTimes: [],
    rate: 0,
    pulseTimer: null,
    lastRefreshAt: 0,
    action: null,
    activitySig: null,
  },
  reportsSeeded: false,
  seenAt: new Map(),
  unread: new Map(),
  find: { open: false, query: '', matches: [], index: 0, returnRoot: null, returnExpanded: null, returnSelected: null },
  inspectorSig: null,
  inspectorTaskId: null,
  inspectorTrajectorySig: null,
  inspectorInstruction: null,
  inspectorTab: 'run',
  inspectorResetRunTop: false,
  inspectorResetReportTop: false,
  inspectorLastPaintedTab: null,
  trajectoryFollow: true,
  scrollGuardUntil: 0,
  error: null,
};

const reduceMotion = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function el(t, p = {}, c = []) {
  const n = document.createElement(t);
  for (const [k, v] of Object.entries(p)) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (v != null) n.setAttribute(k, v);
  }
  for (const x of [].concat(c)) {
    if (x == null) continue;
    n.append(typeof x === 'string' ? document.createTextNode(x) : x);
  }
  return n;
}

function normKind(k) { return k === 'root' ? 'root' : (k === 'complex' ? 'complex' : 'simple'); }
function normState(s) { return s === 'awake' ? 'awake' : (s === 'failing' ? 'failing' : 'sleeping'); }
function nodeRadius(t) { return t && (t.kind === 'root' || t.kind === 'complex') ? 26 : 18; }
function ghostId(title) { return '__sit__:' + title; }

function wrapTitle(title, maxChars = LABEL_MAX_CHARS) {
  const words = String(title || '').split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  const push = s => { if (s) lines.push(s); };
  for (let word of words) {
    while (word.length > maxChars) {
      if (line) { push(line); line = ''; }
      push(word.slice(0, maxChars));
      word = word.slice(maxChars);
    }
    if (!word) continue;
    const next = line ? `${line} ${word}` : word;
    if (line && next.length > maxChars) { push(line); line = word; }
    else line = next;
  }
  push(line);
  if (!lines.length) lines.push('');
  return lines;
}
function labelMetrics(title) {
  const lines = wrapTitle(title);
  const width = Math.max(1, ...lines.map(l => l.length)) * LABEL_CHAR_W;
  return { lines, width, height: lines.length * LABEL_LINE_H };
}

function normalizeSituation(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const type = typeof raw.type === 'string' ? raw.type : '';
  if (!type) return null;
  const parent = raw.parent != null && String(raw.parent).trim() ? String(raw.parent) : null;
  if (type === 'question') {
    if (raw.question == null || !String(raw.question).trim()) return null;
    return { type, parent, question: String(raw.question) };
  }
  if (type === 'task-proposal') {
    const list = Array.isArray(raw.tasks) ? raw.tasks : [];
    const tasks = list.map(t => ({
      title: String(t.title || t.id || ''),
      task: String(t.task || t.instruction || ''),
      kind: normKind(t.kind),
      dependencies: Array.isArray(t.dependencies) ? t.dependencies.map(String) : [],
      ttl: Number(t.ttl) > 0 ? Number(t.ttl) : DEFAULT_TTL,
    })).filter(t => t.title);
    if (!tasks.length) return null;
    return { type, parent, tasks };
  }
  if (type === 'tool-approval') {
    const tool = raw.tool != null ? String(raw.tool).trim() : '';
    if (!tool) return { type, parent, unsupported: true };
    const args = raw.args == null ? {} : raw.args;
    const justification = raw.justification == null ? '' : String(raw.justification).trim();
    return { type, parent, tool, args, justification };
  }
  if (type === 'review') {
    const task = raw.task != null && String(raw.task).trim() ? String(raw.task) : null;
    return { type, parent: task };
  }
  return { type, parent, unsupported: true };
}
function situationSignature(s) { return s ? JSON.stringify(s) : null; }

function applyTasks(raw) {
  const arr = Array.isArray(raw)
    ? raw
    : (raw && typeof raw === 'object' ? Object.entries(raw).map(([k, v]) => ({ title: k, ...(v || {}) })) : []);
  const map = new Map();
  for (const t of arr) {
    const id = t.title || t.id || t.name;
    if (!id) continue;
    map.set(id, {
      id,
      title: id,
      instruction: t.task || t.instruction || '',
      deps: Array.isArray(t.dependencies) ? t.dependencies.map(String) : [],
      kind: normKind(t.kind),
      state: normState(t.state),
      ttl: Number(t.ttl) > 0 ? Number(t.ttl) : DEFAULT_TTL,
      report: t.report || '',
      reportAt: t.timestamp || t.reportAt || null,
      trajectory: Array.isArray(t.trajectory) ? t.trajectory.map(String) : [],
    });
  }
  return map;
}

function getRoots() { return [...state.tasks.values()].filter(t => t.kind === 'root'); }
function rootGraph() {
  const roots = getRoots().map(r => r.id).sort((a, b) => a.localeCompare(b));
  const ids = new Set(roots);
  const adj = new Map(roots.map(id => [id, new Set()]));
  for (const id of roots) {
    const t = state.tasks.get(id);
    for (const d of t.deps) {
      if (!ids.has(d)) continue;
      adj.get(id).add(d);
      adj.get(d).add(id);
    }
  }
  return { roots, adj };
}
function orderedRoots(graph = rootGraph()) {
  const seen = new Set(), out = [];
  for (const start of graph.roots) {
    if (seen.has(start)) continue;
    const q = [start];
    seen.add(start);
    while (q.length) {
      const id = q.shift();
      out.push(id);
      const next = [...(graph.adj.get(id) || [])]
        .filter(n => !seen.has(n))
        .sort((a, b) => a.localeCompare(b));
      for (const n of next) { seen.add(n); q.push(n); }
    }
  }
  return out;
}
function rootComponentOf(rootId, graph = rootGraph()) {
  if (!graph.adj.has(rootId)) return new Set();
  const seen = new Set([rootId]);
  const q = [rootId];
  while (q.length) {
    const id = q.shift();
    for (const n of graph.adj.get(id) || []) {
      if (!seen.has(n)) { seen.add(n); q.push(n); }
    }
  }
  return seen;
}
function getClosure(rid) {
  const seen = new Set([rid]);
  const q = [rid];
  while (q.length) {
    const c = q.shift();
    const t = state.tasks.get(c);
    if (!t) continue;
    for (const d of t.deps) {
      if (!seen.has(d) && state.tasks.has(d)) { seen.add(d); q.push(d); }
    }
  }
  return seen;
}
function rootOwning(id) {
  if (!state.tasks.has(id)) return null;
  if (state.tasks.get(id).kind === 'root') return id;
  for (const r of getRoots()) {
    if (getClosure(r.id).has(id)) return r.id;
  }
  return null;
}
function visibleSet(rootId) {
  const seen = new Set([rootId]);
  const q = [rootId];
  while (q.length) {
    const id = q.shift();
    if (!state.expanded.has(id)) continue;
    const t = state.tasks.get(id);
    if (!t) continue;
    for (const d of t.deps) {
      if (!seen.has(d) && state.tasks.has(d)) { seen.add(d); q.push(d); }
    }
  }
  return seen;
}
function ensurePath(rootId, taskId) {
  if (!rootId || !taskId || !state.tasks.has(rootId) || !state.tasks.has(taskId)) return false;
  state.expanded.add(rootId);
  if (rootId === taskId) return true;
  const parent = new Map();
  const seen = new Set([rootId]);
  const q = [rootId];
  let found = false;
  while (q.length) {
    const id = q.shift();
    if (id === taskId) { found = true; break; }
    const t = state.tasks.get(id);
    if (!t) continue;
    for (const d of t.deps) {
      if (state.tasks.has(d) && !seen.has(d)) {
        seen.add(d);
        parent.set(d, id);
        q.push(d);
      }
    }
  }
  if (!found) return false;
  let cur = taskId;
  while (parent.has(cur)) {
    cur = parent.get(cur);
    state.expanded.add(cur);
  }
  return true;
}

function currentAwakeTask() {
  for (const t of state.tasks.values()) if (t.state === 'awake') return t;
  return null;
}
function awakeCount() {
  let n = 0;
  for (const t of state.tasks.values()) if (t.state === 'awake') n++;
  return n;
}
function markSeen(id) {
  const t = state.tasks.get(id);
  if (!t || !t.report) return;
  state.seenAt.set(id, t.reportAt || '');
  state.unread.delete(id);
}
function observeReports(prevTasks, nextTasks) {
  const first = !state.reportsSeeded;
  for (const [id, t] of nextTasks) {
    if (!t.report) {
      state.seenAt.delete(id);
      state.unread.delete(id);
      continue;
    }
    const at = t.reportAt || '';
    if (first) { state.seenAt.set(id, at); continue; }
    const prev = prevTasks.get(id);
    const prevAt = prev ? (prev.reportAt || '') : null;
    if (prevAt !== at && (!state.seenAt.has(id) || state.seenAt.get(id) !== at)) {
      state.unread.set(id, at);
    }
  }
  for (const id of [...state.seenAt.keys()]) if (!nextTasks.has(id)) state.seenAt.delete(id);
  for (const id of [...state.unread.keys()]) {
    if (!nextTasks.has(id) || !nextTasks.get(id).report) state.unread.delete(id);
  }
  state.reportsSeeded = true;
}
function activeGhosts() {
  const s = state.situation;
  if (!s || s.type !== 'task-proposal' || s.unsupported || state.situationStatus === 'settled') return [];
  return s.tasks;
}
function revealAnchor(parent) {
  if (!parent || !state.tasks.has(parent)) { state.anchorId = null; return; }
  state.anchorId = parent;
  const root = rootOwning(parent);
  if (!root) return;
  const changedRoot = state.activeRoot !== root;
  if (changedRoot) {
    state.activeRoot = root;
    state.expanded = new Set([root]);
  }
  const before = state.expanded.size;
  ensurePath(root, parent);
  if (changedRoot || state.expanded.size !== before) state.fitPending = true;
}

function applyEnvelope(env) {
  const prevTasks = state.tasks;
  const prevSelected = state.selected ? prevTasks.get(state.selected) : null;
  const hadSelectedFocus = !!state.selected && (
    actionRing.node().contains(document.activeElement) ||
    document.getElementById('inspector').contains(document.activeElement) ||
    document.getElementById('delete-popover').contains(document.activeElement) ||
    document.getElementById('run-popover').contains(document.activeElement)
  );
  state.tasks = applyTasks(env.tasks);
  const lostSelected = !!state.selected && !state.tasks.has(state.selected);
  if (state.gesture.mode !== 'idle') {
    const src = state.gesture.sourceId;
    const tgt = state.gesture.targetId;
    if ((src && !state.tasks.has(src)) || (tgt && !state.tasks.has(tgt))) cancelGesture();
  }
  const next = normalizeSituation(env.situation);
  const sig = situationSignature(next);
  if (sig !== state.situationSig) {
    state.situation = next;
    state.situationSig = sig;
    state.situationStatus = 'idle';
    state.reviewDraft = '';
    state.reviewError = null;
    state.actionLock = true;
    if (state.actionLockTimer) clearTimeout(state.actionLockTimer);
    state.actionLockTimer = setTimeout(() => {
      state.actionLock = false;
      state.actionLockTimer = null;
      render();
    }, 300);
    if (next && !next.unsupported) revealAnchor(next.parent);
    else state.anchorId = null;
    if (state.gesture.mode !== 'idle') cancelGesture();
    if (state.deleteConfirm.open) { state.deleteConfirm = { open: false, pending: false, error: null }; }
  }
  observeReports(prevTasks, state.tasks);

  if (state.activeRoot && (!state.tasks.has(state.activeRoot) || state.tasks.get(state.activeRoot).kind !== 'root')) {
    state.activeRoot = null;
    state.expanded.clear();
    state.selected = null;
  }
  if (state.selected && !state.tasks.has(state.selected)) {
    state.selected = null;
    state.deleteConfirm = { open: false, pending: false, error: null };
    state.runPrompt = { open: false, taskId: null, tip: '', tipSaved: false, pending: false, error: null };
  }
  if (state.selected && !lostSelected) {
    const nextSelected = state.tasks.get(state.selected);
    if (prevSelected && nextSelected) {
      const reportChanged = !!nextSelected.report && (
        prevSelected.report !== nextSelected.report ||
        prevSelected.reportAt !== nextSelected.reportAt
      );
      const becameAwake = prevSelected.state !== 'awake' && nextSelected.state === 'awake';
      if (reportChanged) {
        state.inspectorTab = 'report';
        state.inspectorResetReportTop = true;
      } else if (becameAwake) {
        state.inspectorTab = 'run';
        state.inspectorResetRunTop = true;
        state.trajectoryFollow = true;
      }
    }
  }
  if (state.anchorId && !state.tasks.has(state.anchorId)) state.anchorId = null;
  if (state.selected && state.unread.has(state.selected)) markSeen(state.selected);

  const awake = currentAwakeTask();
  if (awake && state.activeRoot && !state.find.open && rootOwning(awake.id) === state.activeRoot) {
    const before = state.expanded.size;
    ensurePath(state.activeRoot, awake.id);
    if (state.expanded.size !== before) state.fitPending = true;
  }
  if (state.find.query.trim()) recomputeFind(false);
  render();
  if (lostSelected && hadSelectedFocus) focusGraph();
  syncStream();
  if (state.stop.confirming && !workInFlight()) closeStopConfirm();
}

const svg = d3.select('#graph'), g = svg.append('g'), edgeL = g.append('g'), edgeHitL = g.append('g'), gestureL = g.append('g'), nodeL = g.append('g'), actionL = g.append('g');
actionL.attr('id', 'node-actions-layer');
const actionRing = actionL.append('g').attr('id', 'node-actions').style('display', 'none');
const runAction = actionRing.append('g').attr('class', 'node-action run').attr('role', 'button').attr('tabindex', '0');
runAction.append('circle').attr('class', 'action-disc').attr('r', 15);
runAction.append('path').attr('class', 'action-icon').attr('d', 'M-4,-6.5 L7,0 L-4,6.5 Z');
runAction.append('title');
const deleteAction = actionRing.append('g').attr('class', 'node-action delete').attr('role', 'button').attr('tabindex', '0');
deleteAction.append('circle').attr('class', 'action-disc').attr('r', 15);
deleteAction.append('path').attr('class', 'action-icon').attr('d', 'M-5,-5 L5,5 M5,-5 L-5,5');
deleteAction.append('title');
runAction.on('pointerdown', e => e.stopPropagation());
deleteAction.on('pointerdown', e => e.stopPropagation());
runAction.on('click', e => { e.stopPropagation(); activateNodeAction('run'); });
deleteAction.on('click', e => { e.stopPropagation(); activateNodeAction('delete'); });
runAction.on('keydown', e => actionKeydown(e, 'run'));
deleteAction.on('keydown', e => actionKeydown(e, 'delete'));
gestureL.attr('id', 'gesture-layer').style('display', 'none');
const gestureArrow = gestureL.append('path').attr('class', 'gesture-arrow').attr('marker-end', 'url(#arr)');
const gestureEmpty = gestureL.append('g').attr('class', 'gesture-empty-group');
gestureEmpty.append('circle').attr('class', 'gesture-empty').attr('r', 12);
gestureEmpty.append('text').attr('class', 'gesture-empty-mark').text('+');
const edgeProgress = gestureL.append('path').attr('class', 'edge-progress').style('display', 'none');
const edgeDefs = svg.append('defs');
for (const [id, size] of [['arr', 6], ['arr-root', 8]]) {
  edgeDefs.append('marker')
    .attr('id', id)
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 9)
    .attr('refY', 5)
    .attr('markerWidth', size)
    .attr('markerHeight', size)
    .attr('orient', 'auto-start-reverse')
    .append('path')
    .attr('d', 'M0,0 L10,5 L0,10 z')
    .attr('fill', 'context-stroke');
}
const zoom = d3.zoom().scaleExtent([.12, 4]).filter(event => {
  if (event.type !== 'wheel' && event.button) return false;
  const target = event.target;
  if (event.type !== 'wheel' && target && target.closest && (target.closest('g.node') || target.closest('.edge-hit') || target.closest('.node-action'))) return false;
  return !event.ctrlKey || event.type === 'wheel';
}).on('zoom', e => {
  state.viewport = e.transform;
  g.attr('transform', e.transform);
  paintNodeActions();
  if (state.deleteConfirm.open) positionDeletePopover();
  if (state.runPrompt.open) positionRunPrompt();
});
svg.call(zoom).on('dblclick.zoom', null);
svg.on('click', e => {
  if (e.target !== svg.node()) return;
  if (state.find.open) return;
  if (state.form.open || state.link.open) return;
  if (Date.now() < state.gesture.suppressClickUntil) return;
  if (state.selected || state.activeRoot) { overview(); return; }
  openCreateForm(e.clientX, e.clientY, []);
});

function build() {
  const L = { nodes: new Map(), edges: [], ghosts: new Set(), visible: new Set() };
  const ghosts = activeGhosts();
  const ghostTitles = new Set(ghosts.map(x => x.title));
  const findMode = state.find.open;
  let visible = null;
  const contextRoots = new Set();
  if (findMode) visible = new Set(state.tasks.keys());
  else if (state.activeRoot && state.tasks.has(state.activeRoot)) visible = visibleSet(state.activeRoot);
  if (visible && !findMode && state.activeRoot) {
    const graph = rootGraph();
    for (const id of rootComponentOf(state.activeRoot, graph)) {
      if (!visible.has(id)) { visible.add(id); contextRoots.add(id); }
    }
  }

  if (visible) {
    L.visible = new Set(visible);
    const real = [...visible].map(id => state.tasks.get(id)).filter(Boolean);
    const gg = new dagre.graphlib.Graph();
    gg.setGraph({ rankdir: 'BT', nodesep: 50, ranksep: 80, marginx: 60, marginy: 60 });
    gg.setDefaultEdgeLabel(() => ({}));
    for (const t of real) {
      const r = nodeRadius(t), lm = labelMetrics(t.title);
      gg.setNode(t.id, { width: Math.max(r * 2 + 50, lm.width + 24), height: r * 2 + LABEL_GAP + lm.height });
    }
    for (const t of real) {
      for (const d of t.deps) {
        if (visible.has(d) && state.tasks.has(d)) gg.setEdge(d, t.id);
      }
    }
    for (const ghost of ghosts) {
      const gid = ghostId(ghost.title), lm = labelMetrics(ghost.title);
      L.ghosts.add(gid);
      gg.setNode(gid, { width: Math.max(18 * 2 + 50, lm.width + 24), height: 18 * 2 + LABEL_GAP + lm.height });
    }
    for (const ghost of ghosts) {
      const gid = ghostId(ghost.title);
      for (const d of ghost.dependencies) {
        if (visible.has(d) && state.tasks.has(d)) gg.setEdge(d, gid);
        else if (ghostTitles.has(d)) gg.setEdge(ghostId(d), gid);
      }
    }
    dagre.layout(gg);
    for (const id of gg.nodes()) {
      const nd = gg.node(id);
      if (L.ghosts.has(id)) {
        const ghost = ghosts.find(x => ghostId(x.title) === id);
        const title = ghost ? ghost.title : id;
        L.nodes.set(id, { x: nd.x, y: nd.y, r: 18, ghost: true, title, kind: ghost ? ghost.kind : 'simple', label: labelMetrics(title) });
      } else {
        const t = state.tasks.get(id);
        L.nodes.set(id, { x: nd.x, y: nd.y, r: nodeRadius(t), label: labelMetrics(t.title), context: contextRoots.has(id) });
      }
    }
    for (const e of gg.edges()) {
      const s = state.tasks.get(e.v), t = state.tasks.get(e.w);
      L.edges.push({
        source: e.v,
        target: e.w,
        proposal: L.ghosts.has(e.v) || L.ghosts.has(e.w),
        root: !!s && !!t && s.kind === 'root' && t.kind === 'root',
        context: contextRoots.has(e.v) || contextRoots.has(e.w),
      });
    }
  } else {
    const rootIds = orderedRoots();
    L.visible = new Set(rootIds);
    const R = Math.max(200, rootIds.length * 60);
    rootIds.forEach((id, i) => {
      const ang = (i / Math.max(1, rootIds.length)) * Math.PI * 2 - Math.PI / 2;
      const t = state.tasks.get(id);
      L.nodes.set(id, { x: R * Math.cos(ang), y: R * Math.sin(ang), r: 26, label: labelMetrics(t.title), context: false });
    });
    for (const id of rootIds) {
      const t = state.tasks.get(id);
      for (const d of t.deps) {
        if (L.visible.has(d) && state.tasks.get(d).kind === 'root') {
          L.edges.push({ source: d, target: id, root: true, context: false });
        }
      }
    }
    if (ghosts.length) {
      const y = -R - 140;
      ghosts.forEach((ghost, i) => {
        const gid = ghostId(ghost.title);
        L.ghosts.add(gid);
        L.nodes.set(gid, { x: (i - (ghosts.length - 1) / 2) * 150, y, r: 18, ghost: true, title: ghost.title, kind: ghost.kind, label: labelMetrics(ghost.title) });
      });
      for (const ghost of ghosts) {
        const gid = ghostId(ghost.title);
        for (const d of ghost.dependencies) {
          if (ghostTitles.has(d)) L.edges.push({ source: ghostId(d), target: gid, proposal: true });
          else if (state.tasks.has(d)) L.edges.push({ source: d, target: gid, proposal: true });
        }
      }
    }
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of L.nodes.values()) {
    const lm = p.label || { width: 0, height: 0 };
    minX = Math.min(minX, p.x - Math.max(p.r, lm.width / 2));
    maxX = Math.max(maxX, p.x + Math.max(p.r, lm.width / 2));
    minY = Math.min(minY, p.y - p.r);
    maxY = Math.max(maxY, p.y + p.r + LABEL_GAP + lm.height);
  }
  if (L.nodes.size) L.bounds = { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
  state.layoutChanged = true;
  return L;
}

function canvasWidth() {
  const graph = document.getElementById('graph');
  return graph.clientWidth || 800;
}
function applyZoom(transform, animate = true) {
  if (!animate || reduceMotion()) { svg.call(zoom.transform, transform); return; }
  svg.transition().duration(300).call(zoom.transform, transform);
}
function fit() {
  const L = state.layout;
  if (!L || !L.bounds) return;
  const b = L.bounds;
  const w = canvasWidth();
  const h = document.getElementById('graph').clientHeight || 600;
  const k = Math.min(w / (b.width + 140), h / (b.height + 140), 2);
  const target = d3.zoomIdentity.translate(w / 2, h / 2).scale(k).translate(-(b.minX + b.maxX) / 2, -(b.minY + b.maxY) / 2);
  applyZoom(target);
}
function centerOn(id) {
  const p = state.layout && state.layout.nodes.get(id);
  if (!p) return;
  const graph = document.getElementById('graph');
  const w = canvasWidth(), h = graph.clientHeight || 600;
  const t = state.viewport;
  const sx = t.applyX(p.x), sy = t.applyY(p.y);
  if (sx > 60 && sx < w - 60 && sy > 60 && sy < h - 60) return;
  const k = Math.max(.6, Math.min(t.k || 1, 1.4));
  applyZoom(d3.zoomIdentity.translate(w / 2, h / 2).scale(k).translate(-p.x, -p.y));
}

/* ---- direct manipulation ---- */
function focusGraph() {
  const graph = document.getElementById('graph');
  if (graph) graph.focus({ preventScroll: true });
}
function actionRingVisible() {
  const t = state.selected ? state.tasks.get(state.selected) : null;
  return !!(
    t &&
    state.layout && state.layout.nodes.has(state.selected) &&
    state.mode !== 'sandbox' &&
    state.gesture.mode === 'idle' &&
    !state.form.open &&
    !state.link.open
  );
}
function selectedNodePos() {
  const p = state.layout && state.selected ? state.layout.nodes.get(state.selected) : null;
  if (!p) return null;
  let x = p.x, y = p.y;
  const n = nodeL.selectAll('g.node').filter(d => d.id === state.selected).node();
  if (n && n.transform && n.transform.baseVal.numberOfItems) {
    const m = n.transform.baseVal.consolidate();
    if (m) { x = m.matrix.e; y = m.matrix.f; }
  }
  return { x, y, r: p.r };
}
function paintNodeActions() {
  const t = state.selected ? state.tasks.get(state.selected) : null;
  const p = selectedNodePos();
  const visible = !!(t && p && actionRingVisible());
  actionRing.style('display', visible ? null : 'none');
  if (!visible) {
    if (!state.selected && actionRing.node().contains(document.activeElement)) focusGraph();
    return;
  }
  const k = state.viewport.k || 1;
  const offset = p.r + 24 / k;
  actionRing.attr('transform', `translate(${p.x},${p.y}) scale(${1 / k})`);
  runAction.attr('transform', `translate(${-offset},0)`);
  deleteAction.attr('transform', `translate(${offset},0)`);

  const running = state.run.id === t.id && state.run.pending;
  const rr = runReason(t);
  runAction
    .attr('aria-disabled', String(!!rr))
    .attr('aria-label', rr ? `Run unavailable: ${rr}` : `Run ${t.title}`)
    .classed('pending', running);
  runAction.select('title').text(rr || (running ? `Starting ${t.title}` : `Run ${t.title}`));

  const dr = deleteReason(t) || (state.deleteConfirm.open ? 'Delete confirmation is open' : '');
  deleteAction
    .attr('aria-disabled', String(!!dr))
    .attr('aria-label', dr ? `Delete unavailable: ${dr}` : `Delete ${t.title}`);
  deleteAction.select('title').text(dr || `Delete ${t.title}`);
  if (state.runPrompt.open) positionRunPrompt();
}
let actionTrackToken = 0;
function trackNodeActions(ms = 360) {
  const token = ++actionTrackToken;
  const until = performance.now() + ms;
  const tick = now => {
    if (token !== actionTrackToken) return;
    paintNodeActions();
    if (now < until) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
function activateNodeAction(which) {
  const t = state.selected ? state.tasks.get(state.selected) : null;
  if (!t) return;
  if (which === 'run') {
    if (runAction.attr('aria-disabled') === 'true') return;
    openRunPrompt(t.id);
  } else {
    if (deleteAction.attr('aria-disabled') === 'true') return;
    openDeleteConfirm(t.id);
  }
}
function focusAction(which) {
  if (which === 'node') { focusNode(state.selected); return; }
  const el = which === 'run' ? runAction.node() : deleteAction.node();
  if (el) el.focus({ preventScroll: true });
}
function focusNextNodeAfter(id) {
  const groups = [...nodeL.selectAll('g.node').nodes()];
  const i = groups.findIndex(n => n.getAttribute('data-id') === id);
  const next = i >= 0 ? groups[i + 1] : null;
  if (next) next.focus({ preventScroll: true });
  else focusGraph();
}
function actionKeydown(e, which) {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    e.stopPropagation();
    activateNodeAction(which);
    return;
  }
  if (e.key === 'Tab') {
    e.preventDefault();
    e.stopPropagation();
    if (e.shiftKey) focusAction(which === 'delete' ? 'run' : 'node');
    else if (which === 'run') focusAction('delete');
    else focusNextNodeAfter(state.selected);
  }
}
function positionDeletePopover() {
  const pop = document.getElementById('delete-popover');
  if (!pop || !state.deleteConfirm.open) return;
  const p = selectedNodePos();
  if (!p) return;
  const rect = svg.node().getBoundingClientRect();
  const k = state.viewport.k || 1;
  const sx = rect.left + state.viewport.applyX(p.x) + p.r * k + 26;
  const sy = rect.top + state.viewport.applyY(p.y);
  positionOverlay(pop, sx, sy);
}
function renderDeletePopover() {
  const pop = document.getElementById('delete-popover');
  pop.classList.toggle('hidden', !state.deleteConfirm.open);
  if (!state.deleteConfirm.open) return;
  const t = state.selected ? state.tasks.get(state.selected) : null;
  document.getElementById('delete-popover-title').textContent = t ? `Delete "${t.title}"?` : 'Delete this task?';
  const err = document.getElementById('delete-popover-error');
  err.textContent = state.deleteConfirm.error || '';
  err.classList.toggle('hidden', !state.deleteConfirm.error);
  const yes = document.getElementById('delete-popover-yes');
  yes.textContent = state.deleteConfirm.pending ? 'Deleting…' : 'Delete';
  yes.disabled = state.deleteConfirm.pending || topologyLocked() || !!state.error || (mutationInFlight() && !state.deleteConfirm.pending);
  positionDeletePopover();
}
function clientPoint(clientX, clientY) {
  const node = svg.node();
  const rect = node.getBoundingClientRect();
  return state.viewport.invert([clientX - rect.left, clientY - rect.top]);
}
function nodeAtPoint(gx, gy) {
  if (!state.layout) return null;
  let best = null, bestD = Infinity;
  for (const [id, p] of state.layout.nodes) {
    if (!p || !p.label) continue;
    const dx = gx - p.x, dy = gy - p.y;
    const inCore = Math.hypot(dx, dy) <= p.r + 12;
    const lm = p.label;
    const labelTop = p.y + p.r + LABEL_GAP - LABEL_LINE_H;
    const labelBottom = p.y + p.r + LABEL_GAP + lm.height;
    const inLabel = Math.abs(gx - p.x) <= Math.max(lm.width / 2, p.r) && gy >= labelTop && gy <= labelBottom;
    if (!inCore && !inLabel) continue;
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = { id, p }; }
  }
  return best;
}
function positionOverlay(node, clientX, clientY) {
  const pane = document.getElementById('map-pane').getBoundingClientRect();
  const w = node.offsetWidth || 300, h = node.offsetHeight || 180;
  let left = clientX - pane.left + 12;
  let top = clientY - pane.top + 12;
  left = Math.max(8, Math.min(left, pane.width - w - 8));
  top = Math.max(8, Math.min(top, pane.height - h - 8));
  node.style.left = `${left}px`;
  node.style.top = `${top}px`;
}
function renderFormDeps() {
  const box = document.getElementById('task-form-deps');
  box.replaceChildren();
  for (const dep of state.form.deps) {
    box.append(el('span', { class: 'dep-chip' }, [
      el('span', { class: 'dep-arrow', text: '→ depends on' }),
      el('span', { text: dep }),
    ]));
  }
}
function syncFormState() {
  if (!state.form.open) return;
  const submit = document.getElementById('task-form-submit');
  const err = document.getElementById('task-form-error');
  const locked = topologyLocked();
  const dup = state.form.title.trim() && state.tasks.has(state.form.title.trim());
  submit.disabled = locked || state.form.pending || mutationInFlight() || !!state.error || !state.form.title.trim() || !!dup;
  submit.textContent = state.form.pending ? 'Creating…' : 'Create';
  document.getElementById('task-form-cancel').disabled = state.form.pending;
  let text = state.form.error || '';
  if (!text && dup) text = 'A task with this title already exists.';
  if (!text && locked) text = `${lockReason()} — topology locked`;
  if (!text && state.error) text = 'Backend unreachable';
  err.textContent = text;
  err.classList.toggle('hidden', !text);
}
function openCreateForm(clientX, clientY, deps = [], prefill = '') {
  if (topologyLocked()) { notifyLocked(); return; }
  if (state.runPrompt.open) closeRunPrompt(false);
  state.form = {
    open: true,
    x: clientX,
    y: clientY,
    title: prefill,
    deps: deps.filter(Boolean).slice(0, 1),
    error: null,
    pending: false,
    sourceId: deps[0] || null,
  };
  const box = document.getElementById('task-form');
  const input = document.getElementById('task-form-input');
  input.value = prefill;
  renderFormDeps();
  box.classList.remove('hidden');
  positionOverlay(box, clientX, clientY);
  syncFormState();
  render();
  requestAnimationFrame(() => {
    input.focus({ preventScroll: true });
    input.setSelectionRange(input.value.length, input.value.length);
  });
}
function closeCreateForm(restoreFocus = true) {
  if (!state.form.open || state.form.pending) return;
  const sourceId = state.form.sourceId;
  state.form = { open: false, x: 0, y: 0, title: '', deps: [], error: null, pending: false, sourceId: null };
  document.getElementById('task-form').classList.add('hidden');
  render();
  if (restoreFocus) {
    if (sourceId && state.layout && state.layout.nodes.has(sourceId)) focusNode(sourceId);
    else focusGraph();
  }
}
async function submitCreateForm() {
  if (!state.form.open || state.form.pending) return;
  if (!canMutate()) { render(); return; }
  const title = state.form.title.trim();
  if (!title) { syncFormState(); return; }
  if (state.tasks.has(title)) {
    state.form.error = 'A task with this title already exists.';
    syncFormState();
    return;
  }
  state.form.title = title;
  state.form.error = null;
  state.form.pending = true;
  beginMutation('create');
  syncFormState();
  try {
    await createTask(title, state.form.deps.slice());
    state.form.pending = false;
    endMutation();
    closeCreateForm(false);
    showToast(`Created ${title}`);
    await refresh();
  } catch (e) {
    state.form.pending = false;
    endMutation();
    state.form.error = humanError(e);
    syncFormState();
  }
}

function currentTaskId() {
  if (state.find.open && state.find.matches.length) return currentMatchId();
  return state.selected;
}
function linkCandidates() {
  const target = state.link.taskId;
  const t = target ? state.tasks.get(target) : null;
  if (!t) return [];
  const q = state.link.query.trim().toLowerCase();
  return [...state.tasks.values()]
    .filter(x => x.id !== target)
    .filter(x => !q || x.title.toLowerCase().includes(q))
    .sort((a, b) => {
      const al = t.deps.includes(a.id) ? 1 : 0;
      const bl = t.deps.includes(b.id) ? 1 : 0;
      if (al !== bl) return al - bl;
      return a.title.localeCompare(b.title);
    })
    .slice(0, 8);
}
function renderLinkBox() {
  if (!state.link.open) return;
  const t = state.tasks.get(state.link.taskId);
  document.getElementById('link-box-title').textContent = t ? `Link dependency for ${t.title}` : 'Link dependency';
  const list = document.getElementById('link-suggest');
  const items = state.link.items = linkCandidates();
  if (state.link.index >= items.length) state.link.index = items.length - 1;
  if (state.link.index < -1) state.link.index = -1;
  list.replaceChildren();
  if (!items.length) {
    list.append(el('div', { class: 'link-item', text: 'No matching tasks' }));
  } else {
    items.forEach((task, i) => {
      const linked = !!t && t.deps.includes(task.id);
      const row = el('button', {
        class: `link-item${i === state.link.index ? ' active' : ''}${linked ? ' linked' : ''}`,
        type: 'button',
        role: 'option',
        'aria-selected': String(i === state.link.index),
        'data-id': task.id,
      }, [
        el('span', { text: task.title }),
        el('span', { class: 'link-state', text: linked ? 'linked — choose to remove' : 'add' }),
      ]);
      row.addEventListener('mousedown', e => e.preventDefault());
      row.addEventListener('click', () => toggleDependency(task.id));
      list.append(row);
    });
  }
  const hint = document.getElementById('link-hint');
  hint.textContent = state.link.pending ? 'Saving…' : '↑↓ choose · Enter add/remove · Esc cancel';
  const err = document.getElementById('link-error');
  err.textContent = state.link.error || '';
  err.classList.toggle('hidden', !state.link.error);
}
function openLinkBox() {
  const id = currentTaskId();
  if (!id || !state.tasks.has(id)) return;
  if (topologyLocked()) { notifyLocked(); return; }
  if (state.runPrompt.open) closeRunPrompt(false);
  state.link = { open: true, taskId: id, query: '', items: [], index: -1, error: null, pending: false };
  const box = document.getElementById('link-box');
  const input = document.getElementById('link-input');
  input.value = '';
  box.classList.remove('hidden');
  const pane = document.getElementById('map-pane').getBoundingClientRect();
  positionOverlay(box, pane.left + pane.width / 2, pane.top + pane.height / 2);
  renderLinkBox();
  paintNodeActions();
  requestAnimationFrame(() => input.focus({ preventScroll: true }));
}
function closeLinkBox(restoreFocus = true) {
  if (!state.link.open) return;
  state.link = { open: false, taskId: null, query: '', items: [], index: -1, error: null, pending: false };
  document.getElementById('link-box').classList.add('hidden');
  paintNodeActions();
  if (restoreFocus) focusGraph();
}
async function toggleDependency(depId) {
  if (!state.link.open || state.link.pending) return;
  const target = state.link.taskId;
  const t = state.tasks.get(target);
  if (!t || depId === target) return;
  if (!canMutate()) { renderLinkBox(); return; }
  const has = t.deps.includes(depId);
  const next = has ? t.deps.filter(d => d !== depId) : [...t.deps, depId];
  state.link.pending = true;
  state.link.error = null;
  beginMutation(has ? 'unlink' : 'link');
  renderLinkBox();
  try {
    await putDependencies(target, next);
    endMutation();
    state.link.pending = false;
    closeLinkBox(false);
    showToast(has ? `Removed ${depId} → ${target}` : `Linked ${depId} → ${target}`);
    await refresh();
  } catch (e) {
    endMutation();
    state.link.pending = false;
    state.link.error = humanError(e);
    renderLinkBox();
    if (e.status === 404) refresh();
  }
}

function deleteReason(t) {
  if (!t) return '';
  if (topologyLocked()) return state.situation ? 'Resolve the current situation first' : 'Engine is running';
  if (state.error) return 'Backend unreachable';
  if (state.deleteConfirm.pending) return 'Deleting…';
  if (mutationInFlight()) return 'Another change is in progress';
  return '';
}
function openDeleteConfirm(id) {
  const t = state.tasks.get(id);
  if (!t) return;
  if (topologyLocked()) { notifyLocked(); return; }
  if (state.runPrompt.open) closeRunPrompt(false);
  if (state.selected !== id) selectNode(id);
  state.deleteConfirm = { open: true, pending: false, error: null };
  render();
  requestAnimationFrame(() => {
    const btn = document.getElementById('delete-popover-cancel');
    if (btn) btn.focus({ preventScroll: true });
  });
}
function closeDeleteConfirm(restoreFocus = true) {
  if (!state.deleteConfirm.open) return;
  state.deleteConfirm = { open: false, pending: false, error: null };
  render();
  if (restoreFocus) requestAnimationFrame(() => focusAction('delete'));
}
async function confirmDeleteTask() {
  const t = state.selected ? state.tasks.get(state.selected) : null;
  if (!t || state.deleteConfirm.pending) return;
  if (!canMutate()) { render(); return; }
  state.deleteConfirm.pending = true;
  state.deleteConfirm.error = null;
  beginMutation('delete');
  render();
  try {
    await deleteTaskRequest(t.title);
    endMutation();
    state.deleteConfirm = { open: false, pending: false, error: null };
    closeInspector();
    focusGraph();
    showToast(`Deleted ${t.title}`);
    await refresh();
  } catch (e) {
    endMutation();
    state.deleteConfirm.pending = false;
    state.deleteConfirm.error = humanError(e);
    render();
    if (e.status === 404) refresh();
  }
}

function runReason(t) {
  if (!t) return '';
  if (t.state === 'awake') return 'This task is already running';
  if (state.situation) return 'Resolve the current situation first';
  if (awakeCount() > 0) return 'Engine is running';
  if (state.run.pending && state.run.id === t.id) return 'Starting…';
  if (mutationInFlight()) return 'Another change is in progress';
  if (state.error) return 'Backend unreachable';
  if (state.actionLock) return 'Wait a moment';
  return '';
}
function positionRunPrompt() {
  const pop = document.getElementById('run-popover');
  if (!pop || !state.runPrompt.open) return;
  const p = selectedNodePos();
  if (!p) return;
  const rect = svg.node().getBoundingClientRect();
  const k = state.viewport.k || 1;
  const sx = rect.left + state.viewport.applyX(p.x) - p.r * k - 26;
  const sy = rect.top + state.viewport.applyY(p.y);
  positionOverlay(pop, sx, sy);
}
function paintRunPrompt() {
  const pop = document.getElementById('run-popover');
  const rp = state.runPrompt;
  const t = rp.open && rp.taskId ? state.tasks.get(rp.taskId) : null;
  if (rp.open && !t) {
    state.runPrompt = { open: false, taskId: null, tip: '', tipSaved: false, pending: false, error: null };
    pop.classList.add('hidden');
    return;
  }
  pop.classList.toggle('hidden', !rp.open);
  if (!rp.open) return;

  document.getElementById('run-popover-title').textContent = `Run "${t.title}"`;
  const input = document.getElementById('run-popover-tip');
  if (input.value !== rp.tip) input.value = rp.tip;
  input.readOnly = rp.pending || rp.tipSaved;

  const reason = topologyLocked()
    ? lockReason()
    : state.error ? 'Backend unreachable'
      : (mutationInFlight() && !rp.pending) ? 'Another change is in progress' : '';

  const submit = document.getElementById('run-popover-submit');
  submit.setAttribute('aria-disabled', String(rp.pending || !!reason));
  submit.textContent = rp.pending ? 'Running…' : rp.tipSaved ? 'Retry run' : 'Run';
  submit.title = reason || (rp.tipSaved ? 'Retry the run' : 'Run this task');
  submit.setAttribute('aria-label', reason
    ? `Run unavailable: ${reason}`
    : rp.tipSaved ? 'Retry run' : 'Run this task');
  const cancel = document.getElementById('run-popover-cancel');
  cancel.setAttribute('aria-disabled', String(rp.pending));
  cancel.title = rp.pending ? 'Wait for the run to finish' : 'Cancel';

  const err = document.getElementById('run-popover-error');
  err.textContent = rp.error || '';
  err.classList.toggle('hidden', !rp.error);
  const hint = document.getElementById('run-popover-hint');
  hint.textContent = reason
    ? `${reason} — run disabled`
    : rp.tipSaved
      ? 'Tip saved · Retry run'
      : 'Optional · Enter to run · Shift+Enter for a new line';
  positionRunPrompt();
}
function openRunPrompt(id) {
  const t = state.tasks.get(id);
  if (!t) return;
  const reason = runReason(t);
  if (reason) { showToast(reason); return; }
  if (state.runPrompt.open && state.runPrompt.taskId === id) {
    requestAnimationFrame(() => document.getElementById('run-popover-tip').focus({ preventScroll: true }));
    return;
  }
  if (state.deleteConfirm.open) closeDeleteConfirm(false);
  if (state.link.open) closeLinkBox(false);
  if (state.form.open) closeCreateForm(false);
  if (state.selected !== id) selectNode(id);
  state.runPrompt = { open: true, taskId: id, tip: '', tipSaved: false, pending: false, error: null };
  render();
  requestAnimationFrame(() => {
    positionRunPrompt();
    document.getElementById('run-popover-tip').focus({ preventScroll: true });
  });
}
function closeRunPrompt(restoreFocus = true) {
  if (!state.runPrompt.open || state.runPrompt.pending) return;
  state.runPrompt = { open: false, taskId: null, tip: '', tipSaved: false, pending: false, error: null };
  render();
  if (restoreFocus) requestAnimationFrame(() => focusAction('run'));
}
async function submitRunPrompt() {
  const rp = state.runPrompt;
  if (!rp.open || rp.pending) return;
  if (document.getElementById('run-popover-submit').getAttribute('aria-disabled') === 'true') return;
  const t = state.tasks.get(rp.taskId);
  if (!t) { closeRunPrompt(false); return; }
  const tip = rp.tip.trim();
  rp.pending = true;
  rp.error = null;
  state.run = { id: t.id, pending: true };
  beginMutation('run');
  paintRunPrompt();
  try {
    if (tip && !rp.tipSaved) {
      await putTaskTip(t.title, tip, t.deps);
      rp.tipSaved = true;
      paintRunPrompt();
    }
    await runTaskRequest(t.title);
    await refresh();
    endMutation();
    state.run = { id: t.id, pending: false };
    rp.pending = false;
    if (state.selected === t.id) {
      state.inspectorTab = 'run';
      state.inspectorResetRunTop = true;
      state.trajectoryFollow = true;
    }
    closeRunPrompt();
    render();
  } catch (e) {
    endMutation();
    state.run = { id: t.id, pending: false };
    rp.pending = false;
    rp.error = `Run failed: ${humanError(e)}`;
    paintRunPrompt();
  }
}

function setHoldRing(id, p) {
  nodeL.selectAll('g.node').filter(d => d.id === id).each(function (d) {
    const sel = d3.select(this);
    sel.classed('holding', true);
    const circ = 2 * Math.PI * (d.r + 11);
    sel.select('.hold-ring').attr('stroke-dashoffset', String(circ * (1 - p)));
  });
}
function clearGestureVisuals() {
  if (state.gesture.holdTimer) { clearTimeout(state.gesture.holdTimer); state.gesture.holdTimer = null; }
  if (state.gesture.holdRAF) { cancelAnimationFrame(state.gesture.holdRAF); state.gesture.holdRAF = 0; }
  nodeL.selectAll('g.node').classed('holding', false).classed('drag-source', false).classed('drop-valid', false).classed('drop-invalid', false);
  edgeL.selectAll('path.edge').classed('armed', false);
  gestureL.style('display', 'none');
  edgeProgress.style('display', 'none');
  gestureArrow.classed('invalid', false);
}
function teardownGesture(suppressClick = true) {
  clearGestureVisuals();
  const g = state.gesture;
  g.mode = 'idle';
  g.pointerId = null;
  g.sourceId = null;
  g.edge = null;
  g.targetId = null;
  g.targetState = 'empty';
  g.edgePath = null;
  if (suppressClick) g.suppressClickUntil = Date.now() + 350;
}
function cancelGesture(suppressClick = true) {
  if (state.gesture.mode === 'idle') return;
  teardownGesture(suppressClick);
  render();
}
function startNodeGesture(e, d) {
  const g = state.gesture;
  svg.interrupt();
  clearGestureVisuals();
  g.mode = 'pendingNode';
  g.pointerId = e.pointerId;
  g.sourceId = d.id;
  g.edge = null;
  g.startX = e.clientX;
  g.startY = e.clientY;
  g.x = e.clientX;
  g.y = e.clientY;
  g.targetId = null;
  g.targetState = 'empty';
  paintNodeActions();
  try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  if (topologyLocked()) {
    g.holdTimer = setTimeout(() => {
      if (state.gesture.mode === 'pendingNode') { notifyLocked(); teardownGesture(false); }
    }, HOLD_MS);
    return;
  }
  const started = performance.now();
  const frame = now => {
    if (state.gesture.mode !== 'pendingNode') return;
    const p = Math.min(1, (now - started) / HOLD_MS);
    setHoldRing(d.id, p);
    if (p >= 1) {
      nodeL.selectAll('g.node').classed('holding', false);
      g.mode = 'dragging';
      gestureL.style('display', null);
      paintGesture();
      return;
    }
    g.holdRAF = requestAnimationFrame(frame);
  };
  g.holdRAF = requestAnimationFrame(frame);
}
function startEdgeGesture(e, d, pathEl) {
  if (topologyLocked()) { notifyLocked(); return; }
  const g = state.gesture;
  svg.interrupt();
  clearGestureVisuals();
  g.mode = 'pendingEdge';
  g.pointerId = e.pointerId;
  g.sourceId = d.source;
  g.edge = { source: d.source, target: d.target };
  g.edgePath = pathEl;
  g.startX = e.clientX;
  g.startY = e.clientY;
  paintNodeActions();
  try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  const len = pathEl.getTotalLength ? pathEl.getTotalLength() : 0;
  gestureL.style('display', null);
  edgeProgress.style('display', null)
    .attr('d', pathEl.getAttribute('d'))
    .attr('stroke-dasharray', String(len))
    .attr('stroke-dashoffset', String(len));
  const started = performance.now();
  const frame = now => {
    if (state.gesture.mode !== 'pendingEdge') return;
    const p = Math.min(1, (now - started) / HOLD_MS);
    edgeProgress.attr('stroke-dashoffset', String(len * (1 - p)));
    if (p >= 1) {
      g.mode = 'edgeArmed';
      edgeProgress.style('display', 'none');
      edgeL.selectAll('path.edge')
        .filter(x => x.source === g.edge.source && x.target === g.edge.target)
        .classed('armed', true);
      showToast('Release to remove this dependency');
      return;
    }
    g.holdRAF = requestAnimationFrame(frame);
  };
  g.holdRAF = requestAnimationFrame(frame);
}
function pointInEdge(pathEl, clientX, clientY) {
  try {
    const node = svg.node();
    const rect = node.getBoundingClientRect();
    const pt = node.createSVGPoint();
    pt.x = clientX - rect.left;
    pt.y = clientY - rect.top;
    const inv = node.getScreenCTM().inverse();
    return pathEl.isPointInStroke(pt.matrixTransform(inv));
  } catch (e) {
    return true;
  }
}
function paintGesture() {
  const g = state.gesture;
  if (g.mode !== 'dragging') return;
  const p = clientPoint(g.x, g.y);
  const src = state.layout && state.layout.nodes.get(g.sourceId);
  if (!src) { cancelGesture(); return; }
  const hit = nodeAtPoint(p[0], p[1]);
  let stateName = 'empty', targetId = null, reason = '';
  if (hit) {
    targetId = hit.id;
    const task = state.tasks.get(targetId);
    if (targetId === g.sourceId) { stateName = 'invalid'; reason = 'A task cannot depend on itself'; }
    else if (hit.p.ghost || hit.p.temp) { stateName = 'invalid'; reason = 'Proposal tasks cannot be linked'; }
    else if (task && task.deps.includes(g.sourceId)) { stateName = 'invalid'; reason = 'Already linked'; }
    else if (!task) { stateName = 'invalid'; reason = 'Task is not in the network'; }
    else stateName = 'valid';
  }
  g.targetId = targetId;
  g.targetState = stateName;
  g.targetReason = reason;
  nodeL.selectAll('g.node')
    .classed('drag-source', d => d.id === g.sourceId)
    .classed('drop-valid', d => d.id === targetId && stateName === 'valid')
    .classed('drop-invalid', d => d.id === targetId && stateName === 'invalid');
  const arrowD = `M${src.x},${src.y + src.r} C${src.x},${(src.y + src.r + p[1]) / 2} ${p[0]},${(src.y + src.r + p[1]) / 2} ${p[0]},${p[1]}`;
  gestureArrow.attr('d', arrowD).classed('invalid', stateName === 'invalid');
  if (!hit) {
    gestureEmpty.style('display', null).attr('transform', `translate(${p[0]},${p[1]})`);
    gestureEmpty.select('.gesture-empty').classed('invalid', false);
  } else {
    gestureEmpty.style('display', 'none');
  }
}
function onGestureMove(e) {
  const g = state.gesture;
  if (g.mode === 'idle' || e.pointerId !== g.pointerId) return;
  if (g.mode === 'pendingNode' || g.mode === 'pendingEdge') {
    if (Math.hypot(e.clientX - g.startX, e.clientY - g.startY) > MOVE_SLOP) cancelGesture(true);
    return;
  }
  if (g.mode === 'edgeArmed') {
    if (g.edgePath && !pointInEdge(g.edgePath, e.clientX, e.clientY)) cancelGesture(true);
    return;
  }
  if (g.mode === 'dragging') {
    e.preventDefault();
    g.x = e.clientX;
    g.y = e.clientY;
    paintGesture();
  }
}
function finishNodeDrag() {
  const g = state.gesture;
  const source = g.sourceId, target = g.targetId, targetState = g.targetState, reason = g.targetReason;
  const x = g.x, y = g.y;
  teardownGesture(true);
  if (targetState === 'valid' && target) addDependency(target, source);
  else if (targetState === 'invalid') { showToast(reason || 'Cannot link here'); render(); }
  else openCreateForm(x, y, [source]);
}
function onGestureUp(e) {
  const g = state.gesture;
  if (g.mode === 'idle' || e.pointerId !== g.pointerId) return;
  if (g.mode === 'pendingNode' || g.mode === 'pendingEdge') {
    teardownGesture(false);
    paintNodeActions();
    return;
  }
  if (g.mode === 'dragging') { finishNodeDrag(); return; }
  if (g.mode === 'edgeArmed') {
    const edge = g.edge;
    teardownGesture(true);
    if (edge) removeDependency(edge.target, edge.source);
  }
}
function onGestureCancel(e) {
  const g = state.gesture;
  if (g.mode === 'idle' || e.pointerId !== g.pointerId) return;
  cancelGesture(true);
}
async function addDependency(target, source) {
  const t = state.tasks.get(target);
  if (!t) return;
  if (t.deps.includes(source)) { showToast('Already linked'); render(); return; }
  if (!canMutate()) { render(); return; }
  beginMutation('link');
  try {
    await putDependencies(target, [...t.deps, source]);
    endMutation();
    showToast(`Linked ${source} → ${target}`);
    await refresh();
  } catch (e) {
    endMutation();
    showToast(humanError(e), true);
    if (e.status === 404) refresh();
  }
}
async function removeDependency(target, source) {
  const t = state.tasks.get(target);
  if (!t) return;
  if (!canMutate()) { render(); return; }
  beginMutation('unlink');
  try {
    await putDependencies(target, t.deps.filter(d => d !== source));
    endMutation();
    showToast(`Removed ${source} → ${target}`);
    await refresh();
  } catch (e) {
    endMutation();
    showToast(humanError(e), true);
    if (e.status === 404) refresh();
  }
}
window.addEventListener('pointermove', onGestureMove, { passive: false });
window.addEventListener('pointerup', onGestureUp);
window.addEventListener('pointercancel', onGestureCancel);
document.getElementById('graph').addEventListener('contextmenu', e => {
  if (state.gesture.mode !== 'idle') e.preventDefault();
});

function unreadUnder(id, visible) {
  if (state.unread.has(id)) return true;
  const cl = getClosure(id);
  for (const uid of state.unread.keys()) {
    if (cl.has(uid) && !visible.has(uid)) return true;
  }
  return false;
}

function paint() {
  const L = state.layout;
  if (!L) return;
  const items = [...L.nodes].map(([id, p]) => {
    const t = state.tasks.get(id);
    return {
      id, x: p.x, y: p.y, r: p.r,
      ghost: !!p.ghost, temp: !!p.temp,
      title: p.ghost ? p.title : (t ? t.title : id),
      kind: p.ghost ? p.kind : (t ? t.kind : 'simple'),
      state: p.ghost ? '' : (t ? t.state : 'sleeping'),
      lines: (p.label && p.label.lines) || [p.title || ''],
      hasReport: !p.ghost && !p.temp && !!t?.report,
      unread: !p.ghost && !p.temp && unreadUnder(id, L.visible),
      context: !!p.context,
    };
  });

  const findActive = state.find.open && state.find.query.trim() !== '';
  const currentMatch = findActive ? state.find.matches[state.find.index] : null;
  const reviewAnchor = state.situation && state.situation.type === 'review' ? state.anchorId : null;
  const awakeIds = new Set([...state.tasks.values()].filter(t => t.state === 'awake').map(t => t.id));
  const showRootActivity = !state.activeRoot || state.find.open;
  const animate = state.layoutChanged && !reduceMotion();

  const nJoin = nodeL.selectAll('g.node').data(items, d => d.id);
  nJoin.exit().remove();
  const ne = nJoin.enter().append('g').attr('class', 'node');
  ne.append('circle').attr('class', 'halo');
  ne.append('circle').attr('class', 'ring2');
  ne.append('circle').attr('class', 'anchor-ring');
  ne.append('circle').attr('class', 'core');
  ne.append('circle').attr('class', 'report-dot');
  ne.append('circle').attr('class', 'unread-dot');
  ne.append('circle').attr('class', 'selection-ring');
  ne.append('circle').attr('class', 'cursor-ring');
  ne.append('circle').attr('class', 'hold-ring');
  ne.append('circle').attr('class', 'drop-ring');
  ne.append('text').attr('class', 'label').attr('text-anchor', 'middle');
  const merged = nJoin.merge(ne);
  merged.each(function (d) {
    const sel = d3.select(this);
    const r = d.r;
    sel.select('.core').attr('r', r);
    sel.select('.halo').attr('r', r + 9);
    sel.select('.ring2').attr('r', r + 5);
    sel.select('.anchor-ring').attr('r', r + 13);
    sel.select('.selection-ring').attr('r', r + 8);
    sel.select('.cursor-ring').attr('r', r + 2.5);
    const holdR = r + 11;
    const circ = 2 * Math.PI * holdR;
    sel.select('.hold-ring').attr('r', holdR).attr('stroke-dasharray', `${circ}`).attr('stroke-dashoffset', `${circ}`);
    sel.select('.drop-ring').attr('r', r + 17);
    sel.select('.report-dot')
      .attr('cx', r * 0.68)
      .attr('cy', -r * 0.68)
      .attr('r', 2.6)
      .style('display', d.hasReport && !d.ghost && !d.temp ? null : 'none');
    sel.select('.unread-dot')
      .attr('cx', -r * 0.68)
      .attr('cy', -r * 0.68)
      .attr('r', 3.4);
    const lab = sel.select('.label').attr('y', r + LABEL_GAP);
    lab.text(null);
    const lines = d.lines && d.lines.length ? d.lines : [d.title || ''];
    lines.forEach((line, i) => lab.append('tspan').attr('x', 0).attr('dy', i === 0 ? 0 : LABEL_LINE_H).text(i < lines.length - 1 ? `${line} ` : line));
    const cls = ['node'];
    if (d.temp) cls.push('temp');
    else if (d.ghost) cls.push('proposal', `kind-${d.kind}`);
    else cls.push(`kind-${d.kind}`, `state-${d.state}`);
    if (d.id === state.anchorId) cls.push('anchor');
    if (reviewAnchor && d.id === reviewAnchor) cls.push('review');
    if (findActive && state.find.matches.includes(d.id)) cls.push('match');
    if (currentMatch === d.id) cls.push('match-current');
    if (state.stream.status === 'open' && state.stream.task === d.id) cls.push('live');
    if (d.id === state.selected) cls.push('selected');
    if (d.unread) cls.push('unread');
    if (d.context) cls.push('context-root');
    if (showRootActivity && d.kind === 'root' && [...awakeIds].some(id => getClosure(d.id).has(id))) cls.push('active-closure');
    sel.attr('class', cls.join(' ')).attr('data-id', d.id).attr('aria-description', null);
    if (!animate) sel.attr('transform', `translate(${d.x},${d.y})`);
  });
  merged
    .attr('tabindex', d => (d.ghost || d.temp ? null : '0'))
    .attr('role', d => (d.ghost || d.temp ? null : 'button'))
    .attr('aria-label', d => {
      if (d.ghost || d.temp) return null;
      const bits = [d.title, d.state];
      if (d.context) bits.push('related root context');
      if (d.hasReport) bits.push('reported');
      if (d.unread) bits.push('unread report');
      return bits.join(', ');
    });
  merged.on('click', (e, d) => {
    e.stopPropagation();
    if (Date.now() < state.gesture.suppressClickUntil) return;
    mapActivate(d);
  });
  merged.on('pointerdown', (e, d) => {
    if (d.ghost || d.temp || e.button !== 0 || state.form.open || state.link.open) return;
    e.stopPropagation();
    startNodeGesture(e, d);
  });
  merged.on('keydown', (e, d) => {
    if (e.key === 'Tab' && !e.shiftKey && state.selected === d.id && actionRingVisible()) {
      e.preventDefault();
      e.stopPropagation();
      focusAction('run');
      return;
    }
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    e.stopPropagation();
    mapActivate(d);
  });

  if (animate) {
    ne.style('opacity', 0);
    merged.transition().duration(280).attr('transform', d => `translate(${d.x},${d.y})`).style('opacity', 1);
  } else {
    merged.attr('transform', d => `translate(${d.x},${d.y})`);
  }

  const edgeKey = d => d.source + '>' + d.target;
  const edgePath = d => {
    const s = L.nodes.get(d.source), t = L.nodes.get(d.target);
    if (!s || !t) return '';
    if (d.root) {
      const dx = t.x - s.x, dy = t.y - s.y;
      const dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist, uy = dy / dist;
      const sx = s.x + ux * (s.r + 2), sy = s.y + uy * (s.r + 2);
      const ex = t.x - ux * (t.r + 8), ey = t.y - uy * (t.r + 8);
      const mx = (sx + ex) / 2, my = (sy + ey) / 2;
      const nx = -uy, ny = ux;
      const bend = Math.min(96, Math.max(26, dist * 0.16));
      return `M${sx},${sy} Q${mx + nx * bend},${my + ny * bend} ${ex},${ey}`;
    }
    return `M${s.x},${s.y + s.r} C${s.x},${(s.y + s.r + t.y - t.r) / 2} ${t.x},${(s.y + s.r + t.y - t.r) / 2} ${t.x},${t.y - t.r - 6}`;
  };
  const eJoin = edgeL.selectAll('path.edge').data(L.edges, edgeKey);
  eJoin.exit().remove();
  const ee = eJoin.enter().append('path').attr('class', 'edge').attr('marker-end', 'url(#arr)');
  const em = eJoin.merge(ee);
  em.attr('class', d => {
    const cls = ['edge'];
    if (d.proposal) cls.push('proposal');
    if (d.root) cls.push('root-link');
    if (d.context) cls.push('context');
    if (awakeIds.has(d.source) || awakeIds.has(d.target)) cls.push('awake');
    if (state.gesture.edge && state.gesture.edge.source === d.source && state.gesture.edge.target === d.target) cls.push('armed');
    return cls.join(' ');
  }).attr('marker-end', d => d.root ? 'url(#arr-root)' : 'url(#arr)')
    .attr('d', edgePath).attr('data-source', d => d.source).attr('data-target', d => d.target);
  if (animate) {
    ee.style('opacity', 0);
    em.transition().duration(280).style('opacity', 1);
  }

  const hitJoin = edgeHitL.selectAll('path.edge-hit').data(L.edges, edgeKey);
  hitJoin.exit().remove();
  const hh = hitJoin.enter().append('path').attr('class', 'edge-hit');
  hh.append('title');
  const hm = hitJoin.merge(hh);
  hm.attr('d', edgePath).attr('data-source', d => d.source).attr('data-target', d => d.target);
  hm.select('title').text(d => `Hold to remove dependency: ${d.source} → ${d.target}`);
  hm.on('pointerenter', (e, d) => {
    if (state.gesture.mode !== 'idle') return;
    edgeL.selectAll('path.edge').filter(x => edgeKey(x) === edgeKey(d)).classed('hover', true);
  });
  hm.on('pointerleave', (e, d) => {
    edgeL.selectAll('path.edge').filter(x => edgeKey(x) === edgeKey(d)).classed('hover', false);
  });
  hm.on('pointerdown', (e, d) => {
    if (e.button !== 0 || state.form.open || state.link.open) return;
    if (d.proposal) return;
    e.stopPropagation();
    startEdgeGesture(e, d, e.currentTarget);
  });
  svg.classed('find-active', findActive);
  state.layoutChanged = false;
  if (animate && state.selected) trackNodeActions();
}

function render() {
  paintMode();
  const needMap = state.mode !== 'sandbox' || state.find.open;
  const freeze = state.gesture.mode !== 'idle' || state.form.open;
  if (needMap) {
    if (!freeze) {
      const ghostShown = activeGhosts().length ? state.situationSig : '';
      const view = state.find.open ? 'find' : (state.activeRoot ? `root:${state.activeRoot}` : 'overview');
      const expandedSig = [...state.expanded].sort().join(',');
      const taskSig = [...state.tasks.values()]
        .sort((a, b) => a.id.localeCompare(b.id))
        .map(t => `${t.id}:${t.kind}:${t.deps.join(',')}`)
        .join('|');
      const sig = taskSig + ':' + view + ':' + expandedSig + ':' + ghostShown;
      if (sig !== state.signature) {
        state.signature = sig;
        state.layout = build();
        state.fitPending = true;
      }
      paint();
      if (state.fitPending && state.layout && state.layout.bounds) {
        state.fitPending = false;
        fit();
      }
    } else if (state.gesture.mode !== 'idle') {
      paintGesture();
    }
  }
  renderInspector();
  paintNodeActions();
  renderDeletePopover();
  paintRunPrompt();
  paintTop();
  paintStop();
  renderComposer();
  paintFind();
  paintBreadcrumb();
  paintMapEmpty();
  paintActivity();
  paintMode();
  if (state.form.open) syncFormState();
}

function paintMode() {
  const app = document.getElementById('app');
  app.className = `mode-${state.mode}`;
  const sandbox = state.mode === 'sandbox';
  document.getElementById('sandbox').classList.toggle('hidden', !sandbox);
  document.getElementById('map-pane').toggleAttribute('inert', sandbox);
  const toggle = document.getElementById('sandbox-toggle');
  toggle.textContent = sandbox ? 'Map' : 'Sandbox';
  toggle.setAttribute('aria-pressed', String(sandbox));
  const link = document.getElementById('sandbox-open');
  link.href = SANDBOX_URL;
  link.classList.toggle('hidden', !sandbox);
}

function paintTop() {
  const st = document.getElementById('status'), s = document.getElementById('status-text');
  const awake = awakeCount() > 0;
  st.classList.toggle('error', !!state.error);
  st.classList.toggle('busy', !state.error && awake);
  document.getElementById('brand').classList.toggle('busy', !state.error && awake);
  st.title = state.error ? 'backend unreachable' : (awake ? 'engine working' : 'connected');
  s.textContent = state.error ? 'backend unreachable' : '';
}

function paintBreadcrumb() {
  const b = document.getElementById('breadcrumb');
  const root = state.mode === 'map' && !state.find.open && state.activeRoot ? state.tasks.get(state.activeRoot) : null;
  if (!root) {
    b.classList.add('hidden');
    b.textContent = '';
    return;
  }
  b.classList.remove('hidden');
  b.innerHTML = '';
  b.append(
    el('span', { class: 'bc-back', text: '◀ overview' }),
    el('span', { class: 'bc-sep', text: '·' }),
    el('span', { class: 'bc-root', text: root.title }),
  );
}

function paintMapEmpty() {
  const box = document.getElementById('map-empty');
  const show = state.mode === 'map' && !state.find.open && !state.activeRoot && getRoots().length === 0 && !state.form.open;
  box.classList.toggle('hidden', !show);
}

function paintFind() {
  const node = document.getElementById('find-count');
  const n = state.find.matches.length;
  node.textContent = n ? `${state.find.index + 1}/${n}` : '0/0';
  node.classList.toggle('zero', n === 0);
}

function normalizeAction(text) {
  return String(text == null ? '' : text)
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
function activityDotState(taskId) {
  const s = state.stream;
  if (!taskId) return 'idle';
  if (s.task !== taskId) return 'quiet';
  if (s.status === 'open' || s.status === 'connecting') {
    if (s.lastChunkAt) {
      const age = Date.now() - s.lastChunkAt;
      if (age >= STREAM_STALL_MS) return 'stalled';
      if (age >= STREAM_QUIET_MS) return 'quiet';
    }
    return 'live';
  }
  if (s.lastChunkAt) {
    const age = Date.now() - s.lastChunkAt;
    if (age >= STREAM_STALL_MS) return 'stalled';
  }
  return 'quiet';
}
function paintActivity() {
  const strip = document.getElementById('activity');
  if (!strip) return;
  const dot = document.getElementById('activity-dot');
  const taskBtn = document.getElementById('activity-task');
  const sep = document.getElementById('activity-sep');
  const actionEl = document.getElementById('activity-text');
  const s = state.stream;
  const awake = currentAwakeTask();
  const streamTask = s.task ? state.tasks.get(s.task) : null;
  let taskId = null;
  if (s.task && (!streamTask || streamTask.state === 'awake')) taskId = s.task;
  else if (awake) taskId = awake.id;
  const task = taskId ? state.tasks.get(taskId) : null;
  const title = taskId || '';
  const action = taskId && s.action && s.action.task === taskId ? s.action.text : '';
  const dotState = activityDotState(taskId);
  const clickable = !!task;
  const sig = `${title}|${action}|${dotState}|${clickable ? 1 : 0}`;
  if (sig === s.activitySig) return;
  s.activitySig = sig;
  strip.dataset.task = taskId || '';
  dot.className = `activity-dot ${dotState}`;
  if (!taskId) {
    taskBtn.classList.add('hidden');
    taskBtn.textContent = '';
    taskBtn.setAttribute('aria-disabled', 'true');
    taskBtn.removeAttribute('title');
    sep.classList.add('hidden');
    actionEl.textContent = 'No actions running';
    actionEl.classList.add('idle');
    actionEl.removeAttribute('title');
    actionEl.removeAttribute('aria-label');
    strip.removeAttribute('title');
    return;
  }
  taskBtn.classList.remove('hidden');
  if (taskBtn.textContent !== title) taskBtn.textContent = title;
  taskBtn.setAttribute('aria-disabled', String(!clickable));
  if (clickable) {
    taskBtn.title = `Open ${title}`;
    taskBtn.setAttribute('aria-label', `Open ${title}`);
  } else {
    taskBtn.removeAttribute('title');
    taskBtn.setAttribute('aria-label', title);
  }
  if (action) {
    sep.classList.remove('hidden');
    actionEl.classList.remove('idle');
    actionEl.textContent = action;
    actionEl.title = action;
    actionEl.setAttribute('aria-label', action);
    strip.title = action;
  } else {
    sep.classList.add('hidden');
    actionEl.classList.add('idle');
    actionEl.textContent = '';
    actionEl.removeAttribute('title');
    actionEl.removeAttribute('aria-label');
    strip.removeAttribute('title');
  }
}

function renderReport(text) {
  try {
    const clean = DOMPurify.sanitize(marked.parse(String(text)), {
      FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form', 'input', 'button'],
      FORBID_ATTR: ['style'],
    });
    const node = el('div', { class: 'report-md' });
    node.innerHTML = clean;
    for (const a of node.querySelectorAll('a')) {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    }
    return node;
  } catch (e) {
    return el('pre', { class: 'report' }, [String(text)]);
  }
}

function humanError(e) {
  const m = String((e && e.message) || e || '');
  if (/failed to fetch|networkerror|load failed/i.test(m)) return 'backend unreachable';
  return m;
}
function topologyLocked() { return awakeCount() > 0 || !!state.situation; }
function mutationInFlight() { return !!state.mutation.kind; }
function lockReason() {
  if (state.situation) return 'Resolve the current situation first';
  if (awakeCount() > 0) return 'Engine is running';
  return '';
}
function showToast(text, isError = false, ms = 0) {
  const box = document.getElementById('map-toast');
  if (!box || !text) return;
  box.textContent = text;
  box.classList.remove('hidden');
  box.classList.toggle('error', !!isError);
  clearTimeout(state.toast.timer);
  state.toast.timer = setTimeout(() => {
    box.classList.add('hidden');
    state.toast.timer = null;
  }, ms || (isError ? 6000 : 3000));
}
function notifyLocked() {
  showToast(lockReason() ? `${lockReason()} — topology locked` : 'Topology is locked', false);
}
async function apiError(res) {
  let detail = '';
  try {
    const body = await res.json();
    if (body && body.detail != null) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
  } catch (e) { /* ignore */ }
  const err = new Error(detail || `request failed (${res.status})`);
  err.status = res.status;
  return err;
}
async function api(url, method, body) {
  const opts = { method, headers: { accept: 'application/json' } };
  if (body !== undefined) {
    opts.headers['content-type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) throw await apiError(res);
  try { return await res.json(); } catch (e) { return {}; }
}
function createTask(title, deps) { return api(TASKS_URL, 'POST', { task: title, dependencies: deps }); }
function putDependencies(task, deps) { return api(TASKS_URL, 'PUT', { task, dependencies: deps }); }
function putTaskTip(task, tip, deps) { return api(TASKS_URL, 'PUT', { task, tip, dependencies: deps }); }
function deleteTaskRequest(name) { return api(`${TASKS_URL}/${encodeURIComponent(name)}`, 'DELETE'); }
function runTaskRequest(name) { return api(`${TASKS_URL}/${encodeURIComponent(name)}/run`, 'POST'); }
function beginMutation(kind) { state.mutation.kind = kind; render(); }
function endMutation() { state.mutation.kind = null; render(); }
function canMutate() {
  if (topologyLocked()) { notifyLocked(); return false; }
  if (mutationInFlight()) { showToast('Another change is in progress'); return false; }
  if (state.error) { showToast('Backend unreachable', true); return false; }
  return true;
}
function workInFlight() {
  return awakeCount() > 0 || !!state.situation;
}
function stopReason() {
  if (state.stop.pending) return 'Stopping…';
  if (state.error) return 'Backend unreachable';
  if (!workInFlight()) return 'Nothing is running';
  return '';
}
function canStop() { return !stopReason(); }
function paintStop() {
  const btn = document.getElementById('stop-all');
  if (!btn) return;
  const reason = stopReason();
  const pending = state.stop.pending;
  btn.setAttribute('aria-disabled', String(!!reason));
  btn.classList.toggle('pending', pending);
  btn.setAttribute('aria-expanded', String(state.stop.confirming));
  btn.title = reason || 'Stop all running work';
  btn.setAttribute('aria-label', reason ? `Stop all tasks: ${reason}` : 'Stop all tasks');
  const full = btn.querySelector('.stop-label-full');
  const short = btn.querySelector('.stop-label-short');
  if (full) full.textContent = pending ? 'Stopping…' : 'Stop all';
  if (short) short.textContent = pending ? 'Stopping…' : 'Stop';
  const dialog = document.getElementById('stop-confirm');
  dialog.classList.toggle('hidden', !state.stop.confirming);
  const err = document.getElementById('stop-confirm-error');
  err.classList.toggle('hidden', !state.stop.error);
  err.textContent = state.stop.error || '';
}
function focusStop() {
  const btn = document.getElementById('stop-all');
  if (btn) btn.focus({ preventScroll: true });
}
function openStopConfirm() {
  if (!canStop() || state.stop.confirming) return;
  state.stop.confirming = true;
  state.stop.error = null;
  render();
  requestAnimationFrame(() => {
    const safe = document.getElementById('stop-cancel');
    if (safe) safe.focus({ preventScroll: true });
  });
}
function closeStopConfirm(restoreFocus = true) {
  if (!state.stop.confirming) return;
  state.stop.confirming = false;
  state.stop.error = null;
  render();
  if (restoreFocus) requestAnimationFrame(focusStop);
}
async function cancelAll() {
  if (state.stop.pending) return;
  if (!workInFlight()) { closeStopConfirm(); return; }
  state.stop.confirming = false;
  state.stop.pending = true;
  state.stop.error = null;
  render();
  requestAnimationFrame(focusStop);
  try {
    const res = await fetch(CANCEL_URL, { method: 'POST' });
    if (res.status !== 200 && res.status !== 404) throw new Error(`request failed (${res.status})`);
    await refresh();
    state.stop.pending = false;
    render();
  } catch (e) {
    state.stop.pending = false;
    state.stop.error = humanError(e) === 'backend unreachable'
      ? 'Could not stop tasks — backend unreachable'
      : 'Could not stop tasks.';
    state.stop.confirming = true;
    render();
    requestAnimationFrame(() => {
      const safe = document.getElementById('stop-cancel');
      if (safe) safe.focus({ preventScroll: true });
    });
  }
}
function setInspectorScroll(box, value) {
  state.scrollGuardUntil = performance.now() + 120;
  box.scrollTop = value;
}
function scrollLatestTrajectory(box) {
  const latest = box.querySelector('.trajectory-step:last-child');
  if (!latest) return;
  const c = box.getBoundingClientRect(), r = latest.getBoundingClientRect();
  if (r.top < c.top) setInspectorScroll(box, box.scrollTop + (r.top - c.top - 6));
  else if (r.bottom > c.bottom) setInspectorScroll(box, box.scrollTop + (r.bottom - c.bottom + 6));
}
function renderInspector() {
  const panel = document.getElementById('inspector');
  const t = state.selected ? state.tasks.get(state.selected) : null;
  const open = !!t;
  panel.classList.toggle('open', open);
  panel.setAttribute('aria-hidden', String(!open || state.mode === 'sandbox'));
  panel.toggleAttribute('inert', !open || state.mode === 'sandbox');
  if (!t) {
    state.inspectorSig = null;
    state.inspectorTaskId = null;
    state.inspectorTrajectorySig = null;
    state.inspectorInstruction = null;
    state.inspectorLastPaintedTab = null;
    return;
  }
  const title = document.getElementById('inspector-title');
  const runPanel = document.getElementById('inspector-run-panel');
  const reportPanel = document.getElementById('inspector-report-panel');
  if (title.textContent !== t.title) title.textContent = t.title;

  const instruction = t.instruction || '';
  const traj = t.trajectory || [];
  const trajSig = traj.join('\n');
  const sig = `${t.id}::${instruction}::${t.report || ''}::${t.reportAt || ''}::${trajSig}`;
  if (sig !== state.inspectorSig) {
    const same = state.inspectorTaskId === t.id;
    const taskChanged = !same;
    const trajChanged = same && state.inspectorTrajectorySig !== trajSig;
    const instructionChanged = same && state.inspectorInstruction !== instruction;
    const resetTrajectory = trajChanged && traj.length === 0;
    const resetRun = taskChanged || state.inspectorResetRunTop;
    const resetReport = taskChanged || state.inspectorResetReportTop;
    const oldRunScroll = runPanel.scrollTop;
    const oldReportScroll = reportPanel.scrollTop;
    const oldRunH = runPanel.scrollHeight;
    if (taskChanged || resetTrajectory) state.trajectoryFollow = true;
    state.inspectorSig = sig;
    state.inspectorTaskId = t.id;
    state.inspectorTrajectorySig = trajSig;
    state.inspectorInstruction = instruction;
    state.inspectorResetRunTop = false;
    state.inspectorResetReportTop = false;

    const taskBlock = el('section', { class: 'task-block' }, [
      el('div', { class: 'inspector-label', text: 'Task' }),
      instruction.trim()
        ? el('div', { class: 'task-text', text: instruction })
        : el('div', { class: 'task-empty', text: 'No instructions' }),
    ]);
    const trajBlock = el('section', {}, [
      el('div', { class: 'inspector-label', text: 'Trajectory' }),
      traj.length
        ? el('ol', { class: 'trajectory-list' }, traj.map(step => el('li', { class: 'trajectory-step', text: step })))
        : el('div', { class: 'trajectory-empty', text: 'No trajectory yet' }),
    ]);
    runPanel.replaceChildren(taskBlock, trajBlock);
    reportPanel.replaceChildren();
    if (t.report) reportPanel.append(renderReport(t.report));
    else reportPanel.append(el('div', { class: 'inspector-empty', id: 'inspector-empty', text: 'No report yet.' }));

    const newRunH = runPanel.scrollHeight;
    const runActive = state.inspectorTab === 'run';
    if (resetRun) {
      setInspectorScroll(runPanel, 0);
    } else if (runActive) {
      if (!state.trajectoryFollow) setInspectorScroll(runPanel, oldRunScroll + (newRunH - oldRunH));
    } else {
      runPanel.scrollTop = oldRunScroll;
    }
    reportPanel.scrollTop = resetReport ? 0 : oldReportScroll;

    if (runActive && state.trajectoryFollow && (taskChanged || resetRun || trajChanged || instructionChanged || resetTrajectory)) {
      requestAnimationFrame(() => scrollLatestTrajectory(runPanel));
    }
  }

  if (state.inspectorResetRunTop) {
    state.inspectorResetRunTop = false;
    setInspectorScroll(runPanel, 0);
    if (state.inspectorTab === 'run' && state.trajectoryFollow) {
      requestAnimationFrame(() => scrollLatestTrajectory(runPanel));
    }
  }
  if (state.inspectorResetReportTop) {
    state.inspectorResetReportTop = false;
    reportPanel.scrollTop = 0;
  }
  paintInspectorView();
}

function paintInspectorView() {
  const t = state.selected ? state.tasks.get(state.selected) : null;
  if (!t) return;
  const runPanel = document.getElementById('inspector-run-panel');
  const reportPanel = document.getElementById('inspector-report-panel');
  const tabRun = document.getElementById('inspector-tab-run');
  const tabReport = document.getElementById('inspector-tab-report');
  const tabs = document.getElementById('inspector-tabs');
  const active = state.inspectorTab === 'report' ? 'report' : 'run';
  const prev = state.inspectorLastPaintedTab;
  const activeEl = document.activeElement;
  const inspector = document.getElementById('inspector');
  const inInspector = inspector.contains(activeEl);
  const inTablist = tabs.contains(activeEl);
  const inHiddenPanel = active === 'run' ? reportPanel.contains(activeEl) : runPanel.contains(activeEl);

  runPanel.classList.toggle('hidden', active !== 'run');
  reportPanel.classList.toggle('hidden', active !== 'report');
  tabRun.setAttribute('aria-selected', String(active === 'run'));
  tabReport.setAttribute('aria-selected', String(active === 'report'));
  tabRun.tabIndex = active === 'run' ? 0 : -1;
  tabReport.tabIndex = active === 'report' ? 0 : -1;

  if (prev && prev !== active && inInspector && (inTablist || inHiddenPanel)) {
    (active === 'run' ? tabRun : tabReport).focus({ preventScroll: true });
  }
  state.inspectorLastPaintedTab = active;
}

function setInspectorTab(tab) {
  if (tab !== 'run' && tab !== 'report') return;
  if (state.inspectorTab === tab) return;
  state.inspectorTab = tab;
  render();
  if (tab === 'run' && state.trajectoryFollow) {
    requestAnimationFrame(() => {
      if (state.inspectorTab === 'run') scrollLatestTrajectory(document.getElementById('inspector-run-panel'));
    });
  }
}

document.getElementById('inspector-run-panel').addEventListener('scroll', () => {
  if (!state.selected || state.inspectorTab !== 'run' || performance.now() < state.scrollGuardUntil) return;
  const box = document.getElementById('inspector-run-panel');
  const latest = box.querySelector('.trajectory-step:last-child');
  if (!latest) return;
  const c = box.getBoundingClientRect(), r = latest.getBoundingClientRect();
  state.trajectoryFollow = r.bottom > c.top && r.top < c.bottom;
});

function focusNode(id) {
  requestAnimationFrame(() => {
    const n = document.querySelector(`g.node[data-id="${CSS.escape(String(id))}"]`);
    if (n) n.focus({ preventScroll: true });
  });
}

function closeInspector() {
  const prev = state.selected;
  const hadFocus = document.getElementById('inspector').contains(document.activeElement);
  state.selected = null;
  state.inspectorSig = null;
  state.inspectorTaskId = null;
  state.inspectorTrajectorySig = null;
  state.trajectoryFollow = true;
  state.deleteConfirm = { open: false, pending: false, error: null };
  state.runPrompt = { open: false, taskId: null, tip: '', tipSaved: false, pending: false, error: null };
  render();
  if (hadFocus) focusNode(prev);
}
function overview() {
  state.mode = 'map';
  state.activeRoot = null;
  state.expanded = new Set();
  state.selected = null;
  state.anchorId = null;
  state.inspectorSig = null;
  state.inspectorTaskId = null;
  state.inspectorTrajectorySig = null;
  state.trajectoryFollow = true;
  state.deleteConfirm = { open: false, pending: false, error: null };
  state.runPrompt = { open: false, taskId: null, tip: '', tipSaved: false, pending: false, error: null };
  state.fitPending = true;
  if (state.find.open) closeFind(false);
  render();
}
function selectNode(id) {
  const t = state.tasks.get(id);
  if (!t) return;
  const root = rootOwning(id);
  if (!root) return;
  if (state.runPrompt.open && state.runPrompt.taskId !== id) closeRunPrompt(false);
  if (state.find.open) closeFind(false);
  state.mode = 'map';
  const alreadySelected = state.selected === id;
  const rootChanged = state.activeRoot !== root;
  if (rootChanged) {
    state.activeRoot = root;
    state.expanded = new Set([root]);
  } else {
    state.expanded.add(root);
  }
  if (t.kind !== 'simple') state.expanded.add(id);
  const before = [...state.expanded].sort().join('|');
  ensurePath(root, id);
  const after = [...state.expanded].sort().join('|');
  state.selected = id;
  if (!alreadySelected) {
    state.inspectorSig = null;
    state.inspectorTaskId = null;
    state.inspectorTrajectorySig = null;
    state.inspectorInstruction = null;
    state.inspectorTab = (t.state === 'awake' || !t.report) ? 'run' : 'report';
    state.inspectorResetRunTop = false;
    state.inspectorResetReportTop = false;
    state.inspectorLastPaintedTab = null;
    state.trajectoryFollow = true;
    state.deleteConfirm = { open: false, pending: false, error: null };
  }
  markSeen(id);
  if (rootChanged || before !== after) state.fitPending = true;
  render();
}
function mapActivate(d) {
  if (!d || d.ghost || d.temp) return;
  selectNode(d.id);
}

/* ---- composer ---- */
function baseComposerMode() {
  const s = state.situation;
  if (s) return s.unsupported ? 'notice' : 'review';
  return null;
}
function reviewLocked() {
  return state.situationStatus !== 'idle' || !!state.error || state.actionLock || state.stop.pending;
}
function canReview() {
  const s = state.situation;
  return !!s && !s.unsupported && state.situationStatus === 'idle' && !state.error && !state.actionLock && !state.stop.pending;
}
function growComposer(input) {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight + 2, 168) + 'px';
}
let composerBodySig = null;
function argCountLabel(args) {
  const n = Array.isArray(args) ? args.length
    : (args && typeof args === 'object' ? Object.keys(args).length : 0);
  return n ? `${n} arg${n === 1 ? '' : 's'}` : 'no args';
}
function renderComposerDetails(s) {
  const box = document.getElementById('composer-details');
  const needsTask = s.type === 'review';
  const anchored = needsTask && s.parent ? state.tasks.get(s.parent) : null;
  const bodySig = needsTask
    ? state.situationSig + ':' + state.situationStatus + ':' + (anchored
      ? JSON.stringify([anchored.title, anchored.kind, (anchored.report || '').length])
      : 'missing')
    : state.situationSig + ':' + state.situationStatus;
  if (bodySig === composerBodySig) return;
  composerBodySig = bodySig;
  box.innerHTML = '';

  if (s.type === 'question') {
    box.append(el('div', { class: 'proposal-card' }, [
      el('div', { class: 'proposal-instruction', text: s.question }),
    ]));
    return;
  }
  if (s.type === 'review') {
    if (!anchored) {
      box.append(el('div', { class: 'proposal-card' }, [
        el('div', { class: 'proposal-head' }, [
          el('span', { class: 'proposal-name', text: s.parent || '(root unavailable)' }),
          el('span', { class: 'badge', text: 'review' }),
        ]),
        el('div', { class: 'proposal-instruction', text: s.parent ? 'The reviewed root is not in the network.' : 'No root was named for this review.' }),
      ]));
      return;
    }
    const hasReport = !!(anchored.report || '').length;
    const meta = el('div', { class: 'proposal-deps' }, [
      el('span', { text: hasReport ? 'Report ready' : 'Report pending' }),
    ]);
    if (hasReport) {
      meta.append(el('button', {
        class: 'proposal-link',
        type: 'button',
        title: `Open report for ${anchored.title}`,
        onclick: () => selectNode(anchored.id),
      }, ['Open report →']));
    }
    box.append(el('div', { class: 'proposal-card' }, [
      el('div', { class: 'proposal-head' }, [
        el('span', { class: 'proposal-name', text: anchored.title }),
      ]),
      meta,
      el('div', { class: 'proposal-instruction', text: 'Results are ready. Approve them, or describe a new direction to pivot.' }),
    ]));
    return;
  }
  if (s.type === 'tool-approval') {
    box.append(el('div', { class: 'proposal-card justify-card' }, [
      el('div', { class: 'justify-label', text: 'WHY THIS CALL' }),
      el('div', { class: 'justify-text', text: s.justification }),
    ]));
    box.append(el('details', { class: 'tool-details' }, [
      el('summary', {}, [
        el('span', { text: 'Tool call details' }),
        el('span', { class: 'tool-thumb', text: ` · ${s.tool} · ${argCountLabel(s.args)}` }),
      ]),
      el('div', { class: 'proposal-deps', text: 'arguments' }),
      el('pre', { class: 'report', text: JSON.stringify(s.args, null, 2) }),
    ]));
    return;
  }
  if (s.type === 'task-proposal' && state.situationStatus === 'settled') return;
  for (const t of s.tasks) {
    box.append(el('div', { class: 'proposal-card' }, [
      el('div', { class: 'proposal-head' }, [
        el('span', { class: 'proposal-name', text: t.title }),
        el('span', { class: `badge kind-${t.kind}`, text: t.kind }),
      ]),
      el('div', { class: 'proposal-instruction', text: t.task || '(no instruction)' }),
      el('div', { class: 'proposal-deps', text: `deps: ${t.dependencies.length ? t.dependencies.join(', ') : 'none'}` }),
    ]));
  }
}
function renderComposerSummary(summary, base, s, summaryText) {
  const isAction = base === 'review' && !!s && s.type === 'tool-approval';
  const parent = isAction ? s.parent : null;
  const exists = !!parent && state.tasks.has(parent);
  const sig = isAction ? `action|${parent || ''}|${exists}` : `${base}|${summaryText}`;
  if (summary.dataset.sig === sig) return;
  summary.dataset.sig = sig;
  summary.replaceChildren();
  if (!isAction) {
    summary.textContent = summaryText;
    return;
  }
  summary.append(document.createTextNode('under '));
  if (!parent) {
    summary.append(el('span', { class: 'task-ref-missing', text: '(no parent)' }));
    return;
  }
  if (!exists) {
    summary.append(el('span', { class: 'task-ref-missing', text: `${parent} · not in network` }));
    return;
  }
  summary.append(el('button', {
    class: 'task-ref',
    type: 'button',
    title: `Open ${parent}`,
    onclick: () => selectNode(parent),
  }, [parent]));
}
function renderComposer() {
  const base = baseComposerMode();
  const s = state.situation;
  const section = document.getElementById('composer');
  if (!base) {
    section.className = 'hidden';
    state.composerMode = null;
    return;
  }
  const input = document.getElementById('composer-input');
  const details = document.getElementById('composer-details');
  const kind = document.getElementById('composer-kind');
  const summary = document.getElementById('composer-summary');
  const hint = document.getElementById('composer-hint');
  const status = document.getElementById('composer-status');
  const retry = document.getElementById('composer-retry');
  const row = document.getElementById('composer-row');
  const primary = document.getElementById('composer-primary');

  const eventClass = base === 'review' && s ? EVENT_CLASSES[s.type] : '';
  section.className = `mode-${base}${eventClass ? ' ' + eventClass : ''}`;

  let kindText = 'NOTICE';
  let summaryText = `Unsupported situation: ${s ? s.type : 'unknown'}`;
  if (base === 'review') {
    kindText = EVENT_LABELS[s.type] || 'NOTICE';
    if (s.type === 'question') {
      summaryText = s.parent ? `under ${s.parent}` : '';
    } else if (s.type === 'task-proposal') {
      summaryText = `${s.tasks.length} proposed task${s.tasks.length === 1 ? '' : 's'}${s.parent ? ` under ${s.parent}` : ''}`;
    } else if (s.type === 'tool-approval') {
      summaryText = s.parent ? `under ${s.parent}` : '';
    } else if (s.type === 'review') {
      summaryText = s.parent ? `results ready · ${s.parent}` : 'root unavailable';
    }
  }
  kind.textContent = kindText;
  renderComposerSummary(summary, base, s, summaryText);

  if (base === 'review') {
    details.classList.remove('hidden');
    renderComposerDetails(s);
  } else {
    details.classList.add('hidden');
    details.innerHTML = '';
    composerBodySig = null;
  }

  const value = state.reviewDraft;
  if (input.value !== value) {
    input.value = value;
    growComposer(input);
  }
  input.placeholder = s.type === 'question' ? 'Type your answer…'
    : s.type === 'task-proposal' ? 'Type feedback to request changes…'
      : s.type === 'review' ? 'Type a new direction to pivot…'
        : 'Explain why this tool call should not run…';
  const wasDisabled = input.disabled;
  input.disabled = base === 'notice' || (base === 'review' && state.situationStatus !== 'idle');
  row.classList.toggle('hidden', base === 'notice');
  if (wasDisabled && !input.disabled && base !== 'notice' && state.mode === 'map'
    && document.activeElement === document.body) input.focus();

  const empty = !value.trim();
  primary.classList.remove('hidden');
  if (base === 'review' && s.type === 'question') {
    primary.textContent = 'Send';
    primary.disabled = reviewLocked() || empty;
  } else if (base === 'review' && s.type === 'task-proposal') {
    primary.textContent = empty ? `Approve ${s.tasks.length} task${s.tasks.length === 1 ? '' : 's'}` : 'Request changes';
    primary.disabled = reviewLocked();
  } else if (base === 'review' && s.type === 'review') {
    primary.textContent = empty ? 'Approve results' : 'Pivot';
    primary.disabled = reviewLocked();
  } else if (base === 'review' && s.type === 'tool-approval') {
    primary.textContent = empty ? 'Approve tool call' : 'Deny';
    primary.disabled = reviewLocked();
  } else {
    primary.classList.add('hidden');
  }

  let statusText = '', isError = false;
  if (state.error) {
    statusText = 'backend unreachable';
    isError = true;
  } else if (base === 'review') {
    if (state.situationStatus === 'sending') statusText = 'sending…';
    else if (state.situationStatus === 'settled') statusText = s.type === 'tool-approval'
      ? 'approved — running tool'
      : s.type === 'review' ? 'results approved' : 'approved — adding to network';
    else if (state.situationStatus === 'awaiting') statusText = s.type === 'question'
      ? 'answer sent — waiting for backend'
      : s.type === 'tool-approval' ? 'denied — waiting for backend'
        : s.type === 'review' ? 'pivot sent — waiting for backend' : 'feedback sent — waiting for revision';
    else if (state.reviewError) { statusText = `failed: ${state.reviewError}`; isError = true; }
  }
  status.textContent = statusText;
  status.classList.toggle('error', isError);
  retry.classList.toggle('hidden', !state.error);

  let hintText = '';
  if (base === 'notice') hintText = 'No action is available for this situation.';
  else if (s.type === 'question') hintText = 'Enter or click to send your answer';
  else if (s.type === 'task-proposal') hintText = empty
    ? 'Click to approve · Type to request changes'
    : 'Enter or click to request changes';
  else if (s.type === 'review') hintText = empty
    ? 'Click to approve the results · Type to pivot'
    : 'Enter or click to pivot';
  else if (s.type === 'tool-approval') hintText = empty
    ? 'Click to approve · Type to deny'
    : 'Enter or click to deny';
  if (base === 'review' && state.situationStatus !== 'idle') hintText = 'Waiting for the backend…';
  hint.textContent = hintText;

  const prev = state.composerMode;
  state.composerMode = base;
  if (prev !== base) maybeFocusComposer(base);
}
function maybeFocusComposer(base) {
  if (base !== 'review') return;
  if (state.mode !== 'map') return;
  if (base === 'review' && state.situationStatus !== 'idle') return;
  const active = document.activeElement;
  if (active && active.closest && (active.closest('#composer') || active.closest('#graph') || active.closest('#inspector') || active.closest('#findbar') || active.closest('#task-form') || active.closest('#link-box'))) return;
  if (active && active !== document.body) return;
  const input = document.getElementById('composer-input');
  if (!input.disabled) input.focus();
}

function recomputeFind(resetIndex) {
  const q = state.find.query.trim().toLowerCase();
  if (!q) { state.find.matches = []; state.find.index = 0; return; }
  state.find.matches = [...state.tasks.values()]
    .filter(t => `${t.title} ${t.instruction} ${t.report || ''}`.toLowerCase().includes(q))
    .map(t => t.id);
  if (resetIndex || state.find.index >= state.find.matches.length) state.find.index = 0;
}
function goToMatch(dir) {
  const m = state.find.matches;
  if (!m.length) { paint(); return; }
  state.find.index = (state.find.index + dir + m.length) % m.length;
  const id = m[state.find.index];
  render();
  centerOn(id);
  paintFind();
}
function currentMatchId() {
  return state.find.matches[state.find.index] || null;
}

const composerInput = document.getElementById('composer-input');
composerInput.addEventListener('input', e => {
  state.reviewDraft = e.target.value;
  state.reviewError = null;
  growComposer(e.target);
  renderComposer();
});
composerInput.addEventListener('keydown', e => {
  if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
  e.preventDefault();
  composerSubmit();
});
document.getElementById('composer-primary').onclick = composerPrimary;
document.getElementById('composer-retry').onclick = refresh;

function composerSubmit() {
  if (state.actionLock) return;
  if (baseComposerMode() !== 'review') return;
  sendReviewText();
}
function composerPrimary() {
  if (state.actionLock) return;
  if (baseComposerMode() !== 'review') return;
  const s = state.situation;
  if (!s || s.unsupported) return;
  if (s.type === 'question') return sendReviewText();
  if (state.reviewDraft.trim()) return sendReviewText();
  return approveSituation();
}
function sendReviewText() {
  const text = state.reviewDraft.trim();
  if (!text || !canReview()) return;
  resolveSituation({ review: text });
}
function approveSituation() {
  const s = state.situation;
  if (!s || s.unsupported || !canReview()) return;
  if (s.type !== 'task-proposal' && s.type !== 'tool-approval' && s.type !== 'review') return;
  resolveSituation({});
}
async function postReview(body) {
  const res = await fetch(REVIEW_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(String(res.status));
}
async function resolveSituation(body) {
  if (!state.situation || state.situationStatus !== 'idle' || state.error || state.stop.pending) return;
  const type = state.situation.type;
  state.situationStatus = 'sending';
  state.reviewError = null;
  renderComposer();
  try {
    await postReview(body);
    const reviewType = type === 'task-proposal' || type === 'tool-approval' || type === 'review';
    if (reviewType && !('review' in body)) state.situationStatus = 'settled';
    else state.situationStatus = 'awaiting';
    render();
  } catch (e) {
    state.situationStatus = 'idle';
    state.reviewError = humanError(e);
    render();
  }
}

function setMode(mode) {
  if (mode === 'sandbox' && state.mode !== 'sandbox') {
    state.sandboxReturnMode = state.mode;
    const frame = document.getElementById('sandbox-frame');
    if (!frame.dataset.loaded) {
      frame.dataset.loaded = '1';
      frame.src = SANDBOX_URL;
    }
    state.mode = 'sandbox';
    if (state.runPrompt.open) closeRunPrompt(false);
    if (state.find.open) closeFind(false);
    render();
    return;
  }
  if (mode === 'map') {
    state.mode = 'map';
    state.fitPending = true;
    render();
  }
}
document.getElementById('sandbox-toggle').onclick = () => setMode(state.mode === 'sandbox' ? 'map' : 'sandbox');
document.getElementById('sandbox-open').href = SANDBOX_URL;
document.getElementById('brand').onclick = overview;
document.getElementById('brand').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); overview(); }
});
document.getElementById('breadcrumb').onclick = overview;
document.getElementById('inspector-close').onclick = closeInspector;
document.getElementById('inspector-tab-run').onclick = () => setInspectorTab('run');
document.getElementById('inspector-tab-report').onclick = () => setInspectorTab('report');
document.getElementById('inspector-tabs').addEventListener('keydown', e => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
  e.preventDefault();
  const next = e.key === 'Home' ? 'run'
    : e.key === 'End' ? 'report'
      : state.inspectorTab === 'run' ? 'report' : 'run';
  setInspectorTab(next);
  document.getElementById(`inspector-tab-${next}`).focus({ preventScroll: true });
});
document.getElementById('delete-popover-cancel').onclick = () => closeDeleteConfirm();
document.getElementById('delete-popover-yes').onclick = confirmDeleteTask;
document.getElementById('run-popover-cancel').onclick = () => closeRunPrompt();
document.getElementById('run-popover-submit').onclick = submitRunPrompt;
document.getElementById('run-popover-tip').addEventListener('input', e => {
  state.runPrompt.tip = e.target.value;
  if (!state.runPrompt.tipSaved) state.runPrompt.error = null;
  paintRunPrompt();
});
document.getElementById('run-popover-tip').addEventListener('keydown', e => {
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeRunPrompt(); return; }
  if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
  e.preventDefault();
  e.stopPropagation();
  submitRunPrompt();
});
document.getElementById('task-form-cancel').onclick = () => closeCreateForm();
document.getElementById('task-form-submit').onclick = submitCreateForm;
document.getElementById('task-form-input').addEventListener('input', e => {
  state.form.title = e.target.value;
  state.form.error = null;
  syncFormState();
});
document.getElementById('task-form-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); submitCreateForm(); }
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeCreateForm(); }
});
document.getElementById('link-cancel').onclick = () => closeLinkBox();
document.getElementById('link-input').addEventListener('input', e => {
  state.link.query = e.target.value;
  state.link.index = -1;
  renderLinkBox();
});
document.getElementById('link-input').addEventListener('keydown', e => {
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeLinkBox(); return; }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const n = state.link.items.length;
    if (!n) return;
    const dir = e.key === 'ArrowDown' ? 1 : -1;
    state.link.index = state.link.index < 0
      ? (dir > 0 ? 0 : n - 1)
      : (state.link.index + dir + n) % n;
    renderLinkBox();
    return;
  }
  if (e.key === 'Enter' && !e.isComposing) {
    e.preventDefault();
    const item = state.link.items[state.link.index] || state.link.items[0];
    if (item) toggleDependency(item.id);
  }
});
document.getElementById('stop-all').onclick = openStopConfirm;
document.getElementById('stop-cancel').onclick = () => closeStopConfirm();
document.getElementById('stop-confirm-yes').onclick = cancelAll;
document.addEventListener('click', e => {
  if (!state.stop.confirming) return;
  const dialog = document.getElementById('stop-confirm');
  const btn = document.getElementById('stop-all');
  if (dialog.contains(e.target) || btn.contains(e.target)) return;
  closeStopConfirm();
});
document.addEventListener('pointerdown', e => {
  if (state.form.open) {
    const box = document.getElementById('task-form');
    if (!box.contains(e.target)) {
      state.gesture.suppressClickUntil = Date.now() + 400;
      closeCreateForm();
    }
  }
  if (state.link.open) {
    const box = document.getElementById('link-box');
    if (!box.contains(e.target)) {
      state.gesture.suppressClickUntil = Date.now() + 400;
      closeLinkBox();
    }
  }
  if (state.deleteConfirm.open) {
    const box = document.getElementById('delete-popover');
    const btn = document.querySelector('#node-actions .node-action.delete');
    if (!box.contains(e.target) && !(btn && btn.contains(e.target))) { closeDeleteConfirm(); }
  }
  if (state.runPrompt.open) {
    const box = document.getElementById('run-popover');
    const actions = document.querySelector('#node-actions');
    if (!box.contains(e.target) && !(actions && actions.contains(e.target))) {
      state.gesture.suppressClickUntil = Date.now() + 400;
      closeRunPrompt();
    }
  }
}, true);
document.getElementById('activity-task').onclick = () => {
  const btn = document.getElementById('activity-task');
  if (btn.getAttribute('aria-disabled') === 'true') return;
  const id = document.getElementById('activity').dataset.task;
  if (!id || !state.tasks.has(id)) return;
  selectNode(id);
};

function openFind() {
  if (state.mode === 'sandbox') return;
  if (state.find.open) { document.getElementById('find-input').focus(); return; }
  state.find.returnRoot = state.activeRoot;
  state.find.returnExpanded = new Set(state.expanded);
  state.find.returnSelected = state.selected;
  state.find.open = true;
  state.find.query = '';
  state.find.matches = [];
  state.find.index = 0;
  state.mode = 'map';
  state.selected = null;
  state.inspectorSig = null;
  state.inspectorTaskId = null;
  state.fitPending = true;
  document.getElementById('find-input').value = '';
  document.getElementById('findbar').classList.remove('hidden');
  render();
  document.getElementById('find-input').focus();
}
function closeFind(restore = true) {
  document.getElementById('findbar').classList.add('hidden');
  const input = document.getElementById('find-input');
  input.value = '';
  input.blur();
  const saved = state.find;
  state.find = { open: false, query: '', matches: [], index: 0, returnRoot: null, returnExpanded: null, returnSelected: null };
  if (!restore) return;
  if (saved.returnRoot && state.tasks.has(saved.returnRoot) && state.tasks.get(saved.returnRoot).kind === 'root') {
    state.activeRoot = saved.returnRoot;
    state.expanded = saved.returnExpanded && saved.returnExpanded.size
      ? new Set([...saved.returnExpanded].filter(id => state.tasks.has(id)))
      : new Set([saved.returnRoot]);
    state.expanded.add(saved.returnRoot);
    state.selected = saved.returnSelected && state.tasks.has(saved.returnSelected) ? saved.returnSelected : null;
  } else {
    state.activeRoot = null;
    state.expanded.clear();
    state.selected = null;
  }
  state.fitPending = true;
  render();
}
document.getElementById('find-close').onclick = () => closeFind();
document.getElementById('find-input').addEventListener('input', e => {
  state.find.query = e.target.value;
  recomputeFind(true);
  state.fitPending = true;
  render();
  if (state.find.matches.length) centerOn(currentMatchId());
});
document.getElementById('find-input').addEventListener('keydown', e => {
  const has = state.find.matches.length > 0;
  if (e.key === 'Enter') {
    e.preventDefault();
    const id = currentMatchId();
    if (id) {
      closeFind(false);
      selectNode(id);
      centerOn(id);
      return;
    }
    const q = e.target.value.trim();
    if (!q) return;
    if (topologyLocked()) { notifyLocked(); return; }
    closeFind(false);
    const pane = document.getElementById('map-pane').getBoundingClientRect();
    openCreateForm(pane.left + pane.width / 2, pane.top + pane.height / 2, [], q);
    return;
  }
  if (!has) return;
  if (e.key === 'Tab') { e.preventDefault(); goToMatch(e.shiftKey ? -1 : 1); return; }
  if (e.key === 'ArrowDown') { e.preventDefault(); goToMatch(1); return; }
  if (e.key === 'ArrowUp') { e.preventDefault(); goToMatch(-1); }
});
document.getElementById('find-prev').onclick = () => goToMatch(-1);
document.getElementById('find-next').onclick = () => goToMatch(1);

document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
    if (state.mode === 'sandbox' || state.form.open || state.link.open || state.runPrompt.open) return;
    e.preventDefault();
    openFind();
    return;
  }
  if (e.key === 'Escape') {
    if (state.stop.confirming) { closeStopConfirm(); return; }
    if (state.form.open) { closeCreateForm(); return; }
    if (state.link.open) { closeLinkBox(); return; }
    if (state.deleteConfirm.open) { closeDeleteConfirm(); return; }
    if (state.runPrompt.open) { closeRunPrompt(); return; }
    if (state.gesture.mode !== 'idle') { cancelGesture(); return; }
    if (state.find.open) { closeFind(); return; }
    if (state.mode === 'sandbox') { setMode(state.sandboxReturnMode || 'map'); return; }
    if (state.selected) { closeInspector(); return; }
    if (state.activeRoot) { overview(); return; }
    return;
  }
  const a = document.activeElement, tag = a && a.tagName;
  const typing = tag === 'INPUT' || tag === 'TEXTAREA' || (a && a.isContentEditable);
  if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey) {
    if (typing || state.mode === 'sandbox' || state.form.open || state.link.open || state.deleteConfirm.open || state.runPrompt.open) return;
    e.preventDefault();
    openFind();
    return;
  }
  if (typing || state.mode === 'sandbox' || state.form.open || state.link.open || state.deleteConfirm.open || state.runPrompt.open || state.stop.confirming) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'l' || e.key === 'L') {
    const id = currentTaskId();
    if (!id) return;
    e.preventDefault();
    openLinkBox();
    return;
  }
  if (e.key === 'Delete') {
    const id = currentTaskId();
    if (!id) return;
    e.preventDefault();
    openDeleteConfirm(id);
  }
});

/* ---- token heartbeat ---- */
let streamAbort = null, streamRetryTimer = null, streamBackoff = 1000;

function syncStream() {
  const awake = currentAwakeTask();
  if (!awake || document.hidden) {
    if (streamAbort || streamRetryTimer || state.stream.status !== 'idle') closeStream();
    return;
  }
  if (streamAbort || streamRetryTimer) return;
  connectStream();
}
function connectStream() {
  const awake = currentAwakeTask();
  if (!awake || document.hidden || streamAbort) return;
  const ctrl = new AbortController();
  streamAbort = ctrl;
  const s = state.stream;
  s.status = 'connecting';
  if (s.task && s.task !== awake.id) {
    s.action = null;
  }
  s.task = awake.id;
  s.lastChunkAt = null;
  s.chunkTimes = [];
  s.rate = 0;
  if (s.pulseTimer) { clearTimeout(s.pulseTimer); s.pulseTimer = null; }
  runStream(ctrl);
}
async function runStream(ctrl) {
  const s = state.stream;
  try {
    const res = await fetch(STREAM_URL, {
      signal: ctrl.signal,
      headers: { accept: 'text/event-stream' },
      cache: 'no-store',
    });
    if (ctrl.signal.aborted) return;
    if (res.status === 404) {
      if (streamAbort === ctrl) streamAbort = null;
      if (currentAwakeTask()) {
        s.status = 'ended';
        scheduleStreamRetry(2000);
      } else {
        closeStream();
      }
      return;
    }
    if (!res.ok || !res.body) throw new Error(String(res.status));
    s.status = 'open';
    streamBackoff = 1000;
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        handleFrame(buf.slice(0, idx));
        buf = buf.slice(idx + 2);
      }
    }
    if (!ctrl.signal.aborted) onStreamEnd();
  } catch (e) {
    if (!ctrl.signal.aborted) onStreamEnd();
  } finally {
    if (streamAbort === ctrl) streamAbort = null;
  }
}
function handleFrame(frame) {
  let event = '';
  const data = [];
  for (const line of frame.split('\n')) {
    if (!line) continue;
    if (line.startsWith(':')) continue;
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
  }
  if (event) onChunk(event, data.join('\n'));
}
function onChunk(task, rawAction) {
  const s = state.stream;
  const now = Date.now();
  if (s.task !== task) {
    s.task = task;
    s.chunkTimes = [];
    s.rate = 0;
    s.action = null;
    streamRefreshSoon();
  }
  const text = normalizeAction(rawAction);
  if (text) s.action = { task, text, at: now };
  s.lastChunkAt = now;
  s.chunkTimes.push(now);
  schedulePulse();
}
function schedulePulse() {
  const s = state.stream;
  if (s.pulseTimer || reduceMotion()) return;
  s.pulseTimer = setTimeout(() => {
    s.pulseTimer = null;
    if (state.stream.status === 'open' && state.stream.task) emitPulse(state.stream.task);
  }, 95);
}
function emitPulse(taskId) {
  if (reduceMotion()) return;
  nodeL.selectAll('g.node').filter(d => d.id === taskId).each(function () {
    const core = d3.select(this).select('.core');
    const r = +core.attr('r') || 18;
    d3.select(this).append('circle')
      .attr('class', 'token-pulse')
      .attr('r', r + 2)
      .attr('opacity', .55)
      .transition()
      .duration(650)
      .ease(d3.easeCubicOut)
      .attr('r', r + 34)
      .attr('opacity', 0)
      .remove();
  });
}
function streamRefreshSoon() {
  const now = Date.now();
  if (now - state.stream.lastRefreshAt < 600) return;
  state.stream.lastRefreshAt = now;
  refresh();
}
function onStreamEnd() {
  const s = state.stream;
  if (s.status === 'idle') return;
  s.status = 'ended';
  s.rate = 0;
  scheduleStreamRetry();
  streamRefreshSoon();
}
function scheduleStreamRetry(delay) {
  if (streamRetryTimer) return;
  const wait = delay != null ? delay : streamBackoff;
  streamBackoff = Math.min(streamBackoff * 2, 10000);
  streamRetryTimer = setTimeout(() => {
    streamRetryTimer = null;
    if (currentAwakeTask() && !document.hidden) connectStream();
    else closeStream();
  }, wait);
}
function closeStream() {
  if (streamAbort) { const a = streamAbort; streamAbort = null; a.abort(); }
  if (streamRetryTimer) { clearTimeout(streamRetryTimer); streamRetryTimer = null; }
  const s = state.stream;
  if (s.pulseTimer) { clearTimeout(s.pulseTimer); s.pulseTimer = null; }
  s.status = 'idle';
  s.task = null;
  s.chunkTimes = [];
  s.rate = 0;
  s.lastChunkAt = null;
  s.action = null;
  s.activitySig = null;
  nodeL.selectAll('.token-pulse').remove();
  updateNodeActivity(0, Date.now());
  paintActivity();
}
function updateNodeActivity(level, now) {
  const s = state.stream;
  const action = s.action && s.task && s.action.task === s.task ? s.action.text : '';
  nodeL.selectAll('g.node.state-awake').each(function (d) {
    const on = s.status === 'open' && s.task === d.id;
    const age = s.lastChunkAt ? now - s.lastChunkAt : null;
    const quiet = on && age != null && age >= STREAM_QUIET_MS;
    const stalled = on && age != null && age >= STREAM_STALL_MS;
    const sel = d3.select(this);
    sel.classed('live', on);
    sel.classed('quiet', quiet);
    sel.classed('stalled', stalled);
    if (on) sel.style('--activity', String(level));
    else sel.style('--activity', null);
    if (!on) {
      sel.attr('aria-description', null);
      sel.node().__descAt = 0;
      return;
    }
    const base = stalled ? 'Output has been quiet for a while' : 'Engine is working';
    const desc = action ? `${base}: ${action}` : base;
    const prev = sel.attr('aria-description');
    if (desc !== prev && (now - (sel.node().__descAt || 0) >= 1000 || !prev)) {
      sel.attr('aria-description', desc);
      sel.node().__descAt = now;
    }
  });
}
function tickHeartbeat() {
  const now = Date.now();
  const s = state.stream;
  s.chunkTimes = s.chunkTimes.filter(t => now - t < 3000);
  const instant = s.chunkTimes.filter(t => now - t < 1000).length;
  s.rate = s.rate * 0.7 + instant * 0.3;
  const live = s.status === 'open' && s.task;
  updateNodeActivity(live ? Math.min(1, s.rate / 8) : 0, now);
  paintActivity();
}

let pab = null, pt = null;
async function getTasks(sig) {
  const res = await fetch(TASKS_URL, { signal: sig, headers: { accept: 'application/json' }, cache: 'no-store' });
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}
async function refresh() {
  if (pab) pab.abort();
  pab = new AbortController();
  try {
    const env = await getTasks(pab.signal);
    if (pab.signal.aborted) return;
    state.error = null;
    applyEnvelope(env);
  } catch (e) {
    if (e.name === 'AbortError') return;
    state.error = String(e);
    render();
  }
}
function poll() {
  if (!document.hidden) refresh();
  pt = setTimeout(poll, POLL_INTERVAL);
}

let mapW = 0, resizeTimer = null;
const mapObserver = new ResizeObserver(entries => {
  const w = entries[0].contentRect.width;
  if (!mapW) { mapW = w; return; }
  if (Math.abs(w - mapW) < 40) return;
  mapW = w;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.mode === 'map' && state.gesture.mode === 'idle' && !state.form.open && state.layout && state.layout.bounds) fit();
  }, 180);
});
mapObserver.observe(document.getElementById('map-pane'));

document.addEventListener('visibilitychange', () => {
  if (document.hidden) closeStream();
  else refresh();
});
window.addEventListener('pagehide', () => closeStream());
setInterval(tickHeartbeat, 100);

render();
refresh();
pt = setTimeout(poll, POLL_INTERVAL);
