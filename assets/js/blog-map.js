(() => {
  const canvas = document.querySelector('.blog-map-canvas');
  if (!canvas) return;
  const svg = canvas.querySelector('svg');
  const list = canvas.querySelector('.blog-map-nodes');
  const nodes = new Map([...canvas.querySelectorAll('[data-node]')].map(node => [node.dataset.node, node]));
  const edges = JSON.parse(document.getElementById('blog-map-data').textContent)
    .filter(edge => nodes.has(edge.from) && nodes.has(edge.to));

  // Every edge constrains the layout, including transitions through stop nodes.
  // Work backwards from terminal nodes so predecessors always sit below successors.
  const remaining = new Map([...nodes.keys()].map(id => [id, 0]));
  const parents = new Map([...nodes.keys()].map(id => [id, []]));
  const ranks = new Map([...nodes.keys()].map(id => [id, 0]));
  edges.forEach(({ from, to }) => {
    remaining.set(from, remaining.get(from) + 1);
    parents.get(to).push(from);
  });
  const queue = [...nodes.keys()].filter(id => remaining.get(id) === 0);
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i];
    for (const parent of parents.get(id)) {
      ranks.set(parent, Math.max(ranks.get(parent), ranks.get(id) + 1));
      remaining.set(parent, remaining.get(parent) - 1);
      if (remaining.get(parent) === 0) queue.push(parent);
    }
  }
  // A cycle cannot form a reading tree. Keep the chronological list usable instead.
  if (queue.length !== nodes.size) return;
  const layers = [];
  for (const [id] of nodes) {
    const rank = ranks.get(id);
    (layers[rank] ??= []).push(id);
  }
  const columns = Math.max(...layers.map(layer => layer.length));
  list.style.setProperty('--columns', columns * 2);
  layers.forEach((layer, rank) => layer.forEach((id, index) => {
    const node = nodes.get(id);
    node.style.setProperty('--row', rank + 1);
    node.style.setProperty('--column', columns - layer.length + 1 + index * 2);
  }));
  canvas.classList.add('is-laid-out');

  const tooltip = document.createElement('div');
  tooltip.className = 'blog-map-edge-tooltip';
  tooltip.hidden = true;
  canvas.append(tooltip);
  let activeEdge = null;
  function hideLabel() {
    if (activeEdge) activeEdge.group.classList.remove('is-active');
    activeEdge = null;
    tooltip.hidden = true;
  }
  function showLabel(item, event) {
    hideLabel();
    activeEdge = item;
    item.group.classList.add('is-active');
    tooltip.textContent = item.edge.label || 'Related idea';
    tooltip.hidden = false;
    const bounds = canvas.getBoundingClientRect();
    const point = item.path.getPointAtLength(item.path.getTotalLength() / 2);
    const x = event?.clientX != null ? event.clientX - bounds.left : point.x;
    const y = event?.clientY != null ? event.clientY - bounds.top : point.y;
    const half = tooltip.offsetWidth / 2;
    tooltip.style.left = `${Math.max(half + 4, Math.min(bounds.width - half - 4, x))}px`;
    tooltip.style.top = `${y + 12}px`;
  }
  const paths = edges.map(edge => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.dataset.kind = edge.kind || 'development';
    group.setAttribute('tabindex', '0');
    group.setAttribute('role', 'img');
    const from = nodes.get(edge.from).querySelector('h2').textContent.trim();
    const to = nodes.get(edge.to).querySelector('h2').textContent.trim();
    group.setAttribute('aria-label', `${from} → ${to}: ${edge.label || 'Related idea'}`);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.dataset.from = edge.from;
    path.dataset.to = edge.to;
    path.dataset.kind = edge.kind || 'development';
    const hit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hit.setAttribute('class', 'blog-map-edge-hit');
    hit.setAttribute('aria-hidden', 'true');
    const head = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    head.setAttribute('class', 'blog-map-arrowhead');
    head.setAttribute('d', 'M -8 -4 L 0 0 L -8 4');
    head.setAttribute('aria-hidden', 'true');
    group.append(path, head, hit);
    svg.append(group);
    const item = { edge, path, hit, head, group };
    group.addEventListener('mouseenter', event => showLabel(item, event));
    group.addEventListener('mouseleave', () => {
      if (document.activeElement === group) showLabel(item);
      else if (activeEdge === item) hideLabel();
    });
    group.addEventListener('focus', () => showLabel(item));
    group.addEventListener('blur', () => { if (activeEdge === item) hideLabel(); });
    group.addEventListener('keydown', event => { if (event.key === 'Escape') hideLabel(); });
    return item;
  });
  const sidePaths = paths.filter(({ edge }) => ranks.get(edge.from) - ranks.get(edge.to) > 1);
  const laneEnds = [];
  // Reserve a separate lane for routes whose vertical spans overlap.
  [...sidePaths].sort((a, b) => ranks.get(a.edge.to) - ranks.get(b.edge.to)
    || ranks.get(a.edge.from) - ranks.get(b.edge.from)).forEach(item => {
    const top = ranks.get(item.edge.to);
    const bottom = ranks.get(item.edge.from);
    let lane = laneEnds.findIndex(end => end < top);
    if (lane === -1) lane = laneEnds.length;
    laneEnds[lane] = bottom;
    item.lane = lane;
  });
  list.style.setProperty('--side-lanes', laneEnds.length);
  const sidePorts = new Map([...nodes.keys()].map(id => [id, {
    outgoing: sidePaths.filter(({ edge }) => edge.from === id),
    incoming: sidePaths.filter(({ edge }) => edge.to === id),
  }]));
  function port(item, direction) {
    const id = direction === 'outgoing' ? item.edge.from : item.edge.to;
    const siblings = sidePorts.get(id)[direction];
    // Outgoing connections attach above incoming ones, with separate slots per edge.
    return (direction === 'outgoing' ? 0.16 : 0.60)
      + 0.24 * (siblings.indexOf(item) + 1) / (siblings.length + 1);
  }
  const mobile = window.matchMedia('(max-width: 700px)');
  function draw() {
    hideLabel();
    if (mobile.matches) return;
    const bounds = canvas.getBoundingClientRect();
    for (const item of paths) {
      const { edge, path, hit, head } = item;
      const a = nodes.get(edge.from).querySelector('.blog-map-post').getBoundingClientRect();
      const b = nodes.get(edge.to).querySelector('.blog-map-post').getBoundingClientRect();
      const x1 = a.left + a.width / 2 - bounds.left;
      const x2 = b.left + b.width / 2 - bounds.left;
      const y1 = a.top - bounds.top - 4;
      const y2 = b.bottom - bounds.top + 10;
      if (ranks.get(edge.from) - ranks.get(edge.to) > 1) {
        // Long edges travel outside the cards instead of crossing intervening posts.
        const lane = bounds.width - 24 - item.lane * 24;
        const start = a.top + a.height * port(item, 'outgoing') - bounds.top;
        const end = b.top + b.height * port(item, 'incoming') - bounds.top;
        const startX = a.right - bounds.left + 4;
        const endX = b.right - bounds.left + 10;
        const radius = Math.max(0, Math.min(18, (lane - Math.max(startX, endX)) / 2, (start - end) / 4));
        path.setAttribute('d', `M ${startX} ${start} H ${lane - radius} Q ${lane} ${start} ${lane} ${start - radius} V ${end + radius} Q ${lane} ${end} ${lane - radius} ${end} H ${endX}`);
      } else {
        const dx = x2 - x1;
        const lift = (y1 - y2) * 0.48;
        path.setAttribute('d', `M ${x1} ${y1} C ${x1 + dx * 0.12} ${y1 - lift}, ${x2 - dx * 0.22} ${y2 + lift}, ${x2} ${y2}`);
      }
      hit.setAttribute('d', path.getAttribute('d'));
      // Rotate a fixed-size chevron to the actual tangent at the curve's end.
      const length = path.getTotalLength();
      const tip = path.getPointAtLength(length);
      const before = path.getPointAtLength(Math.max(0, length - 0.75));
      const angle = Math.atan2(tip.y - before.y, tip.x - before.x) * 180 / Math.PI;
      head.setAttribute('transform', `translate(${tip.x} ${tip.y}) rotate(${angle})`);
    }
  }
  for (const [id, node] of nodes) {
    const highlight = active => paths.forEach(({ edge, group }) => group.classList.toggle('is-related', active && (edge.from === id || edge.to === id)));
    node.addEventListener('mouseenter', () => highlight(true));
    node.addEventListener('mouseleave', () => highlight(false));
    node.addEventListener('focusin', () => highlight(true));
    node.addEventListener('focusout', () => highlight(false));
  }
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  nodes.forEach(node => observer.observe(node));
  mobile.addEventListener('change', draw);
  document.fonts.ready.then(draw);
  draw();
})();
