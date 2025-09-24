#!/usr/bin/env python3
import argparse
import json
import re
import os
import sys
from pathlib import Path


HTML_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Pipeline DAG Visualization</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #111827; }
    #container { display: grid; grid-template-columns: 1fr 320px; grid-template-rows: auto 1fr; height: 100vh; transition: grid-template-columns 220ms ease; }
    #container.no-details { grid-template-columns: 1fr 0px; }
    header { grid-column: 1 / -1; padding: 8px 12px; background: #0f172a; color: #e2e8f0; display:flex; align-items:center; gap:12px; }
    header h1 { font-size: 16px; margin: 0; font-weight: 600; }
    header .meta { opacity: .8; font-size: 12px; }
    #graph { position: relative; background: #111827; height: 100%; min-height: 480px; }
    #sidebar { border-left: none; border: 1px solid #e2e8f0; background: rgba(248,250,252,0.96); overflow: auto; transition: transform 220ms ease, opacity 220ms ease; margin: 12px 12px 12px 0; border-radius: 12px; box-shadow: 0 12px 28px rgba(0,0,0,0.28); }
    #container.no-details #sidebar { transform: translateX(100%); opacity: 0; pointer-events: none; }
    #sidebar h2 { font-size: 14px; margin: 0; }
    .panel-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }
    #hideDetails { display:flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius: 50%; border: 1px solid #cbd5e1; background: #e2e8f0; color: #0f172a; cursor: pointer; }
    #hideDetails:hover { background: #e5e7eb; }
    #hideDetails svg { width:16px; height:16px; stroke:#0f172a; }
    #details { padding: 0 12px 12px; font-size: 12px; }
    .kv { margin: 8px 0; }
    .kv .k { color: #334155; font-weight: 600; }
    .kv .v { color: #0f172a; word-break: break-all; }

    svg { width: 100%; height: 100%; }
    marker path { fill: #94a3b8; }
    .link { stroke: #64748b; stroke-opacity: .7; }
    .link.tensor { stroke-dasharray: 2 2; }
    .node circle { stroke: #1e293b; stroke-width: 1; }
    .node rect { stroke: #0b1324; stroke-width: 1; }
    .node text { font-size: 11px; pointer-events: none; }

    .op circle { fill: #60a5fa; }
    .tensor rect { fill: #34d399; rx: 6; ry: 6; }
    .io rect { fill: #fbbf24; rx: 6; ry: 6; }
    .external rect { fill: #f87171; rx: 6; ry: 6; }
    .selected circle, .selected rect { stroke: #3b82f6; stroke-width: 2; }
    .node:hover circle, .node:hover rect { stroke: #93c5fd; stroke-width: 2.2; }
    .node:hover { cursor: pointer; }
    .node.adjacent circle, .node.adjacent rect { stroke: #93c5fd; stroke-width: 2.2; }
    .link.adjacent { stroke: #93c5fd; stroke-opacity: .95; stroke-width: 2.2; }
    .tensorlabel.adjacent { fill: #93c5fd; font-weight: 700; }

    .legend { position:absolute; right:12px; top:12px; background:#0f172a; color:#e2e8f0; border: 1px solid #1f2937; border-radius: 6px; padding: 8px 10px; font-size: 12px; }
    .legend-item { display:flex; align-items:center; gap:6px; margin: 4px 0; }
    .legend-swatch { width: 12px; height: 12px; border:1px solid #1e293b; }
    .legend .sw-op { background:#60a5fa; border-radius: 50%; width: 12px; height: 12px; }
    .legend .sw-tensor { background:#34d399; border-radius: 2px; }
    .legend .sw-io { background:#fbbf24; border-radius: 2px; }
    .legend .sw-external { background:#f87171; border-radius: 2px; }
  </style>
  <script>/*__D3_BUNDLE__*/</script>
  <script>
    const GRAPH_DATA = __GRAPH_DATA__;
  </script>
</head>
<body>
  <div id=\"container\" class=\"no-details\"> 
    <header>
      <h1>Pipeline DAG Visualization</h1>
      <div class=\"meta\">Source: <span id=\"src\"></span></div>
    </header>
    <div id=\"graph\"></div>
    <aside id=\"sidebar\"> 
      <div class=\"panel-header\"> 
        <h2>Details</h2>
        <button id=\"hideDetails\" title=\"隐藏\" aria-label=\"隐藏\"> 
          <svg viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"> 
            <path d=\"M15 6l6 6-6 6\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/> 
          </svg>
        </button>
      </div>
      <div id=\"details\">Click a node to inspect its properties.</div>
    </aside>
  </div>
  <script>
  window.addEventListener('load', function() {
    function nonZero(n, fallback) { return (typeof n === 'number' && n > 0) ? n : fallback; }
    const gEl = document.getElementById('graph');
    const rect = gEl.getBoundingClientRect();
    const width = nonZero(rect.width, Math.max(600, window.innerWidth - 340));
    const height = nonZero(rect.height, Math.max(480, window.innerHeight - 100));
    const container = d3.select('#graph');
    const root = d3.select('#container');

    d3.select('#src').text(GRAPH_DATA.source_path);

    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Arrow markers
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5');

    // Legend
    const legend = container.append('div').attr('class','legend');
    const legendItems = [
      {label:'Operator', cls:'sw-op'},
      {label:'Tensor', cls:'sw-tensor'},
      {label:'Pipeline IO', cls:'sw-io'},
      {label:'External/Inputless', cls:'sw-external'}
    ];
    legendItems.forEach(it => {
      const row = legend.append('div').attr('class','legend-item');
      row.append('div').attr('class', `legend-swatch ${it.cls}`);
      row.append('div').text(it.label);
    });

    const nodes = GRAPH_DATA.nodes.map(d => Object.assign({}, d));
    const links = GRAPH_DATA.links.map(d => Object.assign({}, d));

    // Ensure D3 loaded
    if (typeof d3 === 'undefined') {
      const warn = document.createElement('div');
      warn.style.color = '#fecaca';
      warn.style.background = '#7f1d1d';
      warn.style.padding = '10px';
      warn.textContent = 'Error: D3 not loaded.';
      gEl.appendChild(warn);
      return;
    }

    // Build layered DAG layout (left-to-right)
    const idToNode = new Map(nodes.map(n => [n.id, n]));
    nodes.forEach(n => { n.in = []; n.out = []; n.layer = undefined; });
    links.forEach(l => {
      const s = typeof l.source === 'string' ? idToNode.get(l.source) : idToNode.get(l.source.id);
      const t = typeof l.target === 'string' ? idToNode.get(l.target) : idToNode.get(l.target.id);
      l.source = s; l.target = t;
      s && s.out.push(t);
      t && t.in.push(s);
    });

    // Kahn-style layering: source nodes start at layer 0
    const indeg = new Map(nodes.map(n => [n.id, n.in.length]));
    const q = [];
    nodes.forEach(n => { if ((indeg.get(n.id) || 0) === 0) { n.layer = 0; q.push(n); } });
    while (q.length) {
      const u = q.shift();
      const base = u.layer ?? 0;
      (u.out || []).forEach(v => {
        v.layer = Math.max(v.layer ?? 0, base + 1);
        indeg.set(v.id, (indeg.get(v.id) || 0) - 1);
        if ((indeg.get(v.id) || 0) === 0) q.push(v);
      });
    }
    // Fallback: any undefined layer -> 0
    nodes.forEach(n => { if (n.layer === undefined) n.layer = 0; });

    // Group by layer and assign positions
    const layers = d3.rollup(nodes, v => v, n => n.layer);
    const layerKeys = Array.from(layers.keys()).sort((a,b)=>a-b);
    const colGap = 220; // horizontal spacing between layers
    const rowGap = 90;  // vertical spacing
    const topPad = 40, leftPad = 60;
    layerKeys.forEach((k, idx) => {
      const col = layers.get(k);
      // Sort: tensors first then ops for readability
      col.sort((a,b)=> (a.type===b.type?0:(a.type==='tensor'||a.type==='io'||a.type==='external')?-1:1));
      col.forEach((n, i) => {
        n.x = leftPad + idx * colGap;
        n.y = topPad + i * rowGap;
      });
    });

    // Vertically center the whole graph within the viewport
    const minY = d3.min(nodes, d => d.y);
    const maxY = d3.max(nodes, d => d.y);
    const graphH = (maxY ?? 0) - (minY ?? 0);
    let deltaY = 0;
    if (Number.isFinite(graphH) && Number.isFinite(minY)) {
      if (graphH < height) {
        deltaY = (height - graphH) / 2 - minY;
      } else {
        // If graph taller than view, keep a small top padding
        deltaY = Math.max(20 - minY, 0);
      }
      nodes.forEach(n => { n.y += deltaY; });
    }

    const g = svg.append('g');
    const zoom = d3.zoom().scaleExtent([0.3, 2]).on('zoom', (event) => {
      g.attr('transform', event.transform);
    });
    svg.call(zoom);

    const link = g.append('g').attr('stroke-width', 1.4).selectAll('line')
      .data(links)
      .join('line')
      .attr('class', d => 'link' + (d.kind === 'tensor' ? ' tensor' : ''))
      .attr('marker-end', 'url(#arrow)')
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    // Tensor labels (above tensor nodes)
    const dtypeName = (code) => ({1:'uint8',5:'int32',6:'float32'})[code] || (code==null?'':String(code));
    function tensorInfo(n){
      const a = n.attributes || {};
      const dims = Array.isArray(a.dimensions) ? a.dimensions.join('×') : '';
      const ch = a.channels!=null ? a.channels : '';
      const dt = dtypeName(a.data_type);
      let parts = [];
      if (dims) parts.push(dims);
      if (ch) parts.push('c'+ch);
      if (dt) parts.push(dt);
      return parts.join(' ');
    }
    const tensorNodes = nodes.filter(n => n.type==='tensor' || n.type==='io' || n.type==='external');
    const tensorLabels = g.append('g').selectAll('text.tensorlabel')
      .data(tensorNodes)
      .join('text')
      .attr('class','tensorlabel')
      .attr('fill', '#e2e8f0')
      .attr('font-size', 11)
      .attr('text-anchor','middle')
      .style('pointer-events','none')
      .text(d => tensorInfo(d));

    const node = g.append('g').selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', d => `node ${d.type}`)
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .call(d3.drag().on('drag', dragged));

    node.append(d => {
      if (d.type === 'op') {
        const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        el.setAttribute('r', 44);
        return el;
      } else {
        const el = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        el.setAttribute('x', -55);
        el.setAttribute('y', -21);
        el.setAttribute('width', 110);
        el.setAttribute('height', 42);
        return el;
      }
    });

    node.append('text')
      .attr('fill', d => d.type==='op' ? '#0b1020' : '#0b1020')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .text(d => d.label);

    node.on('click', (event, d) => {
      node.classed('selected', n => n.id === d.id);
      showDetails(d);
      root.classed('no-details', false);
      event.stopPropagation();
    });

    svg.on('click', () => { node.classed('selected', false); root.classed('no-details', true); });

    // Hover adjacency highlighting
    node.on('mouseenter', (event, d) => {
      const isAdj = (n) => n.id === d.id || (d.in || []).includes(n) || (d.out || []).includes(n);
      node.classed('adjacent', n => isAdj(n));
      link.classed('adjacent', l => l.source === d || l.target === d);
      tensorLabels.classed('adjacent', n => isAdj(n));
    });
    node.on('mouseleave', () => {
      node.classed('adjacent', false);
      link.classed('adjacent', false);
      tensorLabels.classed('adjacent', false);
    });

    // Hide button toggles sidebar visibility
    const hideBtn = document.getElementById('hideDetails');
    if (hideBtn) hideBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      node.classed('selected', false);
      root.classed('no-details', true);
      const el = document.getElementById('details');
      el.textContent = 'Click a node to inspect its properties.';
    });

    function updateLinks() {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      tensorLabels
        .attr('x', d => d.x)
        .attr('y', d => d.y - 26);
    }
    updateLinks();

    function dragged(event, d) {
      d.x = event.x; d.y = event.y;
      d3.select(this).attr('transform', `translate(${d.x},${d.y})`);
      updateLinks();
    }

    function showDetails(d) {
      const el = document.getElementById('details');
      const esc = (v) => (typeof v === 'string' ? v.replace(/[&<>]/g, (c)=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])) : String(v));
      const entries = [];
      entries.push(`<div class=\"kv\"><span class=\"k\">Id:</span> <span class=\"v\">${esc(d.id)}</span></div>`);
      entries.push(`<div class=\"kv\"><span class=\"k\">Type:</span> <span class=\"v\">${esc(d.type)}</span></div>`);
      if (d.subtype) entries.push(`<div class=\"kv\"><span class=\"k\">Op:</span> <span class=\"v\">${esc(d.subtype)}</span></div>`);
      if (d.attributes && Object.keys(d.attributes).length) {
        entries.push('<div class=\"kv\"><span class=\"k\">Attributes:</span></div>');
        entries.push('<pre style=\"background:#e2e8f0;padding:8px;border-radius:6px;overflow:auto;\">'+esc(JSON.stringify(d.attributes, null, 2))+'</pre>');
      }
      if (d.inputs && d.inputs.length) {
        entries.push(`<div class=\"kv\"><span class=\"k\">Inputs:</span> <span class=\"v\">${d.inputs.map(esc).join(', ')}</span></div>`);
      }
      if (d.outputs && d.outputs.length) {
        entries.push(`<div class=\"kv\"><span class=\"k\">Outputs:</span> <span class=\"v\">${d.outputs.map(esc).join(', ')}</span></div>`);
      }
      el.innerHTML = entries.join('\n');
    }
  });
  </script>
  </body>
  </html>
"""


def parse_pipeline(json_obj, source_path):
    # Build bipartite graph: operator nodes and tensor nodes
    nodes = []
    links = []

    tensors = json_obj.get("tensors", {})
    operators = json_obj.get("operators", [])
    pipeline_inputs = set(json_obj.get("inputs", []) or [])
    pipeline_outputs = set(json_obj.get("outputs", []) or [])

    # Create tensor nodes
    for tname, tmeta in tensors.items():
        kind = "tensor"
        label = tname
        node = {
            "id": f"tensor::{tname}",
            "type": "tensor",
            "subtype": None,
            "label": label,
            "attributes": tmeta,
        }
        if tname in pipeline_inputs or tname in pipeline_outputs:
            node["type"] = "io"
        nodes.append(node)

    # Helper to ensure nodes exist for tensors referenced but not declared
    def ensure_tensor_node(name, external=False):
        nid = f"tensor::{name}"
        if any(n for n in nodes if n["id"] == nid):
            return nid
        node = {
            "id": nid,
            "type": "external" if external else ("io" if (name in pipeline_inputs or name in pipeline_outputs) else "tensor"),
            "subtype": None,
            "label": name,
            "attributes": {},
        }
        nodes.append(node)
        return nid

    # Operators
    for idx, op in enumerate(operators):
        op_type = op.get("type", f"op_{idx}")
        op_id = f"op::{idx}:{op_type}"
        # Extract structured inputs/outputs
        raw_inputs = op.get("inputs", []) or []
        in_list = []
        for item in raw_inputs:
            if isinstance(item, dict):
                # object with {name, tensor}
                tname = item.get("tensor") or item.get("name")
                if tname:
                    in_list.append(tname)
            elif isinstance(item, str):
                in_list.append(item)
        raw_outputs = op.get("outputs", []) or []
        out_list = []
        for item in raw_outputs:
            if isinstance(item, dict):
                tname = item.get("tensor") or item.get("name")
                if tname:
                    out_list.append(tname)
            elif isinstance(item, str):
                out_list.append(item)

        # Collect other attributes (non-IO) for display
        attr = {k: v for k, v in op.items() if k not in ("inputs", "outputs", "type")}

        nodes.append({
            "id": op_id,
            "type": "op",
            "subtype": op_type,
            "label": op_type,
            "inputs": in_list,
            "outputs": out_list,
            "attributes": attr,
        })

        # Links from input tensors to op
        for t in in_list:
            src = ensure_tensor_node(t, external=(t not in tensors))
            links.append({"source": src, "target": op_id, "kind": "tensor"})
        # Links from op to output tensors
        for t in out_list:
            dst = ensure_tensor_node(t, external=(t not in tensors))
            links.append({"source": op_id, "target": dst, "kind": "tensor"})

    # Mark pipeline IO on corresponding nodes
    for n in nodes:
        if n["id"].startswith("tensor::"):
            name = n["label"]
            if name in pipeline_inputs or name in pipeline_outputs:
                if n["type"] == "tensor":
                    n["type"] = "io"

    return {
        "source_path": str(source_path),
        "nodes": nodes,
        "links": links,
    }


def _get_d3_bundle() -> str:
    # Try to load a cached local copy first
    local_paths = [
        Path(__file__).with_name('_d3_v7.min.js'),
    ]
    for p in local_paths:
        if p.exists():
            try:
                return p.read_text(encoding='utf-8')
            except Exception:
                pass

    # Fallback to fetching from CDN
    try:
        import urllib.request
        url = 'https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js'
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read().decode('utf-8')
        # Cache it for next time
        try:
            local_paths[0].write_text(data, encoding='utf-8')
        except Exception:
            pass
        return data
    except Exception:
        # As a last resort, keep placeholder; runtime will show an error box
        return ''


def generate_html(graph_data: dict) -> str:
    data_json = json.dumps(graph_data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", data_json)
    d3_js = _get_d3_bundle()
    html = html.replace("/*__D3_BUNDLE__*/", d3_js)
    # Avoid problematic JS newline literal inside HTML (some viewers corrupt newlines)
    html = html.replace("entries.join('\\\\n')", "entries.join('')")  # backslash-n literal
    html = html.replace("""el.innerHTML = entries.join('\n');""", "el.innerHTML = entries.join('');")  # actual newline
    return html


def main():
    p = argparse.ArgumentParser(description="Visualize pipeline JSON as an interactive DAG and save HTML")
    p.add_argument("input", help="Path(s) to pipeline .json")
    p.add_argument("-o", "--output", default=None, help="file to write HTML outputs")
    args = p.parse_args()
    
    if args.output is None:
        args.output = args.input.replace(".json", "_vis.html")
    Path(args.output).resolve().parent.mkdir(exist_ok=True)
    src = Path(args.input)
    if not src.exists():
        print(f"[WARN] File not found: {src}", file=sys.stderr)
        return 1
    try:
        with open(src, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse {src}: {e}", file=sys.stderr)
        return 1

    g = parse_pipeline(obj, source_path=src)
    html = generate_html(g)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
