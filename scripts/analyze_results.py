#!/usr/bin/env python3
"""Analyze vf-eval results.jsonl and write an HTML report with simple charts.

Usage:
  python scripts/analyze_results.py outputs/evals/<env_model>/<uuid>/results.jsonl

Outputs:
  - analysis.html (interactive, no external deps)
  - summary.json (stats)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def quantile(a: list[float], q: float) -> float:
    if not a:
        return 0.0
    if len(a) == 1:
        return a[0]
    a = sorted(a)
    k = (len(a) - 1) * q
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] * (c - k) + a[c] * (k - f)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_results.py <path/to/results.jsonl>")
        sys.exit(1)
    res_path = Path(sys.argv[1]).resolve()
    if not res_path.exists():
        print(f"File not found: {res_path}")
        sys.exit(1)
    rows = load_rows(res_path)
    rewards = [float(r.get("reward", 0.0)) for r in rows]
    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in rewards) / n if n else 0.0
    std = math.sqrt(var)
    mn, mx = (min(rewards), max(rewards)) if n else (0.0, 0.0)
    q25, q50, q75 = (quantile(rewards, 0.25), quantile(rewards, 0.5), quantile(rewards, 0.75))

    # Worst 5 records
    indexed = [
        {
            "i": i,
            "reward": rewards[i],
            "info": rows[i].get("info", {}) or {},
            "prompt": rows[i].get("prompt", ""),
        }
        for i in range(n)
    ]
    worst = sorted(indexed, key=lambda x: x["reward"])[:5]

    # Tag aggregation
    by_tag: dict[str, list[float]] = defaultdict(list)
    for i in range(n):
        info = rows[i].get("info", {}) or {}
        tags = info.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            by_tag[t].append(rewards[i])
    tag_avgs = sorted(
        ((t, sum(v) / len(v), len(v)) for t, v in by_tag.items()),
        key=lambda x: x[1],
    )

    # Summary JSON
    summary = {
        "path": str(res_path),
        "count": n,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(mn, 3),
        "q25": round(q25, 3),
        "median": round(q50, 3),
        "q75": round(q75, 3),
        "max": round(mx, 3),
        "worst": [
            {
                "index": w["i"],
                "reward": round(w["reward"], 3),
                "tags": w["info"].get("tags", []),
                "question": str(w["prompt"])[:200],
            }
            for w in worst
        ],
    }
    out_dir = res_path.parent
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Build a tiny HTML with inline JS (no external deps)
    # Data payload
    data = {
        "rewards": rewards,
        "tag_avgs": tag_avgs[:10],  # show top 10 worst by avg
        "worst": summary["worst"],
        "mean": summary["mean"],
        "std": summary["std"],
        "min": summary["min"],
        "max": summary["max"],
        "median": summary["median"],
    }
    html = f"""
<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"utf-8\" />
<title>banking-mrm eval report</title>
<style>
 body {{ font: 14px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 16px; }}
 h1 {{ margin: 0 0 8px 0; }}
 .row {{ display: flex; gap: 24px; flex-wrap: wrap; }}
 .card {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; }}
 .chart {{ width: 460px; height: 220px; }}
 .small {{ font-size: 12px; color: #555; }}
 .table {{ border-collapse: collapse; width: 100%; }}
 .table th, .table td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }}
 .bar {{ height: 16px; background: #54A24B; }}
 .bar0 {{ background: #4C78A8; }}
 .bar1 {{ background: #F58518; }}
 .tagrow {{ display:flex; gap:8px; align-items:center; margin:4px 0; }}
 .taglabel {{ width: 180px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
 .tagbar {{ flex:1; height: 12px; background:#54A24B; }}
 .badge {{ display:inline-block; padding:1px 6px; background:#eee; border-radius:10px; margin-left:6px; }}
</style></head>
<body>
<h1>Banking MRM – Judge-only Evaluation</h1>
<div class="small">File: {res_path.name}</div>
<div class="row">
  <div class="card">
    <h3>Summary</h3>
    <div>Count: {n}</div>
    <div>Mean: {summary['mean']}, Std: {summary['std']}</div>
    <div>Median: {summary['median']}, Min/Max: {summary['min']} / {summary['max']}</div>
  </div>
</div>

<div class="row">
  <div class="card">
    <h3>Histogram</h3>
    <div id="hist" class="chart"></div>
  </div>
  <div class="card">
    <h3>Reward by Index</h3>
    <div id="line" class="chart"></div>
  </div>
</div>

<div class="row">
  <div class="card" style="min-width:480px;">
    <h3>Worst 5</h3>
    <table class="table"><thead><tr><th>#</th><th>Reward</th><th>Tags</th><th>Question (truncated)</th></tr></thead>
      <tbody id="worst"></tbody>
    </table>
  </div>
  <div class="card" style="min-width:460px;">
    <h3>Per-tag Avg (top by count)</h3>
    <div id="tags"></div>
  </div>
</div>

<script>
const data = {json.dumps(data)};
function elt(tag, attrs={{}}, text='') {{
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text) e.textContent = text;
  return e;
}}

function drawHistogram(containerId, values) {{
  const bins = 10, w=460, h=220, pad=24;
  const counts = Array(bins).fill(0);
  values.forEach(v => {{ const i = Math.min(bins-1, Math.max(0, Math.floor(v*bins))); counts[i]++; }});
  const maxc = Math.max(...counts,1);
  const c = document.getElementById(containerId);
  c.innerHTML='';
  const svgNS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(svgNS,'svg'); svg.setAttribute('width', w); svg.setAttribute('height', h);
  const barw=(w-2*pad)/bins;
  counts.forEach((cnt,i)=>{{
    const bh=(h-2*pad)*(cnt/maxc);
    const rect=document.createElementNS(svgNS,'rect');
    rect.setAttribute('x', pad+i*barw);
    rect.setAttribute('y', h-pad-bh);
    rect.setAttribute('width', barw-2);
    rect.setAttribute('height', bh);
    rect.setAttribute('fill', '#4C78A8');
    svg.appendChild(rect);
  }});
  c.appendChild(svg);
}}

function drawLine(containerId, values) {{
  const w=460, h=220, pad=24;
  const c=document.getElementById(containerId); c.innerHTML='';
  const svgNS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(svgNS,'svg'); svg.setAttribute('width',w); svg.setAttribute('height',h);
  const n=values.length;
  const pts = values.map((v,i)=>[pad + (i*(w-2*pad)/(Math.max(1,n-1))), h-pad - v*(h-2*pad)]);
  for (let i=1;i<pts.length;i++){{
    const [x1,y1]=pts[i-1], [x2,y2]=pts[i];
    const l=document.createElementNS(svgNS,'line');
    l.setAttribute('x1',x1); l.setAttribute('y1',y1); l.setAttribute('x2',x2); l.setAttribute('y2',y2);
    l.setAttribute('stroke','#F58518'); l.setAttribute('stroke-width','2'); svg.appendChild(l);
  }}
  pts.forEach(([x,y])=>{{
    const dot=document.createElementNS(svgNS,'circle'); dot.setAttribute('cx',x); dot.setAttribute('cy',y); dot.setAttribute('r',2.5); dot.setAttribute('fill','#F58518'); svg.appendChild(dot);
  }});
  c.appendChild(svg);
}}

function fillWorst(tableId, worst) {{
  const tbody=document.getElementById(tableId);
  worst.forEach(w=>{{
    const tr=elt('tr');
    tr.appendChild(elt('td',{{}}, String(w.index)));
    tr.appendChild(elt('td',{{}}, String(w.reward)));
    const tags=Array.isArray(w.tags)? w.tags.join(', '):String(w.tags||'');
    tr.appendChild(elt('td',{{}}, tags));
    tr.appendChild(elt('td',{{}}, w.question));
    tbody.appendChild(tr);
  }});
}}

function drawTags(containerId, tagAvgs) {{
  const root=document.getElementById(containerId);
  if (!tagAvgs || !tagAvgs.length) {{ root.textContent='(no tags)'; return; }}
  tagAvgs.forEach(([label, avg, count])=>{{
    const row=elt('div',{{class:'tagrow'}});
    const l=elt('div',{{class:'taglabel'}}, label);
    const bar=elt('div',{{class:'tagbar'}});
    bar.style.width = (Math.max(0,Math.min(1,avg))*100)+'%';
    const badge=elt('span',{{class:'badge'}}, String(count));
    row.appendChild(l); row.appendChild(bar); row.appendChild(badge);
    root.appendChild(row);
  }});
}}

drawHistogram('hist', data.rewards);
drawLine('line', data.rewards);
fillWorst('worst', data.worst);
drawTags('tags', data.tag_avgs);
</script>
</body></html>
"""
    (out_dir / "analysis.html").write_text(html, encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(out_dir / "analysis.html")}, indent=2))


if __name__ == "__main__":
    main()

