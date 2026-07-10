#!/usr/bin/env python3
"""生成 compare.html — 跨网络 R/L/S 分解对比，表头: 网络 → 训练方式+损失配置 → R/L/S。"""

import json, os, re, yaml
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "html"  # _compare/html/
EXP_DIR = ROOT / "experiments"

IMG_PREFIX = "../../"


# ── helpers ───────────────────────────────────────────────

def extract_loss_config(run_name: str) -> str:
    """从 run 目录名提取损失配置部分。"""
    s = re.sub(r"_\d{8}-\d{6}$", "", run_name)
    m = re.search(r"_(pixel|point)_", s)
    if m:
        return s[m.end():]
    return s


def loss_short(loss: str) -> str:
    """压缩损失配置为紧凑列标题。"""
    s = loss
    if s and s[0].isdigit():
        s = '_' + s
    s = re.sub(r"_(\d+\.?\d*)crh",    r" crossH=\1", s)
    s = re.sub(r"_(\d+\.?\d*)crl",    r" crossL=\1", s)
    s = re.sub(r"_(\d+\.?\d*)rh",     r" reconH=\1", s)
    s = re.sub(r"_(\d+\.?\d*)rl",     r" reconL=\1", s)
    s = re.sub(r"_(\d+\.?\d*)r_",     r" recon=\1_", s)
    s = re.sub(r"_(\d+\.?\d*)anchor", r" anchor=\1", s)
    s = re.sub(r"_(\d+\.?\d*)bdsp",   r" bdsp=\1", s)
    s = re.sub(r"_(\d+\.?\d*)sr",     r" sr=\1", s)
    s = re.sub(r"_(\d+\.?\d*)ref",    r" ref=\1", s)
    s = re.sub(r"_(\d+\.?\d*)er",     r" equalR=\1", s)
    s = re.sub(r"_(\d+\.?\d*)sm",     r" sm=\1", s)
    s = s.replace("_", " ")
    return s.strip()


# ── 收集实验 ──────────────────────────────────────────────

# 结构: [{dataset, model, mode, loss, loss_short, exp, iters, psnr, rel_path, img_indices}, ...]
experiments = []
for model_dir in sorted(EXP_DIR.iterdir()):
    if not model_dir.is_dir():
        continue
    model = model_dir.name
    for mode_dir in sorted(model_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        for run_dir in sorted(mode_dir.iterdir()):
            cfg = run_dir / "config.yaml"
            syn = run_dir / "synthesis"
            if not cfg.exists() or not syn.exists():
                continue
            config = yaml.safe_load(cfg.read_text())
            dataset = config["data"]["path"]
            run_name = run_dir.name
            loss_cfg = extract_loss_config(run_name)

            iters = sorted(
                [d.name for d in syn.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda x: int(x),
            )

            # PSNR
            psnr = "-"
            cmp_txt = run_dir / "synthesis_compare.txt"
            if cmp_txt.exists():
                for line in cmp_txt.read_text().splitlines():
                    if not line.startswith("#") and line.strip() and not line.startswith(" " * 8):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                psnr = f"{float(parts[2]):.1f}"
                            except ValueError:
                                pass

            # img 目录 → iter → max_index
            img_indices = {}
            img_dir = run_dir / "img"
            if img_dir.exists():
                for iter_d in img_dir.iterdir():
                    if iter_d.is_dir() and iter_d.name.isdigit():
                        r_files = list(iter_d.glob("*_R_low.png"))
                        if r_files:
                            img_indices[iter_d.name] = max(
                                int(f.stem.split("_")[0]) for f in r_files
                            )

            experiments.append({
                "dataset":     dataset,
                "model":       model,
                "mode":        mode,
                "loss":        loss_cfg,
                "loss_short":  loss_short(loss_cfg),
                "exp":         run_name,
                "iters":       iters,
                "psnr":        psnr,
                "rel_path":    str(run_dir.relative_to(ROOT)),
                "img_indices": img_indices,
            })

# ── 按 dataset → model → mode → loss 分组 ────────────────

datasets = {}
for e in experiments:
    ds = e["dataset"]
    if ds not in datasets:
        datasets[ds] = {"models": [], "test_files": [], "test_count": 0,
                        "_model_idx": {}, "_mode_idx": {}}

    info = datasets[ds]
    model = e["model"]
    mode = e["mode"]
    loss = e["loss"]

    # 找或建 model 槽位
    if model not in info["_model_idx"]:
        info["_model_idx"][model] = len(info["models"])
        info["models"].append({"model": model, "modes": [], "_mode_idx": {}})
    model_slot = info["models"][info["_model_idx"][model]]

    # 找或建 mode 槽位
    if mode not in model_slot["_mode_idx"]:
        model_slot["_mode_idx"][mode] = len(model_slot["modes"])
        model_slot["modes"].append({"mode": mode, "losses": [], "_loss_idx": {}})
    mode_slot = model_slot["modes"][model_slot["_mode_idx"][mode]]

    # 找或建 loss 槽位
    if loss not in mode_slot["_loss_idx"]:
        mode_slot["_loss_idx"][loss] = len(mode_slot["losses"])
        mode_slot["losses"].append({
            "loss":       loss,
            "loss_short": e["loss_short"],
            "runs":       [],
        })
    loss_slot = mode_slot["losses"][mode_slot["_loss_idx"][loss]]
    loss_slot["runs"].append(e)

# 清理临时索引 + test 文件 + all_iters
for ds_key, info in datasets.items():
    del info["_model_idx"]
    for model_slot in info["models"]:
        del model_slot["_mode_idx"]
        for mode_slot in model_slot["modes"]:
            del mode_slot["_loss_idx"]

    test_low = ROOT / ds_key / "test" / "low"
    if test_low.exists():
        info["test_files"] = sorted([f.name for f in test_low.glob("*.*")])
        info["test_count"] = len(info["test_files"])

    all_iters = set()
    for model_slot in info["models"]:
        for mode_slot in model_slot["modes"]:
            for loss_slot in mode_slot["losses"]:
                for r in loss_slot["runs"]:
                    all_iters.update(r["iters"])
    info["all_iters"] = sorted(all_iters, key=lambda x: int(x))

# ── 构建 JS 数据 ──────────────────────────────────────────

js_data = {}
for ds_key, info in datasets.items():
    js_models = []
    for model_slot in info["models"]:
        js_modes = []
        for mode_slot in model_slot["modes"]:
            js_losses = []
            for loss_slot in mode_slot["losses"]:
                js_runs = []
                for r in loss_slot["runs"]:
                    js_runs.append({
                        "exp":         r["exp"],
                        "psnr":        r["psnr"],
                        "rel_path":    r["rel_path"],
                        "img_indices": r["img_indices"],
                    })
                js_losses.append({
                    "loss":       loss_slot["loss"],
                    "loss_short": loss_slot["loss_short"],
                    "runs":       js_runs,
                })
            js_modes.append({
                "mode":   mode_slot["mode"],
                "losses": js_losses,
            })
        js_models.append({
            "model": model_slot["model"],
            "modes": js_modes,
        })
    js_data[ds_key] = {
        "models":     js_models,
        "test_count": info["test_count"],
        "test_files": info["test_files"],
        "all_iters":  info["all_iters"],
    }

# ── HTML ──────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retinex 分解对比</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font:14px/1.4 system-ui,sans-serif; background:#1a1a2e; color:#ddd; }}
header {{ background:#16213e; padding:12px 20px; position:sticky; top:0; z-index:10; }}
header h1 {{ font-size:18px; color:#e94560; }}
.controls {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:8px; }}
.controls select, .controls input, .controls button {{
    padding:4px 8px; background:#0f3460; color:#fff;
    border:1px solid #e94560; border-radius:4px;
}}
.controls button {{ cursor:pointer; }}
.controls button:hover {{ background:#e94560; }}
.tabs {{ display:flex; gap:8px; margin-bottom:8px; }}
.tab {{ padding:6px 16px; background:#0f3460; border-radius:6px 6px 0 0; cursor:pointer; }}
.tab.active {{ background:#e94560; color:#fff; }}

.table-wrap {{ overflow-x:auto; padding:10px; }}
table {{ border-collapse:collapse; width:max-content; }}
th {{ background:#16213e; padding:6px 4px; font-size:11px; vertical-align:bottom; }}
th.model-hdr {{ color:#e94560; font-size:13px; border-bottom:2px solid #e94560; }}
th.mode-hdr {{ font-size:11px; max-width:160px; overflow:hidden; text-overflow:ellipsis; }}
th .mode-label {{ color:#4ecdc4; font-size:12px; }}
th .loss-label {{ color:#f0c060; font-size:10px; }}
th .psnr {{ color:#888; font-size:10px; }}
th.sep-model {{ border-left:3px solid #e94560; }}
th.sep-mode {{ border-left:2px solid #7a7a5a; }}
td {{ padding:2px; text-align:center; }}
td.sep-model {{ border-left:3px solid #e94560; }}
td.sep-mode {{ border-left:2px solid #7a7a5a; }}
td img {{ display:block; width:180px; height:auto; border-radius:3px; cursor:pointer; transition:transform .15s; }}
td img:hover {{ transform:scale(2.5); z-index:20; position:relative; box-shadow:0 0 20px #000; }}

/* ── Lightbox ── */
.lightbox {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.9); z-index:1000; justify-content:center; align-items:center; cursor:pointer; }}
.lightbox.active {{ display:flex; }}
.lightbox img {{ max-width:95vw; max-height:95vh; object-fit:contain;
    border-radius:4px; box-shadow:0 0 40px rgba(233,69,96,0.5); }}

/* ── 左侧固定列 ── */
th.idx, td.idx {{
    position: sticky; left: 0; z-index: 2;
    width: 28px; min-width: 28px;
}}
th.idx {{ background: #16213e; }}
td.idx {{ background: #1a1a2e; }}
th.original, td.original {{
    position: sticky; left: 28px; z-index: 2;
}}
th.original {{ background: #16213e; }}
td.original {{ background: #1a1a2e; }}

td.original img {{ border:2px solid #e94560; }}
td.original {{ border-right:3px solid #e94560; }}
th.original {{ border-right:3px solid #e94560; }}

.info {{ padding:10px 20px; color:#888; font-size:12px; }}
</style>
</head>
<body>

<header>
  <h1>Retinex 分解对比 — 原始 vs R / L / R×L</h1>
  <div class="tabs" id="tabs"></div>
  <div class="controls">
    <label>Iter:</label>
    <select id="iterSelect"></select>
    <label>Img #:</label>
    <input type="number" id="imgIndex" min="0" value="0" style="width:80px">
    <span id="imgRange" style="color:#888"></span>
    <button onclick="prevImg()">◀</button>
    <button onclick="nextImg()">▶</button>
    <label style="margin-left:12px">跳转:</label>
    <input type="range" id="imgSlider" min="0" value="0" style="width:200px">
  </div>
</header>

<div class="table-wrap" id="content"></div>
<div class="lightbox" id="lightbox"></div>
<div class="info">
  R=反射分量 L=光照分量 R×L=重建结果 | PSNR(dB)仅衡量分解重建低光图与输入的相似度，越高说明重建越接近输入
  | 表头: <span style="color:#e94560">网络</span>
  → <span style="color:#4ecdc4">训练方式</span>
  → <span style="color:#f0c060">损失配置</span>
  → R / L / 重建R×L
</div>

<script>
const IMG_PREFIX = "{IMG_PREFIX}";
const DATA = {json.dumps(js_data, ensure_ascii=False)};

let currentDS = '';
let currentIter = '10000';
let currentIdx = 0;

function init() {{
    const tabs = document.getElementById('tabs');
    const dsOrder = Object.keys(DATA).sort();
    for (const ds of dsOrder) {{
        const d = document.createElement('div');
        d.className = 'tab';
        d.textContent = ds.replace('datasets/','');
        d.onclick = () => selectDS(ds);
        tabs.appendChild(d);
    }}
    selectDS(dsOrder[0]);
}}

function selectDS(ds) {{
    currentDS = ds;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    for (const t of document.querySelectorAll('.tab')) {{
        if (t.textContent === ds.replace('datasets/','')) t.classList.add('active');
    }}
    const info = DATA[ds];

    const iterSel = document.getElementById('iterSelect');
    iterSel.innerHTML = info.all_iters.map(i => `<option>${{i}}</option>`).join('');
    let bestIter = info.all_iters[0] || '10000';
    let maxCount = 0;
    for (const it of info.all_iters) {{
        let cnt = 0;
        for (const model of info.models) {{
            for (const md of model.modes) {{
                for (const ls of md.losses) {{
                    for (const r of ls.runs) {{
                        if (r.img_indices[it] !== undefined) cnt++;
                    }}
                }}
            }}
        }}
        if (cnt > maxCount) {{ maxCount = cnt; bestIter = it; }}
    }}
    iterSel.value = bestIter;
    currentIter = bestIter;
    iterSel.onchange = () => {{ currentIter = iterSel.value; updateSlider(); render(); }};

    updateSlider();
    currentIdx = 0;
    render();
}}

function getEffectiveCount() {{
    const info = DATA[currentDS];
    let minMax = info.test_count - 1;
    for (const model of info.models) {{
        for (const md of model.modes) {{
            for (const ls of md.losses) {{
                for (const r of ls.runs) {{
                    const m = r.img_indices[currentIter];
                    if (m !== undefined && m < minMax) minMax = m;
                }}
            }}
        }}
    }}
    return minMax + 1;
}}

function updateSlider() {{
    const n = getEffectiveCount();
    document.getElementById('imgSlider').max = n - 1;
    document.getElementById('imgIndex').max = n - 1;
    document.getElementById('imgRange').textContent = `/ ${{n}} (共${{DATA[currentDS].test_count}}张)`;
    if (currentIdx >= n) currentIdx = n - 1;
}}

const PAGE = 5;
function prevImg() {{ currentIdx = Math.max(0, currentIdx - PAGE); render(); }}
function nextImg() {{
    const max = getEffectiveCount() - 1;
    currentIdx = Math.min(max, currentIdx + PAGE); render();
}}

document.getElementById('imgIndex').onchange = function() {{
    currentIdx = Math.min(parseInt(this.value) || 0, getEffectiveCount() - 1);
    render();
}};
document.getElementById('imgSlider').oninput = function() {{
    currentIdx = Math.min(parseInt(this.value), getEffectiveCount() - 1);
    render();
}};

// 展平当前 iter 下有图的 (model, mode, loss, run) 四元组
function collectColumns(info) {{
    const cols = [];
    for (let mi = 0; mi < info.models.length; mi++) {{
        const model = info.models[mi];
        let modelFirst = true;
        for (const md of model.modes) {{
            let modeFirst = true;
            for (const ls of md.losses) {{
                for (const r of ls.runs) {{
                    if (r.img_indices[currentIter] !== undefined) {{
                        cols.push({{
                            model: model,
                            mode: md,
                            loss: ls,
                            run: r,
                            isFirstInModel: modelFirst,
                            isFirstInMode: modeFirst,
                        }});
                        modelFirst = false;
                        modeFirst = false;
                    }}
                }}
            }}
        }}
    }}
    return cols;
}}

function render() {{
    const info = DATA[currentDS];
    const idx = currentIdx;
    document.getElementById('imgIndex').value = idx;
    document.getElementById('imgSlider').value = idx;

    const cols = collectColumns(info);
    if (cols.length === 0) {{
        document.getElementById('content').innerHTML = '<p style="padding:20px">当前 iter 下无可用图片数据。</p>';
        return;
    }}

    // ── 表头行 1: 网络 超表头 ──
    let hdr1 = '<tr><th rowspan="3" class="idx">#</th><th rowspan="3" class="original">原始 Input</th>';
    for (let ci = 0; ci < cols.length; ci++) {{
        const col = cols[ci];
        if (col.isFirstInModel) {{
            let span = 1;
            for (let cj = ci + 1; cj < cols.length && cols[cj].model === col.model; cj++) span++;
            const sep = ci > 0 ? ' sep-model' : '';
            hdr1 += `<th class="model-hdr${{sep}}" colspan="${{span * 3}}">${{col.model.model}}</th>`;
        }}
    }}
    hdr1 += '</tr>';

    // ── 表头行 2: 训练方式 + 损失配置 ──
    let hdr2 = '<tr>';
    for (let ci = 0; ci < cols.length; ci++) {{
        const col = cols[ci];
        let sepCls = '';
        if (col.isFirstInModel && ci > 0) sepCls = ' sep-model';
        else if (col.isFirstInMode && ci > 0 && !col.isFirstInModel) sepCls = ' sep-mode';
        const label = col.mode.mode + ' · ' + col.loss.loss_short;
        hdr2 += `<th class="mode-hdr${{sepCls}}" colspan="3" title="${{label}}"><span class="mode-label">${{col.mode.mode}}</span><br><span class="loss-label">${{col.loss.loss_short}}</span><br><span class="psnr">${{col.run.psnr}}dB</span></th>`;
    }}
    hdr2 += '</tr>';

    // ── 表头行 3: R / L / S ──
    let hdr3 = '<tr>';
    for (let ci = 0; ci < cols.length; ci++) {{
        const col = cols[ci];
        let sepCls = '';
        if (col.isFirstInModel && ci > 0) sepCls = ' sep-model';
        else if (col.isFirstInMode && ci > 0 && !col.isFirstInModel) sepCls = ' sep-mode';
        hdr3 += `<th${{sepCls}} style="font-size:10px;color:#aaa">R</th><th style="font-size:10px;color:#aaa">L</th><th style="font-size:10px;color:#aaa">重建R×L</th>`;
    }}
    hdr3 += '</tr>';

    let html = '<table><thead>' + hdr1 + hdr2 + hdr3 + '</thead><tbody>';

    const ROWS = 5;
    const maxDisplay = getEffectiveCount() - 1;
    // 按页对齐：每页 ROWS 行，无重叠
    let start = Math.floor(idx / ROWS) * ROWS;
    let end = Math.min(maxDisplay, start + ROWS - 1);
    // 处理最后一页不足 ROWS 行的情况
    if (end - start + 1 < ROWS) {{ start = Math.max(0, end - ROWS + 1); }}
    for (let i = start; i <= end; i++) {{
        html += `<tr><td class="idx" style="font-size:11px;color:#888">${{i}}</td>`;
        const origFile = info.test_files[i] || '';
        html += `<td class="original"><img src="${{IMG_PREFIX}}datasets/${{currentDS.replace('datasets/','')}}/test/low/${{encodeURI(origFile)}}" onerror="this.style.display='none'"></td>`;
        for (let ci = 0; ci < cols.length; ci++) {{
            const col = cols[ci];
            const r = col.run;
            const base = r.rel_path;
            let sepCls = '';
            if (col.isFirstInModel && ci > 0) sepCls = ' sep-model';
            else if (col.isFirstInMode && ci > 0 && !col.isFirstInModel) sepCls = ' sep-mode';
            html += `<td class="${{sepCls}}"><img src="${{IMG_PREFIX}}${{base}}/img/${{currentIter}}/${{i}}_R_low.png" onerror="this.style.display='none'"></td>`;
            html += `<td><img src="${{IMG_PREFIX}}${{base}}/img/${{currentIter}}/${{i}}_L_low.png" onerror="this.style.display='none'"></td>`;
            html += `<td><img src="${{IMG_PREFIX}}${{base}}/synthesis/${{currentIter}}/${{i}}_S_low.png" onerror="this.style.display='none'"></td>`;
        }}
        html += '</tr>';
    }}
    html += '</tbody></table>';
    document.getElementById('content').innerHTML = html;
    document.querySelectorAll('#content img').forEach(img => {{
        img.addEventListener('click', e => {{ e.stopPropagation(); openLightbox(img.src); }});
    }});
}}

function openLightbox(src) {{
    const lb = document.getElementById('lightbox');
    lb.innerHTML = `<img src="${{src}}">`;
    lb.classList.add('active');
}}
document.getElementById('lightbox').addEventListener('click', function() {{
    this.classList.remove('active');
}});

init();
</script>
</body>
</html>"""

out = OUT_DIR / "compare.html"
out.write_text(html)
print(f"生成: {out}")
print(f"数据集: {list(datasets.keys())}")
for ds, info in datasets.items():
    n_runs = sum(1 for m in info["models"] for md in m["modes"] for ls in md["losses"] for _ in ls["runs"])
    print(f"  {ds}: {len(info['models'])} models, {n_runs} runs, {info['test_count']} test images")
