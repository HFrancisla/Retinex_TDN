#!/usr/bin/env python3
"""
生成 {model}_compare.html — 对单个网络的多个训练 run 做 R/L/S 分解对比。

表头层级: 训练方式(mode) → 损失配置(loss config) → R/L/S
用法:
    python _compare/generate_model_compare.py                    # 为 experiments/ 下每个网络都生成
    python _compare/generate_model_compare.py --model RetinexPixelClassic  # 只生成指定网络
    python _compare/generate_model_compare.py -m RetinexPixelTrans -m RetinexPointRaw  # 多个
"""

import argparse, json, os, re, sys, yaml
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "html"  # _compare/html/
EXP_DIR = ROOT / "experiments"

IMG_PREFIX = "../../"

# ── helpers ───────────────────────────────────────────────

# 数据集名称标准化：将 data.path 映射为规范的短名称
DATASET_ALIASES = {
    "BDD_below40": "BDDnight",  # BDDnight 的亮度过滤子集
}

# 数据集在 HTML 中的显示顺序（未列出的数据集按名称排序追加到末尾）
DATASET_ORDER = ["LOLv2", "BDDnight"]


def _dataset_sort_key(ds_name: str):
    """按 DATASET_ORDER 排序，未列出的排到末尾后按名称排序。"""
    try:
        return (0, DATASET_ORDER.index(ds_name), "")
    except ValueError:
        return (1, 0, ds_name)


def normalize_dataset_name(raw_path: str) -> str:
    """将配置中的 data.path 标准化为数据集短名称。"""
    name = os.path.basename(raw_path.rstrip("/\\"))
    return DATASET_ALIASES.get(name, name)


def extract_loss_config(run_name: str) -> str:
    """从 run 目录名提取损失配置部分。

    "BDD_below40_point_1r_0.05anchor_0.05bdsp_0.05rlc_0.05ref_20260625-013114"
    → "1r_0.05anchor_0.05bdsp_0.05rlc_0.05ref"

    兼容两种格式：
    - 旧格式: {dataset}_{pixel|point}_{loss_config}_{timestamp}
    - 新格式: {dataset}_{loss_config}_{timestamp}
    """
    # 去掉末尾时间戳 _YYYYMMDD-HHMMSS
    s = re.sub(r"_\d{8}-\d{6}$", "", run_name)
    # 旧格式兼容: _pixel_ 或 _point_ 分隔
    m = re.search(r"_(pixel|point)_", s)
    if m:
        return s[m.end():]
    # 新格式: 损失配置以数字权重开头 (如 1.0r, 0.3r)
    m = re.search(r"_\d", s)
    if m:
        return s[m.start() + 1:]
    # fallback: 无法识别时返回除去 dataset 前缀的部分
    return s


def loss_short(loss: str) -> str:
    """压缩损失配置为紧凑列标题。

    "1r_0.05anchor_0.05bdsp" → "recon=1 anchor=0.05 bdsp=0.05"
    "1rh_0.3rl_0.001crh_0.001crl_0.1er_0.1sm" → "recon_h=1 recon_l=0.3 cross_h=0.001 cross_l=0.001 equal_R=0.1 sm=0.1"
    """
    s = loss
    if s and s[0].isdigit():
        s = '_' + s
    s = re.sub(r"_(\d+\.?\d*)crh(v\d+)?",    r" crossH\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)crl(v\d+)?",    r" crossL\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)rh(v\d+)?",     r" reconH\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)rl(v\d+)?",     r" reconL\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)r(v\d+)?_",     r" recon\2=\1_", s)
    s = re.sub(r"_(\d+\.?\d*)anchor(v\d+)?", r" anchor\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)bdsp(v\d+)?",   r" bdsp\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)rlc(v\d+)?",    r" redecompL\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)ref(v\d+)?",    r" ref\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)er(v\d+)?",     r" equalR\2=\1", s)
    s = re.sub(r"_(\d+\.?\d*)sm(v\d+)?",     r" sm\2=\1", s)
    s = s.replace("_", " ")
    return s.strip()


def mode_label(mode: str) -> str:
    """训练方式标签，直接使用原始英文名。"""
    return mode


# ── 收集实验 ──────────────────────────────────────────────

def collect_experiments():
    """返回 {model_name: [run_info, ...]}。"""
    models = {}
    for model_dir in sorted(EXP_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        runs = []
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
                dataset_path = config["data"]["path"]
                dataset_id = normalize_dataset_name(dataset_path)
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

                runs.append({
                    "dataset_id":   dataset_id,
                    "dataset_path": dataset_path,
                    "mode":         mode,
                    "loss":         loss_cfg,
                    "loss_short":   loss_short(loss_cfg),
                    "run_name":     run_name,
                    "iters":        iters,
                    "psnr":         psnr,
                    "rel_path":     str(run_dir.relative_to(ROOT)),
                    "img_indices":  img_indices,
                })

        if runs:
            models[model] = runs

    return models


# ── 构建 HTML ─────────────────────────────────────────────

def build_html(model_name: str, runs: list) -> str:
    """为单个网络生成对比页。"""

    # ---- 按 dataset → mode → loss 分组 ----
    # datasets[ds] = {
    #     "modes": [ { "mode": "pure_low_single",
    #                  "losses": [ { "loss": "1r_...", "loss_short": "1r a=...",
    #                                "runs": [run, ...] }, ... ] }, ... ],
    #     "test_files": [...], "test_count": N, "all_iters": [...]
    # }
    datasets = {}
    for r in runs:
        ds_id = r["dataset_id"]
        if ds_id not in datasets:
            datasets[ds_id] = {"modes": [], "test_files": [], "test_count": 0,
                               "dataset_path": r["dataset_path"],  # 用于文件查找
                               "_mode_idx": {}, "_loss_idx": {}}  # 临时索引
        info = datasets[ds_id]
        mode = r["mode"]
        loss = r["loss"]
        # 找或建 mode 槽位
        if mode not in info["_mode_idx"]:
            info["_mode_idx"][mode] = len(info["modes"])
            info["modes"].append({"mode": mode, "losses": [], "_loss_idx": {}})
        mode_slot = info["modes"][info["_mode_idx"][mode]]
        # 找或建 loss 槽位
        if loss not in mode_slot["_loss_idx"]:
            mode_slot["_loss_idx"][loss] = len(mode_slot["losses"])
            mode_slot["losses"].append({
                "loss":       loss,
                "loss_short": r["loss_short"],
                "runs":       [],
            })
        loss_slot = mode_slot["losses"][mode_slot["_loss_idx"][loss]]
        loss_slot["runs"].append(r)

    # 清理临时索引 + 补充 test 文件和 all_iters
    for ds_key, info in sorted(datasets.items(), key=lambda x: _dataset_sort_key(x[0])):
        del info["_mode_idx"]
        for m in info["modes"]:
            del m["_loss_idx"]
        # test 文件
        test_low = ROOT / info["dataset_path"] / "test" / "low"
        if test_low.exists():
            info["test_files"] = sorted([f.name for f in test_low.glob("*.*")])
            info["test_count"] = len(info["test_files"])
        # 所有 iter 的并集 + 最大图片索引（用于截断 test_files）
        all_iters = set()
        max_img_idx = -1
        for mode_slot in info["modes"]:
            for loss_slot in mode_slot["losses"]:
                for r in loss_slot["runs"]:
                    all_iters.update(r["iters"])
                    for v in r["img_indices"].values():
                        if v > max_img_idx:
                            max_img_idx = v
        info["all_iters"] = sorted(all_iters, key=lambda x: int(x))
        info["max_img_idx"] = max_img_idx

    # ---- 构建 JS 数据 ----
    _ordered_ds_keys = [k for k, _ in sorted(datasets.items(), key=lambda x: _dataset_sort_key(x[0]))]
    js_data = {}
    js_data["_dataset_order"] = _ordered_ds_keys
    for ds_key in _ordered_ds_keys:
        info = datasets[ds_key]
        js_modes = []
        for mode_slot in info["modes"]:
            js_losses = []
            for loss_slot in mode_slot["losses"]:
                js_runs = []
                for r in loss_slot["runs"]:
                    js_runs.append({
                        "run_name":    r["run_name"],
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
                "label":  mode_label(mode_slot["mode"]),
                "losses": js_losses,
            })
        # 数据集图片路径前缀（HTML 相对路径 — 相对于项目根目录）
        _raw_path = info.get("dataset_path", ds_key)
        if os.path.isabs(_raw_path):
            try:
                _img_prefix = os.path.relpath(_raw_path, ROOT)
            except ValueError:
                _img_prefix = f"datasets/{os.path.basename(_raw_path)}"
        else:
            _img_prefix = _raw_path
        # 截断 test_files：只保留 run 图片实际能覆盖的范围，避免嵌入数千无用文件名
        _max_need = info.get("max_img_idx", -1) + 1
        _tfiles = info["test_files"][:_max_need] if _max_need > 0 and info["test_files"] else info["test_files"]

        js_data[ds_key] = {
            "modes":         js_modes,
            "test_count":    info["test_count"],
            "test_files":    _tfiles,
            "all_iters":     info["all_iters"],
            "dataset_path":  _raw_path,
            "img_prefix":    _img_prefix,
        }

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{model_name} — 训练 Run 对比</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font:14px/1.4 system-ui,sans-serif; background:#1a1a2e; color:#ddd; }}
header {{ background:#16213e; padding:12px 20px; position:sticky; top:0; z-index:10; }}
header h1 {{ font-size:18px; color:#e94560; }}
header h1 small {{ font-size:14px; color:#888; font-weight:normal; }}
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
th.mode-hdr {{ color:#e94560; font-size:13px; border-bottom:2px solid #e94560; }}
th.loss-hdr {{ font-size:11px; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
th.loss-hdr .loss-label {{ color:#f0c060; }}
th.loss-hdr .psnr {{ color:#888; font-size:10px; }}
th.sep-mode {{ border-left:3px solid #e94560; }}
th.sep-loss {{ border-left:2px solid #7a7a5a; }}
td {{ padding:2px; text-align:center; }}
td.sep-mode {{ border-left:3px solid #e94560; }}
td.sep-loss {{ border-left:2px solid #7a7a5a; }}
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
  <h1>{model_name} <small>— 训练 Run 分解对比</small></h1>
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
  | 表头: <span style="color:#e94560">训练方式</span>
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
    const dsOrder = DATA._dataset_order || Object.keys(DATA).filter(k => k !== '_dataset_order').sort();
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
    // 默认选择覆盖 run 最多的 iter
    let bestIter = info.all_iters[0] || '10000';
    let maxCount = 0;
    for (const it of info.all_iters) {{
        let cnt = 0;
        for (const mode of info.modes) {{
            for (const loss of mode.losses) {{
                for (const r of loss.runs) {{
                    if (r.img_indices[it] !== undefined) cnt++;
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
    // 从 test_count-1 或 run 的 img_indices 中取最小值（交集），
    // 但当 test_count=0 时（无原始图片可用），回退到 run 图片的最大索引
    let minMax = info.test_count > 0 ? info.test_count - 1 : -1;
    for (const mode of info.modes) {{
        for (const loss of mode.losses) {{
            for (const r of loss.runs) {{
                const m = r.img_indices[currentIter];
                if (m !== undefined) {{
                    if (minMax < 0 || m < minMax) minMax = m;
                }}
            }}
        }}
    }}
    return minMax < 0 ? 0 : minMax + 1;
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

// 展平当前 iter 下有图的 (mode, loss, run) 三元组，同时记录分隔类型
function collectColumns(info) {{
    const cols = [];  // {{ mode, loss, run, isFirstInMode, isFirstInLoss }}
    for (let mi = 0; mi < info.modes.length; mi++) {{
        const mode = info.modes[mi];
        let modeHasContent = false;
        for (const loss of mode.losses) {{
            let lossFirst = true;
            for (const r of loss.runs) {{
                if (r.img_indices[currentIter] !== undefined) {{
                    cols.push({{
                        mode: mode,
                        loss: loss,
                        run: r,
                        isFirstInMode: !modeHasContent,
                        isFirstInLoss: lossFirst,
                    }});
                    modeHasContent = true;
                    lossFirst = false;
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

    // ── 表头行 1: Mode 超表头 ──
    let hdr1 = '<tr><th rowspan="3" class="idx">#</th><th rowspan="3" class="original">原始 Input</th>';
    for (let ci = 0; ci < cols.length; ci++) {{
        const col = cols[ci];
        if (col.isFirstInMode) {{
            // 统计该 mode 占多少列
            let span = 1;
            for (let cj = ci + 1; cj < cols.length && cols[cj].mode === col.mode; cj++) span++;
            const sep = ci > 0 ? ' sep-mode' : '';
            hdr1 += `<th class="mode-hdr${{sep}}" colspan="${{span * 3}}">${{col.mode.label}}</th>`;
        }}
    }}
    hdr1 += '</tr>';

    // ── 表头行 2: Loss 配置 ──
    let hdr2 = '<tr>';
    for (let ci = 0; ci < cols.length; ci++) {{
        const col = cols[ci];
        let sepCls = '';
        if (col.isFirstInMode && ci > 0) sepCls = ' sep-mode';
        else if (col.isFirstInLoss && ci > 0 && !col.isFirstInMode) sepCls = ' sep-loss';
        hdr2 += `<th class="loss-hdr${{sepCls}}" colspan="3" title="${{col.loss.loss}}"><span class="loss-label">${{col.loss.loss_short}}</span><br><span class="psnr">${{col.run.psnr}}dB</span></th>`;
    }}
    hdr2 += '</tr>';

    // ── 表头行 3: R / L / S ──
    let hdr3 = '<tr>';
    for (let ci = 0; ci < cols.length; ci++) {{
        const col = cols[ci];
        let sepCls = '';
        if (col.isFirstInMode && ci > 0) sepCls = ' sep-mode';
        else if (col.isFirstInLoss && ci > 0 && !col.isFirstInMode) sepCls = ' sep-loss';
        hdr3 += `<th${{sepCls}} style="font-size:10px;color:#aaa">R</th><th style="font-size:10px;color:#aaa">L</th><th style="font-size:10px;color:#aaa">重建R×L</th>`;
    }}
    hdr3 += '</tr>';

    let html = '<table><thead>' + hdr1 + hdr2 + hdr3 + '</thead><tbody>';

    // ── 数据行 ──
    const ROWS = 5;
    const maxDisplay = getEffectiveCount() - 1;
    // 按页对齐：每页 ROWS 行，无重叠
    let start = Math.floor(idx / ROWS) * ROWS;
    let end = Math.min(maxDisplay, start + ROWS - 1);
    // 处理最后一页不足 ROWS 行的情况
    if (end - start + 1 < ROWS) {{ start = Math.max(0, end - ROWS + 1); }}
    for (let i = start; i <= end; i++) {{
        html += `<tr><td class="idx" style="font-size:11px;color:#888">${{i}}</td>`;
        // Original
        const origFile = info.test_files[i] || '';
        html += `<td class="original"><img src="${{IMG_PREFIX}}${{info.img_prefix}}/test/low/${{encodeURI(origFile)}}" onerror="this.style.display='none'"></td>`;
        // 各列
        for (let ci = 0; ci < cols.length; ci++) {{
            const col = cols[ci];
            const r = col.run;
            const base = r.rel_path;
            let sepCls = '';
            if (col.isFirstInMode && ci > 0) sepCls = ' sep-mode';
            else if (col.isFirstInLoss && ci > 0 && !col.isFirstInMode) sepCls = ' sep-loss';
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


# ── main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="按网络模型分组生成实验对比 HTML")
    parser.add_argument("-m", "--model", action="append", dest="models",
                        help="指定网络名称（可多次使用）；不指定则为所有网络生成")
    args = parser.parse_args()

    all_models = collect_experiments()
    if not all_models:
        print("未找到任何实验数据。", file=sys.stderr)
        sys.exit(1)

    target_models = args.models if args.models else sorted(all_models.keys())
    for m in target_models:
        if m not in all_models:
            print(f"警告: 网络 '{m}' 在 experiments/ 中不存在，跳过。", file=sys.stderr)

    generated = []
    for model_name in target_models:
        if model_name not in all_models:
            continue
        runs = all_models[model_name]
        html = build_html(model_name, runs)
        out_path = OUT_DIR / f"compare_{model_name}.html"
        out_path.write_text(html)
        generated.append(out_path)
        # 统计：按 dataset → mode → loss 层级打印
        ds_map = {}
        for r in runs:
            ds_map.setdefault(r["dataset_id"], {}).setdefault(r["mode"], set()).add(r["loss"])
        print(f"生成: {out_path}")
        for ds, modes in ds_map.items():
            print(f"  {ds}:")
            for mode, losses in modes.items():
                print(f"    {mode}: {len(losses)} 种损失配置")

    if not generated:
        print("没有生成任何文件。", file=sys.stderr)
        sys.exit(1)

    print(f"\n共生成 {len(generated)} 个 HTML 文件。")


if __name__ == "__main__":
    main()
