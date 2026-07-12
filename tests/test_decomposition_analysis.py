import importlib.util
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / '_compare' / 'analyze_decomposition.py'
SPEC = importlib.util.spec_from_file_location('analyze_decomposition', SCRIPT_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def save_component(path, value, channels=3):
    shape = (4, 5, channels) if channels == 3 else (4, 5)
    image = np.full(shape, round(value * 255), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_paired_analysis_computes_low_high_consistency(tmp_path):
    iteration_dir = tmp_path / 'img' / '100'
    iteration_dir.mkdir(parents=True)
    save_component(iteration_dir / '0_R_low.png', 0.5)
    save_component(iteration_dir / '0_L_low.png', 0.2)
    save_component(iteration_dir / '0_R_high.png', 0.75)
    save_component(iteration_dir / '0_L_high.png', 0.8)

    rows, warnings = analysis.analyze_iteration(iteration_dir, paired=True)

    assert not warnings
    assert len(rows) == 1
    assert rows[0]['r_low_std'] == 0
    assert rows[0]['r_low_gradient'] == 0
    assert rows[0]['l_low_tv'] == 0
    assert rows[0]['r_consistency_l1'] > 0.24
    assert 'r_high_mean' in rows[0]
    assert 'l_high_mean' in rows[0]

    summaries = analysis.summarize(rows)
    report_path = tmp_path / 'analysis.txt'
    analysis.write_report(report_path, tmp_path, 'paired', summaries, warnings)
    report = report_path.read_text(encoding='utf-8')
    assert 'Rlow_mean' in report
    assert 'R_LH_L1' in report


def test_nonpaired_analysis_ignores_high_files(tmp_path):
    iteration_dir = tmp_path / 'img' / '100'
    iteration_dir.mkdir(parents=True)
    save_component(iteration_dir / '0_R_low.png', 0.5)
    save_component(iteration_dir / '0_L_low.png', 0.2)
    save_component(iteration_dir / '0_R_high.png', 0.75)
    save_component(iteration_dir / '0_L_high.png', 0.8)

    rows, warnings = analysis.analyze_iteration(iteration_dir, paired=False)

    assert not warnings
    assert 'r_high_mean' not in rows[0]
    assert 'l_high_mean' not in rows[0]
    assert 'r_consistency_l1' not in rows[0]


def test_paired_analysis_warns_when_legacy_results_have_no_high(tmp_path):
    iteration_dir = tmp_path / 'img' / '100'
    iteration_dir.mkdir(parents=True)
    save_component(iteration_dir / '0_R_low.png', 0.5)
    save_component(iteration_dir / '0_L_low.png', 0.2)

    rows, warnings = analysis.analyze_iteration(iteration_dir, paired=True)

    assert len(rows) == 1
    assert warnings
    assert 'R_high' in warnings[0]
