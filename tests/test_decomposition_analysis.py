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


def test_paired_analysis_separates_consistency_from_high_reference(tmp_path):
    iteration_dir = tmp_path / 'img' / '100'
    input_dir = tmp_path / 'dataset'
    iteration_dir.mkdir(parents=True)
    input_dir.mkdir()
    # Both reflectances are mutually consistent but jointly much brighter than
    # the matched normal-light reference.
    save_component(iteration_dir / '0_R_low.png', 0.8)
    save_component(iteration_dir / '0_L_low.png', 0.25)
    save_component(iteration_dir / '0_R_high.png', 0.8)
    save_component(iteration_dir / '0_L_high.png', 0.5)
    save_component(input_dir / 'low.png', 0.2)
    save_component(input_dir / 'high.png', 0.4)

    records = [analysis.InputRecord(input_dir / 'low.png', input_dir / 'high.png')]
    rows, warnings = analysis.analyze_iteration(
        iteration_dir, paired=True, input_records=records, mode='paired'
    )

    assert not warnings
    row = rows[0]
    assert row['r_consistency_psnr'] == 100
    assert row['r_low_highref_l1'] > 0.39
    assert row['r_low_highref_mean_ratio'] > 1.9
    assert row['r_low_highref_overbright_010'] == 1
    assert row['self_low_psnr'] > 50
    assert 'r_high_highref_psnr' in row
    assert 'l_high_input_gray_corr' in row


def test_pure_low_single_uses_no_reference_metrics_without_high(tmp_path):
    iteration_dir = tmp_path / 'img' / '100'
    input_dir = tmp_path / 'dataset'
    iteration_dir.mkdir(parents=True)
    input_dir.mkdir()
    save_component(iteration_dir / '0_R_low.png', 0.5)
    save_component(iteration_dir / '0_L_low.png', 0.4)
    save_component(input_dir / 'low.png', 0.2)

    rows, warnings = analysis.analyze_iteration(
        iteration_dir,
        input_records=[analysis.InputRecord(input_dir / 'low.png')],
        mode='pure_low_single',
        anchor_version='v2',
    )

    assert not warnings
    row = rows[0]
    assert row['self_low_psnr'] > 50
    assert row['anchor_abs_error'] > 0.19
    assert 'r_low_tv_to_input' in row
    assert 'l_low_input_gray_corr' in row
    assert 'r_consistency_psnr' not in row
    assert 'r_low_highref_psnr' not in row


def test_anchor_diagnostics_match_point_and_pixel_loss_definitions():
    illumination = np.full((2, 2), 0.4, dtype=np.float32)
    image = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]],
    ], dtype=np.float32)

    point_target, point_v1 = analysis.anchor_diagnostics(
        illumination, image, 'v1', 'point'
    )
    expected_map = image.max(axis=2)
    assert np.isclose(point_target, expected_map.mean())
    assert np.isclose(point_v1, np.abs(0.4 - expected_map).mean())

    point_v2_target, point_v2 = analysis.anchor_diagnostics(
        illumination, image, 'v2', 'point'
    )
    assert np.isclose(point_v2_target, image.max())
    assert np.isclose(point_v2, abs(0.4 - image.max()))

    pixel_target, pixel_v2 = analysis.anchor_diagnostics(
        illumination, image, 'v2', 'pixel'
    )
    assert np.isclose(pixel_target, image.mean())
    assert np.isclose(pixel_v2, abs(0.4 - image.mean()))


def test_validation_record_matching_is_exact_and_mode_appropriate(tmp_path):
    dataset = tmp_path / 'dataset'
    low_dir = dataset / 'test' / 'low'
    high_dir = dataset / 'test' / 'high'
    low_dir.mkdir(parents=True)
    high_dir.mkdir(parents=True)
    save_component(low_dir / 'scene.png', 0.2)
    save_component(high_dir / 'scene.png', 0.5)
    config = {'data': {'path': str(dataset), 'mode': 'pure_low_single'}}

    records = analysis.resolve_validation_records(tmp_path, config)
    assert len(records) == 1
    assert records[0].high == high_dir / 'scene.png'

    (high_dir / 'scene.png').rename(high_dir / 'different.png')
    records = analysis.resolve_validation_records(tmp_path, config)
    assert records[0].high is None

    config['data']['mode'] = 'paired'
    try:
        analysis.resolve_validation_records(tmp_path, config)
    except ValueError as error:
        assert 'filenames do not match' in str(error)
    else:
        raise AssertionError('paired mode accepted mismatched low/high filenames')
