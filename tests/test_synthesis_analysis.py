import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


COMPARE_DIR = Path(__file__).parents[1] / '_compare'


def load_script(name):
    path = COMPARE_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


synthesis = load_script('synthesize_retinex')
psnr_synthesis = load_script('psnr_synthesis')


def save_image(path, value, grayscale=False):
    shape = (4, 5) if grayscale else (4, 5, 3)
    image = np.full(shape, round(value * 255), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_synthesis_refreshes_when_component_is_newer(tmp_path):
    r_path = tmp_path / '0_R_low.png'
    l_path = tmp_path / '0_L_low.png'
    out_path = tmp_path / 'synthesis' / '0_S_low.png'
    save_image(r_path, 0.5)
    save_image(l_path, 0.2, grayscale=True)

    assert synthesis.synthesize_and_save(r_path, l_path, out_path)
    first = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    assert np.isclose(first.mean() / 255, 0.1, atol=0.01)
    assert not synthesis.synthesize_and_save(r_path, l_path, out_path)

    save_image(r_path, 0.8)
    newer = out_path.stat().st_mtime + 10
    os.utime(r_path, (newer, newer))
    assert synthesis.synthesize_and_save(r_path, l_path, out_path)
    second = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    assert np.isclose(second.mean() / 255, 0.16, atol=0.01)


def test_process_iteration_removes_orphan_synthesis(tmp_path):
    image_dir = tmp_path / 'img' / '100'
    synthesis_dir = tmp_path / 'synthesis' / '100'
    save_image(image_dir / '0_R_low.png', 0.5)
    save_image(image_dir / '0_L_low.png', 0.2, grayscale=True)
    save_image(synthesis_dir / '7_S_low.png', 0.9)

    stats = synthesis.process_iteration(image_dir, synthesis_dir)

    assert stats['low'] == 1
    assert (synthesis_dir / '0_S_low.png').is_file()
    assert not (synthesis_dir / '7_S_low.png').exists()


def test_psnr_report_supports_pure_low_images_layout_and_rejects_gaps(tmp_path):
    run_dir = tmp_path / 'model' / 'pure_low_single' / 'run'
    dataset = tmp_path / 'dataset'
    input_path = dataset / 'test' / 'images' / 'scene.png'
    image_dir = run_dir / 'img' / '100'
    synthesis_dir = run_dir / 'synthesis' / '100'
    save_image(input_path, 0.2)
    save_image(image_dir / '0_R_low.png', 0.5)
    save_image(image_dir / '0_L_low.png', 0.4, grayscale=True)
    save_image(synthesis_dir / '0_S_low.png', 0.2)
    config = {
        'data': {'path': str(dataset), 'mode': 'pure_low_single'},
        'loss': {'mode': 'pure_low_single_pixel'},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')

    psnr_synthesis.process_run(run_dir)
    report = (run_dir / 'synthesis_compare.txt').read_text(encoding='utf-8')
    assert 'Reconstruction integrity only' in report
    assert '100.00' in report

    (synthesis_dir / '0_S_low.png').unlink()
    try:
        psnr_synthesis.process_run(run_dir)
    except ValueError as error:
        assert 'incomplete/stale' in str(error)
    else:
        raise AssertionError('incomplete synthesis was accepted')
