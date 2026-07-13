import math
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from data import PureLowDataSet, transforms as T
from loss import PairedLoss, PureLowDoubleLoss, PureLowSingleLoss, UnpairedLoss
from loss.bsdp import BDSP_Face
from loss.decomposition_loss import (
    _pixel_anchor_loss,
    _point_anchor_loss,
    _retinex_smooth,
)
from models import RetinexPixelClassic, RetinexPointRaw
from train import generate_experiment_name, validate_pipeline_config
from utils import (
    _build_loss_function, calculate_psnr, create_lr_scheduler, evaluate, train_step,
)


def test_psnr_is_averaged_per_image():
    pred = torch.tensor([[[[0.0]]], [[[0.9]]]])
    target = torch.ones_like(pred)
    assert calculate_psnr(pred, target) == pytest.approx(10.0, abs=1e-5)


def test_point_anchor_supports_v1_and_v2_versions():
    image = torch.tensor(
        [[[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.9]]]], dtype=torch.float32
    )
    point_l = torch.full((1, 1, 1, 1), 0.5)
    assert _point_anchor_loss(point_l, image, 'v1').item() == pytest.approx(0.2)
    assert _point_anchor_loss(point_l, image, 'v2').item() == pytest.approx(0.4)


def test_pixel_anchor_supports_v1_and_v2_versions():
    image = torch.tensor(
        [[[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.9]]]], dtype=torch.float32
    )
    pixel_l = torch.full((1, 1, 1, 2), 0.4)
    assert _pixel_anchor_loss(pixel_l, image, 'v1').item() == pytest.approx(0.3)
    assert _pixel_anchor_loss(pixel_l, image, 'v2').item() == pytest.approx(0.0)


def test_anchor_rejects_unknown_version():
    image = torch.rand(1, 3, 2, 2)
    with pytest.raises(ValueError, match='anchor_version'):
        _point_anchor_loss(torch.rand(1, 1, 1, 1), image, 'unknown')


@pytest.mark.parametrize(
    ('mode', 'version'),
    [
        ('pure_low_single_point', 'v1'),
        ('pure_low_single_point', 'v2'),
        ('pure_low_single_pixel', 'v1'),
        ('pure_low_single_pixel', 'v2'),
    ],
)
def test_anchor_version_is_switchable_from_loss_config(mode, version):
    config = {
        'mode': mode,
        'recon_weight': 1.0,
        'anchor_weight': 0.05,
        'anchor_version': version,
        'bdsp_weight': 0.05,
    }
    if mode.endswith('_pixel'):
        config['smooth_weight'] = 0.1
        config['smooth_version'] = 'v1'
    loss = _build_loss_function(config)
    assert loss.anchor_version == version


@pytest.mark.parametrize(
    'mode',
    [
        'unpaired_point', 'unpaired_pixel',
        'pure_low_double_point', 'pure_low_double_pixel',
        'pure_low_single_point', 'pure_low_single_pixel',
    ],
)
def test_anchor_version_must_be_explicitly_declared(mode):
    valid_fields = {
        'recon_weight': 1.0,
        'anchor_weight': 0.05,
        'bdsp_weight': 0.05,
        'smooth_weight': 0.1,
        'smooth_version': 'v1',
        'self_recon_weight': 0.05,
        'reflect_weight': 0.1,
    }
    from utils import _VALID_LOSS_FIELDS

    config = {'mode': mode}
    config.update({key: valid_fields[key] for key in _VALID_LOSS_FIELDS[mode]
                   if key != 'anchor_version'})
    with pytest.raises(ValueError, match='anchor_version'):
        _build_loss_function(config)


def test_paired_loss_does_not_accept_anchor_version():
    config = {
        'mode': 'paired_point',
        'recon_weight_high': 1.0,
        'recon_weight_low': 0.3,
        'cross_recon_weight_low': 0.001,
        'cross_recon_weight_high': 0.001,
        'equal_r_weight': 0.1,
        'anchor_version': 'v2',
    }
    with pytest.raises(ValueError, match='不支持'):
        _build_loss_function(config)


@pytest.mark.parametrize(
    'mode',
    ['paired_pixel', 'unpaired_pixel', 'pure_low_double_pixel', 'pure_low_single_pixel'],
)
def test_smooth_version_must_be_explicitly_declared_for_pixel_modes(mode):
    from utils import _VALID_LOSS_FIELDS

    values = {
        'recon_weight_high': 1.0,
        'recon_weight_low': 0.3,
        'cross_recon_weight_low': 0.001,
        'cross_recon_weight_high': 0.001,
        'equal_r_weight': 0.1,
        'recon_weight': 1.0,
        'anchor_weight': 0.05,
        'anchor_version': 'v2',
        'bdsp_weight': 0.05,
        'smooth_weight': 0.1,
        'self_recon_weight': 0.05,
        'reflect_weight': 0.1,
    }
    config = {'mode': mode}
    config.update({key: values[key] for key in _VALID_LOSS_FIELDS[mode]
                   if key != 'smooth_version'})
    with pytest.raises(ValueError, match='smooth_version'):
        _build_loss_function(config)


@pytest.mark.parametrize(
    'mode',
    ['paired_point', 'unpaired_point', 'pure_low_double_point', 'pure_low_single_point'],
)
def test_point_modes_reject_smooth_version(mode):
    from utils import _VALID_LOSS_FIELDS

    values = {
        'recon_weight_high': 1.0,
        'recon_weight_low': 0.3,
        'cross_recon_weight_low': 0.001,
        'cross_recon_weight_high': 0.001,
        'equal_r_weight': 0.1,
        'recon_weight': 1.0,
        'anchor_weight': 0.05,
        'anchor_version': 'v2',
        'bdsp_weight': 0.05,
        'self_recon_weight': 0.05,
        'reflect_weight': 0.1,
    }
    config = {'mode': mode}
    config.update({key: values[key] for key in _VALID_LOSS_FIELDS[mode]})
    config['smooth_version'] = 'v1'
    with pytest.raises(ValueError, match='不支持'):
        _build_loss_function(config)


def test_bdsp_is_independent_of_other_batch_samples():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 12, 10)
    together = BDSP_Face(images)
    assert torch.allclose(together[0], BDSP_Face(images[:1])[0], atol=1e-6)
    assert torch.allclose(together[1], BDSP_Face(images[1:])[0], atol=1e-6)


@pytest.mark.parametrize('version', ['v2', 'v3'])
def test_detached_smooth_versions_do_not_backpropagate_into_reflectance(version):
    illumination = torch.rand(1, 1, 8, 8, requires_grad=True)
    reflectance = torch.rand(1, 3, 8, 8, requires_grad=True)
    _retinex_smooth(illumination, reflectance, version).backward()
    assert illumination.grad is not None
    assert reflectance.grad is None


def test_raw_smooth_version_backpropagates_into_reflectance():
    illumination = torch.rand(1, 1, 8, 8, requires_grad=True)
    reflectance = torch.rand(1, 3, 8, 8, requires_grad=True)
    _retinex_smooth(illumination, reflectance, 'v1').backward()
    assert illumination.grad is not None
    assert reflectance.grad is not None
    assert torch.isfinite(reflectance.grad).all()


def test_smooth_versions_have_expected_constant_boundary_behavior():
    illumination = torch.full((1, 1, 8, 8), 0.5)
    reflectance = torch.full((1, 3, 8, 8), 0.5)
    assert _retinex_smooth(illumination, reflectance, 'v1').item() > 0
    assert _retinex_smooth(illumination, reflectance, 'v2').item() == pytest.approx(0.0)
    assert _retinex_smooth(illumination, reflectance, 'v3').item() == pytest.approx(0.0)


def test_smooth_rejects_unknown_version():
    with pytest.raises(ValueError, match='smooth_version'):
        _retinex_smooth(torch.rand(1, 1, 8, 8), torch.rand(1, 3, 8, 8), 'unknown')


def test_smooth_v1_matches_raw_reference_formula():
    torch.manual_seed(11)
    illumination = torch.rand(2, 1, 7, 9)
    reflectance = torch.rand(2, 3, 7, 9)
    gray = (0.299 * reflectance[:, 0:1] + 0.587 * reflectance[:, 1:2]
            + 0.114 * reflectance[:, 2:3])
    kx = illumination.new_tensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2)
    ky = kx.permute(0, 1, 3, 2)

    def gradient(tensor, kernel):
        return torch.nn.functional.conv2d(tensor, kernel, padding=1).abs()

    guide_x = torch.nn.functional.avg_pool2d(
        gradient(gray, kx), kernel_size=3, stride=1, padding=1
    )
    guide_y = torch.nn.functional.avg_pool2d(
        gradient(gray, ky), kernel_size=3, stride=1, padding=1
    )
    expected = torch.mean(
        gradient(illumination, kx) * torch.exp(-10 * guide_x)
        + gradient(illumination, ky) * torch.exp(-10 * guide_y)
    )
    assert torch.allclose(_retinex_smooth(illumination, reflectance, 'v1'), expected)


@pytest.mark.parametrize('version,use_average', [('v2', False), ('v3', True)])
def test_smooth_v2_v3_match_finite_difference_reference(version, use_average):
    torch.manual_seed(12)
    illumination = torch.rand(2, 1, 7, 9)
    reflectance = torch.rand(2, 3, 7, 9)
    gray = (0.299 * reflectance[:, 0:1] + 0.587 * reflectance[:, 1:2]
            + 0.114 * reflectance[:, 2:3])

    def average(gradient):
        if not use_average:
            return gradient
        return torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(gradient, (1, 1, 1, 1), mode='replicate'),
            kernel_size=3, stride=1,
        )

    grad_l_x = (illumination[:, :, :, 1:] - illumination[:, :, :, :-1]).abs()
    grad_l_y = (illumination[:, :, 1:, :] - illumination[:, :, :-1, :]).abs()
    grad_r_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
    grad_r_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
    expected = (
        (grad_l_x * torch.exp(-10 * average(grad_r_x))).mean()
        + (grad_l_y * torch.exp(-10 * average(grad_r_y))).mean()
    )
    assert torch.allclose(_retinex_smooth(illumination, reflectance, version), expected)


@pytest.mark.parametrize('version', ['v1', 'v2', 'v3'])
def test_smooth_version_is_switchable_from_pixel_loss_config(version):
    config = {
        'mode': 'pure_low_single_pixel',
        'recon_weight': 1.0,
        'anchor_weight': 0.05,
        'anchor_version': 'v2',
        'bdsp_weight': 0.05,
        'smooth_weight': 0.1,
        'smooth_version': version,
    }
    loss = _build_loss_function(config)
    assert loss.smooth_version == version


def test_loss_output_exposes_only_enabled_weighted_components():
    torch.manual_seed(1)
    low = torch.rand(2, 3, 8, 8)
    high = torch.rand(2, 3, 8, 8)
    r_low = torch.rand_like(low)
    r_high = torch.rand_like(high)
    l_low = torch.rand(2, 1, 8, 8)
    l_high = torch.rand(2, 1, 8, 8)
    loss = PairedLoss(
        l_type='pixel', recon_weight_high=1.0, recon_weight_low=0.3,
        cross_recon_weight_low=0.01, cross_recon_weight_high=0.02,
        smooth_weight=0.1, equal_r_weight=0.2,
    )
    output = loss(r_low, r_high, l_low, l_high, low, high)
    weighted = sum(
        value for key, value in output.items()
        if key.endswith('_weighted_loss')
    )
    assert torch.allclose(output['total_loss'], weighted, atol=1e-6)
    assert output['cross_recon_weighted_loss'].item() > 0
    assert output['equal_r_weighted_loss'].item() > 0
    assert 'anchor_weighted_loss' not in output
    assert 'bdsp_weighted_loss' not in output
    assert 'self_recon_weighted_loss' not in output
    assert 'reflect_weighted_loss' not in output


@pytest.mark.parametrize(
    ('mode', 'expected_components'),
    [
        ('paired_point', {'recon', 'cross_recon', 'equal_r'}),
        ('paired_pixel', {'recon', 'cross_recon', 'smooth', 'equal_r'}),
        ('unpaired_point', {'recon', 'anchor', 'bdsp', 'self_recon'}),
        ('unpaired_pixel', {'recon', 'anchor', 'bdsp', 'smooth', 'self_recon'}),
        ('pure_low_double_point', {'recon', 'anchor', 'bdsp', 'self_recon', 'reflect'}),
        ('pure_low_double_pixel', {
            'recon', 'anchor', 'bdsp', 'smooth', 'self_recon', 'reflect',
        }),
        ('pure_low_single_point', {'recon', 'anchor', 'bdsp'}),
        ('pure_low_single_pixel', {'recon', 'anchor', 'bdsp', 'smooth'}),
    ],
)
def test_every_loss_mode_reports_exactly_its_enabled_components(mode, expected_components):
    if mode.startswith('paired_'):
        config = {
            'mode': mode,
            'recon_weight_high': 1.0,
            'recon_weight_low': 0.3,
            'cross_recon_weight_low': 0.01,
            'cross_recon_weight_high': 0.02,
            'equal_r_weight': 0.1,
        }
    else:
        config = {
            'mode': mode,
            'recon_weight': 1.0,
            'anchor_weight': 0.05,
            'bdsp_weight': 0.05,
            'anchor_version': 'v2',
        }
        if mode.startswith('unpaired_') or mode.startswith('pure_low_double_'):
            config['self_recon_weight'] = 0.05
        if mode.startswith('pure_low_double_'):
            config['reflect_weight'] = 0.1
    if mode.endswith('_pixel'):
        config.update(smooth_weight=0.1, smooth_version='v1')

    loss = _build_loss_function(config)
    image1 = torch.rand(2, 3, 8, 8)
    image2 = torch.rand(2, 3, 8, 8)
    reflectance1 = torch.rand_like(image1)
    reflectance2 = torch.rand_like(image2)
    illumination_shape = (2, 1, 8, 8) if mode.endswith('_pixel') else (2, 1, 1, 1)
    illumination1 = torch.rand(illumination_shape)
    illumination2 = torch.rand(illumination_shape)

    if mode.startswith('pure_low_single_'):
        output = loss(reflectance1, illumination1, image1)
    else:
        output = loss(
            reflectance1, reflectance2, illumination1, illumination2,
            image1, image2, illumination1, illumination2,
        )

    reported_components = {
        key.removesuffix('_weighted_loss')
        for key in output
        if key.endswith('_weighted_loss')
    }
    assert reported_components == expected_components
    assert all(f'{name}_loss' in output for name in expected_components)
    weighted_total = sum(output[f'{name}_weighted_loss'] for name in expected_components)
    assert torch.allclose(output['total_loss'], weighted_total, atol=1e-6)


def test_zero_weight_loss_components_are_not_reported():
    image = torch.rand(1, 3, 4, 4)
    reflectance = torch.rand_like(image)
    illumination = torch.rand(1, 1, 4, 4)
    loss = PureLowSingleLoss(
        l_type='pixel', recon_weight=1.0, anchor_weight=0,
        bdsp_weight=0, smooth_weight=0,
    )
    output = loss(reflectance, illumination, image)
    assert set(output) == {'total_loss', 'recon_loss', 'recon_weighted_loss'}

    paired = PairedLoss(
        l_type='point', recon_weight_high=1, recon_weight_low=1,
        cross_recon_weight_low=0, cross_recon_weight_high=0,
        equal_r_weight=0,
    )
    point_l = torch.rand(1, 1, 1, 1)
    output = paired(
        reflectance, reflectance, point_l, point_l, image, image
    )
    assert 'cross_recon_loss' not in output
    assert 'cross_recon_low_loss' not in output
    assert 'equal_r_loss' not in output


def test_all_loss_families_total_equals_weighted_components():
    torch.manual_seed(2)
    image1 = torch.rand(2, 3, 8, 8)
    image2 = torch.rand(2, 3, 8, 8)
    r1 = torch.rand_like(image1)
    r2 = torch.rand_like(image2)
    l1 = torch.rand(2, 1, 8, 8)
    l2 = torch.rand(2, 1, 8, 8)
    lr1 = torch.rand(2, 1, 8, 8)
    lr2 = torch.rand(2, 1, 8, 8)
    outputs = [
        PureLowSingleLoss(l_type='pixel', smooth_weight=0.1)(r1, l1, image1),
        UnpairedLoss(l_type='pixel', smooth_weight=0.1)(
            r1, r2, l1, l2, image1, image2, lr1, lr2
        ),
        PureLowDoubleLoss(l_type='pixel', smooth_weight=0.1, reflect_weight=0.2)(
            r1, r2, l1, l2, image1, image2, lr1, lr2
        ),
    ]
    for output in outputs:
        weighted = sum(
            value for key, value in output.items()
            if key.endswith('_weighted_loss')
        )
        assert torch.allclose(output['total_loss'], weighted, atol=1e-6)


def test_reflect_stop_gradient_has_single_l1_scale():
    zeros = torch.zeros(1, 3, 4, 4)
    ones = torch.ones_like(zeros)
    l = torch.ones(1, 1, 4, 4)
    loss = PureLowDoubleLoss(
        recon_weight=0, anchor_weight=0, bdsp_weight=0,
        self_recon_weight=0, reflect_weight=1,
    )
    output = loss(zeros, ones, l, l, zeros, ones, None, None)
    assert output['reflect_loss'].item() == pytest.approx(1.0)
    assert output['total_loss'].item() == pytest.approx(1.0)


def test_pure_low_views_share_spatial_transform(tmp_path):
    values = torch.arange(16, dtype=torch.uint8).reshape(4, 4).numpy()
    rgb = torch.stack([torch.from_numpy(values)] * 3, dim=-1).numpy()
    image_path = tmp_path / 'sample.png'
    Image.fromarray(rgb).save(image_path)
    dataset = PureLowDataSet(
        [str(image_path)],
        transform=T.Compose([
            T.RandomCrop(3), T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5), T.ToTensor(),
        ]),
    )
    view1, view2 = dataset[0]
    assert torch.equal(view1, view2)


def test_model_accepts_non_multiple_of_four_input():
    model = RetinexPixelClassic(dim=8).eval()
    with torch.no_grad():
        reflectance, illumination = model(torch.rand(1, 3, 31, 30))
    assert reflectance.shape == (1, 3, 31, 30)
    assert illumination.shape == (1, 1, 31, 30)


def test_point_model_returns_scalar_illumination():
    model = RetinexPointRaw(dim=8).eval()
    with torch.no_grad():
        reflectance, illumination = model(torch.rand(2, 3, 31, 30))
    assert reflectance.shape == (2, 3, 31, 30)
    assert illumination.shape == (2, 1, 1, 1)


def test_paired_evaluate_saves_low_and_high_for_every_image(tmp_path):
    class IdentityRetinex(torch.nn.Module):
        def forward(self, image):
            return image, torch.ones_like(image[:, :1])

    low = torch.rand(4, 3, 8, 8)
    high = torch.rand(4, 3, 8, 8)
    loader = DataLoader(TensorDataset(low, high), batch_size=4)
    loss = PairedLoss(
        l_type='point', recon_weight_high=1, recon_weight_low=1,
        cross_recon_weight_low=0, cross_recon_weight_high=0,
        equal_r_weight=0,
    )
    metrics, _ = evaluate(
        IdentityRetinex(), loader, torch.device('cpu'), 0.0, str(tmp_path),
        loss_function=loss, save_images=True, global_iter=1,
    )
    assert math.isfinite(metrics['total_loss'])
    output_dir = tmp_path / '1'
    assert len(list(output_dir.glob('*_R_low.png'))) == 4
    assert len(list(output_dir.glob('*_L_low.png'))) == 4
    assert len(list(output_dir.glob('*_R_high.png'))) == 4
    assert len(list(output_dir.glob('*_L_high.png'))) == 4


def test_nonfinite_loss_never_updates_parameters():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, image):
            r = self.value.expand_as(image)
            return r, torch.ones_like(image[:, :1])

    class NanLoss(PureLowSingleLoss):
        def forward(self, r, l, image):
            return {'total_loss': r.mean() * torch.tensor(float('nan'))}

    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = create_lr_scheduler(optimizer, 10)
    before = model.value.detach().clone()
    with pytest.raises(FloatingPointError):
        train_step(
            model, optimizer, NanLoss(), torch.rand(1, 3, 4, 4),
            torch.device('cpu'), scheduler,
        )
    assert torch.equal(model.value.detach(), before)
    assert scheduler.last_epoch == 0


def test_pipeline_config_rejects_mode_mismatch():
    config = {
        'data': {'mode': 'paired', 'crop_size': 8, 'batch_size': 1, 'val_batch_size': 1},
        'model': {'name': 'RetinexPointRaw'},
        'training': {'max_iterations': 1},
        'loss': {
            'mode': 'unpaired_point', 'recon_weight': 1,
            'anchor_weight': 0.05, 'bdsp_weight': 0.05,
            'self_recon_weight': 0.05,
        },
    }
    with pytest.raises(ValueError, match='inconsistent'):
        validate_pipeline_config(config)


def test_all_repository_configs_are_pipeline_consistent():
    paths = list(Path('configs').rglob('*.yaml'))
    assert paths
    for path in paths:
        validate_pipeline_config(yaml.safe_load(path.read_text(encoding='utf-8')))


def test_every_network_has_strictly_paired_v1_v2_anchor_benchmarks():
    v1_paths = sorted(Path('configs').glob('*/pure_low_single/*anchorv1_*.yaml'))
    assert len(v1_paths) == 8  # 4 networks × LOLv2/BDD
    assert len({path.parts[-3] for path in v1_paths}) == 4

    for v1_path in v1_paths:
        v2_path = v1_path.with_name(v1_path.name.replace('anchorv1_', 'anchorv2_'))
        assert v2_path.is_file()
        v1_cfg = yaml.safe_load(v1_path.read_text(encoding='utf-8'))
        v2_cfg = yaml.safe_load(v2_path.read_text(encoding='utf-8'))
        assert v1_cfg['loss']['anchor_version'] == 'v1'
        assert v2_cfg['loss']['anchor_version'] == 'v2'

        v1_normalized = yaml.safe_load(v1_path.read_text(encoding='utf-8'))
        v1_normalized['loss']['anchor_version'] = 'v2'
        v1_normalized['experiment']['name'] = v1_normalized['experiment']['name'].replace(
            'anchorv1_', 'anchorv2_'
        )
        assert v1_normalized == v2_cfg

        assert '0.05anchorv1_' in generate_experiment_name(v1_cfg)
        assert '0.05anchorv2_' in generate_experiment_name(v2_cfg)


def test_every_anchor_config_filename_and_auto_name_expose_version():
    for path in Path('configs').rglob('*.yaml'):
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        version = config['loss'].get('anchor_version')
        if version is None:
            continue
        marker = f'anchor{version}_'
        assert marker in path.name, f'{path} does not contain {marker}'
        assert marker in generate_experiment_name(config)
        assert config['experiment']['name'] == path.stem
        assert config['model']['name'] == path.parts[-3]


def test_every_pixel_config_filename_and_auto_name_expose_smooth_version():
    versioned_count = 0
    for path in Path('configs').rglob('*.yaml'):
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        version = config['loss'].get('smooth_version')
        if version is None:
            assert config['loss']['mode'].endswith('_point')
            continue
        versioned_count += 1
        assert version == 'v1'  # 当前全部配置默认复现 Raw
        marker = f"sm{version}"
        assert marker in path.name, f'{path} does not contain {marker}'
        assert marker in generate_experiment_name(config)
        assert config['experiment']['name'] == path.stem
        assert config['model']['name'] == path.parts[-3]
    assert versioned_count == 25
