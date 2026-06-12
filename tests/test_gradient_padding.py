import torch

from loss.decomposition_loss import gradient, gradient_no_abs


def check_shapes_and_stability():
    cases = [
        ("sobel", 1, 8, 8),
        ("sobel", 3, 7, 9),
        ("robert", 1, 5, 6),
        ("robert", 3, 9, 10),
    ]

    for kernel, channels, h, w in cases:
        x = torch.randn(2, channels, h, w)

        gx = gradient(x, "x", kernel=kernel)
        gy = gradient(x, "y", kernel=kernel)

        assert gx.shape == x.shape, f"shape mismatch: {gx.shape} vs {x.shape} [{kernel},{channels},{h},{w}]"
        assert gy.shape == x.shape, f"shape mismatch: {gy.shape} vs {x.shape} [{kernel},{channels},{h},{w}]"
        assert torch.isfinite(gx).all(), f"non-finite in gx [{kernel},{channels},{h},{w}]"
        assert torch.isfinite(gy).all(), f"non-finite in gy [{kernel},{channels},{h},{w}]"
        assert gx.min() >= 0.0
        assert gx.max() <= 1.0 + 1e-6
        assert gy.min() >= 0.0
        assert gy.max() <= 1.0 + 1e-6

        gx_noabs = gradient_no_abs(x, "x", kernel=kernel)
        gy_noabs = gradient_no_abs(x, "y", kernel=kernel)

        assert gx_noabs.shape == x.shape
        assert gy_noabs.shape == x.shape
        assert torch.isfinite(gx_noabs).all()
        assert torch.isfinite(gy_noabs).all()
        assert (gx_noabs < 0).any(), "signed gradient should include negatives"
        assert (gy_noabs < 0).any(), "signed gradient should include negatives"


def check_constant_image_gradient_normalization():
    x = torch.full((1, 3, 10, 12), 0.5)
    gx = gradient(x, "x", kernel="sobel")
    gy = gradient(x, "y", kernel="robert")

    assert gx.shape == x.shape
    assert gy.shape == x.shape
    assert torch.isfinite(gx).all()
    assert torch.isfinite(gy).all()
    assert gx.min() >= 0.0
    assert gx.max() <= 1.0 + 1e-6
    assert gy.min() >= 0.0
    assert gy.max() <= 1.0 + 1e-6


if __name__ == "__main__":
    check_shapes_and_stability()
    check_constant_image_gradient_normalization()
    print("gradient padding tests passed")
