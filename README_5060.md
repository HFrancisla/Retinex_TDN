# RTX 5060 本机测试环境

这组文件只描述 RTX 5060 本机 CUDA 测试环境，不是项目的通用依赖约定：

- `pyproject_5060.toml`
- `uv_5060.lock`
- `python-version_5060`

由于 uv 只识别标准项目文件名，需要先生成本机副本：

```bash
cp pyproject_5060.toml pyproject.toml
cp uv_5060.lock uv.lock
cp python-version_5060 .python-version
uv sync --locked
```

上述标准名副本和 `.venv/` 已通过 `.git/info/exclude` 在本机忽略。

验证 CUDA：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

当前已验证组合：Python 3.12、PyTorch 2.12.0、torchvision 0.27.0、CUDA 13.0，
RTX 5060 Laptop GPU 计算能力为 `sm_120`。
