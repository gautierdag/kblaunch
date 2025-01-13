# kblaunch

[![Test](https://github.com/gautierdag/kblaunch/actions/workflows/test.yaml/badge.svg)](https://github.com/gautierdag/kblaunch/actions/workflows/test.yaml)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![Ruff](https://img.shields.io/badge/linter-ruff-blue)
[![PyPI Version](https://img.shields.io/pypi/v/kblaunch)](https://pypi.org/project/kblaunch/)

A CLI tool for launching Kubernetes jobs with environment variable and secret management.

## Installation

```bash
pip install kblaunch
```

Or using `uv`:

```bash
uv add kblaunch
```

You can even use `uvx` to use the cli without installing it:

```bash
uvx kblaunch --help
```

## Usage

### Basic Usage

Launch a simple job:

```bash
kblaunch \
    --email your.email@ed.ac.uk \
    --job-name myjob \
    --command "python script.py"
```

### With Environment Variables

1. From local environment:

    ```bash
    kblaunch \
        --job-name myjob \
        --command "python script.py" \
        --local-env-vars PATH \
        --local-env-vars PYTHONPATH
    ```

2. From Kubernetes secrets:

    ```bash
    kblaunch \
        --job-name myjob \
        --command "python script.py" \
        --secrets-env-vars mysecret1 \
        --secrets-env-vars mysecret2
    ```

3. From .env file:

    ```bash
    kblaunch \
        --job-name myjob \
        --command "python script.py" \
        --load-dotenv
    ```

### GPU Jobs

Specify GPU requirements:

```bash
kblaunch \
    --job-name gpu-job \
    --command "python train.py" \
    --gpu-limit 2 \
    --gpu-product "NVIDIA-A100-80GB"
```

### Interactive Mode

Launch an interactive job:

```bash
kblaunch \
    --job-name interactive \
    --interactive
```

## Options

- `--email`: User email [required]
- `--job-name`: Name of the Kubernetes job [required]
- `--docker-image`: Docker image (default: "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04")
- `--namespace`: Kubernetes namespace (default: "informatics")
- `--queue-name`: Kueue queue name
- `--interactive`: Run in interactive mode (default: False)
- `--command`: Command to run in the container [required]
- `--cpu-request`: CPU request (default: "1")
- `--ram-request`: RAM request (default: "8Gi")
- `--gpu-limit`: GPU limit (default: 1)
- `--gpu-product`: GPU product (default: "NVIDIA-A100-80GB-PCIe")
- `--secrets-env-vars`: List of secret environment variables
- `--local-env-vars`: List of local environment variables
- `--load-dotenv`: Load environment variables from .env file (default: True)

## Features

- Kubernetes job management
- Environment variable handling from multiple sources
- Kubernetes secrets integration
- GPU job support
- Interactive mode
- Automatic job cleanup
- Slack notifications (when configured)
