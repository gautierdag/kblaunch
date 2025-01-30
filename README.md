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

### Setup

Run the setup command to configure the tool (email and slack webhook):

```bash
kblaunch setup
```

This will go through the following steps:

1. Set the user (optional): This is used to identify the user and required by the cluster. The default is set to $USER.
2. Set the email (required): This is used to identify the user and required by the cluster.
3. Set up Slack notifications (optional): This will send a test message to the webhook, and setup the webhook in the config. When your job starts you will receive a message at the webhook
4. Set up a PVC (optional): This will create a PVC for the user to use in their jobs
5. Set the default PVC to use (optional): Note only one pod can use the PVC at a time

### Basic Usage

Launch a simple job:

```bash
kblaunch launch
    --job-name myjob \
    --command "python script.py"
```

### With Environment Variables

1. From local environment:

    ```bash
    export PATH=...
    export OPENAI_API_KEY=...
    # pass the environment variables to the job
    kblaunch launch \
        --job-name myjob \
        --command "python script.py" \
        --local-env-vars PATH \
        --local-env-vars OPENAI_API_KEY
    ```

2. From Kubernetes secrets:

    ```bash
    kblaunch launch \
        --job-name myjob \
        --command "python script.py" \
        --secrets-env-vars mysecret1 \
        --secrets-env-vars mysecret2
    ```

3. From .env file (default behavior):

    ```bash
    kblaunch launch \
        --job-name myjob \
        --command "python script.py" \
        --load-dotenv
    ```

    If a .env exists in the current directory, it will be loaded and passed as environment variables to the job.

### GPU Jobs

Specify GPU requirements:

```bash
kblaunch launch \
    --job-name gpu-job \
    --command "python train.py" \
    --gpu-limit 2 \
    --gpu-product "NVIDIA-A100-SXM4-80GB"
```

### Interactive Mode

Launch an interactive job:

```bash
kblaunch launch \
    --job-name interactive \
    --interactive
```

## Options

- `--email`: User email
- `--job-name`: Name of the Kubernetes job [required]
- `--docker-image`: Docker image (default: "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04")
- `--namespace`: Kubernetes namespace (default: "informatics")
- `--queue-name`: Kueue queue name
- `--interactive`: Run in interactive mode (default: False)
- `--command`: Command to run in the container [required if not interactive]
- `--cpu-request`: CPU request (default: "1")
- `--ram-request`: RAM request (default: "8Gi")
- `--gpu-limit`: GPU limit (default: 1)
- `--gpu-product`: GPU product (default: "NVIDIA-A100-SXM4-80GB")
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
