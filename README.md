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

## Launch Options

Launch command options:
- `--email`: User email (overrides config)
- `--job-name`: Name of the Kubernetes job [required]
- `--docker-image`: Docker image (default: "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04")
- `--namespace`: Kubernetes namespace (default: "informatics")
- `--queue-name`: Kueue queue name (default: "informatics-user-queue")
- `--interactive`: Run in interactive mode (default: False)
- `--command`: Command to run in the container [required if not interactive]
- `--cpu-request`: CPU request (default: "1")
- `--ram-request`: RAM request (default: "8Gi")
- `--gpu-limit`: GPU limit (default: 1)
- `--gpu-product`: GPU product type (default: "NVIDIA-A100-SXM4-40GB")
    - Available options: 
        - NVIDIA-A100-SXM4-80GB
        - NVIDIA-A100-SXM4-40GB
        - NVIDIA-A100-SXM4-40GB-MIG-3g.20gb
        - NVIDIA-A100-SXM4-40GB-MIG-1g.5gb
        - NVIDIA-H100-80GB-HBM3
- `--secrets-env-vars`: List of secret environment variables (default: [])
- `--local-env-vars`: List of local environment variables (default: [])
- `--load-dotenv`: Load environment variables from .env file (default: True)
- `--nfs-server`: NFS server address
- `--pvc-name`: Persistent Volume Claim name
- `--dry-run`: Print job YAML without creating it (default: False)
- `--priority`: Priority class name (default: "default")
    - Available options: default, batch, short
- `--vscode`: Install VS Code CLI in container (default: False)
- `--tunnel`: Start VS Code SSH tunnel on startup (requires SLACK_WEBHOOK and --vscode)
- `--startup-script`: Path to startup script to run in container

Monitor command options:
- `--namespace`: Kubernetes namespace (default: "informatics")



## Monitoring Commands

The `kblaunch monitor` command provides several subcommands to monitor cluster resources:
    
Displays aggreate GPU statistics for the cluster:

```bash
kblaunch monitor gpus
```

Displays queued jobs (jobs which are waiting for GPUs):

```bash
kblaunch monitor queue
```

Displays per-user statistics:

```bash
kblaunch monitor users
```

Displays per-job statistics:

```bash
kblaunch monitor jobs
```

Note that `users` and `jobs` commands will run `nvidia-smi` on pods to obtain GPU usage is not recommended for frequent use.


## Features

- Kubernetes job management
- Environment variable handling from multiple sources
- Kubernetes secrets integration
- GPU job support
- Interactive mode
- Automatic job cleanup
- Slack notifications (when configured)
- Persistent Volume Claim (PVC) management
- VS Code integration (with Code tunnelling support)
- Monitoring commands
