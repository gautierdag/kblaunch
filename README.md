# kblaunch

[![Test](https://github.com/gautierdag/kblaunch/actions/workflows/test.yaml/badge.svg)](https://github.com/gautierdag/kblaunch/actions/workflows/test.yaml)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![Ruff](https://img.shields.io/badge/linter-ruff-blue)
[![PyPI Version](https://img.shields.io/pypi/v/kblaunch)](https://pypi.org/project/kblaunch/)
[![Documentation](https://img.shields.io/badge/docs-pdoc-blue)](https://gautierdag.github.io/kblaunch/)

A CLI tool for launching Kubernetes jobs with environment variable and secret management.

## Installation

### Using uv (recommended)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, you can install `uv` using pip:

```bash
pip install uv
```

2. Use `uvx` to use the cli (the `uvx` command invokes a tool without installing it to the local .venv):

```bash
uvx kblaunch --help
```

When using the `kblaunch` command always prepend with `uvx` command.

## Usage

### Setup

Run the setup command to configure the tool (email and slack webhook):

```bash
uvx kblaunch setup
```

This will go through the following steps:

1. Set the user (optional): This is used to identify the user and required by the cluster. The default is set to $USER.
2. Set the email (required): This is used to identify the user and required by the cluster.
3. Set up Slack notifications (optional): This will send a test message to the webhook, and setup the webhook in the config. When your job starts you will receive a message at the webhook. Note a slack webhook is also required for automatic vscode tunnelling. 
4. Set up a PVC (optional): This will create a PVC for the user to use in their jobs.
5. Set the default PVC to use (optional): Note only one pod can use the PVC at a time. The default pvc will be passed to the job. The pvc will always be mounted at `/pvc`.
6. Set up git credentials (optional): If the user has set up a git/rsa key on the head node. We can export it as a secret for them and automatically load it and setup git credentials in their launched pods. This requires having setup git/rsa credentials before hand.

The outcome of `kblaunch setup` is a `.json` file stored in `.cache/.kblaunch/config.json. It should look something like this:

```json
{
  "email": "XXX@ed.ac.uk",
  "user": "sXXX-infk8s",
  "slack_webhook": "https://hooks.slack.com/services/XXX/XXX/XXX",
  "default_pvc": "sXXX-infk8s-pvc",
  "git_secret": "sXXX-infk8s-git-ssh"
}
```

When you later use `kblaunch` to launch a job, it will use the values stored in that `config.json.`

### Basic Usage

Launch a simple job:

```bash
uvx kblaunch launch
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
        --local-env-vars PATH,OPENAI_API_KEY
    ```

2. From Kubernetes secrets:

    ```bash
    uvx kblaunch launch \
        --job-name myjob \
        --command "python script.py" \
        --secrets-env-vars mysecret1,mysecret2
    ```

3. From .env file (default behavior):

    ```bash
    uvx kblaunch launch \
        --job-name myjob \
        --command "python script.py" \
        --load-dotenv
    ```

    If a .env exists in the current directory, it will be loaded and passed as environment variables to the job.

### GPU Jobs

Specify GPU requirements:

```bash
uvx kblaunch launch \
    --job-name gpu-job \
    --command "python train.py" \
    --gpu-limit 2 \
    --gpu-product "NVIDIA-A100-SXM4-80GB"
```

### Interactive Mode

Launch an interactive job:

```bash
uvx kblaunch launch \
    --job-name interactive \
    --interactive
```

## Launch Options

Launch command options:

- `--email`: User email (overrides config)
- `--job-name`: Name of the Kubernetes job [required]
- `--docker-image`: Docker image (default: "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04")
- `--namespace`: Kubernetes namespace (default: $KUBE_NAMESPACE)
- `--queue-name`: Kueue queue name (default: $KUBE_QUEUE_NAME)
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
- `--nfs-server`: NFS server address (default: set to environment variable $INFK8S_NFS_SERVER_IP)
- `--pvc-name`: Persistent Volume Claim name (default: `default_pvc` if present in `config.json`)
- `--dry-run`: Print job YAML without creating it (default: False)
- `--priority`: Priority class name (default: "default")
  - Available options: `default`, `batch`, `short`
- `--vscode`: Install VS Code CLI in container (default: False)
- `--tunnel`: Start VS Code SSH tunnel on startup (requires `$SLACK_WEBHOOK` and --vscode flag)
- `--startup-script`: Path to startup script to run in container

Monitor command options:

- `--namespace`: Kubernetes namespace (default: $KUBE_NAMESPACE)

## Monitoring Commands

The `kblaunch monitor` command provides several subcommands to monitor cluster resources:

Displays aggregate GPU statistics for the cluster:

```bash
uvx kblaunch monitor gpus
```

Displays queued jobs (jobs which are waiting for GPUs):

```bash
uvx kblaunch monitor queue
```

Displays per-user statistics:

```bash
uvx kblaunch monitor users
```

Displays per-job statistics:

```bash
uvx kblaunch monitor jobs
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
