"""kblaunch - A CLI tool for launching and monitoring GPU jobs on Kubernetes clusters.

## Commands
* Launching GPU jobs with various configurations
* Monitoring GPU usage and job statistics
* Setting up user configurations and preferences
* Managing persistent volumes and Git authentication

## Features
* Interactive and batch job support
* GPU resource management and constraints
* Environment variable handling from multiple sources
* Persistent Volume Claims (PVC) for storage
* Git SSH authentication
* VS Code integration with remote tunneling
* Slack notifications for job status
* Real-time cluster monitoring

## Resource Types
* A100 GPUs (40GB and 80GB variants)
* H100 GPUs (80GB variant)
* MIG GPU instances
* CPU and RAM allocation
* Persistent storage volumes

## Job Priority Classes
* default: Standard priority for most workloads
* batch: Lower priority for long-running jobs
* short: High priority for quick jobs (with GPU constraints)

## Environment Integration
* Kubernetes secrets
* Local environment variables
* .env file support
* SSH key management
* NFS workspace mounting
"""

import importlib.metadata

__version__ = importlib.metadata.version("kblaunch")

__all__ = [
    "setup",
    "launch",
    "monitor_gpus",
    "monitor_users",
    "monitor_jobs",
    "monitor_queue",
]

from .cli import setup, launch, monitor_gpus, monitor_users, monitor_jobs, monitor_queue
