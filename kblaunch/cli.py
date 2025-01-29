import os
import subprocess
from typing import List, Optional
import json
from pathlib import Path

import typer
import yaml
from kubernetes import client, config
from loguru import logger
from enum import Enum

MAX_CPU = 192
MAX_RAM = 890
MAX_GPU = 8

CONFIG_DIR = Path.home() / ".cache" / ".kblaunch"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Load configuration from file."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error reading config file {CONFIG_FILE}")
        return {}


def save_config(config: dict):
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


class GPU_PRODUCTS(str, Enum):
    a100_80gb = "NVIDIA-A100-SXM4-80GB"
    a100_40gb = "NVIDIA-A100-SXM4-40GB"
    a100_40gb_mig_3g_20gb = "NVIDIA-A100-SXM4-40GB-MIG-3g.20gb"
    a100_40gb_mig_1g_5gb = "NVIDIA-A100-SXM4-40GB-MIG-1g.5gb"
    h100_80gb_hbm3 = "NVIDIA-H100-80GB-HBM3"


PRIORITY_CLASSES = ["default", "batch", "short"]
NFS_SERVER = os.getenv("INFK8S_NFS_SERVER_IP", "10.24.1.255")

app = typer.Typer()


def is_mig_gpu(gpu_product: str) -> bool:
    """Check if the GPU product is a MIG instance."""
    return "MIG" in gpu_product


def validate_gpu_constraints(gpu_product: str, gpu_limit: int, priority: str):
    """Validate GPU constraints for MIG and H100 instances."""
    # Check MIG constraint
    if is_mig_gpu(gpu_product) and gpu_limit > 1:
        raise ValueError("Cannot request more than one MIG instance in a single job")

    # Check H100 priority constraint
    if "H100" in gpu_product and priority == "short":
        raise ValueError(
            "Cannot request H100 GPUs in the short-workload-high-priority class"
        )


class KubernetesJob:
    def __init__(
        self,
        name: str,
        image: str,
        kueue_queue_name: str,
        command: List[str] = None,
        args: Optional[List[str]] = None,
        cpu_request: Optional[str] = None,
        ram_request: Optional[str] = None,
        storage_request: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_product: Optional[str] = None,
        gpu_limit: Optional[int] = None,
        backoff_limit: int = 4,
        restart_policy: str = "Never",
        shm_size: Optional[str] = None,
        env_vars: Optional[dict] = None,
        secret_env_vars: Optional[dict] = None,
        volume_mounts: Optional[dict] = None,
        job_deadlineseconds: Optional[int] = None,
        privileged_security_context: bool = False,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        namespace: Optional[str] = None,
        image_pull_secret: Optional[str] = None,
        priority: str = "default",
    ):
        # Validate gpu_limit first
        assert (
            gpu_limit is not None
        ), f"gpu_limit must be set to a value between 1 and {MAX_GPU}, not {gpu_limit}"
        assert (
            0 < gpu_limit <= MAX_GPU
        ), f"gpu_limit must be between 1 and {MAX_GPU}, got {gpu_limit}"

        self.name = name
        self.image = image
        self.command = command
        self.args = args
        self.gpu_limit = gpu_limit
        self.gpu_type = gpu_type
        self.gpu_product = gpu_product

        self.cpu_request = cpu_request if cpu_request else 12 * gpu_limit
        self.ram_request = ram_request if ram_request else f"{80 * gpu_limit}G"
        assert (
            int(self.cpu_request) <= MAX_CPU
        ), f"cpu_request must be less than {MAX_CPU}"

        # Safe calculation for shm_size with fallback
        self.shm_size = (
            shm_size
            if shm_size is not None
            else ram_request
            if ram_request is not None
            else f"{max(1, MAX_RAM // gpu_limit)}G"  # Ensure minimum 1G and avoid division by zero
        )

        self.env_vars = env_vars
        self.secret_env_vars = secret_env_vars

        self.storage_request = storage_request
        self.backoff_limit = backoff_limit
        self.restart_policy = restart_policy
        self.image_pull_secret = image_pull_secret

        self.volume_mounts = volume_mounts
        self.job_deadlineseconds = job_deadlineseconds
        self.privileged_security_context = privileged_security_context

        self.user_name = user_name or os.environ.get("USER", "unknown")
        self.user_email = user_email  # This is now a required field.
        self.kueue_queue_name = kueue_queue_name

        assert (
            priority in PRIORITY_CLASSES
        ), f"priority_class_name must be one of {PRIORITY_CLASSES}, not {priority}"
        if priority == "high" and (self.gpu_limit > 1 or "H100" in self.gpu_product):
            logger.error(
                "Priority class 'high' is not allowed for multi-GPU jobs or H100 GPUs."
            )
            logger.error("Using 'default' priority class instead.")
            priority = "default"

        self.labels = {
            "eidf/user": self.user_name,
            "kueue.x-k8s.io/queue-name": self.kueue_queue_name,
            "kueue.x-k8s.io/priority-class": f"{priority}-workload-priority",
        }
        self.annotations = {"eidf/user": self.user_name, "eidf/email": self.user_email}
        self.namespace = namespace

    def _add_shm_size(self, container: dict):
        """Adds shared memory volume if shm_size is set."""
        if self.shm_size:
            container["volumeMounts"].append({"name": "dshm", "mountPath": "/dev/shm"})
        return container

    def _add_env_vars(self, container: dict):
        """Adds secret and normal environment variables to the
        container."""
        # Ensure that the POD_NAME environment variable is set
        container["env"] = [
            {
                "name": "POD_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
            }
        ]
        # Add the environment variables
        if self.env_vars:
            for key, value in self.env_vars.items():
                container["env"].append({"name": key, "value": value})

        # pass kubernetes secrets as environment variables
        if self.secret_env_vars:
            for key, secret_name in self.secret_env_vars.items():
                container["env"].append(
                    {
                        "name": key,
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": secret_name,
                                "key": key,
                            }
                        },
                    }
                )

        return container

    def _add_volume_mounts(self, container: dict):
        """Adds volume mounts to the container."""
        if self.volume_mounts:
            for mount_name, mount_data in self.volume_mounts.items():
                container["volumeMounts"].append(
                    {
                        "name": mount_name,
                        "mountPath": mount_data["mountPath"],
                    }
                )

        return container

    def _add_privileged_security_context(self, container: dict):
        """Adds privileged security context to the container."""
        if self.privileged_security_context:
            container["securityContext"] = {
                "privileged": True,
            }

        return container

    def generate_yaml(self):
        container = {
            "name": self.name,
            "image": self.image,
            "imagePullPolicy": "Always",
            "volumeMounts": [],
            "resources": {
                "requests": {},
                "limits": {},
            },
        }

        if self.command is not None:
            container["command"] = self.command

        if self.args is not None:
            container["args"] = self.args

        if not (
            self.gpu_type is None or self.gpu_limit is None or self.gpu_product is None
        ):
            container["resources"] = {"limits": {f"{self.gpu_type}": self.gpu_limit}}

        container = self._add_shm_size(container)
        container = self._add_env_vars(container)
        container = self._add_volume_mounts(container)
        container = self._add_privileged_security_context(container)

        if (
            self.cpu_request is not None
            or self.ram_request is not None
            or self.storage_request is not None
        ):
            if "resources" not in container:
                container["resources"] = {"requests": {}}

            if "requests" not in container["resources"]:
                container["resources"]["requests"] = {}

        if self.cpu_request is not None:
            container["resources"]["requests"]["cpu"] = self.cpu_request
            container["resources"]["limits"]["cpu"] = self.cpu_request

        if self.ram_request is not None:
            container["resources"]["requests"]["memory"] = self.ram_request
            container["resources"]["limits"]["memory"] = self.ram_request

        if self.storage_request is not None:
            container["resources"]["requests"]["storage"] = self.storage_request

        if self.gpu_type is not None and self.gpu_limit is not None:
            container["resources"]["limits"][f"{self.gpu_type}"] = self.gpu_limit

        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.name,
                "labels": self.labels,  # Add labels here
                "annotations": self.annotations,  # Add metadata here
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": self.labels,  # Add labels to Pod template as well
                        "annotations": self.annotations,  # Add metadata to Pod template as well
                    },
                    "spec": {
                        "containers": [container],
                        "restartPolicy": self.restart_policy,
                        "volumes": [],
                    },
                },
                "backoffLimit": self.backoff_limit,
            },
        }

        if self.image_pull_secret:
            job["spec"]["imagePullSecrets"] = {"name": self.image_pull_secret}

        if self.job_deadlineseconds:
            job["spec"]["activeDeadlineSeconds"] = self.job_deadlineseconds

        if self.namespace:
            job["metadata"]["namespace"] = self.namespace

        if not (
            self.gpu_type is None or self.gpu_limit is None or self.gpu_product is None
        ):
            job["spec"]["template"]["spec"]["nodeSelector"] = {
                f"{self.gpu_type}.product": self.gpu_product
            }
        # Add shared memory volume if shm_size is set
        if self.shm_size:
            job["spec"]["template"]["spec"]["volumes"].append(
                {
                    "name": "dshm",
                    "emptyDir": {
                        "medium": "Memory",
                        "sizeLimit": self.shm_size,
                    },
                }
            )

        # Add volumes for the volume mounts
        if self.volume_mounts:
            for mount_name, mount_data in self.volume_mounts.items():
                volume = {"name": mount_name}
                if mount_name == "nfs":
                    volume["nfs"] = {
                        "server": mount_data["server"],
                        "path": mount_data["mountPath"],
                    }
                # TODO: verify if this works for pvc
                elif mount_name == "pvc":
                    volume["persistentVolumeClaim"] = {"claimName": mount_data}

                # Add more volume types here if needed
                job["spec"]["template"]["spec"]["volumes"].append(volume)

        return yaml.dump(job)

    def run(self):
        config.load_kube_config()

        job_yaml = self.generate_yaml()

        # Save the generated YAML to a temporary file
        with open("temp_job.yaml", "w") as temp_file:
            temp_file.write(job_yaml)

        # Run the kubectl command with --validate=False
        cmd = ["kubectl", "apply", "-f", "temp_job.yaml"]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Remove the temporary file
            os.remove("temp_job.yaml")
            return result.returncode
        except subprocess.CalledProcessError as e:
            logger.info(
                f"Command '{' '.join(cmd)}' failed with return code {e.returncode}."
            )
            logger.info(f"Stdout:\n{e.stdout}")
            logger.info(f"Stderr:\n{e.stderr}")
            # Remove the temporary file
            os.remove("temp_job.yaml")
            return e.returncode  # return the exit code
        except Exception:
            logger.exception(
                f"An unexpected error occurred while running '{' '.join(cmd)}'."
            )  # This logs the traceback too
            # Remove the temporary file
            os.remove("temp_job.yaml")
            return 1  # return the exit code


def check_if_completed(job_name: str, namespace: str = "informatics") -> bool:
    # Load the kube config
    config.load_kube_config()

    # Create an instance of the API class
    api = client.BatchV1Api()

    job_exists = False
    is_completed = True

    # Check if the job exists in the specified namespace
    jobs = api.list_namespaced_job(namespace)
    if job_name in {job.metadata.name for job in jobs.items}:
        job_exists = True

    if job_exists is True:
        job = api.read_namespaced_job(job_name, namespace)
        is_completed = False

        # Check the status conditions
        if job.status.conditions:
            for condition in job.status.conditions:
                if condition.type == "Complete" and condition.status == "True":
                    is_completed = True
                elif condition.type == "Failed" and condition.status == "True":
                    logger.error(f"Job {job_name} has failed.")
        else:
            logger.info(f"Job {job_name} still running or status is unknown.")

        if is_completed:
            api_res = api.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
            logger.info(f"Job '{job_name}' deleted. Status: {api_res.status}")
    return is_completed


def send_message_command(env_vars: set) -> str:
    """
    Send a message to Slack when the job starts if the SLACK_WEBHOOK environment variable is set.
    """
    if "SLACK_WEBHOOK" not in env_vars:
        logger.debug("SLACK_WEBHOOK not found in env_vars.")
        return ""
    return (
        """apt-get update && apt-get install -y curl;"""  # Install the curl command
        + """curl -X POST -H 'Content-type: application/json' --data '{"text":"Job started in '"$POD_NAME"'"}' $SLACK_WEBHOOK ;"""
    )


def install_vscode_command() -> str:
    """Generate command to install VS Code CLI."""
    return (
        """curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && """
        """tar -xf vscode_cli.tar.gz && """
        """rm vscode_cli.tar.gz && """
    )


def get_env_vars(
    local_env_vars: list[str],
    load_dotenv: bool = False,
) -> dict[str, str]:
    """Get environment variables from local environment and secrets."""

    if load_dotenv:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception as e:
            logger.warning(f"Error loading .env file: {e}")

    env_vars = {}
    for var_name in local_env_vars:
        try:
            env_vars[var_name] = os.environ[var_name]
        except KeyError:
            logger.warning(
                f"Environment variable {var_name} not found in local environment"
            )
    return env_vars


def get_secret_env_vars(
    secrets_names: list[str],
    namespace: str = "informatics",
) -> dict[str, str]:
    """
    Get secret environment variables from Kubernetes secrets
    """
    secrets_env_vars = {}
    for secret_name in secrets_names:
        try:
            v1 = client.CoreV1Api()
            secret = v1.read_namespaced_secret(name=secret_name, namespace=namespace)
            for key in secret.data.keys():
                if key in secrets_env_vars:
                    logger.warning(f"Key {key} already set in env_vars.")
                secrets_env_vars[key] = secret_name
        except Exception as e:
            logger.warning(f"Error reading secret {secret_name}: {e}")
    return secrets_env_vars


@app.command()
def setup():
    """Interactive setup for kblaunch configuration."""
    config = load_config()

    # Get email
    email = typer.prompt("Please enter your email")
    config["email"] = email

    # Get Slack webhook
    if typer.confirm("Would you like to set up Slack notifications?", default=False):
        webhook = typer.prompt("Enter your Slack webhook URL")
        config["slack_webhook"] = webhook

    # validate slack webhook
    if "slack_webhook" in config:
        # test post to slack
        try:
            logger.info("Sending test message to Slack")
            import requests

            message = "Hello :wave: from kblaunch"
            response = requests.post(
                config["slack_webhook"],
                json={"text": message},
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error sending test message to Slack: {e}")

    # Save config
    save_config(config)
    logger.info(f"Configuration saved to {CONFIG_FILE}")


@app.command()
def launch(
    email: str = typer.Option(None, help="User email (overrides config)"),
    job_name: str = typer.Option(..., help="Name of the Kubernetes job"),
    docker_image: str = typer.Option(
        "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04", help="Docker image"
    ),
    namespace: str = typer.Option("informatics", help="Kubernetes namespace"),
    queue_name: str = typer.Option("informatics-user-queue", help="Kueue queue name"),
    interactive: bool = typer.Option(False, help="Run in interactive mode"),
    command: str = typer.Option(
        "", help="Command to run in the container"
    ),  # Made optional
    cpu_request: str = typer.Option("1", help="CPU request"),
    ram_request: str = typer.Option("8Gi", help="RAM request"),
    gpu_limit: int = typer.Option(1, help="GPU limit"),
    gpu_product: GPU_PRODUCTS = typer.Option(
        "NVIDIA-A100-SXM4-40GB",
        help="GPU product type to use",
        show_choices=True,
        show_default=True,
    ),
    secrets_env_vars: list[str] = typer.Option(
        [],  # Use empty list as default instead of None
        help="List of secret environment variables to export to the container",
    ),
    local_env_vars: list[str] = typer.Option(
        [],  # Use empty list as default instead of None
        help="List of local environment variables to export to the container",
    ),
    load_dotenv: bool = typer.Option(
        True, help="Load environment variables from .env file"
    ),
    nfs_server: str = typer.Option(NFS_SERVER, help="NFS server"),
    dry_run: bool = typer.Option(False, help="Dry run"),
    priority: str = typer.Option("default", help="Priority class name"),
    vscode: bool = typer.Option(False, help="Install VS Code CLI in the container"),
):
    """Launch a Kubernetes job with the specified configuration."""

    # Load config
    config = load_config()

    # Use email from config if not provided
    if email is None:
        email = config.get("email")
        if email is None:
            raise typer.BadParameter(
                "Email not provided and not found in config. "
                "Please provide --email or run 'kblaunch setup'"
            )

    # Add SLACK_WEBHOOK to local_env_vars if configured
    if "slack_webhook" in config:
        os.environ["SLACK_WEBHOOK"] = config["slack_webhook"]
        if "SLACK_WEBHOOK" not in local_env_vars:
            local_env_vars.append("SLACK_WEBHOOK")

    # Add validation for command parameter
    if not interactive and command == "":
        raise typer.BadParameter("--command is required when not in interactive mode")

    is_completed = check_if_completed(job_name, namespace=namespace)

    if is_completed is True:
        logger.info(f"Job '{job_name}' is completed. Launching a new job.")

        # Validate GPU constraints before creating job
        try:
            validate_gpu_constraints(gpu_product.value, gpu_limit, priority)
        except ValueError as e:
            raise typer.BadParameter(str(e))

        if interactive:
            cmd = "while true; do sleep 60; done;"
        else:
            cmd = command
            logger.info(f"Command: {cmd}")

        # Get local environment variables
        env_vars_dict = get_env_vars(
            local_env_vars=local_env_vars,
            load_dotenv=load_dotenv,
        )
        secrets_env_vars_dict = get_secret_env_vars(
            secrets_names=secrets_env_vars,
            namespace=namespace,
        )

        # Check for overlapping keys in local and secret environment variables
        intersection = set(secrets_env_vars_dict.keys()).intersection(
            env_vars_dict.keys()
        )
        if intersection:
            logger.warning(
                f"Overlapping keys in local and secret environment variables: {intersection}"
            )
        # Combine the environment variables
        union = set(secrets_env_vars_dict.keys()).union(env_vars_dict.keys())

        logger.info(f"Creating job for: {cmd}")
        # Build the full command with optional VS Code installation
        full_cmd = ""
        if vscode:
            full_cmd += install_vscode_command()
        full_cmd += send_message_command(union) + cmd

        job = KubernetesJob(
            name=job_name,
            cpu_request=cpu_request,
            ram_request=ram_request,
            image=docker_image,
            gpu_type="nvidia.com/gpu",
            gpu_limit=gpu_limit,
            gpu_product=gpu_product.value,
            backoff_limit=0,
            command=["/bin/bash", "-c", "--"],
            args=[full_cmd],
            env_vars=env_vars_dict,
            secret_env_vars=secrets_env_vars_dict,
            user_email=email,
            namespace=namespace,
            kueue_queue_name=queue_name,
            volume_mounts={
                "nfs": {"mountPath": "/nfs", "server": nfs_server, "path": "/"}
            },
            priority=priority,
        )
        job_yaml = job.generate_yaml()
        logger.info(job_yaml)
        # Run the Job on the Kubernetes cluster
        if not dry_run:
            job.run()
    else:
        logger.info(f"Job '{job_name}' is still running.")


def cli():
    """Entry point for the application"""
    app()


if __name__ == "__main__":
    cli()
