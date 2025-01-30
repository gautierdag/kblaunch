import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional

import requests
import typer
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from loguru import logger

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


def delete_namespaced_job_safely(
    job_name: str,
    namespace: str = "informatics",
    user: Optional[str] = None,
) -> bool:
    """
    Delete a namespaced job if it exists and the user owns it.

    Args:
        job_name: Name of the job to delete
        namespace: Kubernetes namespace
        user: Username to verify ownership (if None, no ownership check)

    Returns:
        bool: True if job was deleted, False otherwise
    """
    try:
        api = client.BatchV1Api()
        job = api.read_namespaced_job(name=job_name, namespace=namespace)

        # Check ownership if user is provided
        if user is not None:
            job_user = job.metadata.labels.get("eidf/user")
            if job_user != user:
                logger.error(
                    f"Job '{job_name}' belongs to user '{job_user}', not '{user}'"
                )
                return False

        # Delete the job
        api.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        logger.info(f"Job '{job_name}' deleted successfully")
        return True

    except ApiException as e:
        if e.status == 404:
            logger.warning(f"Job '{job_name}' not found")
            return False
        else:
            logger.error(f"Error deleting job: {e}")
            return False


def read_startup_script(script_path: str) -> str:
    """Read and validate startup script."""
    try:
        script_path = Path(script_path).resolve()
        if not script_path.exists():
            raise typer.BadParameter(f"Startup script not found: {script_path}")
        if not script_path.is_file():
            raise typer.BadParameter(f"Not a file: {script_path}")
        logger.info(f"Using startup script: {script_path}")
        return script_path.read_text()
    except Exception as e:
        raise typer.BadParameter(f"Error reading startup script: {e}")


def create_git_secret(
    secret_name: str,
    private_key_path: str,
    namespace: str = "informatics",
) -> bool:
    """
    Create a Kubernetes secret containing SSH private key for Git authentication.

    Args:
        secret_name: Name of the secret
        private_key_path: Path to SSH private key file
        namespace: Kubernetes namespace

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(private_key_path, "r") as f:
            private_key = f.read()

        # Load the kube config
        config.load_kube_config()
        api = client.CoreV1Api()

        # Create the secret
        secret = client.V1Secret(
            metadata=client.V1ObjectMeta(name=secret_name),
            string_data={"ssh-privatekey": private_key},
            type="kubernetes.io/ssh-auth",
        )

        try:
            api.create_namespaced_secret(namespace=namespace, body=secret)
            logger.info(f"Secret '{secret_name}' created successfully")
            return True
        except ApiException as e:
            if e.status == 409:  # Secret already exists
                if typer.confirm(
                    f"Secret '{secret_name}' already exists. Replace it?",
                    default=False,
                ):
                    api.patch_namespaced_secret(
                        name=secret_name, namespace=namespace, body=secret
                    )
                    logger.info(f"Secret '{secret_name}' updated successfully")
                    return True
            else:
                logger.error(f"Error creating secret: {e}")
            return False

    except Exception as e:
        logger.error(f"Error creating Git secret: {e}")
        return False


def setup_git_command() -> str:
    """Generate command to setup Git with SSH key."""
    return (
        """apt-get update && apt-get install -y git openssh-client && """
        """mkdir -p ~/.ssh && """
        """cp /etc/ssh-key/ssh-privatekey ~/.ssh/id_rsa && """
        """chmod 600 ~/.ssh/id_rsa && """
        """ssh-keyscan github.com >> ~/.ssh/known_hosts && """
        """git config --global core.sshCommand 'ssh -i ~/.ssh/id_rsa' && """
        """git config --global user.name "${USER}" && """
        """git config --global user.email "${GIT_EMAIL}" && """
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
        gpu_type: Optional[str] = None,
        gpu_product: Optional[str] = None,
        gpu_limit: Optional[int] = None,
        env_vars: Optional[dict] = None,
        secret_env_vars: Optional[dict] = None,
        nfs_server: str = NFS_SERVER,
        pvc_name: Optional[str] = None,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        namespace: Optional[str] = None,
        priority: str = "default",
        startup_script: Optional[str] = None,
        git_secret: Optional[str] = None,
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

        self.volume_mounts = [
            {"name": "workspace", "mountPath": "/workspace", "readOnly": True},
            {"name": "publicdata", "mountPath": "/public", "readOnly": True},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]
        if pvc_name is not None:
            self.volume_mounts.append({"name": "writeable", "mountPath": "/pvc"})

        USER = os.getenv("USER", "unknown")
        self.volumes = [
            {
                "name": "workspace",
                "nfs": {"path": f"/user/{USER}", "server": nfs_server},
            },
            {
                "name": "publicdata",
                "nfs": {"path": "/public", "server": nfs_server},
            },
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        ]
        if pvc_name is not None:
            self.volumes.append(
                {"name": "writeable", "persistentVolumeClaim": {"claimName": pvc_name}}
            )

        self.env_vars = env_vars
        self.secret_env_vars = secret_env_vars

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

        self.startup_script = startup_script
        if startup_script:
            self.volume_mounts.append(
                {
                    "name": "startup-script",
                    "mountPath": "/startup.sh",
                    "subPath": "startup.sh",
                }
            )
            self.volumes.append(
                {
                    "name": "startup-script",
                    "configMap": {
                        "name": f"{self.name}-startup",
                        "defaultMode": 0o755,  # Make script executable
                    },
                }
            )

        self.git_secret = git_secret
        if git_secret:
            self.volume_mounts.append(
                {
                    "name": "git-ssh",
                    "mountPath": "/etc/ssh-key",
                    "readOnly": True,
                }
            )
            self.volumes.append(
                {
                    "name": "git-ssh",
                    "secret": {
                        "secretName": git_secret,
                        "defaultMode": 0o600,
                    },
                }
            )

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

        container = self._add_env_vars(container)
        container["volumeMounts"] = self.volume_mounts

        if self.cpu_request is not None or self.ram_request is not None:
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
                        "restartPolicy": "Never",
                        "volumes": [],
                    },
                },
                "backoffLimit": 0,
            },
        }

        if self.namespace:
            job["metadata"]["namespace"] = self.namespace

        if not (
            self.gpu_type is None or self.gpu_limit is None or self.gpu_product is None
        ):
            job["spec"]["template"]["spec"]["nodeSelector"] = {
                f"{self.gpu_type}.product": self.gpu_product
            }

        job["spec"]["template"]["spec"]["volumes"] = self.volumes
        return yaml.dump(job)

    def run(self):
        """Create or update the job using the Kubernetes API."""
        config.load_kube_config()
        api = client.BatchV1Api()

        # Convert YAML to dict
        job_dict = yaml.safe_load(self.generate_yaml())

        # log the job yaml
        logger.info(yaml.dump(job_dict))

        try:
            # Try to create the job
            api.create_namespaced_job(
                namespace=self.namespace or "default", body=job_dict
            )
            logger.info(f"Job '{self.name}' created successfully")
            return 0
        except ApiException as e:
            if e.status == 409:  # Conflict - job already exists
                if not typer.confirm(
                    f"Job '{self.name}' already exists. Do you want to delete it and create a new one?",
                    default=False,
                ):
                    logger.info("Operation cancelled by user")
                    return 1

                # Try to delete and recreate
                if delete_namespaced_job_safely(
                    self.name, self.namespace or "default", self.user_name
                ):
                    try:
                        api.create_namespaced_job(
                            namespace=self.namespace or "default", body=job_dict
                        )
                        logger.info(f"Job '{self.name}' recreated successfully")
                        return 0
                    except ApiException as e:
                        logger.error(f"Failed to recreate job: {e}")
                        return 1
                return 1
            else:
                logger.error(f"Failed to create job: {e}")
                return 1
        except Exception as e:
            logger.exception(f"Unexpected error creating job: {e}")
            return 1


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
            delete_namespaced_job_safely(job_name, namespace)
    return is_completed


def send_message_command(env_vars: set) -> str:
    """
    Send a message to Slack when the job starts if the SLACK_WEBHOOK environment variable is set.
    """
    if "SLACK_WEBHOOK" not in env_vars:
        logger.debug("SLACK_WEBHOOK not found in env_vars.")
        return ""

    message_json = json.dumps(
        {
            "text": "Job started in ${POD_NAME}. To connect to the pod, run ```kubectl exec -it ${POD_NAME} -- /bin/bash```"
        }
    )
    # escape double quotes, backticks, and newlines
    message_json = (
        message_json.replace('"', '\\"').replace("`", "\\`").replace("\n", "\\n")
    )
    return (
        """apt-get update && apt-get install -y curl;"""  # Install the curl command
        + f"""curl -X POST -H 'Content-type: application/json' --data "{message_json}" $SLACK_WEBHOOK ;"""
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


def check_if_pvc_exists(pvc_name: str, namespace: str = "informatics") -> bool:
    """
    Check if a Persistent Volume Claim (PVC) exists in the specified namespace.
    """
    # Load the kube config
    config.load_kube_config()
    # Create an instance of the API class
    api = client.CoreV1Api()
    pvc_exists = False
    # Check if the PVC exists in the specified namespace
    pvcs = api.list_namespaced_persistent_volume_claim(namespace)
    if pvc_name in {pvc.metadata.name for pvc in pvcs.items}:
        pvc_exists = True
    return pvc_exists


def validate_storage(storage: str) -> bool:
    """
    Validate storage string format (e.g., 10Gi, 100Mi, 1Ti).

    Args:
        storage: String representing storage size (e.g., "10Gi")

    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    pattern = r"^([0-9]+)(Mi|Gi|Ti)$"
    match = re.match(pattern, storage)

    if not match:
        raise ValueError(
            "Invalid storage format. Must be a number followed by Mi, Gi, or Ti (e.g., 10Gi)"
        )

    size = int(match.group(1))
    unit = match.group(2)

    # Add some reasonable limits
    max_sizes = {
        "Mi": 1024 * 1024,  # 1 TiB in MiB
        "Gi": 1024,  # 1 TiB in GiB
        "Ti": 1,  # 1 TiB
    }

    if size <= 0 or size > max_sizes[unit]:
        raise ValueError(f"Storage size must be between 1 and {max_sizes[unit]}{unit}")

    return True


def create_pvc(
    user: str,
    pvc_name: str,
    storage: str,
    namespace: str = "informatics",
    storage_class: str = "csi-rbd-sc",
) -> bool:
    """
    Create a Persistent Volume Claim.

    Args:
        user: Username for labeling
        pvc_name: Name of the PVC
        storage: Storage size (e.g., "10Gi")
        namespace: Kubernetes namespace
        storage_class: Storage class name

    Returns:
        bool: True if successful, False otherwise
    """
    # Validate storage format
    validate_storage(storage)

    # Load the kube config
    config.load_kube_config()

    # Create an instance of the API class
    api = client.CoreV1Api()

    # Define the PVC
    pvc = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(
            name=pvc_name, namespace=namespace, labels={"eidf/user": user}
        ),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(requests={"storage": storage}),
            storage_class_name=storage_class,
        ),
    )
    try:
        # Create the PVC
        api.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
        logger.info(f"PVC '{pvc_name}' created successfully")
        return True

    except ApiException as e:
        if e.status == 409:  # Conflict - PVC already exists
            logger.warning(f"PVC '{pvc_name}' already exists")
            return False
        else:
            logger.error(f"Error creating PVC: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error creating PVC: {e}")
        raise


@app.command()
def setup():
    """Interactive setup for kblaunch configuration."""
    config = load_config()

    # validate user
    default_user = os.getenv("USER")
    if "user" in config:
        default_user = config["user"]

    if typer.confirm(
        f"Would you like to set the user? (default: {default_user})", default=False
    ):
        user = typer.prompt("Please enter your user", default=default_user)
        config["user"] = user

    # Get email
    existing_email = config.get("email", None)
    email = typer.prompt(
        f"Please enter your email (existing: {existing_email})", default=existing_email
    )
    config["email"] = email

    # Get Slack webhook
    if typer.confirm("Would you like to set up Slack notifications?", default=False):
        existing_webhook = config.get("slack_webhook", None)
        webhook = typer.prompt(
            f"Enter your Slack webhook URL (existing: {existing_webhook})",
            default=existing_webhook,
        )
        config["slack_webhook"] = webhook

    if typer.confirm("Would you like to create a PVC?", default=False):
        user = config["user"]
        current_default = config.get("default_pvc", f"{user}-pvc")

        pvc_name = typer.prompt(
            f"Enter the desired PVC name (default: {current_default})",
            default=current_default,
        )

        # Check if PVC exists
        if check_if_pvc_exists(pvc_name):
            logger.warning(f"PVC '{pvc_name}' already exists")
        else:
            pvc_size = typer.prompt(
                "Enter the desired PVC size (e.g. 10Gi)", default="10Gi"
            )
            try:
                if create_pvc(user, pvc_name, pvc_size):
                    config["pvc_name"] = pvc_name
            except (ValueError, ApiException) as e:
                logger.error(f"Failed to create PVC: {e}")

        use_default = typer.confirm(
            f"Would you like set {pvc_name} as the default PVC? "
            f"Note that only one pod can use the PVC at a time. "
            f"The current default is {current_default}",
            default=True,
        )
        if use_default:
            config["default_pvc"] = pvc_name

    # Git authentication setup
    if typer.confirm("Would you like to set up Git SSH authentication?", default=False):
        default_key_path = str(Path.home() / ".ssh" / "id_rsa")
        key_path = typer.prompt(
            "Enter the path to your SSH private key",
            default=default_key_path,
        )
        secret_name = f"{config['user']}-git-ssh"
        if create_git_secret(secret_name, key_path):
            config["git_secret"] = secret_name

    # validate slack webhook
    if "slack_webhook" in config:
        # test post to slack
        try:
            logger.info("Sending test message to Slack")
            message = "Hello :wave: from ```kblaunch```"
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
    pvc_name: str = typer.Option(None, help="Persistent Volume Claim name"),
    dry_run: bool = typer.Option(False, help="Dry run"),
    priority: str = typer.Option("default", help="Priority class name"),
    vscode: bool = typer.Option(False, help="Install VS Code CLI in the container"),
    startup_script: str = typer.Option(
        None, help="Path to startup script to run in container"
    ),
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

    if "user" in config and os.getenv("USER") is None:
        os.environ["USER"] = config["user"]

    if pvc_name is None:
        pvc_name = config.get("default_pvc")

    if pvc_name is not None:
        if not check_if_pvc_exists(pvc_name):
            logger.error(f"Provided PVC '{pvc_name}' does not exist")
            return

    # Add validation for command parameter
    if not interactive and command == "":
        raise typer.BadParameter("--command is required when not in interactive mode")

    # Validate GPU constraints before creating job
    try:
        validate_gpu_constraints(gpu_product.value, gpu_limit, priority)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    is_completed = check_if_completed(job_name, namespace=namespace)
    if not is_completed:
        if typer.confirm(
            f"Job '{job_name}' already exists. Do you want to delete it and create a new one?",
            default=False,
        ):
            if not delete_namespaced_job_safely(
                job_name,
                namespace=namespace,
                user=config.get("user"),
            ):
                logger.error("Failed to delete existing job")
                return 1
        else:
            logger.info("Operation cancelled by user")
            return 1

    logger.info(f"Job '{job_name}' is completed. Launching a new job.")

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

    # Add USER and GIT_EMAIL to env_vars if git_secret is configured
    if config.get("git_secret"):
        env_vars_dict["USER"] = config.get("user", os.getenv("USER", "unknown"))
        env_vars_dict["GIT_EMAIL"] = email

    secrets_env_vars_dict = get_secret_env_vars(
        secrets_names=secrets_env_vars,
        namespace=namespace,
    )

    # Check for overlapping keys in local and secret environment variables
    intersection = set(secrets_env_vars_dict.keys()).intersection(env_vars_dict.keys())
    if intersection:
        logger.warning(
            f"Overlapping keys in local and secret environment variables: {intersection}"
        )
    # Combine the environment variables
    union = set(secrets_env_vars_dict.keys()).union(env_vars_dict.keys())

    # Handle startup script
    script_content = None
    if startup_script:
        script_content = read_startup_script(startup_script)
        # Create ConfigMap for startup script
        try:
            api = client.CoreV1Api()
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=f"{job_name}-startup", namespace=namespace
                ),
                data={"startup.sh": script_content},
            )
            try:
                api.create_namespaced_config_map(namespace=namespace, body=config_map)
            except ApiException as e:
                if e.status == 409:  # Already exists
                    api.patch_namespaced_config_map(
                        name=f"{job_name}-startup", namespace=namespace, body=config_map
                    )
                else:
                    raise
        except Exception as e:
            raise typer.BadParameter(f"Failed to create startup script ConfigMap: {e}")

    logger.info(f"Creating job for: {cmd}")

    # Modify command to include startup script
    if script_content:
        cmd = f"bash /startup.sh && {cmd}"

    # Build the full command with optional VS Code installation
    full_cmd = ""
    if config.get("git_secret"):
        full_cmd += setup_git_command()
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
        command=["/bin/bash", "-c", "--"],
        args=[full_cmd],
        env_vars=env_vars_dict,
        secret_env_vars=secrets_env_vars_dict,
        user_email=email,
        namespace=namespace,
        kueue_queue_name=queue_name,
        nfs_server=nfs_server,
        pvc_name=pvc_name,
        priority=priority,
        startup_script=script_content,
        git_secret=config.get("git_secret"),
    )
    job_yaml = job.generate_yaml()
    logger.info(job_yaml)
    # Run the Job on the Kubernetes cluster
    if not dry_run:
        job.run()


def cli():
    """Entry point for the application"""
    app()


if __name__ == "__main__":
    cli()
