import importlib.metadata
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
from typing_extensions import Annotated

from kblaunch.bash_utils import (
    install_vscode_command,
    send_message_command,
    setup_git_command,
    start_vscode_tunnel_command,
)
from kblaunch.plots import (
    print_gpu_total,
    print_job_stats,
    print_queue_stats,
    print_user_stats,
)

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


def get_current_namespace(config: Optional[dict] = None) -> Optional[str]:
    """Get the current namespace from environment variable only, no subprocess."""
    if config is None:
        config = load_config()
        if "namespace" in config:
            return config["namespace"]
    return os.getenv("KUBE_NAMESPACE")


def get_user_queue(namespace: Optional[str] = None) -> Optional[str]:
    """Get the user queue name based on namespace."""
    # First check if KUBE_USER_QUEUE is set
    queue = os.getenv("KUBE_USER_QUEUE")
    if queue:
        return queue
    # Otherwise construct from namespace if available
    if namespace:
        return f"{namespace}-user-queue"
    return None


class GPU_PRODUCTS(str, Enum):
    a100_80gb = "NVIDIA-A100-SXM4-80GB"
    a100_40gb = "NVIDIA-A100-SXM4-40GB"
    a100_40gb_mig_3g_20gb = "NVIDIA-A100-SXM4-40GB-MIG-3g.20gb"
    a100_40gb_mig_1g_5gb = "NVIDIA-A100-SXM4-40GB-MIG-1g.5gb"
    h100_80gb_hbm3 = "NVIDIA-H100-80GB-HBM3"
    h200 = "NVIDIA-H200"


class PRIORITY(str, Enum):
    default = "default"
    batch = "batch"
    short = "short"


PRIORITY_MAPPING = {
    "default": "default-workload-priority",
    "batch": "batch-workload-priority",
    "short": "short-workload-high-priority",
}

# Get NFS server from environment or use default
NFS_SERVER = os.getenv("INFK8S_NFS_SERVER_IP", None)

app = typer.Typer()


def is_mig_gpu(gpu_product: str) -> bool:
    """Check if the GPU product is a MIG instance."""
    return "MIG" in gpu_product


def validate_gpu_constraints(gpu_product: str, gpu_limit: int, priority: str):
    """Validate GPU constraints for MIG and H100 instances."""
    # Skip validation for non-GPU jobs
    if gpu_limit == 0:
        return

    # Check MIG constraint
    if is_mig_gpu(gpu_product) and gpu_limit > 1:
        raise ValueError("Cannot request more than one MIG instance in a single job")

    # Check H100 priority constraint
    if ("H100" in gpu_product or gpu_limit > 1) and priority == "short":
        raise ValueError(
            "Cannot request H100 GPUs or multiple GPUs in the short-workload-high-priority class"
        )


def validate_ram_request(ram_request: str) -> bool:
    """Validate RAM request format (e.g., 8Gi, 16Gi, 32Gi)."""
    pattern = r"^([0-9]+)(Gi)$"
    match = re.match(pattern, ram_request)
    if not match:
        raise ValueError(
            "Invalid RAM request format. Must be a number followed by Gi (e.g., 8Gi)"
        )
    size = int(match.group(1))
    if size <= 0 or size > MAX_RAM:
        raise ValueError(f"RAM request must be between 1 and {MAX_RAM}Gi")
    return True


def delete_namespaced_job_safely(
    job_name: str,
    namespace: str,
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
    namespace: str,
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
        nfs_server: Optional[str] = None,
        pvc_name: Optional[str] = None,
        pvcs: Optional[List[dict]] = None,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        namespace: Optional[str] = None,
        priority: str = "default",
        startup_script: Optional[str] = None,
        git_secret: Optional[str] = None,
    ):
        # Validate gpu_limit first
        assert gpu_limit is not None, "gpu_limit must be specified"
        assert 0 <= gpu_limit <= MAX_GPU, (
            f"gpu_limit must be between 0 and {MAX_GPU}, got {gpu_limit}"
        )

        self.name = name
        self.image = image
        self.command = command
        self.args = args
        self.gpu_limit = gpu_limit
        self.gpu_type = gpu_type
        self.gpu_product = gpu_product

        self.cpu_request = (
            cpu_request if cpu_request else (12 * gpu_limit if gpu_limit > 0 else 1)
        )
        self.ram_request = (
            ram_request
            if ram_request
            else f"{80 * gpu_limit if gpu_limit > 0 else 8}Gi"
        )
        # Validate RAM request
        validate_ram_request(self.ram_request)

        assert int(self.cpu_request) <= MAX_CPU, (
            f"cpu_request must be less than {MAX_CPU}"
        )

        if gpu_limit > 0:
            ram_per_gpu = 80
            cpu_per_gpu = 12
            if int(self.cpu_request) < (cpu_per_gpu * gpu_limit) * 0.5:
                logger.warning(f"Low number of cpus detected ({self.cpu_request}).")
                logger.warning(
                    f"Note the recommended CPU request for {gpu_limit} GPUs is {cpu_per_gpu * gpu_limit}."
                )
            elif int(self.cpu_request) > (cpu_per_gpu * gpu_limit) * 1.5:
                logger.warning(f"High number of cpus detected ({self.cpu_request}).")
                logger.warning(
                    f"Note the recommended CPU request for {gpu_limit} GPUs is {cpu_per_gpu * gpu_limit}."
                )
            if int(self.ram_request.rstrip("Gi")) < (ram_per_gpu * gpu_limit) * 0.5:
                logger.warning(f"Low amount of RAM detected ({self.ram_request}).")
                logger.warning(
                    f"Note the recommended RAM request for {gpu_limit} GPUs is {ram_per_gpu * gpu_limit}Gi."
                )
            elif int(self.ram_request.rstrip("Gi")) > (ram_per_gpu * gpu_limit) * 1.5:
                logger.warning(f"High amount of RAM detected ({self.ram_request}).")
                logger.warning(
                    f"Note the recommended RAM request for {gpu_limit} GPUs is {ram_per_gpu * gpu_limit}Gi."
                )

        self.volume_mounts = [
            {"name": "workspace", "mountPath": "/workspace", "readOnly": True},
            {"name": "publicdata", "mountPath": "/public", "readOnly": True},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]

        # Handle the legacy single PVC parameter
        if pvc_name is not None:
            self.volume_mounts.append({"name": "writeable", "mountPath": "/pvc"})

        # Handle multiple PVCs with customizable mount paths
        self.pvcs = pvcs or []
        for pvc in self.pvcs:
            # assert that pvcs name is not pvc_name
            if pvc["name"] == pvc_name:
                raise ValueError(
                    f"PVC name '{pvc['name']}' conflicts with the pvc_name parameter. You cannot mount the same PVC twice."
                )
            self.volume_mounts.append(
                {"name": f"pvc-{pvc['name']}", "mountPath": pvc["mount_path"]}
            )

        USER = os.getenv("USER", "unknown")
        self.volumes = [
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        ]
        if nfs_server is not None:
            self.volumes.append(
                {
                    "name": "workspace",
                    "nfs": {"path": f"/user/{USER}", "server": nfs_server},
                }
            )
            self.volumes.append(
                {
                    "name": "publicdata",
                    "nfs": {"path": "/public", "server": nfs_server},
                }
            )

        # Legacy single PVC support
        if pvc_name is not None:
            self.volumes.append(
                {"name": "writeable", "persistentVolumeClaim": {"claimName": pvc_name}}
            )

        # Add multiple PVCs to volumes
        for pvc in self.pvcs:
            self.volumes.append(
                {
                    "name": f"pvc-{pvc['name']}",
                    "persistentVolumeClaim": {"claimName": pvc["name"]},
                }
            )

        self.env_vars = env_vars
        self.secret_env_vars = secret_env_vars

        self.user_name = user_name or os.environ.get("USER", "unknown")
        self.user_email = user_email  # This is now a required field.
        self.kueue_queue_name = kueue_queue_name

        self.labels = {
            "eidf/user": self.user_name,
            "kueue.x-k8s.io/queue-name": self.kueue_queue_name,
            "kueue.x-k8s.io/priority-class": PRIORITY_MAPPING.get(
                priority, "default-workload-priority"
            ),
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

        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "generateName": self.name,
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

        # Only add GPU configuration if gpu_limit > 0
        if self.gpu_limit > 0 and not (
            self.gpu_type is None or self.gpu_product is None
        ):
            job["spec"]["template"]["spec"]["nodeSelector"] = {
                f"{self.gpu_type}.product": self.gpu_product
            }
            job["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"][
                f"{self.gpu_type}"
            ] = self.gpu_limit

        if self.namespace:
            job["metadata"]["namespace"] = self.namespace

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
                logger.info(f"Job '{self.name}' already exists")
                return 1
            else:
                logger.error(f"Failed to create job: {e}")
                return 1
        except Exception as e:
            logger.exception(f"Unexpected error creating job: {e}")
            return 1


def check_if_completed(job_name: str, namespace: str) -> bool:
    # Load the kube config
    config.load_kube_config()

    # Create an instance of the API class
    api = client.BatchV1Api()

    is_completed = True

    # Check if the job exists in the specified namespace
    jobs = api.list_namespaced_job(namespace)

    if job_name in {job.metadata.name for job in jobs.items}:
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


def get_env_vars(
    local_env_vars: list[str],
    load_dotenv: bool = False,
) -> dict[str, str]:
    """Get environment variables from local environment and secrets."""

    if load_dotenv:
        try:
            from dotenv import load_dotenv as ld_dotenv

            path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(path):
                ld_dotenv(path)
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
    namespace: str,
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
            raise typer.BadParameter(f"Error reading secret {secret_name}: {e}")
    return secrets_env_vars


def check_if_pvc_exists(pvc_name: str, namespace: str) -> bool:
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
    namespace: str,
    storage_class: str = "csi-cephfs-sc",
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
            access_modes=["ReadWriteMany"],
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
    """
     `kblaunch setup`

    Interactive setup wizard for kblaunch configuration.
    No arguments - all configuration is done through interactive prompts.

    This command walks users through the initial setup process, configuring:
    - User identity and email
    - Namespace and queue settings
    - Slack notifications webhook
    - Persistent Volume Claims (PVC) for storage
    - Git SSH authentication
    - NFS server configuration

    The configuration is stored in ~/.cache/.kblaunch/config.json.

    Configuration includes:
    - User: Kubernetes username for job ownership
    - Email: User email for notifications and Git configuration
    - Namespace: Kubernetes namespace for job deployment
    - Queue: Kueue queue name for job scheduling
    - Slack webhook: URL for job status notifications
    - PVC: Persistent storage configuration
    - Git SSH: Authentication for private repositories
    - NFS: Server address for mounting storage
    """
    config = load_config()

    # validate user
    default_user = os.getenv("USER")
    if "user" in config:
        default_user = config["user"]
    else:
        config["user"] = default_user

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

    # Configure namespace
    existing_namespace = config.get("namespace", os.getenv("KUBE_NAMESPACE"))
    if typer.confirm("Would you like to configure your namespace?", default=True):
        namespace = typer.prompt(
            f"Please enter your namespace (existing: {existing_namespace})",
            default=existing_namespace,
        )
        config["namespace"] = namespace
        # Now that we have namespace, ask about queue
        existing_queue = config.get("queue", get_user_queue(namespace))
        if typer.confirm("Would you like to configure your queue?", default=True):
            queue = typer.prompt(
                f"Please enter your queue name (existing: {existing_queue})",
                default=existing_queue or f"{namespace}-user-queue",
            )
            config["queue"] = queue

    # Get NFS Server
    # Get the current NFS server from config or default
    current_nfs = config.get("nfs_server", NFS_SERVER)
    if typer.confirm("Would you like to configure the NFS server?", default=False):
        nfs_server = typer.prompt(
            f"Enter your NFS server address (existing: {current_nfs})",
            default=current_nfs,
        )
        config["nfs_server"] = nfs_server

    # Get Slack webhook
    if typer.confirm("Would you like to set up Slack notifications?", default=False):
        existing_webhook = config.get("slack_webhook", None)
        webhook = typer.prompt(
            f"Enter your Slack webhook URL (existing: {existing_webhook})",
            default=existing_webhook,
        )
        config["slack_webhook"] = webhook

    if typer.confirm("Would you like to use a PVC?", default=False):
        user = config["user"]
        current_default = config.get("default_pvc", f"{user}-pvc")

        pvc_name = typer.prompt(
            f"Enter the PVC name to use (default: {current_default}). We will help you create it if it does not exist.",
            default=current_default,
        )

        namespace = config.get("namespace", get_current_namespace(config))
        if check_if_pvc_exists(pvc_name, namespace):
            if typer.confirm(
                f"Would you like to set {pvc_name} as the default PVC?",
                default=True,
            ):
                config["default_pvc"] = pvc_name
        else:
            if typer.confirm(
                f"PVC '{pvc_name}' does not exist. Would you like to create it?",
                default=True,
            ):
                pvc_size = typer.prompt(
                    "Enter the desired PVC size (e.g. 10Gi)", default="10Gi"
                )
                try:
                    if create_pvc(user, pvc_name, pvc_size, namespace):
                        config["default_pvc"] = pvc_name
                except (ValueError, ApiException) as e:
                    logger.error(f"Failed to create PVC: {e}")

    # Git authentication setup
    if typer.confirm("Would you like to set up Git SSH authentication?", default=False):
        default_key_path = str(Path.home() / ".ssh" / "id_rsa")
        key_path = typer.prompt(
            "Enter the path to your SSH private key",
            default=default_key_path,
        )
        secret_name = f"{config['user']}-git-ssh"
        namespace = config.get("namespace", get_current_namespace(config))
        if create_git_secret(secret_name, key_path, namespace):
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
    namespace: str = typer.Option(
        None, help="Kubernetes namespace (defaults to KUBE_NAMESPACE)"
    ),
    queue_name: str = typer.Option(
        None, help="Kueue queue name (defaults to KUBE_USER_QUEUE)"
    ),
    interactive: bool = typer.Option(False, help="Run in interactive mode"),
    command: str = typer.Option(
        "", help="Command to run in the container"
    ),  # Made optional
    cpu_request: str = typer.Option("6", help="CPU request"),
    ram_request: str = typer.Option("40Gi", help="RAM request"),
    gpu_limit: int = typer.Option(1, help="GPU limit (0 for non-GPU jobs)"),
    gpu_product: GPU_PRODUCTS = typer.Option(
        "NVIDIA-A100-SXM4-40GB",
        help="GPU product type to use (ignored for non-GPU jobs)",
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
    nfs_server: Optional[str] = typer.Option(
        None, help="NFS server (overrides config and environment)"
    ),
    pvc_name: str = typer.Option(None, help="Persistent Volume Claim name"),
    pvcs: str = typer.Option(
        None,
        help='Multiple PVCs with mount paths in JSON format (e.g., \'[{"name":"my-pvc","mount_path":"/data"}]\')',
    ),
    dry_run: bool = typer.Option(False, help="Dry run"),
    priority: PRIORITY = typer.Option(
        "default", help="Priority class name", show_default=True, show_choices=True
    ),
    vscode: bool = typer.Option(False, help="Install VS Code CLI in the container"),
    tunnel: bool = typer.Option(
        False,
        help="Start a VS Code SSH tunnel on startup. Requires SLACK_WEBHOOK and --vscode",
    ),
    startup_script: str = typer.Option(
        None, help="Path to startup script to run in container"
    ),
):
    """
    `kblaunch launch`
    Launch a Kubernetes job with specified configuration.

    This command creates and deploys a Kubernetes job with the given specifications,
    handling GPU allocation, resource requests, and environment setup.

    Args:
    * email (str, optional): User email for notifications
    * job_name (str, required): Name of the Kubernetes job
    * docker_image (str, default="nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04"): Container image
    * namespace (str, default="KUBE_NAMESPACE"): Kubernetes namespace
    * queue_name (str, default="KUBE_USER_QUEUE"): Kueue queue name
    * interactive (bool, default=False): Run in interactive mode
    * command (str, default=""): Command to run in container
    * cpu_request (str, default="6"): CPU cores request
    * ram_request (str, default="40Gi"): RAM request
    * gpu_limit (int, default=1): Number of GPUs
    * gpu_product (GPU_PRODUCTS, default="NVIDIA-A100-SXM4-40GB"): GPU type
    * secrets_env_vars (List[str], default=[]): Secret environment variables
    * local_env_vars (List[str], default=[]): Local environment variables
    * load_dotenv (bool, default=True): Load .env file
    * nfs_server (str, optional): NFS server IP (overrides config)
    * pvc_name (str, optional): PVC name for single PVC mounting at /pvc
    * pvcs (str, optional): Multiple PVCs with mount paths in JSON format (used for mounting multiple PVCs)
    * dry_run (bool, default=False): Print YAML only
    * priority (PRIORITY, default="default"): Job priority
    * vscode (bool, default=False): Install VS Code
    * tunnel (bool, default=False): Start VS Code tunnel
    * startup_script (str, optional): Path to startup script

    Examples:
        ```bash
        # Launch an interactive GPU job
        kblaunch launch --job-name test-job --interactive

        # Launch a batch GPU job with custom command
        kblaunch launch --job-name batch-job --command "python train.py"

        # Launch a CPU-only job
        kblaunch launch --job-name cpu-job --gpu-limit 0

        # Launch with VS Code support
        kblaunch launch --job-name dev-job --interactive --vscode --tunnel

        # Launch with multiple PVCs
        kblaunch launch --job-name multi-pvc-job --pvcs '[{"name":"data-pvc","mount_path":"/data"},{"name":"models-pvc","mount_path":"/models"}]'
        ```

    Notes:
    - Interactive jobs keep running until manually terminated
    - GPU jobs require appropriate queue and priority settings
    - VS Code tunnel requires Slack webhook configuration
    - Multiple PVCs can be mounted with custom paths using the --pvcs option
    """

    # Load config
    config = load_config()

    # Determine namespace if not provided
    if namespace is None:
        namespace = get_current_namespace(config)
        if namespace is None:
            raise typer.BadParameter(
                "Namespace not provided.",
                "Please provide --namespace or run 'kblaunch setup' to configure.",
            )

    # Determine queue name if not provided
    if queue_name is None:
        queue_name = get_user_queue(namespace)
        if queue_name is None:
            raise typer.BadParameter(
                "Queue name not provided.",
                "Please provide --queue-name or run 'kblaunch setup' to configure.",
            )

    # Use email from config if not provided
    if email is None:
        email = config.get("email")
        if email is None:
            raise typer.BadParameter(
                "Email not provided and not found in config. "
                "Please provide --email or run 'kblaunch setup' to configure."
            )

    # Determine which NFS server to use (priority: command-line > config > env var > default)
    if nfs_server is None:
        nfs_server = config.get("nfs_server", NFS_SERVER)
        if nfs_server is None:
            # warn if NFS server is not set
            logger.warning(
                "NFS server not set/found. Please provide --nfs-server or run 'kblaunch setup' mount the NFS partition."
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
        if not check_if_pvc_exists(pvc_name, namespace):
            raise typer.BadParameter(
                f"PVC '{pvc_name}' does not exist in namespace '{namespace}'"
            )

    # Parse multiple PVCs if provided
    parsed_pvcs = []
    if pvcs:
        try:
            parsed_pvcs = json.loads(pvcs)
            # Validate the format
            for pvc in parsed_pvcs:
                if (
                    not isinstance(pvc, dict)
                    or "name" not in pvc
                    or "mount_path" not in pvc
                ):
                    raise typer.BadParameter(
                        "Each PVC entry must be a dictionary with 'name' and 'mount_path' keys"
                    )
                # Validate that the PVC exists
                if not check_if_pvc_exists(pvc["name"], namespace):
                    raise typer.BadParameter(
                        f"PVC '{pvc['name']}' does not exist in namespace '{namespace}'"
                    )

        except json.JSONDecodeError:
            raise typer.BadParameter("Invalid JSON format for pvcs parameter")

    # Add validation for command parameter
    if not interactive and command == "":
        raise typer.BadParameter("--command is required when not in interactive mode")

    # Validate GPU constraints only if requesting GPUs
    if gpu_limit > 0:
        try:
            validate_gpu_constraints(gpu_product.value, gpu_limit, priority.value)
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

    if interactive:
        cmd = "while true; do sleep 60; done;"
    else:
        cmd = command
        logger.info(f"Command: {cmd}")

    logger.info(f"Creating job for: {cmd}")

    # Modify command to include startup script
    if script_content:
        cmd = f"bash /startup.sh && {cmd}"

    # Build the start command with optional VS Code installation
    start_command = send_message_command(union)
    if config.get("git_secret"):
        start_command += setup_git_command()
    if vscode:
        start_command += install_vscode_command()
        if tunnel:
            start_command += start_vscode_tunnel_command(union)
    elif tunnel:
        logger.error("Cannot start tunnel without VS Code installation")

    full_cmd = start_command + cmd

    job = KubernetesJob(
        name=job_name,
        cpu_request=cpu_request,
        ram_request=ram_request,
        image=docker_image,
        gpu_type="nvidia.com/gpu" if gpu_limit > 0 else None,
        gpu_limit=gpu_limit,
        gpu_product=gpu_product.value if gpu_limit > 0 else None,
        command=["/bin/bash", "-c", "--"],
        args=[full_cmd],
        env_vars=env_vars_dict,
        secret_env_vars=secrets_env_vars_dict,
        user_email=email,
        namespace=namespace,
        kueue_queue_name=queue_name,
        nfs_server=nfs_server,
        pvc_name=pvc_name,
        pvcs=parsed_pvcs,
        priority=priority.value,
        startup_script=script_content,
        git_secret=config.get("git_secret"),
    )
    job_yaml = job.generate_yaml()
    logger.info(job_yaml)
    # Run the Job on the Kubernetes cluster
    if not dry_run:
        job.run()


@app.command(name="create-pvc")
def create_pvc_command(
    pvc_name: str = typer.Option(..., help="Name of the PVC to create"),
    storage: str = typer.Option("10Gi", help="Storage size (e.g., 10Gi, 100Mi, 1Ti)"),
    namespace: str = typer.Option(
        None, help="Kubernetes namespace (defaults to configured namespace)"
    ),
    storage_class: str = typer.Option(
        "csi-cephfs-sc", help="Storage class name to use"
    ),
):
    """
    `kblaunch create-pvc`
    Simple command to create a new Persistent Volume Claim (PVC).

    Creates a PVC with the specified name, size, and storage class in the
    specified namespace. The PVC will be labeled with the current user.

    Args:
    * pvc_name (str, required): Name of the PVC to create
    * storage (str, default="10Gi"): Storage size (e.g., 10Gi, 100Mi, 1Ti)
    * namespace (str, optional): Kubernetes namespace
    * storage_class (str, default="csi-cephfs-sc"): Storage class name

    Examples:
        ```bash
        # Create a standard 10Gi PVC
        kblaunch create-pvc --pvc-name my-data-pvc

        # Create a larger PVC with custom storage
        kblaunch create-pvc --pvc-name my-big-pvc --storage 50Gi

        # Create PVC in a specific namespace with custom storage class
        kblaunch create-pvc --pvc-name models-pvc --namespace ml-team --storage-class nfs-sc
        ```
    """
    # Load config
    config = load_config()

    # Get user from config or environment
    user = config.get("user", os.getenv("USER", "unknown"))

    # Determine namespace if not provided
    if namespace is None:
        namespace = get_current_namespace(config)
        if namespace is None:
            raise typer.BadParameter(
                "Namespace not provided and not found in config. "
                "Please provide --namespace or run 'kblaunch setup'"
            )

    # Check if PVC already exists
    if check_if_pvc_exists(pvc_name, namespace):
        logger.warning(f"PVC '{pvc_name}' already exists in namespace '{namespace}'")
        return 0

    # Create the PVC
    try:
        logger.info(f"Creating PVC '{pvc_name}' in namespace '{namespace}'")
        logger.info(f"Storage: {storage}, Storage Class: {storage_class}")
        if create_pvc(user, pvc_name, storage, namespace, storage_class):
            logger.info(f"PVC '{pvc_name}' created successfully")
            # Ask if user wants to set this as the default PVC
            if typer.confirm(
                f"Do you want to set '{pvc_name}' as your default PVC?", default=True
            ):
                config["default_pvc"] = pvc_name
                save_config(config)
                logger.info(f"PVC '{pvc_name}' set as default in config")
            return 0
        else:
            logger.error(f"Failed to create PVC '{pvc_name}'")
            return 1
    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


monitor_app = typer.Typer()
app.add_typer(monitor_app, name="monitor", help="Monitor Kubernetes resources")


@monitor_app.command("gpus")
def monitor_gpus(
    namespace: str = typer.Option(
        None, help="Kubernetes namespace (defaults to KUBE_NAMESPACE)"
    ),
):
    """
    `kblaunch monitor gpus`
    Display overall GPU statistics and utilization by type.

    Shows a comprehensive view of GPU allocation and usage across the cluster,
    including both running and pending GPU requests.

    Args:
    - namespace: Kubernetes namespace to monitor (default: KUBE_NAMESPACE)

    Output includes:
    - Total GPU count by type
    - Running vs. pending GPUs
    - Details of pending GPU requests
    - Wait times for pending requests

    Examples:
        ```bash
        kblaunch monitor gpus
        kblaunch monitor gpus --namespace custom-namespace
        ```
    """
    try:
        namespace = namespace or get_current_namespace(config)
        print_gpu_total(namespace=namespace)
    except Exception as e:
        print(f"Error displaying GPU stats: {e}")


@monitor_app.command("users")
def monitor_users(
    namespace: str = typer.Option(
        None, help="Kubernetes namespace (defaults to KUBE_NAMESPACE)"
    ),
):
    """
    `kblaunch monitor users`
    Display GPU usage statistics grouped by user.

    Provides a user-centric view of GPU allocation and utilization,
    helping identify resource usage patterns across users.

    Args:
    - namespace: Kubernetes namespace to monitor (default: KUBE_NAMESPACE)

    Output includes:
    - GPUs allocated per user
    - Average memory usage per user
    - Inactive GPU count per user
    - Overall usage totals

    Examples:
        ```bash
        kblaunch monitor users
        kblaunch monitor users --namespace custom-namespace
        ```
    """
    try:
        namespace = namespace or get_current_namespace(config)
        print_user_stats(namespace=namespace)
    except Exception as e:
        print(f"Error displaying user stats: {e}")


@monitor_app.command("jobs")
def monitor_jobs(
    namespace: str = typer.Option(
        None, help="Kubernetes namespace (defaults to KUBE_NAMESPACE)"
    ),
):
    """
    `kblaunch monitor jobs`
    Display detailed job-level GPU statistics.

    Shows comprehensive information about all running GPU jobs,
    including resource usage and job characteristics.

    Args:
    - namespace: Kubernetes namespace to monitor (default: KUBE_NAMESPACE)

    Output includes:
    - Job identification and ownership
    - Resource allocation (CPU, RAM, GPU)
    - GPU memory usage
    - Job status (active/inactive)
    - Job mode (interactive/batch)
    - Resource totals and averages

    Examples:
        ```bash
        kblaunch monitor jobs
        kblaunch monitor jobs --namespace custom-namespace
        ```
    """
    try:
        namespace = namespace or get_current_namespace(config)
        print_job_stats(namespace=namespace)
    except Exception as e:
        print(f"Error displaying job stats: {e}")


@monitor_app.command("queue")
def monitor_queue(
    namespace: str = typer.Option(
        None, help="Kubernetes namespace (defaults to KUBE_NAMESPACE)"
    ),
    reasons: bool = typer.Option(False, help="Display queued job event messages"),
    include_cpu: bool = typer.Option(False, help="Show CPU jobs in the queue"),
):
    """
    `kblaunch monitor queue`
    Display statistics about queued workloads.

    Shows information about jobs waiting in the Kueue scheduler,
    including wait times and resource requests.

    Args:
    - namespace: Kubernetes namespace to monitor (default: KUBE_NAMESPACE)
    - reasons: Show detailed reason messages for queued jobs
    - include_cpu: Include CPU jobs in the queue

    Output includes:
    - Queue position and wait time
    - Resource requests (CPU, RAM, GPU)
    - Job priority
    - Queueing reasons (if --reasons flag is used)

    Examples:
        ```bash
        kblaunch monitor queue
        kblaunch monitor queue --reasons
        kblaunch monitor queue --namespace custom-namespace
        ```
    """
    try:
        namespace = namespace or get_current_namespace(config)
        print_queue_stats(namespace=namespace, reasons=reasons, include_cpu=include_cpu)
    except Exception as e:
        print(f"Error displaying queue stats: {e}")


def version_callback(value: bool):
    if value:
        typer.echo(importlib.metadata.version("kblaunch"))
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", callback=version_callback, help="Show version information"
        ),
    ] = None,
):
    """Entry point for the application"""
    pass  # The callback doesn't need to do anything else
