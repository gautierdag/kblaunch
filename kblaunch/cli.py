import base64
import os

import typer
from kubejobs.jobs import KubernetesJob, KueueQueue
from kubernetes import client, config
from loguru import logger

app = typer.Typer()


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


def send_message_command(env_vars: dict) -> str:
    """
    Send a message to Slack when the job starts if the SLACK_WEBHOOK environment variable is set.
    """
    if "SLACK_WEBHOOK" not in env_vars:
        logger.debug("SLACK_WEBHOOK not found in env_vars.")
        return ""
    webhook = env_vars["SLACK_WEBHOOK"]
    return (
        """curl -X POST -H 'Content-type: application/json' --data '{"text":"Job started in '"$POD_NAME"'"}' """
        + webhook
        + " ; "
    )


def get_env_vars(
    local_env_vars: list[str],
    secrets_env_vars: list[str],
    load_dotenv: bool = False,
    namespace: str = "informatics",
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

    for secret_name in secrets_env_vars:
        try:
            v1 = client.CoreV1Api()
            secret = v1.read_namespaced_secret(name=secret_name, namespace=namespace)
            for key, value in secret.data.items():
                decoded_value = base64.b64decode(value).decode("utf-8")
                if key in env_vars:
                    logger.warning(f"Key {key} already set in env_vars.")
                env_vars[key] = decoded_value
        except Exception as e:
            logger.warning(f"Error reading secret {secret_name}: {e}")

    return env_vars


def export_env_vars(env_vars: dict) -> str:
    """Export environment variables."""
    cmd = ""
    for key, value in env_vars.items():
        cmd += f" export {key}='{value}' &&"
    cmd = cmd.strip(" &&") + " ; "
    return cmd


@app.command()
def launch(
    email: str = typer.Option(..., help="User email"),
    job_name: str = typer.Option(..., help="Name of the Kubernetes job"),
    docker_image: str = typer.Option(
        "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04", help="Docker image"
    ),
    namespace: str = typer.Option("informatics", help="Kubernetes namespace"),
    queue_name: str = typer.Option(KueueQueue.INFORMATICS, help="Kueue queue name"),
    interactive: bool = typer.Option(False, help="Run in interactive mode"),
    command: str = typer.Option(..., help="Command to run in the container"),
    cpu_request: str = typer.Option("1", help="CPU request"),
    ram_request: str = typer.Option("8Gi", help="RAM request"),
    gpu_limit: int = typer.Option(1, help="GPU limit"),
    gpu_product: str = typer.Option("NVIDIA-A100-SXM4-80GB", help="GPU product"),
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
):
    """Launch a Kubernetes job with the specified configuration."""

    is_completed = check_if_completed(job_name, namespace=namespace)

    if is_completed is True:
        logger.info(f"Job '{job_name}' is completed. Launching a new job.")

        if interactive:
            cmd = "while true; do sleep 60; done;"
        else:
            cmd = command
            logger.info(f"Command: {cmd}")

        # Get local environment variables
        env_vars = get_env_vars(
            local_env_vars=local_env_vars,
            secrets_env_vars=secrets_env_vars,
            load_dotenv=load_dotenv,
            namespace=namespace,
        )

        logger.info(f"Creating job for: {cmd}")
        job = KubernetesJob(
            name=job_name,
            cpu_request=cpu_request,
            ram_request=ram_request,
            image=docker_image,
            gpu_type="nvidia.com/gpu",
            gpu_limit=gpu_limit,
            gpu_product=gpu_product,
            backoff_limit=0,
            command=["/bin/bash", "-c", "--"],
            args=[export_env_vars(env_vars) + send_message_command(env_vars) + cmd],
            user_email=email,
            namespace=namespace,
            kueue_queue_name=queue_name,
            volume_mounts={
                "nfs": {"mountPath": "/nfs", "server": "10.24.1.255", "path": "/"}
            },
        )
        job_yaml = job.generate_yaml()
        logger.info(job_yaml)
        # Run the Job on the Kubernetes cluster
        job.run()
    else:
        logger.info(f"Job '{job_name}' is still running.")


def cli():
    """Entry point for the application"""
    app()


if __name__ == "__main__":
    cli()
