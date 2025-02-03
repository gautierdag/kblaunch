import pandas as pd
from kubernetes import client, config, stream
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
import warnings
from datetime import datetime, timezone


def get_default_metrics() -> dict:
    """Return default metrics when actual metrics cannot be obtained."""
    return {
        "memory_used": 0,
        "memory_total": 80 * 1024,  # 80GB for A100
        "gpu_mem_used": 0,
        "inactive": True,
    }


def get_gpu_metrics(v1, pod_name: str, namespace: str, permission_errors: dict) -> dict:
    """Get GPU metrics from nvidia-smi for a specific pod."""
    try:
        # Command to get GPU memory usage from nvidia-smi
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]

        exec_stream = stream.stream(
            v1.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            command=["/bin/sh", "-c", " ".join(command)],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )

        # Parse nvidia-smi output
        output = exec_stream.strip()
        if not output or "[Insufficient Permissions]" in output:
            permission_errors["count"] += 1
            return get_default_metrics()

        # Split output into lines and get the first GPU's metrics
        first_gpu = output.split("\n")[0].strip()
        try:
            memory_used, memory_total = map(int, first_gpu.split(","))
            return {
                "memory_used": memory_used,
                "memory_total": memory_total,
                "gpu_mem_used": (memory_used / memory_total) * 100,
                "inactive": memory_used < (0.01 * memory_total),
            }
        except ValueError:
            logger.error(
                f"Failed to parse nvidia-smi output from pod {pod_name}: {output}"
            )
            return get_default_metrics()
    except Exception as e:
        if "permission" not in str(e).lower():
            logger.error(f"Failed to get GPU metrics for pod {pod_name}: {e}")
        else:
            permission_errors["count"] += 1
        return get_default_metrics()


def get_data(load_gpu_metrics=False, namespace="informatics") -> pd.DataFrame:
    """Get live GPU usage data from Kubernetes pods."""
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # Get pods with queue label filter
    pods = v1.list_namespaced_pod(
        namespace=namespace,
    )
    records = []

    # Filter running pods with GPUs first to get accurate total
    gpu_pods = [
        pod
        for pod in pods.items
        if pod.status.phase == "Running"
        and sum(
            int(c.resources.requests.get("nvidia.com/gpu", 0))
            for c in pod.spec.containers
        )
        > 0
    ]

    permission_errors = {"count": 0}  # Track permission errors

    # Create progress bars with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message='install "ipywidgets" for Jupyter support'
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=Console(),
            transient=True,
        ) as progress:
            collect_task = progress.add_task(
                "[cyan]Collecting pod info...", total=len(gpu_pods)
            )

            for pod in gpu_pods:
                #         # Get basic pod info
                namespace = pod.metadata.namespace
                pod_name = pod.metadata.name
                node_name = pod.spec.node_name
                username = pod.metadata.labels.get("eidf/user", "unknown")

                progress.update(
                    collect_task, advance=1, description=f"[cyan]Processing {pod_name}"
                )

                # Skip pods that aren't running
                if pod.status.phase != "Running":
                    continue

                # Skip pods without GPU requests
                gpu_requests = sum(
                    int(c.resources.requests.get("nvidia.com/gpu", 0))
                    for c in pod.spec.containers
                )
                if gpu_requests == 0:
                    continue

                gpu_name = pod.spec.node_selector["nvidia.com/gpu.product"]

                # Get resource requests
                container = pod.spec.containers[0]  # Assuming single container
                cpu_requested = int(float(container.resources.requests.get("cpu", "0")))
                memory_requested = int(
                    float(container.resources.requests.get("memory", "0").rstrip("Gi"))
                )

                # Check if pod is interactive
                container = pod.spec.containers[0]
                command = container.command if container.command else []
                args = container.args if container.args else []
                full_command = " ".join(command + args).lower()
                interactive_patterns = [
                    "sleep infinity",
                    "while true",
                    "tail -f /dev/null",
                    "sleep 60",
                ]
                is_interactive = any(
                    pattern in full_command for pattern in interactive_patterns
                )

                # Get GPU metrics
                gpu_metrics = get_default_metrics()
                if load_gpu_metrics:
                    gpu_metrics = get_gpu_metrics(
                        v1, pod_name, namespace, permission_errors
                    )

                # Create a record for each GPU assigned
                for gpu_id in range(gpu_requests):
                    record = {
                        "pod_name": pod_name,
                        "namespace": namespace,
                        "node_name": node_name,
                        "username": username,
                        "cpu_requested": cpu_requested,
                        "memory_requested": memory_requested,
                        "gpu_name": gpu_name,
                        "gpu_id": gpu_id,
                        "interactive": is_interactive,  # Add the interactive field
                        **gpu_metrics,
                    }
                    records.append(record)

    if permission_errors["count"] > 0:
        logger.info(
            f"Skipped GPU metrics for {permission_errors['count']} pods due to insufficient permissions"
        )

    # Create DataFrame
    df = pd.DataFrame(records)

    if len(df) == 0:
        # Return empty DataFrame with correct columns if no GPU pods found
        return pd.DataFrame(
            columns=[
                "pod_name",
                "namespace",
                "node_name",
                "username",
                "cpu_requested",
                "memory_requested",
                "gpu_name",
                "gpu_id",
                "memory_used",
                "memory_total",
                "gpu_mem_used",
                "inactive",
                "interactive",
            ]
        )

    # Calculate derived fields
    df["gpu_mem_used"] = (df["memory_used"] / df["memory_total"]) * 100
    df["inactive"] = df["gpu_mem_used"] < 1

    return df


def check_job_events_for_errors(api_instance, job_name: str, namespace: str) -> bool:
    """Check if a job has error events in Kubernetes."""
    try:
        # Get events related to the job
        events = api_instance.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={job_name}",
        )

        # Error event types and reasons to check
        error_types = {"Warning", "Error", "Failed"}
        error_reasons = {
            "FailedScheduling",
            "BackOff",
            "Failed",
            "Error",
            "FailedCreate",
            "FailedMount",
            "FailedValidation",
            "InvalidImageName",
            "ImagePullBackOff",
            "CreateContainerError",
        }

        for event in events.items:
            if event.type in error_types or event.reason in error_reasons:
                logger.debug(
                    f"Found error event for job {job_name}: "
                    f"type={event.type}, reason={event.reason}, "
                    f"message={event.message}"
                )
                return True

    except Exception as e:
        logger.debug(f"Error checking events for job {job_name}: {e}")

    return False


def check_job_events_for_queue(api_instance, job_name: str, namespace: str) -> bool:
    """Check if a job is genuinely queued due to resource constraints."""
    try:
        # Get events related to the job
        events = api_instance.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={job_name}",
        )
        for event in events.items:
            # Look specifically for resource quota exceeded events
            if (
                event.type == "Warning"
                and event.reason == "FailedCreate"
                and "exceeded quota" in event.message
                and "compute-resources" in event.message
            ):
                return True

    except Exception as e:
        logger.debug(f"Error checking events for job {job_name}: {e}")
    return False


def get_queue_data(namespace="informatics") -> pd.DataFrame:
    """Get data about queued workloads."""
    config.load_kube_config()
    v1 = client.CustomObjectsApi()
    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()

    try:
        workloads = v1.list_namespaced_custom_object(
            group="kueue.x-k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="workloads",
        )

        records = []
        for wl in workloads["items"]:
            # Extract resource requests early to filter non-GPU workloads
            resources = wl["spec"]["podSets"][0]["template"]["spec"]["containers"][0][
                "resources"
            ]
            gpu_request = int(
                float(resources.get("limits", {}).get("nvidia.com/gpu", "0"))
            )

            # Skip workloads that don't request GPUs
            if gpu_request == 0:
                continue

            # Get job name from workload
            workload_name = wl["metadata"]["name"]
            job_name = workload_name.replace("job-", "", 1)
            # remove last hyphen and everything after it
            job_name = job_name.rsplit("-", 1)[0]

            # Check job status
            try:
                job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)
                job_status = job.status
                # logger.debug(f"Got job status for {job_name}: {job_status}")
                # Skip if job is actively running or completed/failed
                if (
                    job_status.active
                    or job_status.succeeded
                    or job_status.failed
                    or any(
                        c.type in ["Complete", "Failed"]
                        for c in (job_status.conditions or [])
                    )
                ):
                    continue

                # Only include jobs that have resource quota exceeded events
                if not check_job_events_for_queue(core_v1, job_name, namespace):
                    continue

            except Exception as e:
                logger.debug(f"Error checking job {job_name}: {e}")
                continue

            # Get creation timestamp and convert to local time
            created = (
                datetime.strptime(
                    wl["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                )
                .replace(tzinfo=timezone.utc)
                .astimezone()
            )

            # Extract resource requests
            requests = resources.get("requests", {})
            record = {
                "name": job_name,
                "user": wl["spec"]["podSets"][0]["template"]["metadata"]["labels"].get(
                    "eidf/user", "unknown"
                ),
                "queue": wl["spec"]["podSets"][0]["template"]["metadata"]["labels"].get(
                    "kueue.x-k8s.io/queue-name", "unknown"
                ),
                "priority": wl["spec"]["podSets"][0]["template"]["metadata"][
                    "labels"
                ].get("kueue.x-k8s.io/priority-class", "default"),
                "created": created,
                "wait_time": (
                    datetime.now(timezone.utc).astimezone() - created
                ).total_seconds()
                / 60,
                "cpus": int(float(requests.get("cpu", "0"))),
                "memory": int(float(requests.get("memory", "0").rstrip("Gi"))),
                "gpus": gpu_request,
                "gpu_type": wl["spec"]["podSets"][0]["template"]["spec"]
                .get("nodeSelector", {})
                .get("nvidia.com/gpu.product", "unknown"),
            }
            records.append(record)

        # Sort by creation time before returning
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("created")
        return df

    except Exception as e:
        logger.error(f"Failed to get queue data: {e}")
        return pd.DataFrame()


def print_gpu_total(namespace="informatics"):
    latest = get_data(load_gpu_metrics=False, namespace=namespace)
    console = Console()

    gpu_counts = latest["gpu_name"].value_counts()
    gpu_table = Table(title="GPU Count by Type", show_footer=True)
    gpu_table.add_column("GPU Type", style="cyan", footer="TOTAL")
    gpu_table.add_column(
        "Count",
        style="green",
        justify="right",
        footer=str(sum(gpu_counts)),
    )

    for gpu_type, count in gpu_counts.items():
        gpu_table.add_row(gpu_type, str(count))

    console.print(gpu_table)


def print_user_stats(namespace="informatics"):
    latest = get_data(load_gpu_metrics=True, namespace=namespace)
    console = Console()

    user_stats = (
        latest.groupby("username")
        .agg({"gpu_name": "count", "gpu_mem_used": "mean", "inactive": "sum"})
        .round(2)
    )

    user_table = Table(title="User Statistics", show_footer=True)
    user_table.add_column("Username", style="cyan", footer="TOTAL")
    user_table.add_column("GPUs in use", style="green", justify="right")
    user_table.add_column("Avg Memory Usage (%)", style="yellow", justify="right")
    user_table.add_column("Inactive GPUs", style="red", justify="right")

    total_gpus = 0
    total_inactive = 0
    weighted_mem_usage = 0

    for user, row in user_stats.iterrows():
        user_table.add_row(
            user,
            str(row["gpu_name"]),
            f"{row['gpu_mem_used']:.1f}",
            str(int(row["inactive"])),
        )
        total_gpus += row["gpu_name"]
        total_inactive += row["inactive"]
        weighted_mem_usage += row["gpu_mem_used"] * row["gpu_name"]

    avg_mem_usage = weighted_mem_usage / total_gpus if total_gpus > 0 else 0

    user_table.columns[1].footer = str(int(total_gpus))
    user_table.columns[2].footer = f"{avg_mem_usage:.1f}"
    user_table.columns[3].footer = str(int(total_inactive))

    console.print(user_table)


def print_job_stats(namespace="informatics"):
    latest = get_data(load_gpu_metrics=True, namespace=namespace)
    console = Console()

    job_stats = (
        latest.groupby("pod_name")
        .agg(
            {
                "username": "first",
                "gpu_name": "count",
                "gpu_mem_used": "mean",
                "inactive": "all",
                "node_name": "first",
                "cpu_requested": "first",
                "memory_requested": "first",
                "interactive": "first",  # Add interactive to aggregation
            }
        )
        .round(2)
    )

    job_table = Table(title="Job Statistics", show_footer=True)
    job_table.add_column("Job Name", style="cyan", footer="TOTAL")
    job_table.add_column("User", style="blue", justify="left")
    job_table.add_column("Node", style="magenta", justify="left")
    job_table.add_column("CPUs", style="green", justify="right")
    job_table.add_column("RAM (GB)", style="green", justify="right")
    job_table.add_column("GPUs", style="green", justify="right")
    job_table.add_column("GPU Mem (%)", style="yellow", justify="right")
    job_table.add_column("Status", style="red", justify="center")
    job_table.add_column("Mode", style="blue", justify="center")

    total_gpus = 0
    total_jobs = 0
    total_cpus = 0
    total_ram = 0
    total_inactive = 0
    total_interactive = 0
    weighted_mem_usage = 0

    for job_name, row in job_stats.iterrows():
        status = "🔴 Inactive" if row["inactive"] else "🟢 Active"
        mode = (
            "🔤 Interactive" if row["interactive"] else "🔢 Batch"
        )  # Use the field from DataFrame
        ram_gb = int(row["memory_requested"])

        job_table.add_row(
            job_name,
            row["username"],
            row["node_name"],
            str(int(row["cpu_requested"])),
            str(ram_gb),
            str(row["gpu_name"]),
            f"{row['gpu_mem_used']:.1f}",
            status,
            mode,
        )
        total_gpus += row["gpu_name"]
        total_cpus += row["cpu_requested"]
        total_ram += ram_gb
        total_jobs += 1
        total_interactive += 1 if row["interactive"] else 0
        total_inactive += 1 if row["inactive"] else 0
        weighted_mem_usage += row["gpu_mem_used"] * row["gpu_name"]

    avg_mem_usage = weighted_mem_usage / total_gpus if total_gpus > 0 else 0

    job_table.columns[0].footer = f"Jobs: {total_jobs}"
    job_table.columns[3].footer = str(int(total_cpus))
    job_table.columns[4].footer = str(int(total_ram))
    job_table.columns[5].footer = str(int(total_gpus))
    job_table.columns[6].footer = f"{avg_mem_usage:.1f}"
    job_table.columns[7].footer = f"Inactive: {total_inactive}"
    job_table.columns[8].footer = f"Interactive: {total_interactive}"

    console.print(job_table)


def print_queue_stats(namespace="informatics"):
    """Display statistics about queued workloads."""
    df = get_queue_data(namespace=namespace)
    if df.empty:
        logger.info("No workloads in queue")
        return

    console = Console()

    # Sort by creation time
    df = df.sort_values("created")

    queue_table = Table(title="Queue Statistics", show_footer=True)
    queue_table.add_column("Position", style="cyan", justify="right")
    queue_table.add_column("Job Name", style="blue")
    queue_table.add_column("User", style="magenta")
    queue_table.add_column("Wait Time", style="yellow")
    queue_table.add_column("CPUs", style="green", justify="right")
    queue_table.add_column("RAM (GB)", style="green", justify="right")
    queue_table.add_column("GPUs", style="green", justify="right")
    queue_table.add_column("GPU Type", style="green")
    queue_table.add_column("Priority", style="red")

    for idx, row in df.iterrows():
        # Format wait time
        hours = int(row["wait_time"] // 60)
        mins = int(row["wait_time"] % 60)
        wait_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

        queue_table.add_row(
            str(idx + 1),
            row["name"],
            row["user"],
            wait_str,
            str(row["cpus"]),
            str(row["memory"]),
            str(row["gpus"]),
            row["gpu_type"],
            row["priority"],
        )

    # Add summary footer
    queue_table.columns[0].footer = f"Total: {len(df)}"
    queue_table.columns[4].footer = str(df["cpus"].sum())
    queue_table.columns[5].footer = str(df["memory"].sum())
    queue_table.columns[6].footer = str(df["gpus"].sum())

    console.print(queue_table)
