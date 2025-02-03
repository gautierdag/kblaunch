import pandas as pd
from kubernetes import client, config
from datetime import datetime
from rich.console import Console
from rich.table import Table


def check_if_interactive(pod_name: str, namespace: str = "informatics") -> bool:
    """Check if a pod is running in interactive mode by examining its command."""
    try:
        api = client.CoreV1Api()
        pod = api.read_namespaced_pod(pod_name, namespace)

        # Get the command and args from the pod spec
        container = pod.spec.containers[0]
        command = container.command if container.command else []
        args = container.args if container.args else []

        # Combine command and args into a single string for searching
        full_command = " ".join(command + args).lower()

        # Check for common interactive patterns
        interactive_patterns = [
            "sleep infinity",
            "while true",
            "tail -f /dev/null",
            "sleep 60",
        ]
        return any(pattern in full_command for pattern in interactive_patterns)
    except Exception:
        return False


def get_data() -> pd.DataFrame:
    """Get live GPU usage data from Kubernetes pods."""
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # Get all pods in one call
    pods = v1.list_pod_for_all_namespaces(
        label_selector="kueue.x-k8s.io/queue-name"  # Filter for our GPU jobs
    )

    current_time = datetime.now()
    records = []

    for pod in pods.items:
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

        # Get basic pod info
        namespace = pod.metadata.namespace
        pod_name = pod.metadata.name
        node_name = pod.spec.node_name
        username = pod.metadata.labels.get("eidf/user", "unknown")

        # Get resource requests
        container = pod.spec.containers[0]  # Assuming single container
        cpu_requested = int(float(container.resources.requests.get("cpu", "0")))
        memory_requested = int(
            float(container.resources.requests.get("memory", "0").rstrip("Gi"))
        )

        # Get GPU info from node
        node = v1.read_node(node_name)
        gpu_product = node.metadata.labels.get("nvidia.com/gpu.product", "unknown")

        # Get GPU metrics (mocked for now, would need metrics server)
        # In real implementation, you'd get this from prometheus/nvidia-smi
        gpu_metrics = {
            "memory_used": 0,  # Would come from metrics
            "memory_total": 80 * 1024,  # 80GB for A100
            "gpu_mem_used": 0,  # Will be calculated
            "inactive": True,  # Will be set based on usage
        }

        # Create a record for each GPU assigned
        for gpu_id in range(gpu_requests):
            record = {
                "timestamp": current_time,
                "pod_name": pod_name,
                "namespace": namespace,
                "node_name": node_name,
                "username": username,
                "cpu_requested": cpu_requested,
                "memory_requested": memory_requested,
                "gpu_name": gpu_product,
                "gpu_id": gpu_id,
                **gpu_metrics,
            }
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    if len(df) == 0:
        # Return empty DataFrame with correct columns if no GPU pods found
        return pd.DataFrame(
            columns=[
                "timestamp",
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
            ]
        )

    # Calculate derived fields
    df["gpu_mem_used"] = (df["memory_used"] / df["memory_total"]) * 100
    df["inactive"] = df["gpu_mem_used"] < 1

    return df


def print_gpu_total():
    df = get_data()
    console = Console()
    latest = df[df["timestamp"] == df["timestamp"].max()]

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


def print_user_stats():
    df = get_data()
    console = Console()
    latest = df[df["timestamp"] == df["timestamp"].max()]

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


def print_job_stats():
    df = get_data()
    console = Console()
    latest = df[df["timestamp"] == df["timestamp"].max()]

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
        is_interactive = check_if_interactive(job_name)
        mode = "🔤 Interactive" if is_interactive else "🔢 Batch"
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
        total_interactive += 1 if is_interactive else 0
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
