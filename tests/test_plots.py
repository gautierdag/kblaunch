from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from kblaunch.plots import (
    get_data,
    get_default_metrics,
    get_gpu_metrics,
    print_gpu_total,
    print_job_stats,
    print_user_stats,
    print_queue_stats,
)


@pytest.fixture
def mock_k8s_api():
    """Mock Kubernetes API responses"""
    with (
        patch("kubernetes.config.load_kube_config"),
        patch("kubernetes.client.CoreV1Api") as mock_api,
    ):
        # Create mock pod
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_pod.metadata.name = "test-pod"
        mock_pod.metadata.namespace = "namespace"
        mock_pod.metadata.labels = {"eidf/user": "test-user"}
        mock_pod.spec.node_name = "gpu-node-1"
        mock_pod.spec.node_selector = {
            "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-40GB"
        }

        # Mock container
        mock_container = MagicMock()
        mock_container.resources.requests = {
            "cpu": "12",
            "memory": "80Gi",
            "nvidia.com/gpu": "1",
        }
        mock_container.command = ["python"]
        mock_container.args = ["train.py"]
        mock_pod.spec.containers = [mock_container]

        # Setup API response
        mock_api.return_value.list_namespaced_pod.return_value.items = [mock_pod]
        yield mock_api


@pytest.fixture
def mock_namespace():
    """Mock namespace for testing"""
    return "namespace"


@pytest.fixture
def mock_nvidia_smi():
    """Mock nvidia-smi command output"""
    with patch("kubernetes.stream.stream") as mock_stream:
        mock_stream.return_value = "16384, 81920"  # 16GB used out of 80GB
        yield mock_stream


def test_get_default_metrics():
    """Test default metrics generation"""
    metrics = get_default_metrics()
    assert metrics["memory_used"] == 0
    assert metrics["memory_total"] == 80 * 1024
    assert metrics["gpu_mem_used"] == 0
    assert metrics["inactive"] is True


def test_get_gpu_metrics(mock_k8s_api, mock_nvidia_smi):
    """Test GPU metrics collection"""
    permission_errors = {"count": 0}
    metrics = get_gpu_metrics(
        mock_k8s_api(), "test-pod", "informatics", permission_errors
    )

    assert metrics["memory_used"] == 16384
    assert metrics["memory_total"] == 81920
    assert metrics["gpu_mem_used"] == 20.0  # (16384/81920) * 100
    assert metrics["inactive"] is False
    assert permission_errors["count"] == 0


def test_get_gpu_metrics_permission_error(mock_k8s_api):
    """Test GPU metrics collection with permission error"""
    with patch("kubernetes.stream.stream") as mock_stream:
        mock_stream.return_value = "[Insufficient Permissions]"
        permission_errors = {"count": 0}
        metrics = get_gpu_metrics(
            mock_k8s_api(), "test-pod", "informatics", permission_errors
        )

        assert metrics == get_default_metrics()
        assert permission_errors["count"] == 1


def test_get_data(mock_k8s_api, mock_nvidia_smi, mock_namespace):
    """Test data collection for all pods"""
    df = get_data(namespace=mock_namespace, load_gpu_metrics=True)

    assert not df.empty
    assert len(df) == 1  # One GPU pod in our mock
    assert df.iloc[0]["pod_name"] == "test-pod"
    assert df.iloc[0]["username"] == "test-user"
    assert df.iloc[0]["gpu_name"] == "NVIDIA-A100-SXM4-40GB"
    assert bool(df.iloc[0]["interactive"]) is False  # Convert numpy bool to Python bool


@pytest.fixture
def mock_console():
    """Mock rich console output"""
    mock_instance = MagicMock()
    with patch("kblaunch.plots.Console", return_value=mock_instance) as mock_class:
        # Ensure the mock instance is returned when Console() is called
        mock_class.return_value = mock_instance
        yield mock_instance


def test_print_gpu_total(mock_k8s_api, mock_console, mock_namespace):
    """Test GPU total display"""
    # Mock GPU data
    mock_pod = mock_k8s_api.return_value.list_namespaced_pod.return_value.items[0]
    mock_pod.spec.node_selector = {"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-40GB"}

    print_gpu_total(namespace=mock_namespace)

    # Verify the console.print was called
    assert mock_console.print.call_count > 0


def test_print_user_stats(mock_k8s_api, mock_nvidia_smi, mock_console, mock_namespace):
    """Test user statistics display"""
    # Mock user data
    mock_pod = mock_k8s_api.return_value.list_namespaced_pod.return_value.items[0]
    mock_pod.metadata.labels = {"eidf/user": "test-user"}

    print_user_stats(namespace=mock_namespace)

    # Verify the console.print was called
    assert mock_console.print.call_count > 0


def test_print_job_stats(mock_k8s_api, mock_nvidia_smi, mock_console, mock_namespace):
    """Test job statistics display"""
    # Mock job data
    mock_pod = mock_k8s_api.return_value.list_namespaced_pod.return_value.items[0]
    mock_pod.metadata.name = "test-job"

    print_job_stats(namespace=mock_namespace)

    # Verify the console.print was called
    assert mock_console.print.call_count > 0


def test_get_data_empty(mock_k8s_api, mock_namespace):
    """Test data collection with no GPU pods"""
    mock_k8s_api.return_value.list_namespaced_pod.return_value.items = []
    df = get_data(namespace=mock_namespace)

    assert df.empty
    assert list(df.columns) == [
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
        "status",
        "pending_reason",
    ]


def test_get_data_interactive_pod(mock_k8s_api, mock_namespace):
    """Test detection of interactive pods"""
    # Modify mock pod to be interactive
    mock_pod = mock_k8s_api.return_value.list_namespaced_pod.return_value.items[0]
    mock_pod.spec.containers[0].command = ["sleep", "infinity"]
    df = get_data(namespace=mock_namespace)
    assert bool(df.iloc[0]["interactive"]) is True  # Convert numpy bool to Python bool


@patch("kblaunch.plots.get_queue_data")
def test_print_queue_stats(mock_get_queue_data, mock_console, mock_namespace):
    """Test queue statistics display"""
    # Mock queue data as a DataFrame instead of a list with correct fields
    mock_get_queue_data.return_value = pd.DataFrame(
        [
            {
                "name": "test-job",
                "namespace": mock_namespace,
                "queue": "user-queue",
                "priority": "default",
                "cpus": 6,
                "memory": "40Gi",
                "gpus": 1,
                "gpu_type": "NVIDIA-A100-SXM4-40GB",
                "created": "2023-01-01T12:00:00Z",  # Using 'created' instead of 'admit_time'
                "message": "Waiting for resources",
                "wait_time": 10,  # Adding wait_time_minutes field
                "user": "test-user",
            }
        ]
    )

    print_queue_stats(namespace=mock_namespace)

    # Verify the console.print was called
    assert mock_console.print.call_count > 0
