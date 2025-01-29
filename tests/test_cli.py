import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from kubernetes import client, config
from loguru import logger
from typer.testing import CliRunner

from kblaunch.cli import (
    KubernetesJob,
    app,
    check_if_completed,
    get_env_vars,
    send_message_command,
    load_config,
)


@pytest.fixture
def mock_k8s_client(monkeypatch):
    """Mock Kubernetes client for testing."""
    mock_batch_api = MagicMock()
    mock_core_api = MagicMock()

    # Mock the kubernetes config loading
    monkeypatch.setattr(config, "load_kube_config", MagicMock())

    # Mock the kubernetes client APIs
    monkeypatch.setattr(client, "BatchV1Api", lambda: mock_batch_api)
    monkeypatch.setattr(client, "CoreV1Api", lambda: mock_core_api)
    monkeypatch.setattr(client, "V1DeleteOptions", MagicMock)

    return {
        "batch_api": mock_batch_api,
        "core_api": mock_core_api,
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    test_vars = {
        "TEST_VAR": "test_value",
        "PYTHONPATH": "/test/path",
    }
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    return test_vars


@pytest.fixture
def mock_kubernetes_job(monkeypatch):
    """Mock KubernetesJob for testing."""
    mock_job = MagicMock()
    mock_job.generate_yaml.return_value = "mock yaml"
    mock_job.run.return_value = None

    class MockKubernetesJob:
        def __init__(self, *args, **kwargs):
            pass

        def generate_yaml(self):
            return "mock yaml"

        def run(self):
            return None

    monkeypatch.setattr("kblaunch.cli.KubernetesJob", MockKubernetesJob)
    return mock_job


def test_check_if_completed(mock_k8s_client):
    """Test job completion checking."""
    batch_api = mock_k8s_client["batch_api"]

    # Mock job list response
    job_name = "test-job"
    mock_job = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name),
        status=client.V1JobStatus(
            conditions=[client.V1JobCondition(type="Complete", status="True")]
        ),
    )

    # Set up mock returns
    batch_api.list_namespaced_job.return_value.items = [mock_job]
    batch_api.read_namespaced_job.return_value = mock_job

    result = check_if_completed(job_name)
    assert result is True
    batch_api.delete_namespaced_job.assert_called_once()


def test_get_env_vars(mock_env_vars, mock_k8s_client):
    """Test environment variable collection."""
    core_api = mock_k8s_client["core_api"]

    # Mock secret response
    mock_secret = MagicMock()
    mock_secret.data = {"SECRET_KEY": "secret_data"}
    core_api.read_namespaced_secret.return_value = mock_secret

    env_vars = get_env_vars(
        local_env_vars=["TEST_VAR"],
    )
    assert env_vars["TEST_VAR"] == "test_value"


def test_send_message_command():
    """Test Slack message command generation."""
    env_vars = {"SLACK_WEBHOOK": "https://hooks.slack.com/test"}
    result = send_message_command(env_vars)
    assert "curl -X POST" in result
    assert "$SLACK_WEBHOOK" in result


@pytest.mark.parametrize("interactive", [True, False])
@patch("kblaunch.cli.KubernetesJob")
def test_launch_command(mock_kubernetes_job, mock_k8s_client, interactive):
    """Test launch command with different configurations."""
    # Setup mock instance
    mock_job_instance = mock_kubernetes_job.return_value
    mock_job_instance.generate_yaml.return_value = "dummy: yaml"
    mock_job_instance.run.return_value = None

    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    # Prepare arguments
    args = [
        "launch",
    ]
    if interactive:
        args.extend(["--interactive"])
    args.extend(
        [
            "--email",
            "test@example.com",
            "--job-name",
            "test-job",
            "--command",
            "python test.py",
        ]
    )

    # Execute command
    result = runner.invoke(app, args)

    # Verify results
    assert result.exit_code == 0
    mock_kubernetes_job.assert_called_once()
    mock_job_instance.generate_yaml.assert_called_once()
    mock_job_instance.run.assert_called_once()

    # Verify job creation parameters
    job_args = mock_kubernetes_job.call_args[1]
    assert job_args["name"] == "test-job"
    assert job_args["user_email"] == "test@example.com"
    if interactive:
        assert "while true; do sleep 60; done;" in job_args["args"][0]
    else:
        assert "python test.py" in job_args["args"][0]


@patch("kblaunch.cli.KubernetesJob")
def test_launch_with_env_vars(mock_kubernetes_job, mock_k8s_client):
    """Test launch command with environment variables."""
    # Setup mock instance
    mock_job_instance = mock_kubernetes_job.return_value
    mock_job_instance.generate_yaml.return_value = "dummy: yaml"
    mock_job_instance.run.return_value = None

    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    result = runner.invoke(
        app,
        [
            "launch",
            "--email",
            "test@example.com",
            "--job-name",
            "test-job",
            "--command",
            "python test.py",
            "--local-env-vars",
            "TEST_VAR",
        ],
    )

    if result.exit_code != 0:
        # Capture any exceptions for debugging
        logger.error(f"Error output: {result.output}")
        logger.error(f"Exception: {result.exception}")

    assert result.exit_code == 0
    mock_kubernetes_job.assert_called_once()
    mock_job_instance.generate_yaml.assert_called_once()
    mock_job_instance.run.assert_called_once()


@patch("kblaunch.cli.KubernetesJob")
def test_launch_with_vscode(mock_kubernetes_job, mock_k8s_client):
    """Test launch command with VS Code installation."""
    # Setup mock instance
    mock_job_instance = mock_kubernetes_job.return_value
    mock_job_instance.generate_yaml.return_value = "dummy: yaml"
    mock_job_instance.run.return_value = None

    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    result = runner.invoke(
        app,
        [
            "launch",
            "--email",
            "test@example.com",
            "--job-name",
            "test-job",
            "--command",
            "python test.py",
            "--vscode",
        ],
    )

    assert result.exit_code == 0
    mock_kubernetes_job.assert_called_once()

    # Verify VS Code installation command was included
    job_args = mock_kubernetes_job.call_args[1]
    assert (
        "curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64'"
        in job_args["args"][0]
    )


def test_launch_invalid_params():
    """Test launch command with invalid parameters."""
    result = runner.invoke(
        app,
        [
            "launch",
            "--job-name",
            "test-job",  # Missing required params
        ],
    )

    assert result.exit_code != 0


@pytest.fixture
def mock_k8s_config():
    with patch("kubernetes.config.load_kube_config"):
        yield


@pytest.fixture
def mock_subprocess():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run


@pytest.fixture
def basic_job():
    return KubernetesJob(
        name="test-job",
        image="test-image:latest",
        kueue_queue_name="test-queue",
        gpu_limit=1,
        gpu_type="nvidia.com/gpu",
        gpu_product="NVIDIA-A100-SXM4-40GB",
        user_email="test@example.com",
    )


def test_kubernetes_job_init(basic_job):
    assert basic_job.name == "test-job"
    assert basic_job.image == "test-image:latest"
    assert basic_job.gpu_limit == 1
    assert basic_job.cpu_request == 12  # Default CPU request for 1 GPU
    assert basic_job.ram_request == "80G"  # Default RAM request for 1 GPU


def test_kubernetes_job_generate_yaml(basic_job):
    yaml_output = basic_job.generate_yaml()
    job_dict = yaml.safe_load(yaml_output)

    assert job_dict["kind"] == "Job"
    assert job_dict["metadata"]["name"] == "test-job"
    assert (
        job_dict["spec"]["template"]["spec"]["containers"][0]["image"]
        == "test-image:latest"
    )


def test_kubernetes_job_run(mock_k8s_config, mock_subprocess, basic_job):
    with patch("builtins.open", mock_open()) as mock_file, patch(
        "os.remove"
    ) as mock_remove:
        result = basic_job.run()

    assert result == 0
    mock_subprocess.assert_called_once()
    mock_file().write.assert_called_once()
    mock_remove.assert_called_once_with("temp_job.yaml")


@pytest.mark.parametrize("gpu_limit", [-1, 0, 9])
def test_invalid_gpu_limit(gpu_limit):
    with pytest.raises(AssertionError):
        KubernetesJob(
            name="test-job",
            image="test-image:latest",
            kueue_queue_name="test-queue",
            gpu_limit=gpu_limit,
            gpu_type="nvidia.com/gpu",
            gpu_product="NVIDIA-A100-SXM4-40GB",
            user_email="test@example.com",
        )


def test_launch_no_command_non_interactive():
    """Test launch command fails when no command is provided in non-interactive mode."""
    result = runner.invoke(
        app,
        [
            "--email",
            "test@example.com",
            "--job-name",
            "test-job",
        ],
    )
    assert result.exit_code != 0


def test_load_config_no_file():
    """Test loading config when file doesn't exist."""
    with patch("pathlib.Path.exists", return_value=False):
        assert load_config() == {}


def test_load_config_with_file():
    """Test loading config from file."""
    test_config = {
        "email": "test@example.com",
        "slack_webhook": "https://hooks.slack.com/test",
    }
    mock_open_obj = mock_open(read_data=json.dumps(test_config))
    with patch("builtins.open", mock_open_obj):
        with patch("pathlib.Path.exists", return_value=True):
            config = load_config()
            assert config == test_config


def test_setup_command():
    """Test setup command with mock inputs."""
    with patch("typer.confirm", side_effect=[True, True]), patch(
        "typer.prompt", side_effect=["test@example.com", "https://hooks.slack.com/test"]
    ), patch("kblaunch.cli.save_config") as mock_save:
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        mock_save.assert_called_once_with(
            {
                "email": "test@example.com",
                "slack_webhook": "https://hooks.slack.com/test",
            }
        )


runner = CliRunner()
