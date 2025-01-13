import base64
from unittest.mock import MagicMock

import pytest
from kubernetes import client, config
from typer.testing import CliRunner

from kblaunch.cli import (
    app,
    check_if_completed,
    export_env_vars,
    get_env_vars,
    send_message_command,
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

    monkeypatch.setattr("kubejobs.jobs.KubernetesJob", MockKubernetesJob)
    return mock_job


runner = CliRunner()


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
    secret_data = base64.b64encode(b"secret").decode("utf-8")

    # Mock secret response
    mock_secret = MagicMock()
    mock_secret.data = {"SECRET_KEY": secret_data}
    core_api.read_namespaced_secret.return_value = mock_secret

    # Test with only local env vars first
    env_vars = get_env_vars(
        local_env_vars=["TEST_VAR"], secrets_env_vars=[], namespace="test"
    )
    assert env_vars["TEST_VAR"] == "test_value"

    # Test with secret
    env_vars = get_env_vars(
        local_env_vars=[], secrets_env_vars=["test-secret"], namespace="test"
    )
    assert "SECRET_KEY" in env_vars
    assert env_vars["SECRET_KEY"] == "secret"


def test_export_env_vars():
    """Test environment variable export command generation."""
    env_vars = {"TEST_VAR": "test value", "PATH": "/usr/local/bin"}

    result = export_env_vars(env_vars)
    assert "export TEST_VAR='test value'" in result
    assert "export PATH='/usr/local/bin'" in result
    assert result.endswith(" ; ")


def test_send_message_command():
    """Test Slack message command generation."""
    env_vars = {"SLACK_WEBHOOK": "https://hooks.slack.com/test"}
    result = send_message_command(env_vars)
    assert "curl -X POST" in result
    assert "hooks.slack.com/test" in result


@pytest.mark.parametrize("interactive", [True, False])
def test_launch_command(mock_k8s_client, interactive):
    """Test launch command with different configurations."""
    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    args = []
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

    result = runner.invoke(app, args)

    if result.exit_code != 0:
        print(f"Error output: {result.output}")  # For debugging

    assert result.exit_code == 0


def test_launch_with_env_vars(mock_k8s_client):
    """Test launch command with environment variables."""
    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    result = runner.invoke(
        app,
        [
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
        print(f"Error output: {result.output}")  # For debugging

    assert result.exit_code == 0


def test_launch_invalid_params():
    """Test launch command with invalid parameters."""
    result = runner.invoke(
        app,
        [
            "--job-name",
            "test-job",  # Missing required params
        ],
    )

    assert result.exit_code != 0
