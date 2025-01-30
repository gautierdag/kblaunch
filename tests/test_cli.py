import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
import typer
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from loguru import logger
from typer.testing import CliRunner

from kblaunch.cli import (
    KubernetesJob,
    app,
    check_if_completed,
    get_env_vars,
    is_mig_gpu,
    load_config,
    read_startup_script,
    send_message_command,
    validate_gpu_constraints,
    setup_git_command,
    create_git_secret,
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
def basic_job():
    return KubernetesJob(
        name="test-job",
        image="test-image:latest",
        kueue_queue_name="test-queue",
        gpu_limit=1,
        gpu_type="nvidia.com/gpu",
        gpu_product="NVIDIA-A100-SXM4-40GB",
        user_email="test@example.com",
        user_name="test-user",  # Add explicit user name
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


@patch("kubernetes.client.BatchV1Api")
def test_kubernetes_job_run(mock_batch_api, mock_k8s_config, basic_job):
    """Test job run with mocked kubernetes API."""
    # Setup mock instance
    mock_api_instance = MagicMock()
    mock_batch_api.return_value = mock_api_instance

    # Mock successful job creation
    mock_api_instance.create_namespaced_job.return_value = MagicMock()

    # Run the job
    result = basic_job.run()

    # Verify the job was created with correct parameters
    mock_api_instance.create_namespaced_job.assert_called_once()
    call_args = mock_api_instance.create_namespaced_job.call_args

    # Verify namespace and job yaml
    assert call_args[1]["namespace"] == "default"  # or your expected namespace
    job_dict = call_args[1]["body"]
    assert job_dict["kind"] == "Job"
    assert job_dict["metadata"]["name"] == "test-job"

    # Verify return code
    assert result == 0


@patch("kubernetes.client.BatchV1Api")
def test_kubernetes_job_run_existing(
    mock_batch_api, mock_k8s_config, basic_job, monkeypatch
):
    """Test job run when job already exists."""
    # Setup mock instance
    mock_api_instance = MagicMock()
    mock_batch_api.return_value = mock_api_instance

    # Create a mock job with proper metadata and matching user
    mock_job = MagicMock()
    mock_metadata = MagicMock()
    mock_metadata.labels = {
        "eidf/user": "test-user"
    }  # Match the user_name in basic_job
    mock_job.metadata = mock_metadata
    mock_api_instance.read_namespaced_job.return_value = mock_job

    # Mock conflict on job creation
    conflict_exception = ApiException(status=409)
    mock_api_instance.create_namespaced_job.side_effect = [
        conflict_exception,  # First call fails with conflict
        MagicMock(),  # Second call succeeds after deletion
    ]

    # Mock user confirmation
    monkeypatch.setattr("typer.confirm", lambda *args, **kwargs: True)

    # Run the job
    result = basic_job.run()

    # Verify job deletion and recreation
    mock_api_instance.delete_namespaced_job.assert_called_once()
    assert mock_api_instance.create_namespaced_job.call_count == 2
    assert result == 0


@patch("kubernetes.client.BatchV1Api")
def test_kubernetes_job_run_error(mock_batch_api, mock_k8s_config, basic_job):
    """Test job run with API error."""
    # Setup mock instance
    mock_api_instance = MagicMock()
    mock_batch_api.return_value = mock_api_instance

    # Mock API error
    mock_api_instance.create_namespaced_job.side_effect = ApiException(status=500)

    # Run the job
    result = basic_job.run()

    # Verify error handling
    assert result == 1
    mock_api_instance.create_namespaced_job.assert_called_once()


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


@patch("kblaunch.cli.save_config")
@patch("kblaunch.cli.check_if_pvc_exists")
@patch("requests.post")
def test_setup_command(mock_post, mock_check_pvc, mock_save):
    """Test setup command with mock inputs."""
    # Setup mocks
    mock_post.return_value.status_code = 200
    mock_check_pvc.return_value = False

    # Mock all the user interactions
    confirm_responses = [
        True,  # Would you like to set the user?
        True,  # Would you like to set up Slack notifications?
        True,  # Would you like to set up a PVC?
        True,  # Would you like to set as default PVC?
        True,  # Would you like to set up Git SSH authentication?
    ]

    prompt_responses = [
        "user",  # user input
        "test@example.com",  # email input
        "https://hooks.slack.com/test",  # slack webhook
        "user-pvc",  # PVC name
        "10Gi",  # PVC size
        "/home/user/.ssh/id_rsa",  # SSH key path
    ]

    with patch("typer.confirm", side_effect=confirm_responses), patch(
        "typer.prompt", side_effect=prompt_responses
    ), patch("kblaunch.cli.create_pvc", return_value=True), patch(
        "kblaunch.cli.create_git_secret", return_value=True
    ):
        result = runner.invoke(app, ["setup"])

        assert result.exit_code == 0
        mock_save.assert_called_once_with(
            {
                "user": "user",
                "email": "test@example.com",
                "slack_webhook": "https://hooks.slack.com/test",
                "pvc_name": "user-pvc",
                "default_pvc": "user-pvc",
                "git_secret": "user-git-ssh",  # Add git secret to expected config
            }
        )


def test_mig_gpu_validation():
    """Test MIG GPU validation."""
    # Test single MIG instance (should pass)
    validate_gpu_constraints("NVIDIA-A100-SXM4-40GB-MIG-3g.20gb", 1, "default")

    # Test multiple MIG instances (should fail)
    with pytest.raises(ValueError, match="Cannot request more than one MIG instance"):
        validate_gpu_constraints("NVIDIA-A100-SXM4-40GB-MIG-3g.20gb", 2, "default")


def test_h100_priority_validation():
    """Test H100 priority validation."""
    # Test H100 with default priority (should pass)
    validate_gpu_constraints("NVIDIA-H100-80GB-HBM3", 1, "default")

    # Test H100 with short priority (should fail)
    with pytest.raises(ValueError, match="Cannot request H100 GPUs in the short"):
        validate_gpu_constraints("NVIDIA-H100-80GB-HBM3", 1, "short")


def test_is_mig_gpu():
    """Test MIG GPU detection."""
    assert is_mig_gpu("NVIDIA-A100-SXM4-40GB-MIG-3g.20gb") is True
    assert is_mig_gpu("NVIDIA-A100-SXM4-40GB") is False
    assert is_mig_gpu("NVIDIA-H100-80GB-HBM3") is False


def test_read_startup_script(tmp_path):
    """Test reading startup script."""
    # Create a temporary script file
    script_path = tmp_path / "startup.sh"
    script_content = "#!/bin/bash\necho 'Hello, World!'\n"
    script_path.write_text(script_content)

    # Test successful read
    assert read_startup_script(str(script_path)) == script_content

    # Test non-existent file
    with pytest.raises(typer.BadParameter, match="not found"):
        read_startup_script("nonexistent.sh")

    # Test directory instead of file
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    with pytest.raises(typer.BadParameter, match="Not a file"):
        read_startup_script(str(dir_path))


@patch("kblaunch.cli.KubernetesJob")
def test_launch_with_startup_script(mock_kubernetes_job, mock_k8s_client, tmp_path):
    """Test launch command with startup script."""
    # Create a temporary script file
    script_path = tmp_path / "startup.sh"
    script_content = "#!/bin/bash\necho 'Hello, World!'\n"
    script_path.write_text(script_content)

    # Setup mock instances
    mock_job_instance = mock_kubernetes_job.return_value
    mock_job_instance.generate_yaml.return_value = "dummy: yaml"
    mock_job_instance.run.return_value = None

    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    # Mock ConfigMap creation
    core_api = mock_k8s_client["core_api"]
    core_api.create_namespaced_config_map.return_value = MagicMock()

    # Run command with startup script
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
            "--startup-script",
            str(script_path),
        ],
    )

    assert result.exit_code == 0
    mock_kubernetes_job.assert_called_once()

    # Verify ConfigMap was created
    core_api.create_namespaced_config_map.assert_called_once()
    config_map_call = core_api.create_namespaced_config_map.call_args
    assert config_map_call[1]["body"].data["startup.sh"] == script_content

    # Verify job parameters
    job_args = mock_kubernetes_job.call_args[1]
    assert job_args["startup_script"] == script_content
    assert "bash /startup.sh &&" in job_args["args"][0]


@patch("kblaunch.cli.KubernetesJob")
def test_launch_with_startup_script_update(
    mock_kubernetes_job, mock_k8s_client, tmp_path
):
    """Test launch command when ConfigMap already exists."""
    # Create a temporary script file
    script_path = tmp_path / "startup.sh"
    script_content = "#!/bin/bash\necho 'Hello, World!'\n"
    script_path.write_text(script_content)

    # Setup mock instances
    mock_job_instance = mock_kubernetes_job.return_value
    mock_job_instance.generate_yaml.return_value = "dummy: yaml"
    mock_job_instance.run.return_value = None

    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    # Mock ConfigMap creation with conflict
    core_api = mock_k8s_client["core_api"]
    core_api.create_namespaced_config_map.side_effect = ApiException(status=409)

    # Run command with startup script
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
            "--startup-script",
            str(script_path),
        ],
    )

    assert result.exit_code == 0
    mock_kubernetes_job.assert_called_once()

    # Verify ConfigMap was updated
    core_api.patch_namespaced_config_map.assert_called_once()
    patch_call = core_api.patch_namespaced_config_map.call_args
    assert patch_call[1]["body"].data["startup.sh"] == script_content


def test_kubernetes_job_generate_yaml_with_startup_script(basic_job):
    """Test KubernetesJob YAML generation with startup script."""
    script_content = "#!/bin/bash\necho 'test'\n"

    # Create a new job with startup script since basic_job doesn't have startup script setup
    job_with_script = KubernetesJob(
        name=basic_job.name,
        image=basic_job.image,
        kueue_queue_name=basic_job.kueue_queue_name,
        gpu_limit=basic_job.gpu_limit,
        gpu_type=basic_job.gpu_type,
        gpu_product=basic_job.gpu_product,
        user_email=basic_job.user_email,
        user_name=basic_job.user_name,
        startup_script=script_content,  # Add startup script
    )

    yaml_output = job_with_script.generate_yaml()
    job_dict = yaml.safe_load(yaml_output)

    # Verify startup script volume mount and config map are present
    container = job_dict["spec"]["template"]["spec"]["containers"][0]
    volumes = job_dict["spec"]["template"]["spec"]["volumes"]

    startup_mount = next(
        (m for m in container["volumeMounts"] if m["name"] == "startup-script"), None
    )
    assert startup_mount is not None
    assert startup_mount["mountPath"] == "/startup.sh"
    assert startup_mount["subPath"] == "startup.sh"

    startup_volume = next((v for v in volumes if v["name"] == "startup-script"), None)
    assert startup_volume is not None
    assert startup_volume["configMap"]["defaultMode"] == 0o755
    assert startup_volume["configMap"]["name"] == f"{job_with_script.name}-startup"


def test_setup_git_command():
    """Test Git setup command generation."""
    command = setup_git_command()

    # Verify all required Git setup commands are present
    assert "mkdir -p ~/.ssh" in command
    assert "cp /etc/ssh-key/ssh-privatekey ~/.ssh/id_rsa" in command
    assert "chmod 600 ~/.ssh/id_rsa" in command
    assert "ssh-keyscan github.com" in command
    assert "git config --global core.sshCommand" in command
    assert "git config --global user.name" in command
    assert "git config --global user.email" in command


@patch("builtins.open", new_callable=mock_open, read_data="mock-ssh-key")
def test_create_git_secret(mock_file, mock_k8s_client):
    """Test creating Git SSH secret."""
    core_api = mock_k8s_client["core_api"]
    core_api.create_namespaced_secret.return_value = MagicMock()

    result = create_git_secret(
        secret_name="test-git-secret",
        private_key_path="/fake/path/to/key",
        namespace="test-namespace",
    )

    assert result is True
    core_api.create_namespaced_secret.assert_called_once()

    # Verify secret creation arguments
    call_args = core_api.create_namespaced_secret.call_args
    assert call_args[1]["namespace"] == "test-namespace"
    secret = call_args[1]["body"]
    assert secret.metadata.name == "test-git-secret"
    assert secret.string_data["ssh-privatekey"] == "mock-ssh-key"
    assert secret.type == "kubernetes.io/ssh-auth"


@patch("builtins.open", new_callable=mock_open, read_data="mock-ssh-key")
def test_create_git_secret_existing(mock_file, mock_k8s_client):
    """Test updating existing Git SSH secret."""
    core_api = mock_k8s_client["core_api"]

    # Mock conflict on first attempt
    core_api.create_namespaced_secret.side_effect = ApiException(status=409)
    core_api.patch_namespaced_secret.return_value = MagicMock()

    # Mock user confirmation
    with patch("typer.confirm", return_value=True):
        result = create_git_secret(
            secret_name="test-git-secret",
            private_key_path="/fake/path/to/key",
            namespace="test-namespace",
        )

    assert result is True
    core_api.patch_namespaced_secret.assert_called_once()


def test_kubernetes_job_with_git(basic_job):
    """Test KubernetesJob configuration with Git secret."""
    job_with_git = KubernetesJob(
        name=basic_job.name,
        image=basic_job.image,
        kueue_queue_name=basic_job.kueue_queue_name,
        gpu_limit=basic_job.gpu_limit,
        gpu_type=basic_job.gpu_type,
        gpu_product=basic_job.gpu_product,
        user_email=basic_job.user_email,
        git_secret="test-git-secret",
    )

    yaml_output = job_with_git.generate_yaml()
    job_dict = yaml.safe_load(yaml_output)

    # Verify Git secret volume mount
    container = job_dict["spec"]["template"]["spec"]["containers"][0]
    volumes = job_dict["spec"]["template"]["spec"]["volumes"]

    git_mount = next(
        (m for m in container["volumeMounts"] if m["name"] == "git-ssh"), None
    )
    assert git_mount is not None
    assert git_mount["mountPath"] == "/etc/ssh-key"
    assert git_mount["readOnly"] is True

    git_volume = next((v for v in volumes if v["name"] == "git-ssh"), None)
    assert git_volume is not None
    assert git_volume["secret"]["secretName"] == "test-git-secret"
    assert git_volume["secret"]["defaultMode"] == 0o600


@patch("kblaunch.cli.KubernetesJob")
def test_launch_with_git_config(mock_kubernetes_job, mock_k8s_client):
    """Test launch command with Git configuration."""
    # Setup mock instance
    mock_job_instance = mock_kubernetes_job.return_value
    mock_job_instance.generate_yaml.return_value = "dummy: yaml"
    mock_job_instance.run.return_value = None

    # Mock job completion check
    batch_api = mock_k8s_client["batch_api"]
    batch_api.list_namespaced_job.return_value.items = []

    # Create a mock config with Git secret
    mock_config = {
        "user": "test-user",
        "email": "test@example.com",
        "git_secret": "test-git-secret",
    }

    with patch("kblaunch.cli.load_config", return_value=mock_config):
        result = runner.invoke(
            app,
            [
                "launch",
                "--job-name",
                "test-job",
                "--command",
                "python test.py",
            ],
        )

    assert result.exit_code == 0
    mock_kubernetes_job.assert_called_once()

    # Verify Git configuration was included
    job_args = mock_kubernetes_job.call_args[1]
    assert job_args["git_secret"] == "test-git-secret"
    assert job_args["env_vars"]["USER"] == "test-user"
    assert job_args["env_vars"]["GIT_EMAIL"] == "test@example.com"


runner = CliRunner()
