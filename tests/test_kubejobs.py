from unittest import mock
from unittest.mock import patch, MagicMock, mock_open
import pytest
import yaml


from kblaunch.kubejobs import KubernetesJob, create_pvc, create_pv, fetch_user_info


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


@patch("os.getlogin")
@patch("pwd.getpwnam")
@patch("os.getgrouplist")
@patch("grp.getgrgid")
def test_fetch_user_info(
    mock_getgrgid, mock_getgrouplist, mock_getpwnam, mock_getlogin
):
    mock_getlogin.return_value = "testuser"
    mock_pwnam = MagicMock()
    mock_pwnam.pw_dir = "/home/testuser"
    mock_pwnam.pw_shell = "/bin/bash"
    mock_pwnam.pw_gid = 1000
    mock_getpwnam.return_value = mock_pwnam
    mock_getgrouplist.return_value = [1000, 1001]
    mock_group = MagicMock()
    mock_group.gr_name = "testgroup"
    mock_getgrgid.return_value = mock_group

    result = fetch_user_info()

    assert result["login_user"] == "testuser"
    assert result["home"] == "/home/testuser"
    assert result["shell"] == "/bin/bash"
    assert "testgroup testgroup" in result["groups"]


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


def test_create_pvc(mock_k8s_config, mock_subprocess):
    with patch("builtins.open", mock_open()) as mock_file:
        pvc_name = create_pvc("test-pvc", "10Gi")

    assert pvc_name == "test-pvc"
    mock_subprocess.assert_has_calls(
        [
            mock.call(["kubectl", "apply", "-f", "pvc.json"], check=True),
            mock.call(["rm", "pvc.json"], check=True),
        ]
    )
    mock_file.assert_called_once_with("pvc.json", "w")


@patch("kubernetes.client.CoreV1Api")
def test_create_pv(mock_core_v1_api, mock_k8s_config):
    mock_api = MagicMock()
    mock_core_v1_api.return_value = mock_api

    create_pv(
        pv_name="test-pv",
        storage="10Gi",
        storage_class_name="standard",
        access_modes=["ReadWriteOnce"],
        pv_type="local",
        local_path="/test/path",
    )

    mock_api.create_persistent_volume.assert_called_once()


def test_create_pv_invalid_type():
    with pytest.raises(ValueError):
        create_pv(
            pv_name="test-pv",
            storage="10Gi",
            storage_class_name="standard",
            access_modes=["ReadWriteOnce"],
            pv_type="invalid",
        )


def test_create_pv_missing_local_path():
    with pytest.raises(ValueError):
        create_pv(
            pv_name="test-pv",
            storage="10Gi",
            storage_class_name="standard",
            access_modes=["ReadWriteOnce"],
            pv_type="local",
        )
