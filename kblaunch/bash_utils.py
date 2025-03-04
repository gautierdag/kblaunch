import json

from loguru import logger


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
        """apt-get update; apt-get install -y curl;"""  # Install the curl command
        + f"""curl -X POST -H 'Content-type: application/json' --data "{message_json}" $SLACK_WEBHOOK;"""
    )


def install_vscode_command() -> str:
    """Generate command to install VS Code"""
    return (
        """apt-get update && apt-get install -y curl gpg;"""
        """curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg;"""
        """echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list;"""
        """apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y code;"""
        """ln -s /usr/bin/code /usr/local/bin/code;"""
        """code --version --user-data-dir='.' --no-sandbox;"""
    )


def start_vscode_tunnel_command(env_vars: set) -> str:
    """
    Generate command to start the VS Code tunnel.
    This uses the SLACK_WEBHOOK environment variable to send a message to Slack with the device code.
    See vscode.sh for the details.
    """
    if "SLACK_WEBHOOK" not in env_vars:
        logger.debug("SLACK_WEBHOOK required for tunnel.")
        return ""

    # download vscode script from github main branch and run it
    url = "https://raw.githubusercontent.com/gautierdag/kblaunch/refs/heads/main/kblaunch/vscode.sh"
    return (
        """apt-get update && apt-get install -y curl jq;"""  # Install the curl command
        f"""curl -Lk {url} --output vscode.sh;"""
        """chmod +x vscode.sh;"""
        """./vscode.sh &"""
    )


def setup_git_command() -> str:
    """Generate command to setup Git with SSH key."""
    return (
        """apt-get update && apt-get install -y git openssh-client;"""
        """mkdir -p ~/.ssh;"""
        """cp /etc/ssh-key/ssh-privatekey ~/.ssh/id_rsa;"""
        """chmod 600 ~/.ssh/id_rsa;"""
        """ssh-keyscan github.com >> ~/.ssh/known_hosts;"""
        """git config --global core.sshCommand 'ssh -i ~/.ssh/id_rsa';"""
        """git config --global user.name "${USER}";"""
        """git config --global user.email "${GIT_EMAIL}";"""
    )
