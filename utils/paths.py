import os

from config.paths import LOGS_PATH


def get_version_path(version: str) -> str:
    return os.path.join(LOGS_PATH, version)


def get_checkpoint_path(version: str) -> str:
    version_path = get_version_path(version)
    checkpoints_path = os.path.join(version_path, 'checkpoints')

    checkpoint_file_name = os.listdir(checkpoints_path)[0]
    checkpoint_file_path = os.path.join(checkpoints_path, checkpoint_file_name)
    return checkpoint_file_path
