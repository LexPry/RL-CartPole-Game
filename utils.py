import os
from pathlib import Path  # Modern way to handle paths
import datetime  # For timestamping
from typing import Union


class Utils:
    def __init__(self, project_base_path: Path = None,
                 log_subdir: str = "Training/Logs",
                 model_subdir: str = "Model",
                 default_model_filename: str = "PPO_Default.zip"):
        """
        Initializes the Utils class for managing project paths.

        :param project_base_path: The absolute root path of your project.
                                  If None, defaults to the current working directory.
                                  It's often best to explicitly pass Path(__file__).resolve().parent
                                  from your main script if main.py is in the project root.
        :param log_subdir: Name of the subdirectory for TensorBoard logs, relative to project_base_path.
        :param model_subdir: Name of the subdirectory for saved models, relative to project_base_path.
        :param default_model_filename: Default filename for saving/loading a model if no specific name is given.
        """
        if project_base_path is None:
            self.project_root = Path.cwd()  # Defaults to current working directory
            print(
                f"Utils initialized. project_root not provided, defaulting to Current Working Directory: {self.project_root}")
        else:
            self.project_root = Path(project_base_path).resolve()
            print(f"Utils initialized with project_root: {self.project_root}")

        self.log_dir = self.project_root / log_subdir
        self.model_dir = self.project_root / model_subdir
        self.default_model_filename = default_model_filename

        # Ensure these base directories exist upon initialization
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_log_dir_path(self, experiment_name: str = None) -> Path:
        """
        Get the Path object for the TensorBoard log directory.
        Optionally creates a subdirectory for a specific experiment.
        The directory structure is created if it doesn't exist.

        :param experiment_name: Optional name for a specific experiment's subdirectory.
        :return: pathlib.Path object for the log directory.
        """
        if experiment_name:
            path_to_log = self.log_dir / experiment_name
            path_to_log.mkdir(parents=True, exist_ok=True)
            return path_to_log
        return self.log_dir

    def get_model_dir_path(self) -> Path:
        """
        Get the Path object for the directory where models are saved.
        The directory is created if it doesn't exist.

        :return: pathlib.Path object for the model directory.
        """
        return self.model_dir

    def get_model_save_path(self, model_filename: str = None, timestamp: bool = False, agent_name: str = None,
                            env_name: str = None) -> Path:
        """
        Get the full Path for saving a model file within the model directory.
        If model_filename is not provided, uses the default or generates a timestamped one.

        :param model_filename: Optional. The specific name of the model file (e.g., "my_model_v1.zip").
                               If provided, 'timestamp', 'agent_name', 'env_name' are ignored.
        :param timestamp: If True and model_filename is None, generates a timestamped filename.
        :param agent_name: Optional. Used for generating timestamped filename (e.g., "PPO").
        :param env_name: Optional. Used for generating timestamped filename (e.g., "CartPole").
        :return: pathlib.Path object for the model file.
        """
        if model_filename:
            return self.model_dir / model_filename

        final_filename = self.default_model_filename
        if timestamp:
            now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            name_parts = []
            if agent_name:
                name_parts.append(agent_name)
            if env_name:
                # Sanitize env_name for filename
                safe_env_name = env_name.replace('-', '_').replace(':', '_')
                name_parts.append(safe_env_name)
            name_parts.append(now_str)

            base_name = "_".join(name_parts) if name_parts else "model"
            final_filename = f"{base_name}.zip"

        return self.model_dir / final_filename

    def get_latest_model_load_path(self, model_prefix: str = None) -> Union[Path, None]:
        """
        Attempts to find the most recently modified model file in the model directory.
        Can optionally filter by a prefix (e.g., "PPO_CartPole").

        :param model_prefix: Optional. A prefix to filter model files.
        :return: pathlib.Path object for the latest model file, or None if no matching model is found.
        """
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            return None

        # Consider only .zip files as that's what SB3 saves by default
        model_files = [f for f in self.model_dir.glob('*.zip')]

        if model_prefix:
            model_files = [f for f in model_files if f.name.startswith(model_prefix)]

        if not model_files:
            return None

        # Find the most recently modified file
        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        return latest_model