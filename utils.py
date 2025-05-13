import os
from pathlib import Path
import datetime
from typing import Union, Optional


class Utils:
    def __init__(self, project_base_path: Optional[Path] = None,
                 log_subdir: str = "Training/Logs",
                 model_subdir: str = "Model",  # Main directory for models saved by train_agent
                 eval_callback_subdir_name: str = "eval_callback_best_models",  # Subdir name for EvalCallback
                 default_train_model_filename: str = "PPO_Trained.zip"):

        if project_base_path is None:
            self.project_root = Path.cwd()
            print(f"Utils: project_root not provided, defaulting to CWD: {self.project_root}")
        else:
            self.project_root = Path(project_base_path).resolve()
            print(f"Utils: initialized with project_root: {self.project_root}")

        self.log_dir = self.project_root / log_subdir
        self.main_model_dir = self.project_root / model_subdir  # For saves from train_agent
        self.eval_best_model_storage_dir = self.main_model_dir / eval_callback_subdir_name  # Dir for EvalCallback

        self.default_train_model_filename = default_train_model_filename

        # Ensure these directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.main_model_dir.mkdir(parents=True, exist_ok=True)
        self.eval_best_model_storage_dir.mkdir(parents=True, exist_ok=True)  # Crucial for EvalCallback

    def get_log_dir_path(self, experiment_name: Optional[str] = None) -> Path:
        if experiment_name:
            path_to_log = self.log_dir / experiment_name
            path_to_log.mkdir(parents=True, exist_ok=True)
            return path_to_log
        return self.log_dir

    # This method is for the model saved by train_agent itself
    def get_train_agent_model_save_path(self, model_filename: Optional[str] = None, timestamp: bool = False,
                                        agent_name: Optional[str] = None, env_name: Optional[str] = None) -> Path:
        if model_filename:
            return self.main_model_dir / model_filename

        final_filename = self.default_train_model_filename  # Use the specific default
        if timestamp:
            now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            name_parts = []
            if agent_name: name_parts.append(agent_name)
            if env_name: name_parts.append(env_name.replace('-', '_').replace(':', '_'))
            name_parts.append(now_str)
            base_name = "_".join(name_parts) if name_parts else "model"
            final_filename = f"{base_name}.zip"
        return self.main_model_dir / final_filename

    # This method is for loading the model saved by train_agent
    def get_latest_train_agent_model_load_path(self, model_prefix: Optional[str] = None) -> Optional[
        Path]:  # For Python 3.9
        target_dir = self.main_model_dir
        if not target_dir.exists() or not any(target_dir.iterdir()):
            return None
        model_files = [f for f in target_dir.glob('*.zip')]
        if model_prefix:
            model_files = [f for f in model_files if f.name.startswith(model_prefix)]
        if not model_files: return None
        return max(model_files, key=lambda f: f.stat().st_mtime)

    # --- Methods specifically for EvalCallback ---
    def get_eval_callback_save_dir(self) -> Path:
        """
        Returns the DIRECTORY path where EvalCallback should save its 'best_model.zip'.
        """
        return self.eval_best_model_storage_dir

    def get_path_for_loading_eval_best_model(self) -> Path:
        """
        Returns the full FILE path to where 'best_model.zip' (saved by EvalCallback) should be.
        """
        return self.eval_best_model_storage_dir / "best_model.zip"  # EvalCallback saves this filename by default