import os
from os import getcwd


class Utils:
    def __init__(self):
        pass

    def get_log_dir(self):
        """
        Get the log directory for logging files with tensor flow

        :return: The directory path from the current working directory to the log directory
        """
        return os.path.join(getcwd(), "Training", "Logs")

    def get_save_path(self):
        """
        Get the folder where models should be saved
        :return: the directory path from the current working project dir to the saved models folder.
        """
        return os.path.join(getcwd(), "Training", "Saved Model")

    def get_saved_model(self):
        """
        Get the folder where models should be saved
        :return: the directory path from the current working project dir to the saved models folder.
        """
        return os.path.join(getcwd(), "Training", "Saved Model.zip")
