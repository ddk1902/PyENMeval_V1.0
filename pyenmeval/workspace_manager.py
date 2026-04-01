# pyenmeval/workspace_manager.py

from pathlib import Path
import shutil
import tempfile
from datetime import datetime


class WorkspaceManager:
    """
    Gestiona directorios de trabajo para ejecuciones de MaxEnt.
    """

    def __init__(self, base_dir=None):

        # -------------------------------
        # 1. Definir base_dir
        # -------------------------------
        if base_dir is None:
            base_dir = Path(tempfile.gettempdir())
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir

        # -------------------------------
        # 2. Crear workspace único
        # -------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.workspace = self.base_dir / f"pyenmeval_run_{timestamp}"

        self.workspace.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    def create_model_dir(self, model_id, fold_id):
        """
        Crea directorio para un modelo específico y fold.
        """

        model_dir = self.workspace / f"model_{model_id}" / f"fold_{fold_id}"

        model_dir.mkdir(parents=True, exist_ok=True)

        return model_dir  # Path object

    # -------------------------------
    def get_workspace(self):
        return self.workspace

    # -------------------------------
    def cleanup(self):
        if self.workspace.exists():
            shutil.rmtree(self.workspace)