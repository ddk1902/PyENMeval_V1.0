# pyenmeval/maxent_wrapper.py

import subprocess
import tempfile
import os
import shutil
import pandas as pd
from pathlib import Path


class MaxEntWrapper:
    """
    Wrapper para ejecutar MaxEnt (maxent.jar) desde Python.
    """

    def __init__(self, maxent_jar, memory="1024m"):
        """
        maxent_jar : ruta a maxent.jar
        memory : memoria asignada a Java
        """

        if not os.path.exists(maxent_jar):
            raise FileNotFoundError(f"maxent.jar no encontrado: {maxent_jar}")

        self.maxent_jar = maxent_jar
        self.memory = memory

    def _write_samples(self, df, filepath, is_background=False):

     df = df.copy()

        # convertir columnas
     if "lon" in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lon": "longitude", "lat": "latitude"})

            if not is_background:
             if "species" not in df.columns:
                df["species"] = "species"

             df = df[["species", "longitude", "latitude"]]
            else:
             df = df[["longitude", "latitude"]]

             df.to_csv(filepath, index=False)

    def run(
        self,
        occ_df,
        bg_df,
        predictors_dir,
        feature_class="LQH",
        regularization=1,
        output_dir=None,
        threads=1,
        random_seed=True
    ):
        """
        Ejecuta MaxEnt.

        Parameters
        ----------
        occ_df : DataFrame
            Presencias con columnas: species, longitude, latitude
        bg_df : DataFrame
            Background points
        predictors_dir : str
            Carpeta con rasters ambientales
        feature_class : str
        regularization : float
        """

        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = output_dir
            os.makedirs(temp_dir, exist_ok=True)

        samples_file = os.path.join(temp_dir, "occurrences.csv")
        background_file = os.path.join(temp_dir, "background.csv")

        self._write_samples(occ_df, samples_file)
        self._write_samples(bg_df, background_file)
        samples_file_fixed = str(samples_file).replace("\\", "/")
        predictors_dir_fixed = str(predictors_dir).replace("\\", "/")
        output_dir_fixed = str(temp_dir).replace("\\", "/")
        cmd = [
                "java",
                f"-mx{self.memory}",
                "-jar",
                self.maxent_jar,
                f"samplesfile={samples_file_fixed}",
                f"environmentallayers={predictors_dir_fixed}",
                f"outputdirectory={output_dir_fixed}",
                f"betamultiplier={regularization}",
                f"threads={threads}",
                "responsecurves=false",
                "jackknife=false",
                "pictures=false"
            ]

        # Feature classes
        feature_map = {
            "L": "linear=true quadratic=false hinge=false product=false threshold=false",
            "LQ": "linear=true quadratic=true hinge=false product=false threshold=false",
            "H": "linear=false quadratic=false hinge=true product=false threshold=false",
            "LQH": "linear=true quadratic=true hinge=true product=false threshold=false"
        }

        if feature_class in feature_map:
            cmd.extend(feature_map[feature_class].split())

        if random_seed:
            cmd.append("randomseed=true")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Error ejecutando MaxEnt") from e

        prediction_file = os.path.join(temp_dir, "species_prediction.asc")

        if not os.path.exists(prediction_file):
            raise RuntimeError("MaxEnt no generó el archivo de predicción")

        return prediction_file