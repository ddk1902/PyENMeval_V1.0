# examples/run_experiment.py

import pandas as pd

from enmevaluate import ENMevaluate

# ---------------------------------------
# 1. Cargar datos
# ---------------------------------------

occ_df = pd.read_csv("data/occurrences.csv")  
bg_df = pd.read_csv("data/background.csv")

predictors_dir = "data/env_layers/"
maxent_jar = "maxent.jar"

# ---------------------------------------
# 2. Inicializar evaluador
# ---------------------------------------

evaluator = ENMevaluate(
    occ_df=occ_df,
    bg_df=bg_df,
    predictors_dir=predictors_dir,
    maxent_jar=maxent_jar,
    partition_method="block",   # como ENMeval
    cleanup_workspace=True
)

# ---------------------------------------
# 3. Definir grid de parámetros
# ---------------------------------------

feature_classes = ["L", "LQ", "H", "LQH"]
regularization_values = [0.5, 1, 2, 3]

# ---------------------------------------
# 4. Ejecutar tuning
# ---------------------------------------

ranking, best_model = evaluator.run_tuning(
    feature_classes=feature_classes,
    regularization_values=regularization_values
)

# ---------------------------------------
# 5. Resultados
# ---------------------------------------

print("\n=== RANKING DE MODELOS ===")
print(ranking)

print("\n=== MEJOR MODELO ===")
print(best_model)

# Guardar resultados
ranking.to_csv("results/model_ranking.csv", index=False)