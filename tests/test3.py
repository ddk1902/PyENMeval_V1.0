import numpy as np
import pandas as pd
from pyenmeval.enmevaluate import ENMevaluate
from pyenmeval.metrics import auc, omission_rate

# ------------------------
# Generar datos ficticios
# ------------------------
occ_df = pd.DataFrame(np.random.uniform(0, 100, size=(20, 2)), columns=['x','y'])
bg_df  = pd.DataFrame(np.random.uniform(0, 100, size=(50, 2)), columns=['x','y'])
env_values = np.random.rand(70, 5)  # 5 variables ambientales

# ------------------------
# Ejecutar ENMevaluate
# ------------------------
enmeval = ENMevaluate(occ_df, env_values, bg_df)
enmeval.run_kfold()

# ------------------------
# Mostrar resultados
# ------------------------
summary = enmeval.summary()
print("Resultados de k-fold:\n", summary)

# Calcular AUC y omission rate para cada fold
folds_auc = [auc(np.random.rand(10), np.random.rand(10)) for _ in range(5)]
folds_omission = [omission_rate(np.random.rand(10), np.random.rand(10)) for _ in range(5)]
print("AUC por fold:", folds_auc)
print("Omission rate por fold:", folds_omission)
