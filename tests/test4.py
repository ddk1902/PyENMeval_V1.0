import numpy as np
import pandas as pd
from pyenmeval.enmevaluate import ENMevaluate

# Datos ficticios de ocurrencias
occ_df = pd.DataFrame({
    'x': np.random.uniform(0, 100, 10),
    'y': np.random.uniform(0, 100, 10)
})

# Datos ambientales para ocurrencias: 3 variables
env_values = np.random.rand(10, 3)

# Datos de fondo: mismas 3 variables
bg_df = np.random.rand(20, 3)

# Crear objeto
enmeval = ENMevaluate(occ_df, env_values, bg_df=bg_df, k=5)

# Ejecutar k-fold
enmeval.run_kfold()

# Mostrar resumen
print(enmeval.summary())
