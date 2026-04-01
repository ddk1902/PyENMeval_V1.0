# PyENMeval

PyENMeval is a Python implementation inspired by the **ENMeval** workflow for ecological niche modeling.

The library provides tools to evaluate **MaxNet-style models** using feature class combinations and regularization multipliers, similar to the tuning framework available in ENMeval (R).

PyENMeval is designed for researchers who prefer working in **Python-based ecological modeling workflows**.

---

# Features

• Spatial partitioning strategies  
- k-fold  
- spatial block  
- checkerboard

• Hyperparameter tuning  
- Feature Classes (L, Q, H, P, etc.)  
- Regularization Multiplier

• Model evaluation metrics

- AUC (train / test)
- AUC difference
- Omission Rate
- TSS
- AICc
- Delta AICc
- Akaike weights

• Model ranking and selection

• Raster prediction

• Generation of habitat suitability maps

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_user/PyENMeval.git
cd PyENMeval
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or install as a package:

```bash
pip install .
```

---

# Basic Usage

```python
from pyenmeval import ENMevaluate

model = ENMevaluate(
    occ_df=occurrences,
    bg_df=background,
    predictors_dir="data/rasters"
)

ranking, best_model = model.run_tuning(
    feature_classes=["L", "LQ", "LQH"],
    regularization_values=[0.5, 1, 2]
)

model.save_results()
```

---

# Output

PyENMeval produces a results table similar to **ENMeval**, including:

| metric |
|------|
| AUC train |
| AUC test |
| AUC difference |
| Omission rate |
| TSS |
| AICc |
| Delta AICc |
| Akaike weights |
| Model ranking |

---

# Scientific Background

The modeling framework follows the **MaxNet formulation of MaxEnt**, implemented using logistic regression with feature transformations.

References:

Phillips SJ et al. 2017.  
Opening the black box: an open-source release of Maxent.

---

# Project Goal

Provide a **Python-native alternative to ENMeval workflows** for ecological niche modeling and model tuning.

---

# Limitations
This implementation approximates the MaxEnt algorithm using
the MaxNet formulation implemented in Python.

Results may differ slightly from the original MaxEnt Java implementation.

# Author

Diego Gómez