import pandas as pd
from pathlib import Path
from .spatial_utils import prepare_occurrences
from .utils import generate_background


class ENM:

    def __init__(self, occurrences, predictors):

        self.occurrences = pd.read_csv(occurrences)

        self.predictors = Path(predictors)

        self.occ = prepare_occurrences(self.occurrences)

        self.bg = generate_background(self.occ)

        self.model = None


    def tune(self):

        print("Running model tuning...")

        # aquí llamas tu ENMevaluate
        # guardas el mejor modelo

        return self


    def predict(self):

        print("Generating suitability maps...")

        # aquí usas tu script de mapas

        return self


    def variable_importance(self):

        print("Calculating permutation importance")


    def response_curves(self):

        print("Generating response curves")