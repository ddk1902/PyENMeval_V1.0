# pyenmeval/lambda_parser.py

def count_parameters_from_lambdas(lambda_file):
    """
    Cuenta el número de parámetros activos (coeficientes != 0)
    en el archivo .lambdas generado por MaxEnt.

    Parameters
    ----------
    lambda_file : str
        Ruta al archivo species.lambdas

    Returns
    -------
    int
        Número de parámetros del modelo
    """

    param_count = 0

    with open(lambda_file, "r") as f:
        for line in f:

            line = line.strip()

            # ignorar líneas vacías
            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 2:
                continue

            try:
                coef = float(parts[1])
            except ValueError:
                continue

            if coef != 0:
                param_count += 1

    return param_count