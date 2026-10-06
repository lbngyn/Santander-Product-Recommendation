"""Original-information features, frozen categories and paired batch readers."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.products import PRODUCT_COLUMNS

KEYS = ["ncodpers", "fecha_dato"]
PROFILES = ["ind_empleado", "pais_residencia", "sexo", "age", "fecha_alta", "ind_nuevo",
    "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes", "indresi",
    "indext", "conyuemp", "canal_entrada", "indfall", "tipodom", "cod_prov", "nomprov",
    "ind_actividad_cliente", "renta", "segmento"]
NUMERIC = {"age", "antiguedad", "renta"}
DATES = {"fecha_alta", "ult_fec_cli_1t"}
CATEGORICAL = [p for p in PROFILES if p not in NUMERIC | DATES]
FEATURES = [*PROFILES, *("prev_" + p for p in PRODUCT_COLUMNS)]
LABELS = ["acq_" + p for p in PRODUCT_COLUMNS]
COUNT_CANDIDATES = ("n_acquisitions", "n_acquisition", "acquisition_count", "total_acquisitions")


def quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def parquet(path):
    return "read_parquet('" + str(Path(path)).replace("'", "''") + "')"


def model_frame(frame, contract, joint=False):
    """Freeze category vocabulary across products, strategies and inference."""
    names = [*contract["feature_names"], *(["product_id"] if joint else [])]
    result = frame.loc[:, names].copy()
    for name in names:
        if name in contract["category_values"]:
            categories = contract['category_values'][name]
            # Explicit unknown-to-missing conversion also works with newer Pandas.
            values = result[name].where(result[name].isin(categories))
            result[name] = pd.Categorical(values, categories=categories)
        elif name == "product_id":
            result[name] = pd.Categorical(result[name], categories=list(range(len(PRODUCT_COLUMNS))))
        else:
            result[name] = pd.to_numeric(result[name], errors="raise").astype("float32")
    return result


def matrix(frame):
    result = frame.copy()
    for name in result:
        if isinstance(result[name].dtype, pd.CategoricalDtype):
            result[name] = result[name].cat.codes.replace(-1, np.nan).astype("float32")
    return result.to_numpy(dtype="float32", copy=False)


def paired_batches(dataset, split, batch_size):
    from itertools import zip_longest
    inputs = pq.ParquetFile(Path(dataset) / f"{split}_inputs.parquet")
    targets = pq.ParquetFile(Path(dataset) / f"{split}_targets.parquet")
    for left, right in zip_longest(inputs.iter_batches(batch_size=batch_size), targets.iter_batches(batch_size=batch_size)):
        if left is None or right is None:
            raise ValueError("Input/target batch counts differ")
        x, y = left.to_pandas(), right.to_pandas()
        if not x[KEYS].equals(y[KEYS]):
            raise ValueError("Input/target keys or row order differ")
        yield x, y


