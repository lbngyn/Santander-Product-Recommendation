"""CSV encoding detection and Santander schema normalisation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

DATE_COLUMNS = ("fecha_dato", "fecha_alta", "ult_fec_cli_1t")
INTEGER_COLUMNS = ("ncodpers", "ind_nuevo", "antiguedad", "indrel", "tipodom", "cod_prov", "ind_actividad_cliente")
FLOAT_COLUMNS = ("age", "renta")
PRODUCT_COLUMNS = ("ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1", "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1")
MISSING_VALUES = ("", " ", "NA", "N/A", "Unknown")


def detect_encoding(path: str | Path) -> str:
    """Return UTF-8 when valid; otherwise use a Latin-1-compatible encoding."""
    with Path(path).open("rb") as file:
        sample = file.read(1_000_000)
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def parse_santander_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim encoded text and convert known Santander fields to nullable dtypes."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip().replace("", pd.NA)
    for column in DATE_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(result[column], format="%Y-%m-%d", errors="coerce")
    for column in INTEGER_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in FLOAT_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float32")
    for column in PRODUCT_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int8")
    return result


def read_csv_chunks(path: str | Path, chunksize: int, encoding: str | None = None) -> Iterator[pd.DataFrame]:
    """Yield parsed CSV chunks without loading the complete input into memory."""
    reader = pd.read_csv(path, dtype="string", na_values=MISSING_VALUES, encoding=encoding or detect_encoding(path), chunksize=chunksize)
    yield from (parse_santander_dtypes(chunk) for chunk in reader)


def read_csv_dataframe(path: str | Path, encoding: str | None = None) -> pd.DataFrame:
    """Load a complete parsed DataFrame; use only for data that fits RAM."""
    frame = pd.read_csv(path, dtype="string", na_values=MISSING_VALUES, encoding=encoding or detect_encoding(path))
    return parse_santander_dtypes(frame)
