"""CSV encoding detection and Santander schema normalisation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import pandas as pd

DATE_COLUMNS = ("fecha_dato", "fecha_alta", "ult_fec_cli_1t")
INTEGER_COLUMNS = ("ncodpers", "ind_nuevo", "antiguedad", "indrel", "tipodom", "cod_prov", "ind_actividad_cliente")
FLOAT_COLUMNS = ("age", "renta")
PRODUCT_COLUMNS = ("ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1", "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1")
# Santander's CSV contains missing numeric fields written as both ``NA`` and
# `` NA``.  ``na_values`` is applied by pandas *before* ``dtype=`` coercion,
# so these spellings must be declared here; otherwise parsing a nullable
# numeric column fails before our per-chunk normalisation runs.
MISSING_VALUES = ("", " ", "NA", " NA", "NA ", " N/A", "N/A", "N/A ", "Unknown")
NORMALIZED_MISSING_VALUES = frozenset({value.strip() for value in MISSING_VALUES})

# Final nullable dtypes written to the Parquet checkpoint.  These are applied
# after raw text has been normalised for each chunk rather than passed directly
# to ``read_csv``: the source has occasional whitespace-padded missing values
# (for example ``" NA"``), which pandas may try to cast before ``na_values``
# is handled consistently across parser versions.
CSV_DTYPES = {
    **{column: "Int64" for column in INTEGER_COLUMNS},
    **{column: "Float32" for column in FLOAT_COLUMNS},
    **{column: "Int8" for column in PRODUCT_COLUMNS},
}


def detect_encoding(path: str | Path) -> str:
    """Return UTF-8 when valid; otherwise use a Latin-1-compatible encoding."""
    source = Path(path)
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"ZIP input must contain exactly one data file: {source}")
            with archive.open(members[0]) as file:
                sample = file.read(1_000_000)
    else:
        with source.open("rb") as file:
            sample = file.read(1_000_000)
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def parse_santander_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise one chunk to the stable schema used by every Parquet row group."""
    typed_columns = set(DATE_COLUMNS) | set(CSV_DTYPES)
    # CSV inference happens independently per chunk.  A sparse categorical
    # field can therefore be inferred as float in an all-null chunk and string
    # in the next.  Canonicalising all remaining fields prevents schema drift.
    for column in frame.columns:
        if column in typed_columns:
            continue
        cleaned = frame[column].astype("string").str.strip()
        frame[column] = cleaned.mask(cleaned.isin(NORMALIZED_MISSING_VALUES))
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="coerce")
    for column in INTEGER_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(CSV_DTYPES[column])
    for column in FLOAT_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(CSV_DTYPES[column])
    for column in PRODUCT_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(CSV_DTYPES[column])
    return frame


def read_csv_chunks(
    path: str | Path,
    chunksize: int,
    encoding: str | None = None,
    show_progress: bool = False,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield parsed CSV chunks without loading the complete input into memory.

    ``show_progress`` displays a live chunk/row-rate indicator in notebooks and
    terminals without an expensive preliminary full-file row count.
    """
    source = Path(path)
    reader = pd.read_csv(
        source,
        dtype=str,
        na_values=MISSING_VALUES,
        skipinitialspace=True,
        keep_default_na=False,
        na_filter=False,
        encoding=encoding or detect_encoding(source),
        compression="zip" if source.suffix.lower() == ".zip" else "infer",
        chunksize=chunksize,
        usecols=usecols,
        low_memory=False,
    )
    iterator = reader
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError as error:
            raise RuntimeError("Install tqdm with: pip install -r requirements.txt") from error
        iterator = tqdm(reader, desc=f"Parsing {Path(path).name}", unit="chunk", dynamic_ncols=True)
    for chunk in iterator:
        yield parse_santander_dtypes(chunk)


def read_csv_columns(path: str | Path, encoding: str | None = None) -> list[str]:
    """Read only the CSV header, without materialising data rows."""
    source = Path(path)
    header = pd.read_csv(
        source,
        nrows=0,
        dtype=str,
        encoding=encoding or detect_encoding(source),
        compression="zip" if source.suffix.lower() == ".zip" else "infer",
    )
    return list(header.columns)
