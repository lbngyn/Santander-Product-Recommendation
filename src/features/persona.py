"""Stable model representations for Santander's 20 persona fields."""
from __future__ import annotations

from collections.abc import Sequence


# The two raw date fields are deliberately excluded: they need a separately
# versioned lifecycle representation, rather than being passed as timestamps.
PERSONA_FEATURES: tuple[str, ...] = (
    "ind_empleado", "pais_residencia", "sexo", "age", "ind_nuevo",
    "antiguedad", "indrel", "indrel_1mes", "tiprel_1mes", "indresi",
    "indext", "conyuemp", "canal_entrada", "indfall", "tipodom",
    "cod_prov", "nomprov", "ind_actividad_cliente", "renta", "segmento",
)
CATEGORICAL_PERSONA_FEATURES: tuple[str, ...] = (
    "ind_empleado", "pais_residencia", "sexo", "indrel_1mes",
    "tiprel_1mes", "indresi", "indext", "conyuemp", "canal_entrada",
    "indfall", "nomprov", "segmento",
)


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def persona_projection_sql(
    source_alias: str,
    available_columns: Sequence[str],
    *,
    features: Sequence[str] = PERSONA_FEATURES,
) -> str:
    """Project every required persona field into stable numeric model inputs.

    Categorical values use a deterministic non-negative 31-bit hash. This
    avoids a train/test-fitted mapping while LightGBM receives their column
    names separately as categorical features; numeric persona fields preserve
    their original values.
    """
    unknown = set(features).difference(PERSONA_FEATURES)
    if unknown:
        raise ValueError(f"Unknown persona feature(s): {sorted(unknown)}")
    missing = set(features).difference(available_columns)
    if missing:
        raise ValueError(f"Persona source lacks required fields: {sorted(missing)}")
    categorical = set(CATEGORICAL_PERSONA_FEATURES)
    expressions = []
    for feature in features:
        column = f"{source_alias}.{quote(feature)}"
        if feature in categorical:
            expressions.append(
                f"CAST(hash(COALESCE(CAST({column} AS VARCHAR), '__MISSING__')) % 2147483647 AS INTEGER) AS {quote(feature)}"
            )
        else:
            expressions.append(f"{column} AS {quote(feature)}")
    return ", ".join(expressions)
