"""Leakage-safe canonical persona features.

Raw Santander profile names are deliberately confined to this module.  All
model-facing code consumes the names in :data:`PERSONA_FEATURES` instead.
"""
from __future__ import annotations

from collections.abc import Sequence


RAW_PERSONA_COLUMNS: tuple[str, ...] = (
    "fecha_alta", "age", "renta", "ind_nuevo", "indrel", "indrel_1mes",
    "tiprel_1mes", "ult_fec_cli_1t", "ind_actividad_cliente", "ind_empleado",
    "sexo", "indfall", "pais_residencia", "cod_prov", "indresi",
    "canal_entrada", "segmento", "antiguedad", "tipodom", "indext",
)

# These are the only persona columns emitted into a model panel.
PERSONA_FEATURES: tuple[str, ...] = (
    "time_idx", "snapshot_month", "account_age_months", "age", "income_log",
    "is_new_customer", "customer_relationship_status", "is_active_customer",
    "employee_status", "is_male", "is_deceased", "country", "province",
    "is_domestic", "entry_channel", "customer_segment",
    "profile_missing_structural",
)
CATEGORICAL_PERSONA_FEATURES: tuple[str, ...] = (
    "snapshot_month", "customer_relationship_status", "employee_status", "country",
    "province", "entry_channel", "customer_segment",
)


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def raw_persona_projection_sql(source_alias: str, available_columns: Sequence[str]) -> str:
    """Select raw profile values under an internal ``raw_`` prefix.

    Missing legacy fields are NULL so feature-subset artifacts remain usable.
    """
    available = set(available_columns)
    expressions = []
    for name in RAW_PERSONA_COLUMNS:
        target = quote("raw_" + name)
        expressions.append(
            f"{source_alias}.{quote(name)} AS {target}" if name in available else f"NULL AS {target}"
        )
    return ", ".join(expressions)


def persona_feature_projection_sql(*, features: Sequence[str] = PERSONA_FEATURES) -> str:
    """Return canonical persona SQL in a ``customer_time`` window.

    Stable values use previous observations only; dynamic states are never
    forward-filled.  This makes the representation safe for temporal splits.
    """
    unknown = set(features).difference(PERSONA_FEATURES)
    if unknown:
        raise ValueError(f"Unknown canonical persona feature(s): {sorted(unknown)}")

    def raw(name: str) -> str:
        return quote("raw_" + name)

    def text(name: str) -> str:
        return f"NULLIF(TRIM(CAST({raw(name)} AS VARCHAR)), '')"

    def stable(value: str) -> str:
        return f"COALESCE({value}, LAST_VALUE({value} IGNORE NULLS) OVER prior_customer_rows)"

    age = f"TRY_CAST({raw('age')} AS DOUBLE)"
    income = f"CASE WHEN TRY_CAST({raw('renta')} AS DOUBLE) >= 0 THEN TRY_CAST({raw('renta')} AS DOUBLE) END"
    opening_date = f"TRY_CAST({raw('fecha_alta')} AS DATE)"
    relationship = (
        f"COALESCE(REPLACE({text('indrel_1mes')}, '.0', ''), {text('tiprel_1mes')}, "
        f"REPLACE({text('indrel')}, '.0', ''), CASE WHEN {raw('ult_fec_cli_1t')} IS NOT NULL THEN '99' END)"
    )
    structural = ("age", "antiguedad", "ind_nuevo", "indrel", "ind_actividad_cliente",
                  "ind_empleado", "indfall", "pais_residencia", "fecha_alta", "tipodom", "indresi", "indext")
    values: dict[str, str] = {
        "time_idx": "CAST(date_diff('month', MIN(CAST(fecha_dato AS DATE)) OVER (), CAST(fecha_dato AS DATE)) AS INTEGER)",
        "snapshot_month": "CAST(EXTRACT(MONTH FROM CAST(fecha_dato AS DATE)) AS INTEGER)",
        "account_age_months": f"CAST(date_diff('month', {stable(opening_date)}, CAST(fecha_dato AS DATE)) AS INTEGER)",
        "age": f"CAST({stable(age)} AS DOUBLE)",
        "income_log": f"LN(1 + {stable(income)})",
        "is_new_customer": f"CAST(CASE {text('ind_nuevo')} WHEN '1' THEN 1 WHEN '0' THEN 0 END AS TINYINT)",
        "customer_relationship_status": relationship,
        "is_active_customer": f"CAST(CASE {text('ind_actividad_cliente')} WHEN '1' THEN 1 WHEN '0' THEN 0 END AS TINYINT)",
        "employee_status": stable(text("ind_empleado")),
        "is_male": f"CAST(CASE {stable(text('sexo'))} WHEN 'H' THEN 1 WHEN 'V' THEN 0 END AS TINYINT)",
        "is_deceased": f"CAST(CASE {stable(text('indfall'))} WHEN 'S' THEN 1 WHEN 'N' THEN 0 END AS TINYINT)",
        "country": stable(text("pais_residencia")),
        "province": stable(text("cod_prov")),
        "is_domestic": f"CAST(CASE {stable(text('indresi'))} WHEN 'S' THEN 1 WHEN 'N' THEN 0 END AS TINYINT)",
        "entry_channel": stable(text("canal_entrada")),
        "customer_segment": text("segmento"),
        "profile_missing_structural": "CAST((" + " AND ".join(f"{raw(name)} IS NULL" for name in structural) + ") AS TINYINT)",
    }
    projections = []
    for name in features:
        value = values[name]
        # Keep model-panel categoricals as their meaningful values.  The
        # LightGBM adapter turns them into Pandas ``category`` columns at the
        # model boundary, where LightGBM assigns compact internal codes.
        # A missing profile value was previously represented explicitly by the
        # hash input; preserve that semantic state without hashing it.
        if name in CATEGORICAL_PERSONA_FEATURES and name != "snapshot_month":
            value = f"COALESCE(CAST({value} AS VARCHAR), 'MISSING')"
        projections.append(f"{value} AS {quote(name)}")
    return ", ".join(projections)
