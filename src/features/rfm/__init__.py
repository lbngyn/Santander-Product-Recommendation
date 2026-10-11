"""RFM feature builders, with backwards-compatible public exports."""
from .recency import rfm_recency_months_sql, rfm_never_acquired_before_sql
from .frequency import rfm_frequency_sql
from .monetary_porfolio_deep import rfm_monetary_sql
from .history import rfm_acquisition_count_sql, rfm_history_coverage_sql

RFM_FEATURE_NAMES: tuple[str, ...] = (
    "rfm_has_previous_record",
    "rfm_recency_months",
    "rfm_never_acquired_before",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_observed_history_records",
)
