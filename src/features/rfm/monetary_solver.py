"""Select monetary inputs and tier builders without changing legacy features."""
from dataclasses import dataclass

from . import monetary_porfolio_deep, monetary_rental, monetary_rarity_rental


@dataclass(frozen=True)
class MonetaryProxy:
    name: str
    column: str
    transform: str
    module: object

    def value_sql(self):
        from src.features.customer_history import quote
        return (monetary_rental.monetary_value_sql(self.column)
                if self.transform == "log1p" else quote(self.column))

    def tier_sql(self, column, boundaries):
        return self.module.monetary_tier_sql(column, boundaries)


MONETARY_PROXIES = {
    "porfolio_deep": MonetaryProxy("porfolio_deep", "rfm_monetary", "identity", monetary_porfolio_deep),
    "rental": MonetaryProxy("rental", "renta_filled", "log1p", monetary_rental),
    "rarity_rental": MonetaryProxy("rarity_rental", monetary_rarity_rental.FEATURE_NAME, "identity", monetary_rarity_rental),
}


def solve_monetary_proxy(monetary_proxy="porfolio_deep"):
    if monetary_proxy not in MONETARY_PROXIES:
        raise ValueError(f"Unknown monetary_proxy {monetary_proxy!r}; choose {tuple(MONETARY_PROXIES)}")
    return MONETARY_PROXIES[monetary_proxy]
