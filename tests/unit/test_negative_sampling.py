from src.models.lightgbm_binary import _negative_order_sql


def test_random_negative_order_is_deterministic_and_product_specific() -> None:
    first = _negative_order_sql("random", random_state=42, product_ordinal=1)
    assert first == _negative_order_sql("random", random_state=42, product_ordinal=1)
    assert first != _negative_order_sql("random", random_state=42, product_ordinal=2)
    assert "hash(ncodpers, fecha_dato, 43)" in first


def test_head_negative_order_preserves_baseline_behavior() -> None:
    assert _negative_order_sql("head", random_state=42, product_ordinal=1) == ""
