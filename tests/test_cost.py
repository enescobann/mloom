import pytest

from mloom.sdk.cost import PRICING_TABLE, calculate_cost


def test_known_model():
    # (1000 * 0.005 / 1000) + (500 * 0.015 / 1000)
    assert calculate_cost("gpt-4o", 1000, 500) == pytest.approx(0.0125)


def test_every_table_entry_is_priced():
    for model, prices in PRICING_TABLE.items():
        assert calculate_cost(model, 1000, 1000) == pytest.approx(prices["input"] + prices["output"])


def test_unknown_model_costs_nothing():
    assert calculate_cost("my-local-llama", 1000, 1000) == 0.0
    assert calculate_cost(None, 1000, 1000) == 0.0
    assert calculate_cost("", 1000, 1000) == 0.0


@pytest.mark.parametrize("snapshot, base", [
    ("gpt-4o-2024-08-06", "gpt-4o"),
    ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("gpt-3.5-turbo-0125", "gpt-3.5-turbo"),
    ("claude-3-haiku-20240307", "claude-3-haiku"),
])
def test_dated_snapshots_use_base_price(snapshot, base):
    assert calculate_cost(snapshot, 1234, 567) == calculate_cost(base, 1234, 567)


def test_mini_is_not_priced_as_full_model():
    assert calculate_cost("gpt-4o-mini", 1000, 0) < calculate_cost("gpt-4o", 1000, 0)


def test_missing_token_counts_are_treated_as_zero():
    assert calculate_cost("gpt-4o", None, None) == 0.0
    assert calculate_cost("gpt-4o", 1000, None) == pytest.approx(0.005)


def test_zero_tokens():
    assert calculate_cost("gpt-4o", 0, 0) == 0.0
