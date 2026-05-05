"""Tests for BuilderHelperService."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.builder_helper_service import BuilderHelperService
from services.price_service import PriceResult, PriceSource


def _make_price_service(jita_sell_prices: dict[int, float]):
    """Build a price-service stub returning the supplied sell prices."""
    price_service = MagicMock()
    price_service.get_jita_price_data_map.return_value = {
        tid: PriceResult.success_result(
            type_id=tid,
            sell_price=price,
            source=PriceSource.JITA_DATABASE,
        )
        for tid, price in jita_sell_prices.items()
    }
    return price_service


class TestBuilderHelperService:
    @patch("httpx.AsyncClient", side_effect=AssertionError("EverRef must not be called"))
    def test_get_builder_data_uses_all_local_builder_cost_rows(self, mock_async_client):
        first_type_id = 24698
        second_type_id = 999_999

        market_repo = MagicMock()
        market_repo.get_builder_cost_catalog.return_value = pd.DataFrame(
            [
                {
                    "type_id": first_type_id,
                    "type_name": "Included Item",
                    "group_id": 1,
                    "group_name": "Included Group",
                    "category_id": 2,
                    "category_name": "Included Category",
                    "total_cost_per_unit": 12_500_000.0,
                    "time_per_unit": 900.0,
                    "me": 8,
                    "runs": 3,
                    "fetched_at": "2026-05-04 09:15:00",
                },
                {
                    "type_id": second_type_id,
                    "type_name": "Excluded Item",
                    "group_id": 1,
                    "group_name": "Excluded Group",
                    "category_id": 2,
                    "category_name": "Excluded Category",
                    "total_cost_per_unit": 99_999_999.0,
                    "time_per_unit": 900.0,
                    "me": 8,
                    "runs": 3,
                    "fetched_at": "2026-05-04 09:15:00",
                },
            ]
        )
        market_repo.get_all_stats.return_value = pd.DataFrame(
            [
                {"type_id": first_type_id, "price": 15_000_000.0},
                {"type_id": second_type_id, "price": 20_000_000.0},
            ]
        )
        market_repo.get_30day_volume_metrics.return_value = pd.DataFrame(
            [
                {"type_id": first_type_id, "volume_30d": 20.0},
                {"type_id": second_type_id, "volume_30d": 10.0},
            ]
        )

        price_service = _make_price_service({
            first_type_id: 10_000_000.0,
            second_type_id: 12_000_000.0,
        })

        service = BuilderHelperService(market_repo, price_service)
        result = service.get_builder_data()

        assert list(result["type_id"]) == [first_type_id, second_type_id]
        first_row = result.iloc[0]
        second_row = result.iloc[1]
        assert first_row["item_name"] == "Included Item"
        assert first_row["market_sell_price"] == 15_000_000.0
        assert first_row["jita_sell_price"] == 10_000_000.0
        assert first_row["build_cost"] == 12_500_000.0
        assert first_row["cap_utils"] == pytest.approx((15_000_000.0 - 12_500_000.0) / 15_000_000.0)
        assert first_row["profit_30d"] == pytest.approx(50_000_000.0)
        assert first_row["turnover_30d"] == pytest.approx(200_000_000.0)
        assert first_row["volume_30d"] == 20.0

        assert second_row["item_name"] == "Excluded Item"
        assert second_row["market_sell_price"] == 20_000_000.0
        assert second_row["jita_sell_price"] == 12_000_000.0
        assert second_row["build_cost"] == 99_999_999.0
        assert second_row["profit_30d"] == pytest.approx((20_000_000.0 - 99_999_999.0) * 10.0)
        assert second_row["turnover_30d"] == pytest.approx(120_000_000.0)
        assert second_row["volume_30d"] == 10.0

        market_repo.get_builder_cost_catalog.assert_called_once()
        market_repo.get_all_stats.assert_called_once()
        market_repo.get_30day_volume_metrics.assert_called_once_with([first_type_id, second_type_id])
        market_repo.get_sde_info.assert_not_called()
        price_service.get_jita_price_data_map.assert_called_once_with([first_type_id, second_type_id])
        assert mock_async_client.call_count == 0

    @patch("httpx.AsyncClient", side_effect=AssertionError("EverRef must not be called"))
    def test_get_builder_data_returns_empty_when_local_catalog_missing(self, mock_async_client):
        market_repo = MagicMock()
        market_repo.get_builder_cost_catalog.return_value = pd.DataFrame()
        price_service = _make_price_service({})

        service = BuilderHelperService(market_repo, price_service)
        result = service.get_builder_data()

        assert result.empty
        market_repo.get_all_stats.assert_not_called()
        market_repo.get_30day_volume_metrics.assert_not_called()
        market_repo.get_sde_info.assert_not_called()
        price_service.get_jita_price_data_map.assert_not_called()
        assert mock_async_client.call_count == 0

    @patch("httpx.AsyncClient", side_effect=AssertionError("EverRef must not be called"))
    def test_missing_local_market_price_leaves_market_sell_price_none(self, mock_async_client):
        """When no local marketstats row exists, market_sell_price must stay None
        and dependent metrics (cap_utils, profit_30d) must also be None — no
        synthesized fallback is allowed."""
        type_id = 24698

        market_repo = MagicMock()
        market_repo.get_builder_cost_catalog.return_value = pd.DataFrame(
            [
                {
                    "type_id": type_id,
                    "type_name": "Item",
                    "group_id": 1,
                    "group_name": "Group",
                    "category_id": 2,
                    "category_name": "Category",
                    "total_cost_per_unit": 12_500_000.0,
                    "time_per_unit": 900.0,
                    "me": 8,
                    "runs": 3,
                    "fetched_at": "2026-05-04 09:15:00",
                },
            ]
        )
        market_repo.get_all_stats.return_value = pd.DataFrame(columns=["type_id", "price"])
        market_repo.get_30day_volume_metrics.return_value = pd.DataFrame(
            [{"type_id": type_id, "volume_30d": 20.0}]
        )

        price_service = _make_price_service({type_id: 10_000_000.0})

        service = BuilderHelperService(market_repo, price_service)
        result = service.get_builder_data()

        row = result.iloc[0]
        assert row["jita_sell_price"] == 10_000_000.0
        assert row["market_sell_price"] is None
        assert row["cap_utils"] is None
        assert row["profit_30d"] is None
        # turnover_30d uses jita * volume so it should still resolve.
        assert row["turnover_30d"] == pytest.approx(200_000_000.0)
        assert mock_async_client.call_count == 0
