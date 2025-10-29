import json
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestConnectionError

from .logger import get_logger

logger = get_logger()

ParamsT = dict[str, Any] | None
PostDataT = dict[str, Any] | list[dict[str, Any]] | None


class APIClient:
    def __init__(
        self,
        serverurl: str,
        username: str | None = None,
        password: str | None = None,
        *,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
    ) -> None:
        self._serverurl = serverurl
        self._session = requests.Session()

        # allow configuration of pool
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self._session.mount("http://", adapter)

        if username and password:
            self._session.auth = (username, password)

    def _call(
        self, method: str, apipath: str, params: dict[str, Any] | None = None, data: Any = None, files: Any = None
    ) -> dict[str, Any]:
        if str(method).upper() not in ("GET", "POST", "PUT", "DELETE"):
            raise ValueError(f"invalid method <{method}>")
        basepath = f"{self._serverurl}/api/v1/{apipath}"

        hd = {"Accept": "application/json", "Content-Type": "application/json"}

        # Split url
        schema, netloc, path, par, query, fragment = urlparse(basepath)
        # URLEncode query string
        query = urlencode(params) if params else ""
        # recombine url
        url = urlunparse((schema, netloc, path, par, query, fragment))
        res = {}
        try:
            resp = self._session.request(method, url, headers=hd, data=json.dumps(data))
            # return resp.text
            # print(f"resp.text:{resp.text}")
            res = resp.json()
        except RequestConnectionError:
            logger.warning(f"Connection error - could not connect to {netloc}.")
        return res

    def _get(self, apipath: str, params: ParamsT = None) -> dict[str, Any] | None:
        return self._call("GET", apipath, params=params)

    def _delete(self, apipath: str, params: ParamsT = None) -> dict[str, Any] | None:
        return self._call("DELETE", apipath, params=params)

    def _post(self, apipath: str, params: ParamsT = None, data: PostDataT = None) -> dict[str, Any] | None:
        return self._call("POST", apipath, params=params, data=data)

    def start(self) -> dict[str, Any] | None:
        """Start the bot if it's in the stopped state.

        :return: json object
        """
        return self._post("start")

    def stop(self, exit_all: bool = False, reports: bool = True) -> dict[str, Any] | None:
        """Stop the bot. Use `start` to restart.

        :return: json object
        """
        return self._post(
            "stop",
            data={
                "exit_all": exit_all,
                "reports": reports,
            },
        )

    def stop_entry(self) -> dict[str, Any] | None:
        """Stop buying (but handle sells gracefully). Use `reload_config` to reset.

        :return: json object
        """
        return self._post("stopentry")

    def reload_config(self) -> dict[str, Any] | None:
        """Reload configuration.

        :return: json object
        """
        return self._post("reload_config")

    def version(self) -> dict[str, Any] | None:
        """Return the version of the bot.

        :return: json object containing the version
        """
        return self._get("version")

    def show_config(self) -> dict[str, Any] | None:
        """Returns part of the configuration, relevant for trading operations.
        :return: json object containing the version
        """
        return self._get("show_config")

    def ping(self) -> dict[str, str]:
        """simple ping"""
        configstatus = self.show_config()
        if not configstatus:
            return {"status": "not_running"}
        elif configstatus["state"] == "running":
            return {"status": "pong"}
        else:
            return {"status": "not_running"}

    def logs(self, limit: int | None = None) -> dict[str, Any] | None:
        """Show latest logs.

        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.
        :return: json object
        """
        return self._get("logs", params={"limit": limit} if limit else {})

    def performance_entries(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Returns list of dicts containing all Trades, based on buy tag performance
        Can either be average for all pairs or a specific symbol provided

        :return: json object
        """
        return self._get("performance/entries", params={"symbol": symbol} if symbol else None)

    def performance_exits(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Returns list of dicts containing all Trades, based on exit reason performance
        Can either be average for all pairs or a specific symbol provided

        :return: json object
        """
        return self._get("performance/exits", params={"symbol": symbol} if symbol else None)

    def performance_mix_tags(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Returns list of dicts containing all Trades, based on entry_tag + exit_reason performance
        Can either be average for all pairs or a specific symbol provided

        :return: json object
        """
        return self._get("performance/mix_tags", params={"symbol": symbol} if symbol else None)

    def performance_daily(self, days: int | None = None) -> dict[str, Any] | None:
        """Return the profits for each day, and amount of trades.

        :return: json object
        """
        return self._get("performance/daily", params={"timescale": days} if days else None)

    def performance_weekly(self, weeks: int | None = None) -> dict[str, Any] | None:
        """Return the profits for each week, and amount of trades.

        :return: json object
        """
        return self._get("performance/weekly", params={"timescale": weeks} if weeks else None)

    def performance_monthly(self, months: int | None = None) -> dict[str, Any] | None:
        """Return the profits for each month, and amount of trades.

        :return: json object
        """
        return self._get("performance/monthly", params={"timescale": months} if months else None)

    def performance(self) -> dict[str, Any] | None:
        """Return the stats summary.

        :return: json object
        """
        return self._get("performance/stats")

    def performance_generic(self) -> dict[str, Any] | None:
        """Return the stats report (durations, sell-reasons).

        :return: json object
        """
        return self._get("performance/generic")

    def performance_symbol(self) -> dict[str, Any] | None:
        """Return the performance of the different coins.

        :return: json object
        """
        return self._get("performance/symbol")

    def balance(self) -> dict[str, Any] | None:
        """Get the account balance.

        :return: json object
        """
        return self._get("balance")

    def entry_count(self) -> dict[str, Any] | None:
        """Return the amount of open trades.

        :return: json object
        """
        return self._get("entrycount")

    def positions(self, symbol: str | None = None, open_only: bool = False) -> dict[str, Any] | None:
        """Get the status of open positions.

        :return: json object
        """
        return self._get("positions", params={"symbol": symbol, "open_only": open_only})

    def trades(
        self, symbol: str | None = None, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any] | None:
        """Return trades history, sorted by id

        :param limit: Limits trades to the X last trades. Max 500 trades.
        :param offset: Offset by this amount of trades.
        :return: json object
        """
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return self._get("trades", params=params)

    def tradepnls(
        self, symbol: str | None = None, limit: int | None = None, offset: int | None = None, open_only: bool = False
    ) -> dict[str, Any] | None:
        """Return specific trade

        :param symbol: Specify which trade to get.
        :return: json object
        """
        params: dict[str, Any] = {
            "open_only": open_only,
        }
        if symbol:
            params["symbol"] = symbol
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return self._get("tradepnls", params=params)

    def open_orders(
        self, symbol: str | None = None, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any] | None:
        """Return specific trade

        :param symbol: Specify which trade to get.
        :return: json object
        """
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return self._get("orders", params=params)

    def cancel_open_order(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        """Cancel open order for trade.

        :param trade_id: Cancels open orders for this trade.
        :return: json object
        """
        return self._delete(f"trades/{order_id}/{symbol}/open-order")

    def forceenter(
        self,
        symbol: str,
        side: str,
        price: float | None = None,
        *,
        quantity: float | None = None,
        cost: float | None = None,
        amount: float | None = None,
        leverage: float | None = None,
        enter_reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Force entering a trade

        :param symbol: Pair to buy (ETH/BTC)
        :param side: 'long' or 'short'
        :param price: Optional - price to open
        :param amount: Optional keyword argument - amount (as float)
        :param leverage: Optional keyword argument - leverage (as float)
        :param enter_reason: Optional keyword argument - entry reason (as string, default: 'force_enter')
        :return: json object of the trade
        """
        data: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
        }

        if price:
            data["price"] = price

        if quantity:
            data["quantity"] = quantity
        if cost:
            data["cost"] = cost
        if amount:
            data["amount"] = amount

        if leverage:
            data["leverage"] = leverage

        if enter_reason:
            data["enter_reason"] = enter_reason

        return self._post("forceenter", data=data)

    def forceexit(
        self, symbol: str, price: float | None = None, quantity: float | None = None
    ) -> dict[str, Any] | None:
        """Force-exit a trade.

        :param symbol: symbol of the trade (can be received via status command)
        :param price: Optional - price to open
        :param quantity: quantity to sell. Full sell if not given
        :return: json object
        """

        return self._post(
            "forceexit",
            data={
                "symbol": symbol,
                "price": price,
                "quantity": quantity,
            },
        )

    def sysinfo(self) -> dict[str, Any] | None:
        """Provides system information (CPU, RAM usage)

        :return: json object
        """
        return self._get("sysinfo")

    def health(self) -> dict[str, Any] | None:
        """Provides a quick health check of the running bot.

        :return: json object
        """
        return self._get("health")

    # def delete_trade(self, trade_id):
    #     """Delete trade from the database.
    #     Tries to close open orders. Requires manual handling of this asset on the exchange.

    #     :param trade_id: Deletes the trade with this ID from the database.
    #     :return: json object
    #     """
    #     return self._delete(f"trades/{trade_id}")

    # def whitelist(self):
    #     """Show the current whitelist.

    #     :return: json object
    #     """
    #     return self._get("whitelist")

    # def blacklist(self, *args):
    #     """Show the current blacklist.

    #     :param add: list of coins to add (example: "BNB/BTC")
    #     :return: json object
    #     """
    #     if not args:
    #         return self._get("blacklist")
    #     else:
    #         return self._post("blacklist", data={"blacklist": args})
