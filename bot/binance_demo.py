"""
binance_demo.py - Cuenta de Binance Demo Trading (Futuros USDⓈ-M)
=================================================================
Conecta el sistema de trading con una cuenta DEMO de futuros de Binance:

- BinanceDemoClient: peticiones firmadas a la API demo. Se niega a usar
  cualquier dirección que no sea la del entorno demo.
- BinanceDemoExchange: la misma interfaz que `trading_system.PaperExchange`,
  pero el capital, la posición y las órdenes viven en la cuenta demo.

Proyecto: Castle Quant · Método CASTLE (E · Ejecución)
Adaptado de la Mentoría Quant.
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse

import requests

import trading_system as ts

DEMO_URL = "https://demo-fapi.binance.com"
DEMO_HOSTS = {"demo-fapi.binance.com"}       # Únicos destinos permitidos para enviar órdenes
AUTH_ERRORS = {-2014, -2015, -1022, -1002}   # Clave inválida, sin permisos o firma incorrecta
MARGIN_BUFFER = 0.95                         # Nunca se usa más del 95% del saldo: deja margen para comisiones
MIN_ORDER_BUFFER = 1.05                      # La orden mínima se calcula un 5% sobre el mínimo de Binance, por si el precio se mueve

log = logging.getLogger("bot")


class BinanceError(Exception):
    """Error devuelto por la API de Binance."""

    def __init__(self, status: int, code: int, msg: str):
        super().__init__(f"Binance respondió {status} (código {code}): {msg}")
        self.status, self.code, self.msg = status, code, msg

    @property
    def is_auth(self) -> bool:
        return self.code in AUTH_ERRORS or self.status in (401, 403)


class BinanceDemoClient:
    """Cliente REST firmado (HMAC-SHA256) para la API de futuros de Binance Demo Trading."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = DEMO_URL):
        if urlparse(base_url).hostname not in DEMO_HOSTS:
            raise ValueError(f"{base_url} no es el entorno demo de Binance. Este bot solo opera en demo.")
        if not api_key or not api_secret:
            raise ValueError("Faltan las claves de la API demo (BINANCE_DEMO_API_KEY y BINANCE_DEMO_API_SECRET).")
        self.base_url, self._secret = base_url, api_secret.encode()
        self._session = requests.Session()
        self._session.headers["X-MBX-APIKEY"] = api_key
        self._time_offset = 0
        self._exchange_info: Optional[dict] = None
        self.sync_time()

    def sync_time(self) -> None:
        """Mide la diferencia entre el reloj local y el de Binance (las peticiones firmadas llevan la hora)."""
        server = self.request("GET", "/fapi/v1/time", signed=False)["serverTime"]
        self._time_offset = server - int(time.time() * 1000)

    def request(self, method: str, path: str, params: Optional[dict] = None, signed: bool = True):
        try:
            return self._send(method, path, params, signed)
        except BinanceError as e:
            if e.code != -1021:                  # -1021: el reloj local se desfasó; se vuelve a medir y se reintenta
                raise
            self.sync_time()
            return self._send(method, path, params, signed)

    def _send(self, method: str, path: str, params: Optional[dict], signed: bool):
        query = urlencode(params or {})
        if signed:
            query += ("&" if query else "") + f"recvWindow=5000&timestamp={int(time.time() * 1000) + self._time_offset}"
            query += "&signature=" + hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()
        response = self._session.request(method, f"{self.base_url}{path}?{query}", timeout=20)
        try:
            data = response.json()
        except ValueError:
            data = {"code": 0, "msg": response.text[:200]}
        if response.status_code != 200:
            raise BinanceError(response.status_code, data.get("code", 0), data.get("msg", ""))
        return data

    def _latest(self, path: str, params: Optional[dict] = None):
        """Consulta la versión 3 de un endpoint de cuenta y, si no existe en demo, la versión 2."""
        try:
            return self.request("GET", f"/fapi/v3/{path}", params)
        except BinanceError as e:
            if e.status != 404:
                raise
            return self.request("GET", f"/fapi/v2/{path}", params)

    def symbol_filters(self, sym: str) -> dict:
        """Reglas del símbolo en demo: tamaño mínimo y paso de la cantidad, y valor mínimo de una orden."""
        if self._exchange_info is None:              # Se descarga una sola vez: sirve para todos los símbolos
            self._exchange_info = self.request("GET", "/fapi/v1/exchangeInfo", signed=False)
        filters = {f["filterType"]: f for s in self._exchange_info["symbols"] if s["symbol"] == sym for f in s["filters"]}
        if not filters:
            raise ValueError(f"{sym} no existe en Binance Demo Trading")
        return {"step": Decimal(filters["MARKET_LOT_SIZE"]["stepSize"]),
                "min_qty": Decimal(filters["MARKET_LOT_SIZE"]["minQty"]),
                "min_notional": float(filters["MIN_NOTIONAL"]["notional"])}

    def price(self, sym: str) -> float:
        return float(self.request("GET", "/fapi/v1/ticker/price", {"symbol": sym}, signed=False)["price"])

    def margin_balance(self) -> float:
        """Saldo de la cuenta de futuros más la ganancia o pérdida de las posiciones abiertas (USDT)."""
        return float(self._latest("account")["totalMarginBalance"])

    def position_qty(self, sym: str) -> float:
        """Posición neta en el símbolo (positiva = largo, negativa = corto)."""
        return sum(float(p["positionAmt"]) for p in self._latest("positionRisk", {"symbol": sym}) if p["symbol"] == sym)

    def position_qtys(self) -> Dict[str, float]:
        """Posición neta de cada símbolo de la cuenta (los que no aparecen están en cero)."""
        qtys: Dict[str, float] = {}
        for p in self._latest("positionRisk"):
            qtys[p["symbol"]] = qtys.get(p["symbol"], 0.0) + float(p["positionAmt"])
        return qtys

    def is_hedge_mode(self) -> bool:
        return bool(self.request("GET", "/fapi/v1/positionSide/dual")["dualSidePosition"])

    def set_one_way_mode(self) -> None:
        self.request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"})

    def set_leverage(self, sym: str, leverage: int) -> None:
        self.request("POST", "/fapi/v1/leverage", {"symbol": sym, "leverage": leverage})

    def market_order(self, sym: str, signed_qty: Decimal, reduce_only: bool = False) -> dict:
        """
        Orden a mercado. Devuelve la orden ya ejecutada, con su precio promedio (`avgPrice`).

        Con `reduce_only` la orden solo puede reducir la posición; Binance no le exige el valor mínimo.
        """
        params = {"symbol": sym, "side": "BUY" if signed_qty > 0 else "SELL", "type": "MARKET",
                  "quantity": format(abs(signed_qty), "f"), "newOrderRespType": "RESULT"}
        if reduce_only:
            params["reduceOnly"] = "true"
        order = self.request("POST", "/fapi/v1/order", params)
        # En demo la respuesta al enviar no trae el precio de ejecución: se consulta la orden registrada
        return self.request("GET", "/fapi/v1/order", {"symbol": sym, "orderId": order["orderId"]})

    def order_commission(self, sym: str, order_id: int) -> float:
        """Comisión cobrada por una orden ya ejecutada."""
        fills = self.request("GET", "/fapi/v1/userTrades", {"symbol": sym, "orderId": order_id})
        return sum(float(f["commission"]) for f in fills)


class BinanceDemoExchange:
    """
    Cuenta demo con la interfaz de `PaperExchange`: `equity`, `qty`, `position`,
    `mark_to_market` y `market_order`.

    El capital del bot es fijo: `capital` USDT (o el 95% del saldo demo, si es menor).
    Las ganancias y pérdidas se quedan en la cuenta demo y no cambian el tamaño de las posiciones.

    Con apalancamiento, la estrategia pide posiciones de hasta `leverage` veces ese capital.
    Binance reserva como margen posición / `leverage`, que así nunca supera el capital.
    """

    def __init__(self, client: BinanceDemoClient, sym: str, capital: float, leverage: int = 1, fee: float = 0.0):
        self.client, self.sym, self.capital, self.leverage, self.fee = client, sym, capital, leverage, fee
        self.equity, self.qty = 0.0, 0.0
        self.last_price: Optional[float] = None
        self.trades: List[ts.Trade] = []
        self.filters = client.symbol_filters(sym)

    def setup(self) -> None:
        """Deja la cuenta como la necesita el bot: una posición neta por símbolo y el apalancamiento indicado."""
        if self.client.is_hedge_mode():
            try:
                self.client.set_one_way_mode()
            except BinanceError as e:
                raise RuntimeError("La cuenta demo está en modo cobertura (hedge) y no se pudo cambiar. Cierra las posiciones "
                                   "abiertas y activa el modo unidireccional (One-way) en las preferencias de futuros.") from e
        self.client.set_leverage(self.sym, self.leverage)
        self.refresh()

    def refresh(self) -> None:
        """Lee de Binance el saldo y la posición actuales."""
        balance = self.client.margin_balance()
        self.equity = min(self.capital, MARGIN_BUFFER * balance)
        self.qty = self.client.position_qty(self.sym)
        if self.equity <= 0:
            raise RuntimeError(f"La cuenta demo no tiene saldo disponible (saldo: {balance:.2f} USDT)")

    def mark_to_market(self, price: float) -> None:
        self.refresh()
        self.last_price = price

    def position(self, price: float) -> float:
        """Exposición actual como fracción del capital del bot."""
        return self.qty * price / self.equity

    def round_qty(self, signed_qty: float) -> Decimal:
        """Ajusta la cantidad al paso que admite Binance."""
        step = self.filters["step"]
        return (Decimal(str(signed_qty)) / step).to_integral_value(ROUND_HALF_UP) * step

    def min_order_qty(self, price: float) -> Decimal:
        """Menor cantidad que Binance acepta en una orden a ese precio."""
        step = self.filters["step"]
        by_value = Decimal(str(MIN_ORDER_BUFFER * self.filters["min_notional"] / price))
        return max(self.filters["min_qty"], (by_value / step).to_integral_value(ROUND_CEILING) * step)

    def market_order(self, order: ts.Order, price: float, dt: datetime) -> Optional[ts.Trade]:
        """Envía la orden. Si es menor que el mínimo de Binance, envía la cantidad mínima en la misma dirección."""
        qty = self.round_qty(order.signed_qty)
        if qty == 0:                             # La posición ya está en el objetivo: no hay nada que ajustar
            return None
        min_qty = self.min_order_qty(price)
        if abs(qty) < min_qty:
            qty = min_qty.copy_sign(qty)
            log.info("Orden subida al mínimo de Binance: de %+.6f a %s %s (%.0f USDT)",
                     order.signed_qty, format(qty, "+f"), self.sym, float(min_qty) * price)
        return self.send(qty, price, dt)

    def send(self, qty: Decimal, price: float, dt: datetime, reduce_only: bool = False) -> ts.Trade:
        """Envía a Binance una cantidad ya ajustada al paso del símbolo y registra la ejecución."""
        result = self.client.market_order(self.sym, qty, reduce_only)
        filled = float(result["executedQty"]) * (1 if qty > 0 else -1)
        fill_price = float(result["avgPrice"]) or price
        try:
            cost = self.client.order_commission(self.sym, result["orderId"])
        except (BinanceError, requests.RequestException) as e:
            log.warning("No se pudo consultar la comisión de la orden (%s); se usa la estimada", type(e).__name__)
            cost = 0.0
        cost = cost or abs(filled) * fill_price * self.fee    # Si aún no se puede consultar, comisión estimada
        self.qty += filled
        trade = ts.Trade(dt, self.sym, filled, fill_price, cost)
        self.trades.append(trade)
        return trade

    def __repr__(self) -> str:
        return f"BinanceDemoExchange(equity={self.equity:.2f}, qty={self.qty:.6f}, trades={len(self.trades)})"
