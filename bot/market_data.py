"""
market_data.py - Datos del Mercado Real para el Bot
===================================================
Trae de Binance Futures (mercado real, datos públicos, sin claves) las velas ya
cerradas que el bot necesita para decidir.

Proyecto: Castle Quant · Método CASTLE (C · Captura, en vivo)
Adaptado de la Mentoría Quant.
"""

import time
from datetime import datetime, timedelta, timezone

import polars as pl
import requests

MARKET_URL = "https://fapi.binance.com"   # Mercado real: solo se leen datos públicos
WEIGHT_LIMIT = 1800                       # Binance permite 2400 de "peso" por minuto; se pausa antes de llegar


def utc_now() -> datetime:
    """Hora actual en UTC, sin zona horaria (igual que las fechas de los datos)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def floor_time(dt: datetime, period: timedelta) -> datetime:
    """Inicio de la barra de duración `period` que contiene a `dt` (las barras se cuentan desde las 00:00 UTC)."""
    epoch = datetime(1970, 1, 1)
    return epoch + ((dt - epoch) // period) * period


class MarketData:
    """Lector de datos públicos de un símbolo de Binance Futures."""

    def __init__(self, sym: str, base_url: str = MARKET_URL, retries: int = 4):
        self.sym, self.base_url, self.retries = sym, base_url, retries
        self._session = requests.Session()

    def _get(self, path: str, params: dict) -> list:
        """GET con reintentos (espera exponencial); se pausa si el consumo de la API se acerca al límite por minuto."""
        for attempt in range(self.retries):
            try:
                response = self._session.get(self.base_url + path, params=params, timeout=20)
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429:               # Demasiadas peticiones: esperar lo que indique Binance
                time.sleep(int(response.headers.get("Retry-After", 30)))
                continue
            response.raise_for_status()
            if int(response.headers.get("x-mbx-used-weight-1m", 0)) > WEIGHT_LIMIT:
                time.sleep(61 - datetime.now().second)    # Esperar al siguiente minuto
            return response.json()
        raise requests.RequestException(f"Binance sigue limitando las peticiones a {path}")

    def _klines(self, interval: str, start: datetime, end: datetime) -> list:
        """Velas de Binance, tal como las entrega la API, que empiezan en [start, end)."""
        rows, cursor, end_ms = [], to_ms(start), to_ms(end)
        while cursor < end_ms:
            batch = self._get("/fapi/v1/klines", {"symbol": self.sym, "interval": interval, "startTime": cursor,
                                                  "endTime": end_ms - 1, "limit": 1500})
            if not batch:
                break
            rows += batch
            cursor = batch[-1][6] + 1                     # Un milisegundo después del cierre de la última vela recibida
        return rows

    def candles(self, interval: str, period: timedelta, first: datetime, last: datetime, wait: float = 30.0) -> pl.DataFrame:
        """
        Velas OHLC ya cerradas, desde la que empieza en `first` hasta la que empieza en `last`.

        Args:
            interval: Intervalo de Binance ('12h', '1h', ...)
            period: La misma duración, como timedelta
        """
        expected, deadline = (last - first) // period + 1, time.time() + wait
        while True:
            rows = self._klines(interval, first, last + period)
            if len(rows) == expected:
                break
            if time.time() > deadline:
                raise ValueError(f"Binance devolvió {len(rows)} velas de {interval} entre {first} y {last}; se esperaban {expected}")
            time.sleep(2)
        candles = pl.DataFrame([r[:5] for r in rows], schema=["time", "open", "high", "low", "close"], orient="row").select(
            pl.from_epoch(pl.col("time").cast(pl.Int64), time_unit="ms").cast(pl.Datetime("us")).alias("datetime"),
            *[pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")],
        )
        if candles.select((pl.min_horizontal("open", "high", "low", "close") <= 0).any()).item():
            raise ValueError(f"Binance devolvió velas de {interval} con precios inválidos")
        return candles
