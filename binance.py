"""
binance.py - Captura de Datos de Mercado (Binance Futures)
==========================================================
Descarga las velas de 1 minuto del futuro perpetuo que publica Binance en
data.binance.vision (datos públicos, sin claves), las guarda en caché como
archivos parquet y las entrega listas para agregarlas en cualquier temporalidad.

También define las comisiones del exchange.

Proyecto: Castle Quant · Método CASTLE (C · Captura)
Adaptado de la Mentoría Quant y del curso "Build a Quant Trading System" de MemLabs (licencia MIT).
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl
import requests
from tqdm import tqdm

# Comisiones de futuros USDⓈ-M en Binance (nivel regular), por cada lado de la operación
MAKER_FEE = 0.0002      # 0,02 %: orden límite que agrega liquidez
TAKER_FEE = 0.0005      # 0,05 %: orden a mercado que toma liquidez

BASE_URL = "https://data.binance.vision/data/futures/um"
REQUEST_PAUSE = 0.05    # Pausa entre descargas, para no saturar el servidor
MAX_RETRIES = 4         # Reintentos por archivo, con espera exponencial

# Columnas de una vela, en el orden en que las publica Binance
KLINE_SCHEMA = {
    "open_time": pl.Int64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
    "volume": pl.Float64, "close_time": pl.Int64, "quote_volume": pl.Float64, "n_trades": pl.Int64,
    "buy_volume": pl.Float64, "buy_quote_volume": pl.Float64, "ignore": pl.Float64,
}
PRICE_COLS = ["open", "high", "low", "close"]


def _fetch(url: str) -> Optional[bytes]:
    """Descarga un archivo. Devuelve None si Binance aún no lo ha publicado (404)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 404:
                return None
            if response.status_code == 200:
                time.sleep(REQUEST_PAUSE)
                return response.content
            error = f"HTTP {response.status_code}"
        except requests.RequestException as e:
            error = type(e).__name__
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)        # 1 s, 2 s, 4 s
    raise ConnectionError(f"No se pudo descargar {url} tras {MAX_RETRIES} intentos ({error})")


def _read_klines(raw: bytes) -> pl.DataFrame:
    """Convierte el zip de Binance en una tabla de velas, y valida que los precios sean coherentes."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv = zf.read(zf.namelist()[0])
    has_header = not csv[:1].isdigit()      # Los archivos antiguos no traen encabezado
    df = pl.read_csv(io.BytesIO(csv), has_header=has_header, schema=KLINE_SCHEMA).select(
        pl.from_epoch("open_time", time_unit="ms").cast(pl.Datetime("us")).alias("datetime"),
        *PRICE_COLS, "volume", "quote_volume", "n_trades", "buy_volume", "buy_quote_volume",
    )
    valid = df.select(
        (pl.all_horizontal([pl.col(c) > 0 for c in PRICE_COLS])
         & (pl.col("high") >= pl.col("low")) & (pl.col("volume") >= 0)).all()
    ).item()
    if df.is_empty() or not valid:
        raise ValueError("El archivo de velas está vacío o trae precios inválidos")
    return df


def download_klines(symbol: str, period: str, interval: str = "1m", cache_dir: str = "cache") -> Optional[pl.DataFrame]:
    """
    Descarga las velas de un mes ('2026-10') o de un día ('2026-11-24') y las guarda en caché.

    Devuelve None si Binance aún no publica ese archivo (el mes en curso, por ejemplo).
    """
    cache_path = Path(cache_dir) / f"{symbol}-{interval}-{period}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    frequency = "monthly" if len(period) == 7 else "daily"
    raw = _fetch(f"{BASE_URL}/{frequency}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip")
    if raw is None:
        return None

    df = _read_klines(raw)
    cache_path.parent.mkdir(exist_ok=True)
    df.write_parquet(cache_path)
    return df


def _periods(start_date: datetime, end_date: datetime) -> List[str]:
    """Meses completos del rango como 'AAAA-MM' y los días sueltos de los meses incompletos como 'AAAA-MM-DD'."""
    periods, day = [], start_date
    while day <= end_date:
        next_month = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
        if day.day == 1 and next_month - timedelta(days=1) <= end_date:
            periods.append(f"{day:%Y-%m}")
            day = next_month
        else:
            periods.append(f"{day:%Y-%m-%d}")
            day += timedelta(days=1)
    return periods


def _days_of(month: str) -> List[str]:
    first = datetime.strptime(month, "%Y-%m")
    last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return [f"{first + timedelta(days=i):%Y-%m-%d}" for i in range(last.day)]


def download_date_range(symbol: str, start_date: str | datetime, end_date: str | datetime,
                        interval: str = "1m", cache_dir: str = "cache") -> None:
    """
    Descarga las velas de un rango de fechas (ambas incluidas) con barra de progreso.

    Usa un archivo por mes completo y uno por día en los meses incompletos.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    for period in tqdm(_periods(start_date, end_date), desc=f"Downloading {symbol}"):
        try:
            if download_klines(symbol, period, interval, cache_dir) is None:
                # Un mes sin archivo mensual (el mes en curso) se arma con sus archivos diarios
                missing = [d for d in (_days_of(period) if len(period) == 7 else [])
                           if download_klines(symbol, d, interval, cache_dir) is None]
                if missing or len(period) != 7:
                    tqdm.write(f"[AVISO] {symbol} {period}: Binance aún no publica {len(missing) or 1} día(s)")
        except (ConnectionError, ValueError, zipfile.BadZipFile) as e:
            tqdm.write(f"[ERROR] {symbol} {period}: {e}")


def load_klines(symbol: str, start_date: datetime, end_date: datetime,
                interval: str = "1m", cache_dir: str = "cache") -> pl.DataFrame:
    """Velas ya descargadas del rango de fechas (ambas incluidas), ordenadas y sin duplicados."""
    files = sorted(Path(cache_dir).glob(f"{symbol}-{interval}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay velas de {symbol} en '{cache_dir}/'. Ejecuta primero binance.download_date_range")
    df = pl.concat([pl.read_parquet(f) for f in files]).filter(
        (pl.col("datetime") >= start_date) & (pl.col("datetime") < end_date + timedelta(days=1))
    ).unique(subset=["datetime"]).sort("datetime")
    if df.is_empty():
        raise ValueError(f"No hay velas de {symbol} entre {start_date:%Y-%m-%d} y {end_date:%Y-%m-%d}")
    return df
