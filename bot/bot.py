"""
bot.py - El Bot del Método CASTLE en Binance Demo Trading
=========================================================
Ejecuta en vivo la estrategia de los notebooks. Cada 12 horas (00:00 y 12:00 UTC), al cerrar la vela:

    C · Captura   cierres de las últimas velas de 12h del mercado real
    A · Alfa      retornos logarítmicos -> modelo lineal -> predicción
    S T L         signo de la predicción -> posición objetivo (la regla que se simuló en el notebook 2)
    E · Ejecución orden a mercado en la cuenta DEMO, registro en CSV y aviso por Telegram

Uso (desde la carpeta metodo_CASTLE_castlequant.com):

    python bot/bot.py --dry-run --once     Una decisión con datos reales y exchange simulado (no necesita claves)
    python bot/bot.py --check              Verifica las claves y muestra la cuenta demo
    python bot/bot.py --test-order         Compra y vende el mínimo, para probar que la cuenta opera
    python bot/bot.py --telegram           Configura y prueba los avisos por Telegram
    python bot/bot.py                      Enciende el bot en la cuenta demo
    python bot/bot.py --close-position     Cierra la posición abierta en la cuenta demo

Solo opera en demo, con dinero ficticio. Es material educativo, no una recomendación de inversión.

Proyecto: Castle Quant · Método CASTLE (E · Ejecución)
"""

import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Optional, Sequence

BOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = BOT_DIR.parent                   # Carpeta metodo_CASTLE_castlequant.com: ahí están models.py, binance.py y el modelo entrenado
for folder in (BOT_DIR, ROOT_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
import torch

import binance              # Comisiones (metodo_CASTLE_castlequant.com/binance.py)
import models               # Modelo de PyTorch (metodo_CASTLE_castlequant.com/models.py)
import trading_system as ts
from comun import BaseBot, log, run_cli, sleep_until
from market_data import MarketData, floor_time, utc_now

SYM = "BTCUSDT"
INTERVAL = "12h"                # Velas de la estrategia
PERIOD = timedelta(hours=12)
LAGS = (1, 3)                   # Rezagos que usa el modelo: los que eligió el notebook 1


class LogReturnLags(ts.Tick[float, Optional[torch.Tensor]]):
    """
    Recibe precios de cierre uno a uno y entrega los retornos logarítmicos de los rezagos pedidos.
    Es el componente del notebook 3.
    """

    def __init__(self, lags: Sequence[int]):
        self.lags = list(lags)
        self._returns: Deque[float] = deque(maxlen=max(lags))     # Del más reciente al más antiguo
        self._last_price: Optional[float] = None

    def on_tick(self, price: float) -> Optional[torch.Tensor]:
        if self._last_price is not None:
            self._returns.appendleft(float(np.log(price / self._last_price)))
        self._last_price = price
        if len(self._returns) < self._returns.maxlen:
            return None
        return torch.tensor([self._returns[lag - 1] for lag in self.lags], dtype=torch.float32)


class CastleBot(BaseBot):
    """Decide cada 12 horas con el modelo lineal del notebook 1 y registra cada decisión."""

    def __init__(self, exchange, market: MarketData, out_dir: Path, bot_config: dict):
        super().__init__(exchange, out_dir, PERIOD, bot_config["close_delay_seconds"])
        self.market, self.leverage = market, bot_config["leverage"]
        self.model = models.load_model(ROOT_DIR / "model_weights.pth")
        # Con la banda, la posición solo se toca cuando cambia la dirección: lo mismo que se simuló en el notebook 2
        self.strategy = ts.TargetStrategy(SYM, band=bot_config["band"])

    def predict(self, closes: Sequence[float]) -> float:
        """Retorno logarítmico que el modelo espera para la próxima vela, dados los últimos cierres."""
        lags = LogReturnLags(LAGS)
        for price in closes:
            x = lags.on_tick(price)
        with torch.no_grad():
            return float(self.model(x).item())

    def step(self, bar: datetime) -> None:
        """Decide sobre la vela (ya cerrada) que empieza en `bar`."""
        # C · Captura
        closes = self.market.candles(INTERVAL, PERIOD, bar - PERIOD * max(LAGS), bar)["close"].to_list()
        # A · Alfa
        y_hat, price = self.predict(closes), closes[-1]
        # S, T, L · la regla del notebook 2: todo el capital (por el apalancamiento) en la dirección predicha
        self.exchange.mark_to_market(price)
        target = float(np.sign(y_hat)) * self.leverage
        order = self.strategy.on_tick(target, price, self.exchange)
        # E · Ejecución
        trade = self.exchange.market_order(order, price, bar) if order is not None else None

        record = {"datetime": bar, "close": price, "y_hat": y_hat, "target": target,
                  "position": self.exchange.position(price), "equity": self.exchange.equity}
        self.record(record, trade, f"predicción {100 * y_hat:+.3f}% ({'LARGO' if target > 0 else 'CORTO'})")
        self.save_state(bar)

    def start(self) -> datetime:
        """Decide sobre la última vela cerrada, salvo que ya lo haya hecho antes de un reinicio. Devuelve el inicio de esa vela."""
        sleep_until(floor_time(utc_now(), PERIOD) + self.close_delay)
        last_closed = floor_time(utc_now(), PERIOD) - PERIOD
        state = self.load_state()
        if state and state["last_bar"] == last_closed:
            log.info("La vela de %s ya se decidió antes del reinicio; se espera a la siguiente", f"{last_closed:%Y-%m-%d %H:%M}")
        else:
            self.step(last_closed)
        return last_closed


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli("Bot del Método CASTLE en Binance Demo Trading (futuros).", SYM, binance.TAKER_FEE,
            lambda exchange, out_dir, bot_config: CastleBot(exchange, MarketData(SYM), out_dir, bot_config), argv)


if __name__ == "__main__":
    main()
