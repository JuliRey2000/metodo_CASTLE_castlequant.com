"""
comun.py - El Bucle del Bot y sus Comandos
==========================================
Lo que no depende de la estrategia:

- BaseBot: el bucle "esperar el cierre de la vela -> decidir -> registrar -> avisar".
- La cuenta demo: claves, verificación, orden de prueba y cierre de posición.
- run_cli: los comandos que acepta el bot (--check, --dry-run, ...).

La estrategia vive en `bot.py`, que solo define cómo se decide en cada vela.

Proyecto: Castle Quant · Método CASTLE (E · Ejecución)
Adaptado de la Mentoría Quant.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence

import avisos
import trading_system as ts
from binance_demo import BinanceDemoClient, BinanceDemoExchange, BinanceError
from market_data import floor_time, utc_now

BOT_DIR = Path(__file__).resolve().parent
ENV_FILE = BOT_DIR / ".env"               # Claves de la API demo y de Telegram (nunca se sube a GitHub)
CONFIG_FILE = BOT_DIR / "config.json"     # Capital, apalancamiento y banda del bot
LOG_DIR = BOT_DIR / "registros"           # Decisiones y estado del bot (se crea al ejecutar)

log = logging.getLogger("bot")


def sleep_until(target: datetime) -> None:
    """Duerme hasta `target` (UTC) en tramos cortos, para no desfasarse si el computador se suspende."""
    while (remaining := (target - utc_now()).total_seconds()) > 0:
        time.sleep(min(remaining, 30))


# ============================================================================
# BUCLE DEL BOT
# ============================================================================

class BaseBot:
    """
    Bucle común: una decisión cada vez que cierra una vela de duración `period`.

    La estrategia define `start` (ponerse al día y decidir sobre la última vela
    cerrada) y `step` (decidir sobre una vela recién cerrada).
    """

    def __init__(self, exchange, out_dir: Path, period: timedelta, close_delay: float = 5.0):
        self.exchange, self.period = exchange, period
        self.close_delay = timedelta(seconds=close_delay)
        self.state_file, self.decisions_file = out_dir / "estado.json", out_dir / "decisiones.csv"

    def start(self) -> datetime:
        """Pone el bot al día. Devuelve el inicio de la última vela cerrada."""
        raise NotImplementedError

    def step(self, bar: datetime) -> None:
        """Decide sobre la vela (ya cerrada) que empieza en `bar`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Estado y registros
    # ------------------------------------------------------------------

    def load_state(self) -> Optional[dict]:
        if not self.state_file.exists():
            return None
        state = json.loads(self.state_file.read_text())
        return {**state, "last_bar": datetime.fromisoformat(state["last_bar"])}

    def save_state(self, bar: datetime, **extra) -> None:
        """Guarda la última vela procesada y la posición que dejó el bot."""
        self.state_file.write_text(json.dumps({"last_bar": bar.isoformat(), "qty": self.exchange.qty, **extra}))

    def record(self, record: dict, trade: Optional[ts.Trade], signal: str) -> None:
        """Agrega la decisión a `decisiones.csv`, la muestra en pantalla y la envía por Telegram."""
        row = {**record, "order_qty": trade.signed_qty if trade else 0.0,
               "fill_price": trade.price if trade else None, "cost": trade.cost if trade else 0.0}
        is_new = not self.decisions_file.exists()
        with open(self.decisions_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        order = f"orden {trade.signed_qty:+.4f} a {trade.price:,.1f}" if trade else "sin orden"
        log.info("%s | cierre %s | %s | objetivo %+.1f%% | posición %+.1f%% | capital %s | %s",
                 f"{record['datetime']:%Y-%m-%d %H:%M}", f"{record['close']:,.1f}", signal, 100 * record["target"],
                 100 * record["position"], f"{record['equity']:,.2f}", order)
        avisos.send(f"Bot CASTLE · vela {record['datetime']:%Y-%m-%d %H:%M} UTC\n"
                    f"Cierre: {record['close']:,.1f}\n"
                    f"Señal: {signal}\n"
                    f"Objetivo {100 * record['target']:+.0f}% · posición {100 * record['position']:+.1f}%\n"
                    f"{order.capitalize()}")

    def warn_if_position_changed(self) -> None:
        """Avisa si la posición de la cuenta demo no es la que este bot dejó (órdenes manuales u otro bot en la misma cuenta)."""
        state = self.load_state()
        if not isinstance(self.exchange, BinanceDemoExchange) or state is None or "qty" not in state:
            return
        self.exchange.refresh()
        if abs(self.exchange.qty - state["qty"]) > float(self.exchange.filters["step"]) / 2:
            log.warning("La posición de la cuenta (%+.4f) no es la que dejó este bot (%+.4f). Si hay órdenes manuales u otro bot "
                        "en la misma cuenta demo, se estorban.", self.exchange.qty, state["qty"])

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(self, once: bool = False) -> None:
        """Una decisión por vela. Si algo falla, espera (cada vez más) y vuelve a ponerse al día."""
        backoff = 60
        while True:
            try:
                self.warn_if_position_changed()
                last = self.start()
                if once:
                    return
                while True:
                    bar = last + self.period
                    log.info("Próxima decisión: %s UTC", f"{bar + self.period:%Y-%m-%d %H:%M}")
                    sleep_until(bar + self.period + self.close_delay)
                    if floor_time(utc_now(), self.period) - self.period != bar:
                        log.warning("El bot se saltó una decisión (estuvo detenido o suspendido); se pone al día de nuevo")
                        break
                    self.warn_if_position_changed()
                    self.step(bar)
                    last, backoff = bar, 60
            except BinanceError as e:
                if e.is_auth:
                    raise
                log.error("%s. Reintento en %d s", e, backoff)
            except Exception:
                log.exception("Error inesperado. Reintento en %d s", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 900)


# ============================================================================
# CUENTA DEMO: CLAVES, VERIFICACIONES Y ÓRDENES MANUALES
# ============================================================================

def load_env(path: Path = ENV_FILE) -> None:
    """Carga en el entorno las variables de un archivo .env (sin sobrescribir las que ya existan)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_bot_config() -> dict:
    """Configuración del bot (`config.json`), validada."""
    config = json.loads(CONFIG_FILE.read_text())
    leverage = config["leverage"]
    if not isinstance(leverage, int) or isinstance(leverage, bool) or leverage < 1:
        sys.exit(f"`leverage` debe ser un número entero mayor o igual a 1 en {CONFIG_FILE} (valor actual: {leverage!r}).")
    for key in ("capital_usdt", "band", "slippage", "close_delay_seconds"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or config[key] < 0:
            sys.exit(f"`{key}` debe ser un número mayor o igual a 0 en {CONFIG_FILE} (valor actual: {config[key]!r}).")
    return config


def demo_client() -> BinanceDemoClient:
    """Cliente de la cuenta demo, con las claves del archivo .env."""
    key, secret = os.environ.get("BINANCE_DEMO_API_KEY", ""), os.environ.get("BINANCE_DEMO_API_SECRET", "")
    if not key or not secret:
        sys.exit(f"Faltan las claves. Copia bot/.env.example como {ENV_FILE} y pega ahí tus claves de Binance Demo Trading.")
    return BinanceDemoClient(key, secret)


def connect(sym: str, bot_config: dict, fee: float) -> BinanceDemoExchange:
    """Conecta con la cuenta demo y la deja lista para operar `sym`."""
    exchange = BinanceDemoExchange(demo_client(), sym, bot_config["capital_usdt"], bot_config["leverage"], fee)
    exchange.setup()
    return exchange


def print_account(exchange: BinanceDemoExchange) -> float:
    """Muestra el estado de la cuenta demo y devuelve el precio actual."""
    price = exchange.client.price(exchange.sym)
    f = exchange.filters
    print(f"Conexión correcta con {exchange.client.base_url}")
    print(f"  Saldo de la cuenta demo:   {exchange.client.margin_balance():,.2f} USDT")
    print(f"  Capital que usará el bot:  {exchange.equity:,.2f} USDT, apalancamiento {exchange.leverage}x, modo unidireccional")
    print(f"  Posición máxima:           ±{exchange.equity * exchange.leverage:,.2f} USDT ({exchange.leverage * 100}% del capital)")
    print(f"  Posición en {exchange.sym}:       {exchange.qty:+.4f} ({100 * exchange.position(price):+.1f}% del capital) | precio demo: {price:,.1f}")
    print(f"  Orden mínima:              {f['min_qty']} {exchange.sym[:-4]} y {f['min_notional']:.0f} USDT")
    return price


def test_order(exchange: BinanceDemoExchange) -> None:
    """Compra y vende la cantidad mínima: comprueba que la clave tiene permiso para operar futuros."""
    price = print_account(exchange)
    qty = max(float(exchange.filters["min_qty"]), 1.2 * exchange.filters["min_notional"] / price)
    for signed_qty in (qty, -qty):
        trade = exchange.market_order(ts.Order(exchange.sym, signed_qty), price, utc_now())
        print(f"  Orden ejecutada: {trade.signed_qty:+.4f} a {trade.price:,.1f} (comisión {trade.cost:.4f} USDT)")
    exchange.refresh()
    print(f"Prueba completada. Posición final: {exchange.qty:+.4f}")


def close_position(exchange: BinanceDemoExchange) -> None:
    price = print_account(exchange)
    if exchange.qty == 0:
        print("No hay posición abierta.")
        return
    trade = exchange.client.market_order(exchange.sym, exchange.round_qty(-exchange.qty))
    exchange.refresh()
    print(f"Posición cerrada a {float(trade['avgPrice']):,.1f}. Posición final: {exchange.qty:+.4f}")


# ============================================================================
# ARRANQUE
# ============================================================================

def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "bot.log", encoding="utf-8")])


def run_cli(description: str, sym: str, fee: float, make_bot: Callable[[object, Path, dict], BaseBot],
            argv: Optional[Sequence[str]] = None) -> None:
    """
    Lee los comandos, prepara la cuenta (demo o simulada) y enciende el bot.

    Args:
        description: Texto de ayuda del programa
        sym: Símbolo que opera
        fee: Comisión por lado, como fracción del valor negociado
        make_bot: Función (exchange, carpeta de registros, configuración del bot) -> bot
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--check", action="store_true", help="verifica las claves y muestra la cuenta demo, sin operar")
    parser.add_argument("--test-order", action="store_true", help="compra y vende la cantidad mínima en la cuenta demo")
    parser.add_argument("--close-position", action="store_true", help="cierra la posición abierta en la cuenta demo")
    parser.add_argument("--telegram", action="store_true", help="configura y prueba los avisos por Telegram")
    parser.add_argument("--dry-run", action="store_true", help="no envía órdenes: usa datos reales y un exchange simulado (no necesita claves)")
    parser.add_argument("--once", action="store_true", help="toma una sola decisión (la de la última vela cerrada) y termina")
    args = parser.parse_args(argv)
    load_env()
    bot_config = load_bot_config()

    try:
        if args.telegram:
            avisos.setup()
            return
        if args.check or args.test_order or args.close_position:
            command = test_order if args.test_order else close_position if args.close_position else print_account
            command(connect(sym, bot_config, fee))
            return

        out_dir = LOG_DIR / ("simulado" if args.dry_run else "demo")
        setup_logging(out_dir)
        if args.dry_run:
            exchange = ts.PaperExchange(bot_config["capital_usdt"], fee, bot_config["slippage"])
            log.info("MODO SIMULADO: datos reales, sin órdenes. Capital inicial: %s USDT | apalancamiento %dx",
                     f"{exchange.equity:,.2f}", bot_config["leverage"])
        else:
            exchange = connect(sym, bot_config, fee)
            log.info("MODO DEMO: órdenes en %s. Capital del bot: %s USDT | apalancamiento %dx | posición actual: %+.4f %s",
                     exchange.client.base_url, f"{exchange.equity:,.2f}", bot_config["leverage"], exchange.qty, sym)
        log.info("Avisos por Telegram: %s", "activados" if avisos.enabled() else "sin configurar (python bot/bot.py --telegram)")

        try:
            make_bot(exchange, out_dir, bot_config).run(once=args.once)
        except KeyboardInterrupt:
            log.info("Bot detenido. La posición abierta NO se cierra sola: usa --close-position si quieres cerrarla.")
    except BinanceError as e:
        hint = " Revisa que las claves sean de Binance DEMO Trading y tengan permiso de futuros." if e.is_auth else ""
        sys.exit(f"{e}.{hint}")
