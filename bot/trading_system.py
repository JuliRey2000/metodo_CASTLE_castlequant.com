"""
trading_system.py - Órdenes, Exchange Simulado y Estrategia
===========================================================
Las piezas del notebook 3 en la forma que usa el bot: la orden, la operación
ejecutada, una cuenta simulada y la regla que lleva la posición a su objetivo.

Proyecto: Castle Quant · Método CASTLE (E · Ejecución)
Adaptado de la Mentoría Quant y del curso "Build a Quant Trading System" de MemLabs (licencia MIT).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, List, Optional, TypeVar

T = TypeVar('T')  # tipo de entrada
R = TypeVar('R')  # tipo de salida


class Tick(ABC, Generic[T, R]):
    @abstractmethod
    def on_tick(self, val: T) -> R:
        """Procesa un nuevo dato y, opcionalmente, devuelve un resultado."""
        pass


@dataclass(frozen=True)
class Order:
    sym: str
    signed_qty: float

    def __str__(self) -> str:
        side = "BUY" if self.signed_qty > 0 else "SELL"
        return f"Order({side} {abs(self.signed_qty):.6f} {self.sym})"


@dataclass(frozen=True)
class Trade:
    datetime: datetime
    sym: str
    signed_qty: float
    price: float
    cost: float


class PaperExchange:
    """
    Cuenta simulada con una posición neta por símbolo. Cada orden a mercado se ejecuta
    al precio indicado y paga (fee + slippage) sobre el valor negociado.
    """

    def __init__(self, capital: float, fee: float, slippage: float = 0.0):
        self.equity = capital
        self.cost_rate = fee + slippage
        self.qty = 0.0
        self.last_price: Optional[float] = None
        self.trades: List[Trade] = []

    def mark_to_market(self, price: float) -> None:
        """Actualiza el capital con la ganancia o pérdida de la posición abierta."""
        if self.last_price is not None:
            self.equity += self.qty * (price - self.last_price)
        self.last_price = price

    def position(self, price: float) -> float:
        """Exposición actual como fracción del capital."""
        return self.qty * price / self.equity

    def market_order(self, order: Order, price: float, dt: datetime) -> Trade:
        cost = abs(order.signed_qty) * price * self.cost_rate
        self.equity -= cost
        self.qty += order.signed_qty
        trade = Trade(dt, order.sym, order.signed_qty, price, cost)
        self.trades.append(trade)
        return trade

    def __repr__(self) -> str:
        return f"PaperExchange(equity={self.equity:.4f}, qty={self.qty:.6f}, trades={len(self.trades)})"


class TargetStrategy:
    """
    Decide la orden que lleva la posición actual a la objetivo. Si la diferencia no supera
    la banda, no opera: mientras la dirección no cambie, la posición se mantiene.
    """

    def __init__(self, sym: str, band: float):
        self.sym, self.band = sym, band

    def on_tick(self, target: float, price: float, exchange) -> Optional[Order]:
        if abs(target - exchange.position(price)) <= self.band:
            return None
        return Order(self.sym, target * exchange.equity / price - exchange.qty)
