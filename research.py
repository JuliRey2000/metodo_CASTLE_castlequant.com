"""
research.py - Herramientas de Investigación Cuantitativa
========================================================
Lo que usan los notebooks del Método CASTLE: agregación de velas, retornos y
rezagos, entrenamiento de modelos en PyTorch, métricas de trading, simulación
con comisiones y pruebas de sesgos.

Proyecto: Castle Quant · Método CASTLE
Adaptado de la Mentoría Quant y del curso "Build a Quant Trading System" de MemLabs (licencia MIT).
"""

from __future__ import annotations

import itertools
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim

import binance

# ============================================================================
# ESTILO DE LAS GRÁFICAS (colores de Castle Quant)
# ============================================================================

NAVY, LINEA, TEXTO, GRIS, AZUL, DORADO = "#0b1a2e", "#22395a", "#f5f7fa", "#8e9ab5", "#7fa3e6", "#cfb97d"
PALETA = [AZUL, DORADO, "#e07a7a", "#7fd1b9"]

plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": NAVY, "axes.edgecolor": LINEA, "axes.labelcolor": TEXTO,
    "text.color": TEXTO, "xtick.color": GRIS, "ytick.color": GRIS, "grid.color": LINEA, "axes.grid": True,
    "axes.titleweight": "bold", "figure.figsize": (12, 4.5), "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=PALETA),
})


# ============================================================================
# CAPTURA: DE VELAS DE 1 MINUTO A CUALQUIER TEMPORALIDAD
# ============================================================================

OHLC_AGGS = [
    pl.col("open").first(),     # Apertura: la de la primera vela de 1 minuto
    pl.col("high").max(),       # Máximo del intervalo
    pl.col("low").min(),        # Mínimo del intervalo
    pl.col("close").last(),     # Cierre: el de la última vela de 1 minuto
]


def timeseries(df: pl.DataFrame, time_interval: str, aggs: Union[List[pl.Expr], pl.Expr]) -> pl.DataFrame:
    """
    Agrega una tabla con columna 'datetime' en intervalos regulares ('1h', '6h', '12h', '1d', ...).

    Las barras se alinean a las 00:00 UTC y son cerradas por la izquierda: [inicio, fin).
    """
    return df.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(aggs)


def load_timeseries_range(sym: str, time_interval: str, start_date: datetime, end_date: datetime,
                          aggs: Union[List[pl.Expr], pl.Expr], cache_dir: str = "cache") -> pl.DataFrame:
    """Lee las velas de 1 minuto ya descargadas y las agrega con las expresiones dadas."""
    return timeseries(binance.load_klines(sym, start_date, end_date, cache_dir=cache_dir), time_interval, aggs)


def load_ohlc_timeseries_range(sym: str, time_interval: str, start_date: datetime, end_date: datetime,
                               cache_dir: str = "cache") -> pl.DataFrame:
    """Lee las velas de 1 minuto ya descargadas y arma las velas OHLC de la temporalidad pedida."""
    return load_timeseries_range(sym, time_interval, start_date, end_date, OHLC_AGGS, cache_dir)


def sharpe_annualization_factor(interval: str, trading_days_per_year: int = 365,
                                trading_hours_per_day: float = 24) -> float:
    """Raíz cuadrada del número de períodos por año para un intervalo como '1d', '12h', '30m'."""
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("El intervalo debe ser como '1d', '2h', '15m', '30s'")
    value, unit = int(match.group(1)), match.group(2)
    per_day = {"d": 1, "h": trading_hours_per_day, "m": trading_hours_per_day * 60, "s": trading_hours_per_day * 3600}[unit]
    return float(np.sqrt(trading_days_per_year * per_day / value))


# ============================================================================
# VARIABLES: RETORNOS LOGARÍTMICOS Y REZAGOS
# ============================================================================

def log_return_col(col: str) -> str:
    return f"{col}_log_return"


def log_return(col: str, shift_size: int = 1) -> pl.Expr:
    return (pl.col(col) / pl.col(col).shift(shift_size)).log().alias(log_return_col(col))


def lag_col_names(col: str, n: int) -> List[str]:
    return [f"{col}_lag_{i}" for i in range(1, n + 1)]


def add_lags(df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f"{col}_lag_{i}") for i in range(1, max_no_lags + 1)])


def add_log_return_features(df: pl.DataFrame, col: str, forecast_horizon: int, max_no_lags: int = 0) -> pl.DataFrame:
    """Agrega el retorno logarítmico de `col` y, si se piden, sus rezagos."""
    df = df.with_columns(log_return(col, forecast_horizon))
    if max_no_lags > 0:
        df = add_lags(df, log_return_col(col), max_no_lags, forecast_horizon)
    return df


def auto_reg_corr_matrx(df: pl.DataFrame, target: str, max_no_lags: int) -> pl.DataFrame:
    return df.drop_nulls().select([target] + lag_col_names(target, max_no_lags)).corr()


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def _x(df: pl.DataFrame):
    return df["datetime"].to_numpy() if "datetime" in df.columns else np.arange(len(df))


def plot_static_timeseries(ts: pl.DataFrame, sym: str, col: str, time_interval: str) -> None:
    plt.plot(_x(ts), ts[col].to_numpy(), lw=1.1)
    plt.title(f"{sym} · {col} · {time_interval}")
    plt.show()


def plot_column(df: pl.DataFrame, col_name: str, title: str = None) -> None:
    plt.plot(_x(df), df[col_name].to_numpy(), lw=1.3)
    plt.title(title or col_name)
    plt.show()


def plot_multiple_lines(df: pl.DataFrame, col_names: List[str], title: str = "") -> None:
    for col in col_names:
        plt.plot(_x(df), df[col].to_numpy(), lw=1.3, label=col)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_in_out_of_sample(in_sample: pl.DataFrame, out_of_sample: pl.DataFrame, col: str, title: str = "") -> None:
    """La misma curva en dos tramos seguidos: dentro de muestra y fuera de muestra."""
    plt.plot(_x(in_sample), in_sample[col].to_numpy(), lw=1.3, color=AZUL, label="Dentro de muestra")
    plt.plot(_x(out_of_sample), out_of_sample[col].to_numpy(), lw=1.3, color=PALETA[2], label="Fuera de muestra")
    plt.axvline(_x(out_of_sample)[0], color=DORADO, lw=1.2, ls="--")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_distribution(data: pl.DataFrame, col: str, no_bins: int = 100) -> None:
    plt.hist(data[col].drop_nulls().to_numpy(), bins=no_bins, color=AZUL)
    plt.title(f"Distribución de {col}")
    plt.show()


# ============================================================================
# MODELOS
# ============================================================================

def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch para reproducir los mismos resultados."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def init_weights(m: nn.Module) -> None:
    """Misma inicialización siempre: pesos y sesgo en cero (el modelo arranca sin opinión)."""
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def total_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module, model_name: str) -> None:
    print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}\n\nArchitecture:\n  {model}\n")
    print(f"Parameter Count:\n  Total parameters:      {total_model_params(model)}\n{'=' * 60}\n")


def print_model_params(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:\n{param.data.numpy()}")


def get_linear_params(model: nn.Module) -> Tuple[np.ndarray, float]:
    """Pesos y sesgo de un LinearModel como (w, b)."""
    return model.linear.weight.detach().numpy().flatten(), model.linear.bias.detach().numpy().item()


def least_squares_params(X: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Pesos y sesgo exactos de mínimos cuadrados (la fórmula cerrada), para comprobar el entrenamiento."""
    A = np.column_stack([X.numpy().astype(np.float64), np.ones(len(X))])
    solution = np.linalg.lstsq(A, y.numpy().astype(np.float64).ravel(), rcond=None)[0]
    return solution[:-1], float(solution[-1])


# ============================================================================
# PARTICIÓN TEMPORAL: ENTRENAMIENTO / VALIDACIÓN / PRUEBA
# ============================================================================

def timeseries_split(t, val_size: float = 0.2, test_size: float = 0.2):
    """
    Parte una serie en orden cronológico: (entrenamiento, validación, prueba).

    Con el entrenamiento se estiman los pesos, con la validación se elige el modelo
    y la prueba solo se abre una vez, al final.
    """
    if not (0 < val_size and 0 < test_size and val_size + test_size < 1):
        raise ValueError(f"val_size y test_size deben ser positivos y sumar menos de 1 (son {val_size} y {test_size})")
    val_idx, test_idx = int(len(t) * (1 - val_size - test_size)), int(len(t) * (1 - test_size))
    return t[:val_idx], t[val_idx:test_idx], t[test_idx:]


def to_tensors(df: pl.DataFrame, features: List[str], target: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(df[features].to_numpy(), dtype=torch.float32),
            torch.tensor(df[target].to_numpy(), dtype=torch.float32).reshape(-1, 1))


def timeseries_train_val_split(df: pl.DataFrame, features: List[str], target: str, val_size: float = 0.2,
                               test_size: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensores de entrenamiento y validación. El tramo de prueba no se devuelve: sigue sellado."""
    df_train, df_val, _ = timeseries_split(df.drop_nulls(), val_size, test_size)
    X_train, y_train = to_tensors(df_train, features, target)
    X_val, y_val = to_tensors(df_val, features, target)
    return X_train, X_val, y_train, y_val


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def batch_train_reg(model: nn.Module, X_train, X_val, y_train, y_val, no_epochs: int = 3000,
                    criterion=None, lr: float = 0.001, logging: bool = False) -> torch.Tensor:
    """Entrena con descenso de gradiente (Adam) sobre todo el entrenamiento y devuelve la predicción en validación."""
    if criterion is None:
        criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # La tasa de aprendizaje baja poco a poco hasta cero, para que los pesos terminen quietos
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epochs)
    log_tick_size = max(no_epochs // 10, 1)

    model.train()
    for epoch in range(no_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if logging and (epoch + 1) % log_tick_size == 0:
            print(f"Epoch [{epoch + 1}/{no_epochs}], Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_hat = model(X_val)
        if logging:
            print(f"\nValidation Loss: {criterion(y_hat, y_val).item():.6f}, Train Loss: {loss.item():.6f}")
    return y_hat


# ============================================================================
# MÉTRICAS DE TRADING
# ============================================================================

def model_trade_results(y_true, y_pred) -> pl.DataFrame:
    """Una fila por vela: predicción, retorno real, dirección tomada y curva de capital (escala logarítmica)."""
    return pl.DataFrame({
        "y_pred": np.asarray(y_pred, dtype=np.float64).ravel(),
        "y_true": np.asarray(y_true, dtype=np.float64).ravel(),
    }).with_columns(
        (pl.col("y_pred").sign() == pl.col("y_true").sign()).alias("is_won"),
        pl.col("y_pred").sign().alias("position"),
    ).with_columns(
        (pl.col("position") * pl.col("y_true")).alias("trade_log_return"),
    ).with_columns(
        pl.col("trade_log_return").cum_sum().alias("equity_curve"),
    ).with_columns(
        (pl.col("equity_curve") - pl.col("equity_curve").cum_max()).alias("drawdown_log_return"),
    )


def add_tx_fees_log(trades: pl.DataFrame, maker_fee: float, taker_fee: float) -> pl.DataFrame:
    """Curvas netas si cada vela paga una entrada y una salida (2 comisiones), como maker y como taker."""
    return trades.with_columns(
        (pl.col("trade_log_return") + np.log(1 - 2 * maker_fee)).alias("trade_log_return_net_maker"),
        (pl.col("trade_log_return") + np.log(1 - 2 * taker_fee)).alias("trade_log_return_net_taker"),
    ).with_columns(
        pl.col("trade_log_return_net_maker").cum_sum().alias("equity_curve_net_maker"),
        pl.col("trade_log_return_net_taker").cum_sum().alias("equity_curve_net_taker"),
    )


def eval_model_performance(y_actual, y_pred, feature_names: List[str], target_name: str, annualized_rate: float) -> Dict[str, any]:
    """Métricas de trading de un modelo: acierto, valor esperado, Sharpe y drawdown (antes de comisiones)."""
    trades = model_trade_results(y_actual, y_pred)
    returns = trades["trade_log_return"]
    win_rate = trades["is_won"].mean()
    avg_win = trades.filter(pl.col("is_won"))["trade_log_return"].mean()
    avg_loss = trades.filter(~pl.col("is_won"))["trade_log_return"].mean()
    std = returns.std()
    return {
        "features": ",".join(feature_names),
        "target": target_name,
        "no_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": returns.max(),
        "worst_trade": returns.min(),
        "ev": returns.mean(),
        "std": std,
        "total_log_return": returns.sum(),
        "compound_return": float(np.exp(returns.sum())),
        "max_drawdown": trades["drawdown_log_return"].min(),
        "equity_trough": trades["equity_curve"].min(),
        "equity_peak": trades["equity_curve"].max(),
        "sharpe": returns.mean() / std * annualized_rate if std > 0 else 0.0,
    }


def benchmark_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate: float,
                        val_size: float = 0.2, test_size: float = 0.2, loss=None, no_epochs: int = 3000) -> Dict[str, any]:
    """Entrena un modelo con el entrenamiento y lo mide en la validación."""
    X_train, X_val, y_train, y_val = timeseries_train_val_split(df, features, target, val_size, test_size)
    y_hat = batch_train_reg(model, X_train, X_val, y_train, y_val, no_epochs, loss)
    perf = eval_model_performance(y_val, y_hat, features, target, annualized_rate)
    weights, bias = get_linear_params(model)
    perf["weights"], perf["biases"] = str(np.round(weights, 5)), str(np.round(bias, 6))
    return perf


def benchmark_linear_models(ts: pl.DataFrame, target: str, feature_pool: List[str], annualized_rate: float,
                            max_no_features: int = 1, no_epochs: int = 3000, loss=None,
                            val_size: float = 0.2, test_size: float = 0.2) -> pl.DataFrame:
    """Compara en la validación todas las combinaciones de hasta `max_no_features` variables, ordenadas por Sharpe."""
    import models

    combos = []
    for i in range(1, max_no_features + 1):
        combos += list(itertools.combinations(feature_pool, i))

    benchmarks = []
    for features in combos:
        m = models.LinearModel(len(features))
        m.apply(init_weights)
        benchmarks.append(benchmark_reg_model(ts, list(features), target, m, annualized_rate, val_size, test_size, loss, no_epochs))
    return pl.DataFrame(benchmarks).sort("sharpe", descending=True)


def learn_model_trades(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, val_size: float = 0.2,
                       test_size: float = 0.2, loss=None, no_epochs: int = 3000) -> pl.DataFrame:
    """Entrena el modelo y devuelve sus operaciones en la validación."""
    X_train, X_val, y_train, y_val = timeseries_train_val_split(df, features, target, val_size, test_size)
    y_hat = batch_train_reg(model, X_train, X_val, y_train, y_val, no_epochs, loss)
    return model_trade_results(y_val, y_hat)


# ============================================================================
# SIMULACIÓN: DE LA PREDICCIÓN A LAS OPERACIONES
# ============================================================================

def add_model_predictions(trades: pl.DataFrame, model: nn.Module, features: Union[str, List[str]]) -> pl.DataFrame:
    if not isinstance(features, list):
        features = [features]
    with torch.no_grad():
        y_hat = model(torch.tensor(trades[features].to_numpy(), dtype=torch.float32))
    return trades.with_columns(pl.Series("y_hat", y_hat.numpy().ravel().astype(np.float64)))


def add_trade_log_returns(trades: pl.DataFrame) -> pl.DataFrame:
    """Dirección de cada operación, su retorno logarítmico y el acumulado."""
    return trades.with_columns(
        pl.col("y_hat").sign().alias("dir_signal"),
    ).with_columns(
        (pl.col("close_log_return") * pl.col("dir_signal")).alias("trade_log_return"),
    ).with_columns(
        pl.col("trade_log_return").cum_sum().alias("cum_trade_log_return"),
        no_trades_col(),
    )


def no_trades_col() -> pl.Expr:
    """Operaciones que exige cada vela: 1 al abrir, 2 al dar la vuelta (cerrar y abrir la contraria) y 0 si la dirección se mantiene."""
    return (pl.col("dir_signal") - pl.col("dir_signal").shift().fill_null(0)).abs().alias("no_trades")


def add_tx_fee(trades: pl.DataFrame, tx_fee: float, name: str) -> pl.DataFrame:
    """Comisión de cada vela: solo se paga cuando la posición cambia."""
    return trades.with_columns((pl.col("no_trades") * pl.col("entry_trade_value") * tx_fee).alias(f"tx_fee_{name}"))


def add_tx_fees(trades: pl.DataFrame, maker_fee: float, taker_fee: float) -> pl.DataFrame:
    return add_tx_fee(add_tx_fee(trades, maker_fee, "maker"), taker_fee, "taker")


def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
    return trades.with_columns((initial_capital + pl.col(col_name).cum_sum()).alias(f"equity_curve_{suffix}"))


def add_compounding_trades(trades: pl.DataFrame, capital: float, leverage: float, maker_fee: float, taker_fee: float) -> pl.DataFrame:
    """
    Operaciones con tamaño compuesto y apalancamiento: cada vela opera `leverage` veces el capital
    que dejó la anterior. Agrega el valor de entrada y salida, las comisiones y las curvas de capital.
    """
    gross_return = leverage * (pl.col("trade_log_return").exp() - 1)      # Ganancia o pérdida de la vela, sobre el capital

    def growth(fee: float) -> pl.Expr:
        # Factor por el que se multiplica el capital en la vela; si llega a cero, la cuenta queda en cero
        return (1 + gross_return - leverage * fee * pl.col("no_trades")).clip(lower_bound=0)

    trades = trades.with_columns(
        (capital * (1 + gross_return).clip(lower_bound=0).cum_prod()).alias("equity_curve_gross"),
        (capital * growth(taker_fee).cum_prod()).alias("equity_curve_taker"),
        (capital * growth(maker_fee).cum_prod()).alias("equity_curve_maker"),
    ).with_columns(
        (leverage * pl.col("equity_curve_taker").shift().fill_null(capital)).alias("entry_trade_value"),
    ).with_columns(
        (pl.col("entry_trade_value") * pl.col("trade_log_return").exp()).alias("exit_trade_value"),
        (pl.col("entry_trade_value") / pl.col("open") * pl.col("dir_signal")).alias("signed_trade_qty"),
    ).with_columns(
        (pl.col("exit_trade_value") - pl.col("entry_trade_value")).alias("trade_gross_pnl"),
    )
    return add_tx_fees(trades, maker_fee, taker_fee).with_columns(
        (pl.col("trade_gross_pnl") - pl.col("tx_fee_taker")).alias("trade_net_taker_pnl"),
        (pl.col("trade_gross_pnl") - pl.col("tx_fee_maker")).alias("trade_net_maker_pnl"),
    )


# ============================================================================
# LIBRE DE SESGOS
# ============================================================================

def permutation_test(trades: pl.DataFrame, no_permutations: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Retorno logarítmico total de `no_permutations` estrategias que operan las mismas velas
    con las mismas señales, pero barajadas al azar: lo que se gana sin saber nada.
    """
    rng = np.random.default_rng(seed)
    signal, returns = trades["dir_signal"].to_numpy(), trades["close_log_return"].to_numpy()
    return np.array([(rng.permutation(signal) * returns).sum() for _ in range(no_permutations)])


def plot_permutation_test(random_returns: np.ndarray, strategy_return: float) -> None:
    plt.hist(random_returns, bins=50, color=AZUL, label="Señales barajadas")
    plt.axvline(strategy_return, color=DORADO, lw=2.5, label="Nuestra estrategia")
    plt.title("Retorno logarítmico total: la estrategia contra el azar")
    plt.legend()
    plt.show()
