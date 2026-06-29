#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:51:20 2026

@author: davideactis
"""

#!/usr/bin/env python3
"""
Portfolio Optimization and Backtesting

This script implements portfolio optimization using:
- Equal-weight benchmark
- Mean-Variance Optimization
- Mean-Variance + CVaR penalty
- Low-risk Mean-Variance Optimization

The project downloads ETF price data, estimates optimized portfolio weights,
runs an annual rebalancing backtest, and compares out-of-sample performance.
"""

from typing import Dict, Tuple, Optional, Callable
import datetime as dt

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# Configuration
# =============================================================================

TICKERS = ["DBC", "GLD", "IEF", "TLT", "SPY", "VEA", "IEMG"]
PERIOD = "15y"
TRADING_DAYS = 252

FIGURE_DPI = 300


# =============================================================================
# Data
# =============================================================================

def download_price_data(tickers: list[str], period: str = PERIOD) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]

    prices = prices.dropna()
    return prices[tickers]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns."""
    return prices.pct_change().dropna()


# =============================================================================
# Performance Metrics
# =============================================================================

def max_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """Compute drawdown from a cumulative return series."""
    running_max = cumulative_returns.cummax()
    return (cumulative_returns - running_max) / running_max


def performance_stats(
    returns: pd.Series,
    scale: int = TRADING_DAYS,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """Compute annualized return, volatility, Sharpe ratio and max drawdown."""
    cumulative = (1 + returns.fillna(0)).cumprod()

    annual_return = cumulative.iloc[-1] ** (scale / len(cumulative)) - 1
    annual_volatility = returns.std() * np.sqrt(scale)
    sharpe_ratio = (annual_return - risk_free_rate) / (annual_volatility + 1e-8)
    mdd = max_drawdown(cumulative).min()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": mdd,
    }


# =============================================================================
# Portfolio Optimization
# =============================================================================

def mean_variance_optimizer(
    prices: pd.DataFrame,
    lookback: int = 500,
    risk_lambda: float = 5.0,
    cvar_lambda: float = 0.0,
    cvar_level: float = 0.95,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    leverage_constraint: Optional[float] = 1.0,
    concentration_constraint: Optional[float] = 0.7,
) -> np.ndarray:
    """
    Solve a constrained portfolio optimization problem.

    Objective:
        maximize expected return
        minus variance penalty
        minus optional CVaR penalty

    Constraints:
        sum(weights) = 1
        lower_bound <= weights <= upper_bound
        optional leverage constraint
        optional L2 concentration constraint
    """
    returns = prices.tail(lookback).pct_change().dropna()

    mean_returns = returns.mean().values.reshape(-1, 1)
    covariance_matrix = returns.cov().values

    n_assets = len(mean_returns)
    n_observations = len(returns)

    weights = cp.Variable((n_assets, 1))

    portfolio_return = weights.T @ mean_returns
    portfolio_variance = cp.quad_form(weights, covariance_matrix)

    losses = -returns.values @ weights
    var = cp.Variable()
    cvar = var + (1 / ((1 - cvar_level) * n_observations)) * cp.sum(
        cp.pos(losses - var)
    )

    constraints = [cp.sum(weights) == 1]

    lower_bound, upper_bound = weight_bounds
    constraints += [weights >= lower_bound, weights <= upper_bound]

    if leverage_constraint is not None:
        constraints += [cp.norm(weights, 1) <= leverage_constraint]

    if concentration_constraint is not None:
        constraints += [cp.norm(weights, 2) <= concentration_constraint]

    objective = cp.Maximize(
        portfolio_return
        - risk_lambda * portfolio_variance
        - cvar_lambda * cvar
    )

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Optimization failed. Status: {problem.status}")

    return np.asarray(weights.value).flatten()


# =============================================================================
# Backtesting
# =============================================================================

def generate_annual_rebalancing_dates(prices: pd.DataFrame) -> list[dt.date]:
    """Generate annual rebalancing dates after two years of history."""
    years = sorted(set(prices.index.year))

    start_year = years[2]
    end_year = years[-1]

    return [dt.date(year, 1, 1) for year in range(start_year, end_year + 1)]


def backtest_strategy(
    prices: pd.DataFrame,
    optimizer: Callable,
    optimizer_kwargs: Dict,
    rebalancing_dates: Optional[list[dt.date]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run annual rebalanced out-of-sample backtest."""
    if rebalancing_dates is None:
        rebalancing_dates = generate_annual_rebalancing_dates(prices)

    out_of_sample_returns = []
    weights_history = []

    for i, rebalance_date in enumerate(rebalancing_dates):
        in_sample_prices = prices[:rebalance_date]

        if len(in_sample_prices) < 500:
            continue

        if i + 1 < len(rebalancing_dates):
            next_rebalance_date = rebalancing_dates[i + 1]
            out_sample_prices = prices[rebalance_date:next_rebalance_date]
        else:
            out_sample_prices = prices[rebalance_date:]

        weights = optimizer(in_sample_prices, **optimizer_kwargs)
        weights_history.append(weights)

        period_returns = out_sample_prices.pct_change().dropna() @ weights
        out_of_sample_returns.append(period_returns)

    weights_df = pd.DataFrame(
        weights_history,
        index=rebalancing_dates[-len(weights_history):],
        columns=prices.columns,
    )

    return pd.concat(out_of_sample_returns), weights_df


# =============================================================================
# Plotting
# =============================================================================

def save_plot(filename: str) -> None:
    """Save and show figure."""
    plt.tight_layout()
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()


def plot_etf_cumulative_prices(prices: pd.DataFrame) -> None:
    """Plot cumulative ETF price performance."""
    cumulative_prices = (prices.pct_change().fillna(0) + 1).cumprod()

    cumulative_prices.plot(figsize=(9, 5))
    plt.title("Cumulative Performance of ETF Universe")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.legend(title="Ticker")
    save_plot("etf_cumulative_performance.png")


def plot_optimal_weights(weights: pd.Series) -> None:
    """Plot optimized portfolio weights."""
    weights.plot.bar(figsize=(8, 4))
    plt.title("Optimized Portfolio Weights")
    plt.ylabel("Weight")
    plt.xlabel("Ticker")
    save_plot("optimized_portfolio_weights.png")


def plot_strategy_comparison(
    benchmark_returns: pd.Series,
    strategy_returns: pd.Series,
    strategy_name: str,
    filename: str,
) -> None:
    """Plot cumulative performance of benchmark vs strategy."""
    benchmark_cum = (1 + benchmark_returns).cumprod()
    strategy_cum = (1 + strategy_returns).cumprod()

    plt.figure(figsize=(9, 5))
    plt.plot(benchmark_cum, label="Equal Weight")
    plt.plot(strategy_cum, label=strategy_name)
    plt.title(f"Performance Comparison: Equal Weight vs {strategy_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.legend()
    save_plot(filename)


def plot_out_of_sample_performance(
    returns: pd.Series,
    strategy_name: str,
    filename: str,
) -> None:
    """Plot cumulative returns and drawdown for an out-of-sample strategy."""
    stats = performance_stats(returns)
    cumulative = (1 + returns).cumprod()
    drawdown = max_drawdown(cumulative)

    ax = pd.DataFrame(cumulative).plot(
        figsize=(9, 5),
        title=(
            f"{strategy_name} – Out-of-Sample Backtest\n"
            f"Sharpe = {stats['sharpe_ratio']:.2f} | "
            f"MaxDD = {stats['max_drawdown']:.2f} | "
            f"Ann. Return = {stats['annual_return']:.2f} | "
            f"Vol = {stats['annual_volatility']:.2f}"
        ),
    )

    drawdown.plot(ax=ax, secondary_y=True, alpha=0.35, color="orange")
    ax.set_ylabel("Cumulative Value")
    ax.right_ax.set_ylabel("Drawdown")

    save_plot(filename)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run portfolio optimization and backtesting workflow."""
    prices = download_price_data(TICKERS)
    returns = compute_returns(prices)

    equal_weights = np.ones(len(TICKERS)) / len(TICKERS)
    equal_weight_returns = returns @ equal_weights

    plot_etf_cumulative_prices(prices)

    insample_prices = prices[:dt.date(2020, 1, 1)]
    insample_returns = insample_prices.pct_change().dropna()
    equal_weight_insample = equal_weight_returns.loc[insample_returns.index]

    config_mvo_variance = {
        "lookback": 500,
        "risk_lambda": 5.0,
        "cvar_lambda": 0.0,
        "weight_bounds": (0.0, 1.0),
        "leverage_constraint": 1.0,
        "concentration_constraint": 0.7,
    }

    config_mvo_cvar = {
        "lookback": 500,
        "risk_lambda": 3.0,
        "cvar_lambda": 5.0,
        "cvar_level": 0.95,
        "weight_bounds": (0.0, 1.0),
        "leverage_constraint": 1.0,
        "concentration_constraint": 0.7,
    }

    config_mvo_low_risk = {
        "lookback": 500,
        "risk_lambda": 20.0,
        "cvar_lambda": 0.0,
        "weight_bounds": (0.0, 1.0),
        "leverage_constraint": 1.0,
        "concentration_constraint": 0.7,
    }

    static_weights = mean_variance_optimizer(insample_prices, **config_mvo_variance)
    static_weights_series = pd.Series(static_weights, index=TICKERS)

    print("\nStatic optimized weights:")
    print(static_weights_series.round(4))
    print(f"Sum of weights: {static_weights_series.sum():.4f}")

    plot_optimal_weights(static_weights_series)

    static_mvo_returns = insample_returns @ static_weights

    plot_strategy_comparison(
        equal_weight_insample,
        static_mvo_returns,
        "Static MVO",
        "insample_equal_weight_vs_mvo.png",
    )

    rebalancing_dates = generate_annual_rebalancing_dates(prices)

    out_mvo_variance, weights_mvo_variance = backtest_strategy(
        prices,
        optimizer=mean_variance_optimizer,
        optimizer_kwargs=config_mvo_variance,
        rebalancing_dates=rebalancing_dates,
    )

    out_mvo_cvar, weights_mvo_cvar = backtest_strategy(
        prices,
        optimizer=mean_variance_optimizer,
        optimizer_kwargs=config_mvo_cvar,
        rebalancing_dates=rebalancing_dates,
    )

    out_mvo_low_risk, weights_mvo_low_risk = backtest_strategy(
        prices,
        optimizer=mean_variance_optimizer,
        optimizer_kwargs=config_mvo_low_risk,
        rebalancing_dates=rebalancing_dates,
    )

    equal_weight_oos = equal_weight_returns.loc[out_mvo_variance.index]

    strategies = {
        "Equal Weight": equal_weight_oos,
        "MVO Variance": out_mvo_variance,
        "MVO + CVaR": out_mvo_cvar,
        "MVO Low Risk": out_mvo_low_risk,
    }

    summary = pd.DataFrame({
        name: performance_stats(ret)
        for name, ret in strategies.items()
    }).T

    print("\nOut-of-sample performance summary:")
    print(summary.round(4))

    plot_out_of_sample_performance(
        out_mvo_variance,
        "MVO Variance",
        "oos_mvo_variance.png",
    )

    plot_out_of_sample_performance(
        out_mvo_cvar,
        "MVO + CVaR",
        "oos_mvo_cvar.png",
    )

    plot_out_of_sample_performance(
        out_mvo_low_risk,
        "MVO Low Risk",
        "oos_mvo_low_risk.png",
    )


if __name__ == "__main__":
    main()