from zipline.api import order_target_percent, record, symbol, set_benchmark
from zipline import run_algorithm
import warnings
import pyfolio as pf
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

STOCKS = ['AMD', 'CERN', 'COST', 'DELL', 'GPS', 'INTC', 'MMM']


def initialize(algo):
    algo.stocks = STOCKS
    algo.symbol_ids = [algo.symbol(asset) for asset in STOCKS]


def make_weights(algo, data):
    recent = data.history(assets=algo.symbol_ids, fields='close', bar_count=60, frequency='1d')
    short_mean = recent[-10:].mean()
    long_mean = recent[-50:].mean()
    weights = (short_mean - long_mean) / long_mean
    norm_weights = weights / weights.abs().sum()
    return norm_weights


def handle_data(algo, data):
    weights = make_weights(algo, data)
    for security in algo.symbol_ids:
        order_target_percent(security, weights[security])


start = pd.Timestamp('2016-1-1', tz='utc')
end = pd.Timestamp('2017-1-1', tz='utc')
results = run_algorithm(start=start, end=end, capital_base=1000, initialize=initialize, handle_data=handle_data)
print(results.columns)
results.to_csv("backtest.csv")
returns, positions, orders = pf.utils.extract_rets_pos_txn_from_zipline(results)


def plot_returns():
    pf.plot_rolling_returns(returns)  # instantiates plot, basically plt.plot(returns)
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.savefig("returns")
    plt.show()


def plot_positions_tear_sheet():
    pf.create_position_tear_sheet(returns=returns, positions=positions)
    plt.xlabel("Date")
    plt.savefig("positions")
    plt.show()


def plot_drawdown():
    pf.plot_drawdown_periods(returns=returns, top=5)
    plt.xlabel("Date")
    plt.savefig("drawdown")
    plt.show()


def plot_tear_sheet():
    pf.create_full_tear_sheet(returns, positions=positions, transactions=orders)
    plt.savefig("full_tear_sheet")

