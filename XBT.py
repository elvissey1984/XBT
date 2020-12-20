import pandas as pd
import numpy as np
from result import Results
from price import Price


class Data:
    def __init__(self, df: pd.DataFrame, name: str):
        self.df = df
        self.name = name


class Order:
    def __init__(self, action: np.array, price_name: str, order_type: str = 'market', levels: np.array = np.array([])):
        self.action = action
        self.price_name = price_name
        self.order_type = order_type
        self.prices = levels

class Strategy:
    def __init__(self):
        self._data = {}
        self._prices = {}
        self.params = {}

    def next(self):
        raise NotImplementedError('ERROR: The class must have a next function.')

    def add_data(self, data: Data):
        self._data[data.name] = data

    def add_price(self, price: Price):
        self._prices[price.name] = price

    def order(self, order: np.array):
        self.orders.append(order)

    def get_start(self):
        min_price_date = np.array([price.df.index.min() for (price_name, price) in self._prices.items()]).max()
        min_data_date = np.array([data.df.index.min() for (data_name, data) in self._data.items()]).max()
        return np.array([min_price_date, min_data_date]).max()

    def get_end(self):
        max_price_date = np.array([price.df.index.max() for (price_name, price) in self._prices.items()]).min()
        max_data_date = np.array([data.df.index.max() for (data_name, data) in self._data.items()]).min()
        return np.array([max_price_date, max_data_date]).min()

    def _clean_status(self):
        self.has_data = {}
        self.has_prices = {}
        self.prices = None
        self.data = None
        self.pos = np.array([])
        self.pnl = np.array([])
        self.orders = []
        self.num_params = None

    def set_up_params(self) -> int:
        return 0

    def get_periods(self, start, end) -> (int, list):
        price_dates = [price.df.index for price in self._prices]
        data_dates = [data.df.index for data in self._data]
        all_dates = price_dates + data_dates
        all_dates_unique = set([inner for outer in all_dates for inner in outer])
        return len(all_dates_unique), all_dates_unique

class Core:
    def __init__(self):
        self.slippage = 0
        self.commission = 0
        self.strategies = {}

    def add_strategy(self,
                     strategy: Strategy,
                     strategy_name: str):
        if isinstance(strategy, Strategy):
            self.strategies[strategy_name] = strategy
        else:
            print('strategy is not a Strategy object.')

    def run(self):
        pass

    def backtest(self, strategy_name: str, start: str = None, end: str = None) -> Results:
        strategy: Strategy = self._backtest_add_strategy(strategy_name)
        strategy._clean_status()
        start, end, num_periods, list_periods = self._backtest_parse_times(strategy, start, end)
        num_params: int = strategy.set_up_params()
        strategy.num_params = num_params
        pnl, pos = np.ones([num_periods, num_params]), np.ones([num_periods, num_params], np.int16)
        for i, period in enumerate(list_periods):
            strategy.has_data, strategy.data, strategy.has_price, strategy.prices = self._backtest_parse_prices(strategy, period)
            strategy.next()
            strategy.orders, strategy.pnl, strategy.pos = \
                self._execute_orders(strategy.orders, strategy.has_price, strategy.prices, strategy.pnl, strategy.pos)
            pnl[i], pos[i] = strategy.pnl, strategy.pos
        pnl = pd.DataFrame(pnl, index=list_periods)
        pos = pd.DataFrame(pos, index=list_periods)
        strategy._clean_status()
        return Results(strategy_name, strategy.params, pos, pnl)

    def _backtest_add_strategy(self, strategy_name: str) -> Strategy:
        assert strategy_name in self.strategies, f'{strategy_name} is not in Core.strategies. Please use ' \
                                                 f'Core.add_strategy() first. '
        return self.strategies[strategy_name]

    def _backtest_parse_times(self, strategy: Strategy, start: str, end: str):
        if start is None:
            start: pd.Timestamp = strategy.get_start()
        else:
            try:
                start = pd.to_datetime(start)
            except:
                raise Exception('start is not parsable.')
        if end is None:
            end: pd.Timestamp = strategy.get_end()
        else:
            try:
                end = pd.to_datetime(end)
            except:
                raise Exception('end is not parsable.')
        assert start < end, f'{start} is not before {end}.'
        num_periods, list_periods = strategy.get_periods(start, end)
        return start, end, num_periods, list_periods

    def _backtest_parse_prices(self, strategy: Strategy, period: int):
        has_data = {k: period in v.index for (k, v) in strategy._data.items()}
        data = {k: v[:period] for (k, v) in strategy._data.items()}
        has_price = {k: (period in v.index) for (k, v) in strategy._prices.items()}
        prices = {k: v[:period] for (k, v) in strategy._prices.items()}
        return has_data, data, has_price, prices

    def _execute_orders(self, orders: np.array, has_price: dict, prices: dict, pnl: np.array, pos: np.array):
        to_keep = np.zeros(len(orders))
        for i, order in enumerate(orders):
            if has_price[order.price_name]:
                to_keep[i], pnl, pos = self._execute_order(order, prices[order.price_name], pnl, pos)
        orders = orders[to_keep]
        return orders, pnl, pos

    def _execute_order(self, order: Order, price: Price, pnl: np.array, pos: np.array):
        if order.order_type == 'market':
            return self._execute_order_market(order, price, pnl, pos)
        else:
            raise NotImplementedError('order.order_type must be "market"')

    def _execute_order_market(self, order: Order, price: Price, pnl: np.array, pos: np.array):
        if pnl == np.array([]):
            pnl = np.zeros(len(order.action))
            pos = order.action
        else:
            if price.instrument_type == 'futures':
                if price.df.iloc[-2]['roll']:
                    pnl = (price.df.iloc[-1]['close'] + price.df.iloc[-2]['roll'] - price.df.iloc[-2]['close']) * pos
                else:
                    pnl = (price.df.iloc[-1]['close'] - price.df.iloc[-2]['close']) * pos
                pos = order.action + pos
            else:
                raise NotImplemented('Currently only futures are acceptable.')
        return np.zeros(len(order.action)), pnl, pos

