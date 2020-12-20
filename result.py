import pandas as pd
import pyfolio as pf


class Results:
    def __init__(self, strategy_name: str, params: dict, pos: pd.DataFrame, pnl: pd.DataFrame):
        df = pd.DataFrame(self.params)
        df['param'] = df.apply(lambda row: dict(row), axis=1)
        df['Result'] = df.apply(lambda row: Result(strategy_name, row[param], pos[row.name], pnl[row.name]), axis=1)
        self.dict = {row['param']: row['Result'] for row in df.iterrows()}

    def get_result(self, param: dict):
        return self.dict[param]

    def get_summary(self):
        pass

    def get_top(self, metric: str, N: int):
        pass

class Result:
    def __init__(self,
                 strategy_name: str,
                 params: dict,
                 pos: pd.Series,
                 pnl: pd.Series):
        self.strategy_name = strategy_name
        self.params = params
        self.returns = pnl
        self.pos = pos

    def get_summary(self):
        pass

    def plot_returns(self):
        pass

    def plot_rolling_sharpe(self):
        pass

    def plot_rolling_calmar(self):
        pass

    @classmethod
    def combine(cls,
                *results: iter,
                is_equal_volatility: bool = False,
                keep_intersection: bool = True,
                weights: list = None):
        combined = cls(None, None, None, None, None)
        return combined

