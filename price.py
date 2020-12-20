import pandas as pd


class Price:
    def __init__(self, df: pd.DataFrame, name: str, instrument_type: str, lot_size: int = 1,
                 metric_conversion: float = 1):
        self.df = df
        self.name = name
        self.instrument_type = instrument_type
        self.lot_size = lot_size
        self.metric_conversion = metric_conversion


class Futures(Price):
    def __init__(self, df: pd.DataFrame, name: str, lot_size: int = 1, metric_conversion: float = 1):
        Price.__init__(self, df, name, 'futures')
        assert 'roll' in df.columns, Exception('Error: futures price needs to have a "roll" column.')


class FX(Price):
    def __init__(self, df: pd.DataFrame, name: str):
        Price.__init__(self, df, name, 'FX')


class CSVFutures(Futures):
    def __init__(self, filepath: str, header: str, name: str, lot_size: int = 1, metric_conversion: float = 1):
        try:
            df = pd.read_csv(filepath)
        except:
            raise Exception('ERROR: not a valid csv file.')
        df.columns = header
        Futures.__init__(self, df, name, lot_size, metric_conversion)

class Pickle_Futures(Futures):
    def __init__(self, filepath: str, header: [str], name: str, lot_size: int = 1, metric_conversion: float = 1):
        try:
            df = pd.read_pickle(filepath)
        except:
            raise Exception('ERROR: not a valid pickle file.')
        df.columns = header
        Futures.__init__(self, df, name, lot_size, metric_conversion)
