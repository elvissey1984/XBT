import pandas as pd
import price

class EikonFutures(price.Futures):
    def __init__(self, ticker, contract,
                 excluding=None, name=None,
                 start=None, end=None,
                 intraday=False
                 ):
        self.ticker = ticker
        ptype = self._get_price_type(self)
        if ptype == 'flat':
            self.df = self._get_flat()
        elif ptype == 'spread':
            self.df = self._get_spread()
        else:
            raise Exception('EIKON: ptype needs to be either "flat" or "spread".')

    def _get_price_type(self) -> str:
        ticker_type = type(self.ticker)
        contract_type = type(self.contract)
        if ticker_type == str and contract_type == str:
            return 'flat'
        elif type(self.ticker) == list:
            if type(self.contract) == list and len(self.ticker) == len(self.contract):
                return 'spread'
            else:
                raise Exception('EIKON: contract needs to be a list of same length as ticker when ticker is a list.')
        else:
            raise Exception('EIKON: ticker needs to be a string or a list')

    def _get_flat(self, ticker, contract,
                 excluding=None, name=None,
                 start=None, end=None,
                 intraday=False
                 ) -> pd.DataFrame:
        return None

    def _get_spread(self,
                 ticker,
                 contract,
                 excluding=None,
                 name=None,
                 start=None,
                 end=None,
                 intraday=False
                 ) -> pd.DataFrame:
        return None