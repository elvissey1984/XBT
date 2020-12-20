import XBT
import price
import pandas as pd
import numpy as np

class TEST(XBT.Strategy):
    def __init__(self):
        self.params = {'a': np.array(range(5))}
        super().__init__()

    def next(self):
        price = self.prices[-1]
        action = self.params['a'] - price * 100
        self.order(XBT.Order(action, 'po'))

inst = TEST()
pricedata = pd.read_pickle('f:/github/XBT/price.pkl')
price = price.Pickle_Futures('f:/github/XBT/price.pkl',['close','roll'],'PO')
data = XBT.Data(pricedata,'price')
inst.add_price(price)
inst.add_data(data)

core = XBT.Core()
core.add_strategy(inst,'test')
results = core.backtest('test')
