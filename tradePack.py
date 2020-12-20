import pandas as pd
import numpy as np
import datetime as dt

from bokeh.io import save, curdoc,output_file ,show, output_notebook, push_notebook
from bokeh.plotting import figure, gridplot
from bokeh.models import CategoricalColorMapper, HoverTool, ColumnDataSource, Panel,  LinearAxis, Range1d, Legend
from bokeh.models.widgets import Div, CheckboxGroup,Button, MultiSelect,DatePicker, TextInput, Slider, RangeSlider, Tabs, DataTable, DateFormatter,NumberFormatter, StringFormatter,TableColumn, RadioButtonGroup, Select,HTMLTemplateFormatter
from bokeh.layouts import layout, column, row, WidgetBox
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar,FileInput
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.models.callbacks import CustomJS
import sys
import os
import copy
import pickle
sys.path.insert(0, 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/utility/tradePack'.format(os.getlogin()))
from ReutersHub import ReutersHub
output_notebook()




import pyodbc

class dataHander():
    def __init__(self,priceOrTicks='ticks'):
        """Handles all traffic of data. Processes input data to uniform format and is the source of price data.
        
        Parameters
        ----------
        tradingType : str
            Specify which kind of backtest/production run is required. Options are:
            - C       : CLOSE to CLOSE, does not support intraday trading
            - OHLC    : supports intraday execution of orders determined on the OPEN or CLOSE only
            - IDR_N   : supports intraday trading, does not allow (stop) limit orders within N ticks of current value
            - IDM_N   : supports intraday execution of orders determined on the OPEN or CLOSE of every candle of N minutes
        """
        
        self.tradingTimes = {'BO':{'OPEN':'00:00','CLOSE':'19:20'},
                             'SM':{'OPEN':'00:00','CLOSE':'19:20'},
                             'S':{'OPEN':'00:00','CLOSE':'19:20'},
                             'COM':{'OPEN':'10:00','CLOSE':'19:20'},
                             'RS':{'OPEN':'00:00','CLOSE':'23:00'},
                             'LGO':{'OPEN':'00:00','CLOSE':'23:00'},
                             'FCPO':{'OPEN':'04:00','CLOSE':'10:00'},
                             'SOY_BM':{'OPEN':'00:00','CLOSE':'19:20'}}
        
        self.tickSizes =    {'BO':0.01,
                             'SM':0.1,
                             'S':0.25,
                             'COM':0.25,
                             'RS':0.1,
                             'LGO':0.25,
                             'FCPO':1,
                             'SOY_BM':0.25,
                             'MB_SFO':0.25,
                             'MB_RSO':0.25,
                             'MB_PKO':0.25,
                             'MB_PAO':0.25,
                             'MB_SBO':0.25,
                             'MB_CNO':0.25}
        self.npAroundFac = {'BO':2,
                             'SM':2,
                             'S':2,
                             'COM':2,
                             'RS':2,
                             'LGO':2,
                             'FCPO':0,
                             'SOY_BM':2,
                              'MB_SFO':2,
                             'MB_RSO':2,
                             'MB_PKO':2,
                             'MB_PAO':2,
                             'MB_SBO':2,
                             'MB_CNO':2}
        
        self.neovest = {'BO':'ZL',
                             'SM':'ZM',
                             'S':'ZS',
                             'COM':'YECO',
                             'RS':'RS',
                             'LGO':'G',
                             'FCPO':'FCPO',
                             'SOY_BM':'SOY_BM',
                              'MB_SFO':'MB_SFO',
                             'MB_RSO':'MB_RSO',
                             'MB_PKO':'MB_PKO',
                             'MB_PAO':'MB_PAO',
                             'MB_SBO':'MB_SBO',
                             'MB_CNO':'MB_CNO'}
        self.neovestTAS = {'BO':'ZLT',
                             'SM':'ZMT',
                             'S':'SBT',
                             'COM':'YECO',
                             'RS':'RS',
                             'LGO':'G',
                             'FCPO':'FCPO',
                             'SOY_BM':'SOY_BM',
                              'MB_SFO':'MB_SFO',
                             'MB_RSO':'MB_RSO',
                             'MB_PKO':'MB_PKO',
                             'MB_PAO':'MB_PAO',
                             'MB_SBO':'MB_SBO',
                             'MB_CNO':'MB_CNO'} 
        
        self.convDict = {'BO':600,        # if the price moves one 1, resulting PnL in own currency
                             'SM':100,
                             'S':50,
                             'COM':50,
                             'RS':20,
                             'LGO':100,
                             'FCPO':25,
                             'SOY_BM':1,
                              'MB_SFO':1,
                             'MB_RSO':1,
                             'MB_PKO':1,
                             'MB_PAO':1,
                             'MB_SBO':1,
                             'MB_CNO':1} 
        self.lotConvDict = {'BO':27.2155498765775,
                             'SM':90.7184995885916,
                             'S':136.077749382887,
                             'COM':50,
                             'RS':20,
                             'LGO':100,
                             'FCPO':25,
                              'SOY_BM':1,
                              'MB_SFO':1,
                             'MB_RSO':1,
                             'MB_PKO':1,
                             'MB_PAO':1,
                             'MB_SBO':1,
                             'MB_CNO':1} 
        
        self.neoAccount = {'BO':'&17568316CQ',
                             'SM':'&17568316CQ',
                             'S':'&17568316CQ',
                             'COM':'8B203M18CQ',
                             'RS':'8B010',
                             'LGO':'8B010',
                             'FCPO':'WhatsApp',
                             'SOY_BM':'&17568316CQ',
                             'MB_SFO':'CASH',
                             'MB_RSO':'CASH',
                             'MB_PKO':'CASH',
                             'MB_PAO':'CASH',
                             'MB_SBO':'CASH',
                             'MB_CNO':'CASH'}
        
        
        self.tickValues = {'BO':6,
                             'SM':10,
                             'S':12.5,
                             'COM':12.5,
                             'RS':2,
                             'LGO':25,
                             'FCPO':25,
                             'SOY_BM':125,
                             'MB_SFO':1,
                             'MB_RSO':1,
                             'MB_PKO':1,
                             'MB_PAO':1,
                             'MB_SBO':1,
                             'MB_CNO':1}
        
        self.commissions = {'BO': 1.25,
                 'SM': 1.25,
                 'S': 1.25,
                 'RS': 1.75,
                 'FCPO':9.10,
                 'LGO':1.3,
                 'COM':1.40,
                 'SOY_BM':37.5,
                'MB_SFO':0,
                'MB_RSO':0,
                'MB_PKO':0,
                'MB_PAO':0,
                'MB_SBO':0,
                'MB_CNO':0}
        
        self.FX =         {'BO':None,
                             'SM':None,
                             'S':None,
                             'COM':'EURUSD',
                             'RS':'CADUSD',
                             'LGO':None,
                             'FCPO':'MYRUSD',
                             'SOY_BM':None,
                             'MB_SFO':None,
                            'MB_RSO':None,
                            'MB_PKO':None,
                            'MB_PAO':None,
                            'MB_SBO':None,
                            'MB_CNO':None}
        self.FXp =        {}
        self.FXdata = []
        self.groupCount = {}
        self.combined = []
        self.priceUnits =    {'BO':'USc/sP',
                             'SM':'USd/sT',
                             'S':'USd/bushel',
                             'COM':'EUR/T',
                             'RS':'USd/T',
                             'LGO':'USd/T',
                             'FCPO':'MYR/T',
                             'SOY_BM':'USc/bushel',
                             'MB_SFO':'USd/T',
                            'MB_RSO':'USd/T',
                            'MB_PKO':'USd/T',
                            'MB_PAO':'USd/T',
                            'MB_SBO':'USd/T',
                            'MB_CNO':'USd/T'}
        self.currency =    {'BO':'kUSd',
                             'SM':'kUSd',
                             'S':'kUSd',
                             'COM':'kEUR',
                             'RS':'kUSd',
                             'LGO':'kUSd',
                             'FCPO':'kMYR',
                             'SOY_BM':'kUSd',
                             'MB_SFO':'kUSd',
                            'MB_RSO':'kUSd',
                            'MB_PKO':'kUSd',
                            'MB_PAO':'kUSd',
                            'MB_SBO':'kUSd',
                            'MB_CNO':'kUSd'}
        self.exchanges =    {'BO':'cbot',
                             'SM':'cbot',
                             'S':'cbot',
                             'COM':'mat',
                             'RS':'ius',
                             'LGO':'ieu',
                             'FCPO':'mdx',
                             'SOY_BM':'cbot',
                             'MB_SFO':'internal',
                            'MB_RSO':'internal',
                            'MB_PKO':'internal',
                            'MB_PAO':'internal',
                            'MB_SBO':'internal',
                            'MB_CNO':'internal'}
  
        
        self.priceOrTicks = priceOrTicks
        self.signals = []
        self.signalNames = []
        self.tradablePrices = []
        
    def checkDateStr(self,dateSTR):
        """Checks input string for expected date format.
        
        Parameters
        ----------
        dateSTR : str
            date in string format, supported formats:
            - YYYY-MM-DD
            - YYYY-MM-DD HH:MM
            - YYYY-MM-DD HH:MM:SS
            
        Returns
        -------
        bool
        """
        fDict = {10:'%Y-%m-%d',
                16:'%Y-%m-%d %H:%M',
                19:'%Y-%m-%d %H:%M:%S'}
        
        try:
            if dt.datetime.strftime(pd.to_datetime(dateSTR),fDict[len(dateSTR)])==dateSTR:                
                return True
            else:
                print('Date format not recognised, expecting YYYY-MM-DD, YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS format, got: '+dateSTR)
                return False
        except:
            print('Date format not recognised, expecting YYYY-MM-DD, YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS format, got: '+str(dateSTR))
            return False
    def checkTradingType(self,tradingType):
        """Checks input trading type for price requests.
        
        Parameters
        ----------
        tradingType : str
            trading type in str format. Supported options are:
                - 'C'
                - 'OHLC'
                - 'IDM_N', where N is an integer number as str
                - 'IDR_N', where N is an integer number as str
        Returns
        -------
        bool
        """
        if (tradingType in ['C','OHLC']):
            return True        
        elif len(tradingType)>4:
            if(tradingType[:4] in ['IDR_','IDM_']) & (int(tradingType[4:])>0):
                return True
            else:
                print('tradingType '+str(tradingType)+' not supported, possible entries are: '+str(['C','OHLC','IDR_N','IDM_N']))
                return False
        else:
            print('tradingType '+str(tradingType)+' not supported, possible entries are: '+str(['C','OHLC','IDR_N','IDM_N']))
            return False
    def checkContractType(self,CN_YYYYMM):
        """Checks input contract type for price requests.
        
        Parameters
        ----------
        CN_YYYYMM : str or int
            contract type in str or int format. Supported options are:
                - 'cN', where N is an integer number as str
                - YYYYMM, integer
        
        Returns
        -------
        bool
        """
        if type(CN_YYYYMM)==type(''):
            if CN_YYYYMM[0]=='c':
                if int(CN_YYYYMM[1:])>0:
                    return True
            print('Invalid contract definition: '+CN_YYYYMM)
            return False
        elif type(CN_YYYYMM)==type(5):
            if (CN_YYYYMM>199000) & (CN_YYYYMM<1000000):
                return True
            else:
                print('Invalid contract definition: '+str(CN_YYYYMM))
                return False
        else:
            print('Invalid contract definition: '+str(CN_YYYYMM))
            return False
    def addSignal(self,signal_df_in,signal_time=None):
        
        
        """Adds signal to internal signal list. 
        
        Parameters
        ----------
        signal_df : Pandas Dataframe
            Pandas Dataframe with datetime index and signal name(s) as column name(s).
            
        signal_time : None, str
            Time of the day at which the signal needs to be evaluated. Will be added to the index time. None for index as default, format HH:MM in str
            
        """
        cont = True
        signal_df = signal_df_in.copy()
        newSigNames = []
        for sigName in signal_df.columns:
            if sigName in self.signalNames:
                print('Error, signal name '+sigName+' already in signalName list. Aborting..')
                cont=False
            else:
                
                if sigName in ['counter','next_idx','key','t','maxCount']:
                    print('Signal name can not be any of the following: '+str(['counter','next_idx','key','t','maxCount'])+'. Aborting..')
                    cont=False
                else:
                    newSigNames.append(sigName)
                
        if cont:
            print('signal(s) added')
            self.signalNames = self.signalNames+newSigNames
            
            if signal_time!=None:
                tdS = self.calcMinOffset(signal_time)
                idx = signal_df.index
                signal_df.index = pd.DatetimeIndex([i+pd.offsets.Minute(n=tdS) for i in idx])
                
            self.signals.append(signal_df)
    def calcMinOffset(self,inputTime):
        """Calulates minute offset from input time.
        
        Parameters
        ----------
        inputTime : str
            input ime in 'HH:MM' format
        
        Returns
        -------
        int
            number of minutes
        """
        if (len(inputTime)==5) &(inputTime[2]==':'):
            return int(inputTime[:2])*60+int(inputTime[3:])
        else:
            print('Error processing time input, expected HH:MM string format, got: '+str(inputTime))
            return None
    def prepareStart(self):   
        """Removes all in memory data in the data handler that will not be used for future continuation runs. Saves the minumum 
        required data for continuing the run with new data.
        """
        self.signals = []
        self.signalNames = []
        self.tradablePrices = []
        self.FXdata = []
        self.master_idx = None

        self.signalDates = {sigName : self.master_data['signals']['next_idx'][i]+pd.offsets.Second(1) for i,sigName in enumerate(self.master_data['signals']['key'])}

        self.master_data = None
        self.FXp = {}
        self.keepGroupCount = copy.deepcopy(self.groupCount)
        self.groupCount = {}
        self.combined = []
        for pName in list(self.priceMetaDict.keys()):
            if self.priceMetaDict[pName]['dType'] in ['C','OHLC']:
                if self.priceMetaDict[pName]['addTradingHours']:
                    nextT = self.priceMetaDict[pName]['idx'][-1] # added time already makes sure 
                else:
                    nextT = self.priceMetaDict[pName]['idx'][-1]+pd.offsets.Day(1)
            elif self.priceMetaDict[pName]['dType'][:4]=='IDM_':
                if self.priceMetaDict[pName]['atTime'] !=None:
                    nextT = pd.Timestamp(self.priceMetaDict[pName]['idx'][-1].date())+pd.offsets.Minute(self.calcMinOffset(self.tradingTimes[self.priceMetaDict[pName]['RIC']]['CLOSE']))+pd.offsets.Minute()
                else:
                    nextT = self.priceMetaDict[pName]['idx'][-1]+pd.offsets.Minute(1)
            elif self.priceMetaDict[pName]['dType'][:4]=='IDR_':
                nextT = self.priceMetaDict[pName]['idx'][-1]+pd.offsets.Second(1)
            self.priceMetaDict[pName].update({'idx':str(nextT)[:19]})

    def fromStart(self,parent,endDate=None):
        """Gathers new signal and price data from the last datapoint from the last run."""
        for sigName in list(self.signalDates.keys()):
            dfSig,signal_time  = parent.strat.downloadData(sigName,startDate=self.signalDates[sigName])
            if endDate!=None:
                dfSig = dfSig[:endDate]
            self.addSignal(dfSig,signal_time=signal_time)

        for pName in list(self.priceMetaDict.keys()):
            self.addPrice(self.priceMetaDict[pName]['RIC'],
                 self.priceMetaDict[pName]['CN_YYYYMM'],
                 RollDaysPrior=self.priceMetaDict[pName]['RollDaysPrior'],
                 dataType=self.priceMetaDict[pName]['dType'],
                 start_date=self.priceMetaDict[pName]['idx'],
                 end_date=endDate,
                 fromLocal=self.priceMetaDict[pName]['fromLocal'],
                 addTradingHours=self.priceMetaDict[pName]['addTradingHours'],
                 nameOverride=self.priceMetaDict[pName]['nameOverride'],
                 driverName=self.priceMetaDict[pName]['driverName'],
                 pathOfData=self.priceMetaDict[pName]['pathOfData'],
                 factor=self.priceMetaDict[pName]['factor'],
                 FXconv=self.priceMetaDict[pName]['FXconv'],
                 RVvolRatio=self.priceMetaDict[pName]['RVvolRatio'],
                 atTime=self.priceMetaDict[pName]['atTime'],
                 skip=self.priceMetaDict[pName]['skip'])
        self.groupCount = copy.deepcopy(self.keepGroupCount)
    def prep(self,first=True):
        """Creates/merges data structure from inputs."""
        if first:
            self.priceMetaDict = {}
            self.idx_loc = -1
            self.master_data = {'signals':{'counter':[],'next_idx':[],'key':[],'maxCount':[]},'prices':{},'RV':{},'FX':{}}
        else:
            keepMaster_idx = copy.deepcopy(self.master_idx)

        if len(self.tradablePrices)>0:
            if len(list(self.groupCount.keys()))>0: # first time align grouped commodities
                combinations = {}
                IDXcombined = {cName:None for cName in list(self.groupCount.keys())}
                colCheck = {cName:None for cName in list(self.groupCount.keys())}
                for Pl in self.tradablePrices: 
                    df = Pl[0].copy()
                    pName = Pl[1]['nameOverride']
                    if pName in list(self.groupCount.keys()): # part of group        
                        if type(IDXcombined[pName])!=type(None):
                            IDXcombined.update({pName:IDXcombined[pName].intersection(df.index)})
                            if colCheck[pName]!=list(df.columns):
                                print('Columns of to be merged contracts are not identical. Combination failed!')
                                print('Got: '+pName+'| '+str(colCheck[pName]) +' vs '+str(list(df.columns)))
                                print('Check the datatypes per ')
                                self.errorMessage = 'Error: merged contracts column inconsistent'
                                return False
                        else:
                            IDXcombined.update({pName:df.index})
                            colCheck.update({pName:list(df.columns)})
                filterDone = []
                for Pl in self.tradablePrices:  # apply atTime filter after intersection of indexes to ensure a RV datapoint
                    pName = Pl[1]['nameOverride'] 
                    if pName in list(self.groupCount.keys()): # part of group      
                        if Pl[1]['atTime'] != None:    
                            if pName not in filterDone:
                                idx = IDXcombined[pName]  
                                benchM  = int(atTime[:2])*100+int(atTime[3:])
                                HHMM = idx.hour.values*100+idx.minute.values
                                mask = HHMM>=benchM
                                mask[(mask[:-1] & mask[1:]).nonzero()[0]+1] = False
                                idx = idx[mask]
                                IDXcombined.update({pName:idx})
                                filterDone.append(pName)

                for i,Pl in enumerate(self.tradablePrices): # time align all prices in combinations now that the intersect of index is known
                    pName = Pl[1]['nameOverride']
                    idxName = Pl[0].index.name
                    if pName in list(self.groupCount.keys()): # part of group
                        if len(IDXcombined[pName])==0:
                            df = Pl[0].loc[IDXcombined[pName],:].copy()
                            df.index.name = idxName
                        
                            self.tradablePrices[i][0] = df.copy()
                        else:
                            dfAll = Pl[0].copy()
                            
                            rollLocs = (pd.isna(dfAll.Roll)==False).values.nonzero()[0]
                            
                            if len(rollLocs)>0:
                                dfR = dfAll.loc[dfAll.index[rollLocs],['Roll']] # only the datetimes where there is a roll
                                df = Pl[0].loc[IDXcombined[pName],:].copy()
                                
                                dfR = dfR.loc[dfR.index.isin(df.index)==False,:] # when the roll is discarded from the time selection, it should be included
                                if dfR.shape[0]>0:
                                    dfRR = pd.concat([df.loc[:,['Roll']],dfR],axis=1) # the rolls do not align in the concat, but are 1 further down the line
                                    dfRR.columns = ['R1','R2']
                                    dfRR.R2 = dfRR.R2.shift(-1) # move 1 up to align with target date
                                    dfRR.loc[pd.isna(dfRR.R2)==False,'R1'] = dfRR.loc[pd.isna(dfRR.R2)==False,'R2'].copy() # copy them in
                                    df.Roll = dfRR.loc[df.index,'R1'].values # move to df
                            else:
                                df = Pl[0].loc[IDXcombined[pName],:].copy()
                            RIC = Pl[1]['RIC']
                            
                            df.index.name = idxName
                            IDX = df.index
                            loc = (np.diff(IDX.strftime(date_format='%Y%m%d').values.astype(int))!=0).nonzero()[0]
                            df.loc[df.index[loc],'status'] = 'CLOSE' # due to time alignment of intraday data CLOSE status might be lost so adding them here. Omitted the last one as realtime intraday data might not be the last
                                                        
                            self.tradablePrices[i][0] = df.copy()
                                                    
                            oriCols = df.columns
                            if Pl[1]['factor']!=None:
                                fac = Pl[1]['factor']
                            else:
                                fac = 1
                            
                            useCols = ['CLOSE','Roll']
                            copyCols = []
                            for col in oriCols:
                                if col in ['OPEN','HIGH','LOW']:
                                    copyCols.append(col)
                                
                            df.loc[:,useCols] = df.loc[:,useCols]*fac*self.tickSizes[RIC]
                            
                            if Pl[1]['FXconv']:
                                FXname = self.FX[RIC]
                                FX = self.FXp[FXname]
                                keepCols = df.columns
                                dfTemp = pd.concat([df,FX],axis=1).fillna(method='ffill').fillna(method='bfill')
                                df = pd.concat([df,dfTemp.loc[df.index,['c1']]],axis=1)
                                if FXname in ['EURUSD']:
                                    FX = df.loc[:,['c1']].copy()
                                    X = df.loc[:,useCols].values*np.tile(df.loc[:,'c1'].values.reshape([-1,1]),(1,len(useCols)))
                                elif FXname in ['MYRUSD','CADUSD']:
                                    FX = 1/df.loc[:,['c1']]                                    
                                    X = df.loc[:,useCols].values/np.tile(df.loc[:,'c1'].values.reshape([-1,1]),(1,len(useCols)))
                                    
                                FX.columns = ['CLOSE']
                                FX['Roll'] = np.nan
                                FX['CONTRACTS'] = df['CONTRACTS'].copy()
                                
                                FX.index.name = df.index.name
                                self.FXdata.append([FX,Pl[1]])
                                
                                df = df.loc[:,keepCols]
                                df.loc[:,useCols] = X
                            
                            for toCopy in copyCols: # As OHLC bars are not internally aligned, only close is considered.
                                df.loc[:,toCopy] = df.loc[:,'CLOSE'].copy()
                            
                            useCols = useCols+copyCols # all columns need to be considered in the next step
                            
                            if pName in list(combinations.keys()):
                                dfT = combinations[pName][0].copy()
                                X1 = dfT.loc[:,useCols].fillna(0).values
                                mask1 = pd.isna(dfT.loc[:,useCols])
                                X2 = df.loc[:,useCols].fillna(0).values
                                mask2 = pd.isna(df.loc[:,useCols])                            
                                mask = mask1&mask2
                                X = X1+X2
                                X[mask] = np.nan
                                dfT.loc[:,useCols] = X
                                
                                dfT.loc[df['status']=='CLOSE','status'] = 'CLOSE' # 
                            
                                combinations.update({pName:[dfT,Pl[1]]})
                            else:
                                combinations.update({pName:[df,Pl[1]]}) # here contract information is adopted from the first entry for the RV
                
                for pName in list(combinations.keys()):
                    df = combinations[pName][0].copy()
                    df.index.name = pName
                    
                    self.combined.append([df,combinations[pName][1]])
                    
            cont = False
            for Pl in self.tradablePrices:
                if Pl[0].shape[0]>0:
                    cont =True
                    break
            if cont==False:
                print('No price data in prep, aborting')
                self.errorMessage = 'No new data'
                return False
                
                    
            for i,Pl in enumerate(self.tradablePrices):
                pName = Pl[0].index.name
                idx = Pl[0].index.copy()
                
                if first:
                    RIC = Pl[1]['RIC']                
                    metaData = copy.deepcopy(Pl[1])
                    metaData.update({'RIC':RIC,
                                'tickSize' : self.tickSizes[RIC],
                                'tickValues' : self.tickValues[RIC],
                                'FX' : self.FX[RIC],
                                'priceUnits' : self.priceUnits[RIC],
                                'currency':self.currency[RIC],
                                'npAroundFac':self.npAroundFac[RIC],
                                'neovest':self.neovest[RIC],
                                'neoAccount':self.neoAccount[RIC],
                                'commissions':self.commissions[RIC]})
                
                if pName in list(self.priceMetaDict.keys()):
                    newIdx = self.priceMetaDict[pName]['idx'].union(idx)
                    self.priceMetaDict[pName].update({'len':len(newIdx),'idx':newIdx})
                else:
                    self.priceMetaDict.update({pName:{'len':len(idx),'idx':idx}})
                    if first:
                        self.priceMetaDict[pName].update(metaData)
            
                if (i ==0)&(first):
                    self.master_idx = Pl[0].index.copy()
                else:
                    self.master_idx = self.master_idx.union(Pl[0].index)
                
            for sig in self.signals:
                self.master_idx = self.master_idx.union(sig.index)
            if first==False:
                if self.master_idx[len(keepMaster_idx)-1]!=keepMaster_idx[-1]:
                    print('Major error, trying to append older data than previous, which messes up the timeline and data structure. aborting')
                    self.errorMessage = 'Error: data misalignment'
                    return False

                
            dictKey = ['prices','RV','FX']
            for kn, prices in enumerate([self.tradablePrices, self.combined,self.FXdata]):
                for Pl in prices:
                    for colName in Pl[0].columns:
                        if Pl[0].index.name in list(self.master_data[dictKey[kn]].keys()):
                            if first:
                                if (colName not in ['Roll','status']) & (kn==0):
                                    self.master_data[dictKey[kn]][Pl[0].index.name].update({colName:{'idx':Pl[0].index,'val':np.around(Pl[0].loc[:,colName].values,0).astype(int)}})
                                else:
                                    self.master_data[dictKey[kn]][Pl[0].index.name].update({colName:{'idx':Pl[0].index,'val':Pl[0].loc[:,colName].values}})
                                counter = self.master_data[dictKey[kn]][Pl[0].index.name]['counter']
                                next_idx = self.master_data[dictKey[kn]][Pl[0].index.name]['next_idx']
                                keys = self.master_data[dictKey[kn]][Pl[0].index.name]['key']
                                counter.append(0)
                                next_idx.append(Pl[0].index[0])
                                keys.append(colName)
                                self.master_data[dictKey[kn]][Pl[0].index.name].update({'counter':counter})
                                self.master_data[dictKey[kn]][Pl[0].index.name].update({'next_idx':next_idx})   
                                self.master_data[dictKey[kn]][Pl[0].index.name].update({'key':keys})  
                            else:
                                if (colName not in ['Roll','status']) & (kn==0):                                
                                    idx = self.master_data[dictKey[kn]][Pl[0].index.name][colName]['idx'].union(Pl[0].index)
                                    val = np.append(self.master_data[dictKey[kn]][Pl[0].index.name][colName]['val'],np.around(Pl[0].loc[:,colName].values,0).astype(int))
                                    
                                    self.master_data[dictKey[kn]][Pl[0].index.name][colName].update({'idx':idx,'val':val})
                                else:
                                    idx = self.master_data[dictKey[kn]][Pl[0].index.name][colName]['idx'].union(Pl[0].index)
                                    val = np.append(self.master_data[dictKey[kn]][Pl[0].index.name][colName]['val'],Pl[0].loc[:,colName].values)
                                    
                                    self.master_data[dictKey[kn]][Pl[0].index.name][colName].update({'idx':idx,'val':val})
                                
                                next_idx = self.master_data[dictKey[kn]][Pl[0].index.name]['next_idx']
                                keys = self.master_data[dictKey[kn]][Pl[0].index.name]['key']
                                
                                loc = (np.array(keys)==colName).nonzero()[0][0]
                                next_idx[loc] = Pl[0].index[0]
                                
                                self.master_data[dictKey[kn]][Pl[0].index.name].update({'next_idx':next_idx}) 
                                self.master_data[dictKey[kn]][Pl[0].index.name].update({'maxCount':len(idx)}) 
                                
                                
                        else:
                            if (colName not in ['Roll','status']) & (kn==0):
                                self.master_data[dictKey[kn]].update({Pl[0].index.name : {colName:{'idx':Pl[0].index,'val':np.around(Pl[0].loc[:,colName].values,0).astype(int)},'dtype':Pl[1]['dType'] ,'counter':[0],'next_idx':[Pl[0].index[0]],'maxCount':len(Pl[0].index),'key':[colName]}})
                            else:
                                self.master_data[dictKey[kn]].update({Pl[0].index.name : {colName:{'idx':Pl[0].index,'val':Pl[0].loc[:,colName].values},'dtype':Pl[1]['dType'] ,'counter':[0],'next_idx':[Pl[0].index[0]],'maxCount':len(Pl[0].index),'key':[colName]}})
            if first:
                for i,sig in enumerate(self.signals):
                    counter = []
                    next_idx = []
                    maxCount = []
                    keys = []
                    
                    for colName in sig.columns:                    
                        self.master_data['signals'].update({colName : {'idx':sig.index,'val':sig.loc[:,colName].values}})
                        counter.append(0)
                        if sig.shape[0]==0:
                            next_idx.append(None)
                        else:
                            next_idx.append(sig.index[0])
                        maxCount.append(len(sig.index))
                        keys.append(colName)
                    
                    self.master_data['signals'].update({'counter':counter,'next_idx':next_idx,'key':keys,'maxCount':maxCount})
            else:
                next_idx = self.master_data['signals']['next_idx']
                keys = self.master_data['signals']['key']
                maxCount = self.master_data['signals']['maxCount']
                
                for i,sig in enumerate(self.signals):
                    for colName in sig.columns:   
                        loc = (np.array(keys)==colName).nonzero()[0][0]
                        idx = self.master_data['signals'][colName]['idx'].union(sig.index)
                        val = np.append(self.master_data['signals'][colName]['val'],sig.loc[:,colName].values)
                        self.master_data['signals'][colName].update({'idx':idx,'val':val})
                        
                        next_idx[loc] = sig.index[0]
                        maxCount[loc] = len(idx)
                    
                self.master_data['signals'].update({'next_idx':next_idx,'maxCount':maxCount})
            return True
 
        else:
            if hasattr(self,'master_data'):
                print('Data Handler resetted')
            else:
                print('There are no tradable prices imported. Aborting')
                self.errorMessage = 'Error: are no tradable prices imported'
            return False
    def getNextDataPoint(self):
        """Grabs the next data point and saves the price set internally in self.lastDataPoint"""
        
        
        if self.idx_loc+1 !=len(self.master_idx):
            self.idx_loc = self.idx_loc+1        
            currIdx = self.master_idx[self.idx_loc]
        else:
            return False
        
        output = {'t':currIdx,'prices':{},'signals':{},'RV':{},'FX':{}}
        
        newPrice = False
                      
        for dictKey in ['prices','RV','FX']:                       
            for PRD in list(self.master_data[dictKey].keys()):
                counter = self.master_data[dictKey][PRD]['counter']
                next_idx = self.master_data[dictKey][PRD]['next_idx']
                keys = self.master_data[dictKey][PRD]['key']
                maxCount = self.master_data[dictKey][PRD]['maxCount']
                
                pDict = {'dType':self.master_data[dictKey][PRD]['dtype'],'info':False}
                
                for i,ts in enumerate(next_idx):
                    if currIdx==ts: 
                        val = self.master_data[dictKey][PRD][keys[i]]['val'][counter[i]]                        
                        
                        if counter[i]+1!=maxCount:
                            counter[i] = counter[i]+1                
                            next_idx[i] = self.master_data[dictKey][PRD][keys[i]]['idx'][counter[i]]
                        else:
                            counter[i] = counter[i]+1   
                             
                        if keys[i]=='Roll':
                            if np.abs(val)>=0:
                                val = int(val)
                        
                        pDict.update({keys[i]:val})
                        pDict.update({'info':True})
                        newPrice = True
                        
                    elif keys[i]=='CONTRACTS':
                        if counter[i]!=maxCount:
                            val = self.master_data[dictKey][PRD][keys[i]]['val'][counter[i]]
                        else:
                            val = self.master_data[dictKey][PRD][keys[i]]['val'][-1]
                        pDict.update({keys[i]:val})
                    else:
                        pDict.update({keys[i]:np.nan})
                        
                self.master_data[dictKey][PRD].update({'counter':counter,'next_idx':next_idx})
                output[dictKey].update({PRD:pDict})
        
        counter = self.master_data['signals']['counter']
        next_idx = self.master_data['signals']['next_idx'].copy()
        keys = self.master_data['signals']['key']                                
        maxCount = self.master_data['signals']['maxCount']

        for i, ts in enumerate(self.master_data['signals']['next_idx']):
            sigName = self.master_data['signals']['key'][i]    
            if currIdx==ts:
                            
                val = self.master_data['signals'][sigName]['val'][counter[i]]
                newPrice = True
                if counter[i]+1!=maxCount[i]:
                    counter[i] = counter[i]+1                
                    next_idx[i] = self.master_data['signals'][sigName]['idx'][counter[i]]
                else:
                    counter[i] = counter[i]+1                
                    # next_idx wont be updated but expected to be updated when new data is added

                output['signals'].update({sigName:val})
            else:
                output['signals'].update({sigName:np.nan})

        self.master_data['signals'].update({'counter':counter,'next_idx':next_idx})

        if newPrice:
            self.lastProcessed = self.idx_loc  
            self.lastDataPoint = output.copy()
            return True
        else:
            self.idx_loc = self.idx_loc-1
            print('prices exhausted')
            return False
    def fromTicksToActual(self):
        """Converts tick prices to actual prices. Returns converted prices in the same format as self.lastDataPoint"""
        T = copy.deepcopy(self.lastDataPoint)
        
        pNames = list(self.lastDataPoint['prices'].keys())

        for pName in pNames:
            if T['prices'][pName]['info']:
                for key in list(T['prices'][pName].keys()):
                    if key in ['OPEN','HIGH','LOW','CLOSE']:
                        T['prices'][pName].update({key:T['prices'][pName][key]*self.priceMetaDict[pName]['tickSize']})
                    elif key == 'Roll':
                        if np.isnan(T['prices'][pName][key])==False:
                            T['prices'][pName].update({key:T['prices'][pName][key]*self.priceMetaDict[pName]['tickSize']})
        return T
    def addTodaysPrices(self,append=False,info=True):
        """Imports todays data from data saved by intraDayDataGrabber.py
        
        Parameters
        ----------
        append : bool
            Append to current price set in memory if True, replaces current priceset when False
            
        info : bool
            Print out system information if True, otherwise not"""
        self.signals = []
        self.signalNames = []
        self.tradablePrices = []
        
        for pName in list(self.priceMetaDict.keys()):
            RIC = self.priceMetaDict[pName]['RIC']
            CN_YYYYMM = self.priceMetaDict[pName]['CN_YYYYMM']
            dType = self.priceMetaDict[pName]['dType']
            RollDaysPrior = self.priceMetaDict[pName]['RollDaysPrior']
            addTradingHours = self.priceMetaDict[pName]['addTradingHours']
            if RIC=='FCPO':
                RH = ReutersHub('PO',altName='dummy')
            else:
                RH = ReutersHub(RIC,altName='dummy')
            
            if (append) & (dType[:4]=='IDM_'):
                start_datetime = str(self.priceMetaDict[pName]['idx'][-1]+pd.offsets.Minute())
                df = RH.getTodaysIntradayTopresent(CN_YYYYMM,RolldaysPriorExp=RollDaysPrior,start_datetime=start_datetime,info=False)
            else:    
                df = RH.getTodaysIntradayTopresent(CN_YYYYMM,RolldaysPriorExp=RollDaysPrior,info=False)
            if df.shape[0]>0:
                if dType=='C': # if code below is enabled, it will lead to a different behavior than the backtest, hence disabled
                    # df = df.loc[df.index[-1]:,['CLOSE','CONTRACTS']]
                    # df['Roll'] = np.nan
                    # df.loc[:,['CLOSE','Roll']] = df.loc[:,['CLOSE','Roll']]/self.tickSizes[RIC]
                    # df = df.loc[:,['CLOSE','Roll','CONTRACTS']]
                    
                    # df.index.name = pName
                    # df['status'] ='CLOSE'
                    
                    # self.tradablePrices.append([df,self.priceMetaDict[pName]])
                    pass
                elif dType=='OHLC': # if code below is enabled, it will lead to a different behavior than the backtest, hence disabled
                    # df = df.loc[:,['OPEN','HIGH','LOW','CLOSE','CONTRACTS']]
                    # df['Roll'] = np.nan
                    
                    # df.iloc[-1,1] = np.max(df['HIGH'].values)
                    # df.iloc[-1,2] = np.min(df['LOW'].values)
                    
                    # df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']] = df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']]/self.tickSizes[RIC]
                    
                    # if addTradingHours:                        
                        
                    #     dfO = df.loc[:df.index[0],['OPEN']].copy()
                    #     dfC = df.loc[df.index[-1]:,['HIGH','LOW','CLOSE','CONTRACTS','Roll']].copy()
                        
                    #     dfO.index.name = pName
                    #     dfC.index.name = pName
                        
                    #     dfC['status'] ='CLOSE'
                        
                    #     self.tradablePrices.append([dfO,self.priceMetaDict[pName]])
                    #     self.tradablePrices.append([dfC,self.priceMetaDict[pName]])
                    # else:
                    #     df.iloc[-1,0] = df.iloc[0,0]
                    #     df = df.loc[df.index[-1]:,:]
                        
                    #     df.index.name = pName
                    #     df['status'] ='CLOSE'
                    #     self.tradablePrices.append([df,self.priceMetaDict[pName]])
                    pass
                elif dType[:4]=='IDM_':
                    df = df.loc[:,['OPEN','HIGH','LOW','CLOSE','CONTRACTS']]
                    df['Roll'] = np.nan
                    df['status'] = 'OPEN'
                    
                    df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']] = df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']]/self.tickSizes[RIC]
                    Nmins = int(dType[4:])
                    if Nmins!=1:
                        pass
                    
                    df.index.name = pName

                    if self.priceMetaDict[pName]['atTime'] != None:
                        if (self.priceMetaDict[pName]['factor'] != None)|(self.priceMetaDict[pName]['FXconv']):
                            pass # dealt with in prep to ensure a mathing datapoint for RV
                        else:
                            df = self.atTimeProcess(df,self.priceMetaDict[pName]['atTime'])
                    
                    self.tradablePrices.append([df,self.priceMetaDict[pName]])
                    
                elif dType[:4]=='IDR_':
                    pass
            else:
                if info:
                    print('No additional prices to add for '+pName)
                return False
        return True
            
    def addPrice(self,RIC,
                 CN_YYYYMM,
                 RollDaysPrior=0,
                 dataType='OHLC',
                 start_date=None,
                 end_date=None,
                 fromLocal=False,
                 addTradingHours=True,
                 nameOverride=None,
                 driverName='Peanut-process',
                 pathOfData=None,
                 factor = None,
                 FXconv = False,
                 RVvolRatio=1,
                 atTime = None,
                 skip=False):
        """Collects requested prices and adds to the to be processed list by self.prep().
        
        Parameters
        ----------
        RIC : str
            Any of the supported RICs: BO,SM,S,FCPO,RS,COM,LGO
        CN_YYYYMM : str,int
            Contract specification, either provide c1,c2,.. cN or any specific contract in YYYYMM int format
        RollDaysPrior : int
            Specify how many days to roll before spot contract expiration
        dataType : str
            Sets dataType of price source, options: 'C','OHLC','IDM_N' or 'IDR_N'
        start_date : str
            start date of price, see help(self.checkDateStr) for options
        end_date : str
            end date of price, see help(self.checkDateStr) for options
        fromLocal : bool
            Set to True when dataType='IDM_1', CN_YYYYMM='c2' for RIC in BO,SM,S,RS,LGO, 'c3' for FCPO or 'c1' for COM. Downloads presaved data.
        addTradingHours : bool
            When True, adds open/close time to dataType='C' or 'OHLC', when False, all times are set to 00:00 UTC.
        nameOverride : None or str
            A systematic name will be given to the product to refer to in the psDict and pospnlDict when None, otherwise nameOverride will be adopted.
        driverName : str
            Name of your ODBC driver for dataType='IDR_N' and 'IDM_1' data.
        pathOfData : str
            If you are one of these persons to use a Mac in the company, you can set the path to the proper windows path, ask Alex Lefter how to do it.
        factor : None, double, float
            The factor will be used to scale the price of the commodity, on which orders can be placed.
        FXconv : bool
            if True, the price will be converted into USD and can will be traded on that number.
        RVvolRatio : int
            If 1 unit of the RV is traded, trade RVvolRatio lots of this component.
        atTime : None or str
            Only for IDM_N intraday data. If not None, selects 1 minute candle per day at or first after time specified. In HH:MM format.
        skip : bool
            If True, price is used as a signal, not a tradable option. 
        """


        retrieveDetails = { 'RIC':RIC,
                           'dType':dataType,
                            'CN_YYYYMM':CN_YYYYMM,
                            'RollDaysPrior':RollDaysPrior,
                            'addTradingHours':addTradingHours,
                            'nameOverride':nameOverride,
                            'fromLocal':fromLocal,
                            'driverName':driverName,
                            'pathOfData':pathOfData,
                            'factor':factor,
                            'FXconv':FXconv,
                            'RVvolRatio':RVvolRatio,
                            'atTime':atTime,
                            'skip':skip}
        cont=True
        if self.checkTradingType(dataType):
            pass
        else:
            print('Price addition failed1.')
            cont=False
                
        if start_date != None:            
            if self.checkDateStr(start_date)==False:
                print('Price addition failed2.')
                cont= False
                
        if end_date != None:
            if self.checkDateStr(end_date)==False:
                print('Price addition failed3.')
                cont= False
        
        if cont:
            if dataType=='C':
                if RIC=='FCPO':
                    PRD = ReutersHub('PO',pathOfData=pathOfData,altName='dummy')
                else:
                    PRD = ReutersHub(RIC,pathOfData=pathOfData,altName='dummy')
                
                if fromLocal:
                    PRD.fromLocals(Nbdays=1)
                    
                if self.checkContractType(CN_YYYYMM):
                    df = PRD.select(CN_YYYYMM,RolldaysPriorExp=RollDaysPrior)
                    if 'Roll' not in df.columns:
                        df['Roll'] = np.nan
                    
                    df = df.loc[:,['CLOSE','CONTRACTS','Roll']]
                        
                    df.loc[:,['CLOSE','Roll']] = df.loc[:,['CLOSE','Roll']]/self.tickSizes[RIC]
                        
                    if start_date != None:
                        df = df[start_date:]
                        
                    if end_date != None:
                        df = df[:end_date]
                        
                    if addTradingHours:
                        tdC = self.calcMinOffset(self.tradingTimes[RIC]['CLOSE'])
                        idx = df.index
                        df.index = pd.DatetimeIndex([i+pd.offsets.Minute(n=tdC) for i in idx])
                    
                    
                    if nameOverride != None:
                        if (factor!=None)|(FXconv):
                            if nameOverride in list(self.groupCount.keys()):
                                groupN = self.groupCount[nameOverride]
                                addon = '_'+str(groupN+1)
                                self.groupCount.update({nameOverride:groupN+1})
                            else:
                                addon = '_'+str(1)
                                self.groupCount.update({nameOverride:1})
                        else:
                            addon = ''
                        df.index.name = nameOverride+addon
                    else:
                        df.index.name = RIC+'_'+ str(CN_YYYYMM)
                    df['status'] ='CLOSE'
                    self.tradablePrices.append([df,retrieveDetails])
                    if self.FX[RIC]!=None:
                        self.FXp.update({self.FX[RIC]:ReutersHub(self.FX[RIC],altName='dummy').CLOSE})
                else:
                    print('No price data added')
            elif dataType=='OHLC':
                if RIC=='FCPO':
                    PRD = ReutersHub('PO',pathOfData=pathOfData,altName='dummy')
                else:
                    PRD = ReutersHub(RIC,pathOfData=pathOfData,altName='dummy')
                    
                if self.checkContractType(CN_YYYYMM):
                    df = PRD.select(CN_YYYYMM,RolldaysPriorExp=RollDaysPrior)
                    
                    if 'Roll' not in df.columns:
                        df['Roll'] = np.nan
                    
                    df = df.loc[:,['OPEN','HIGH','LOW','CLOSE','CONTRACTS','Roll']]
                    
                    df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']] = df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']]/self.tickSizes[RIC]
                    
                    
                    if start_date != None:
                        df = df[start_date:]
                        
                    if end_date != None:
                        df = df[:end_date]
                    
                    if self.FX[RIC]!=None:
                        self.FXp.update({self.FX[RIC]:ReutersHub(self.FX[RIC],pathOfData=pathOfData,altName='dummy').CLOSE})
                    
                    if addTradingHours:
                        tdO = self.calcMinOffset(self.tradingTimes[RIC]['OPEN'])
                        tdC = self.calcMinOffset(self.tradingTimes[RIC]['CLOSE'])
                        
                        idx = df.index.copy()
                        
                        dfO = df.loc[:,['OPEN']].copy()
                        dfC = df.loc[:,['HIGH','LOW','CLOSE','CONTRACTS','Roll']].copy()
                        
                        
                        
                        dfO.index = pd.DatetimeIndex([i+pd.offsets.Minute(n=tdO) for i in idx])
                        dfC.index = pd.DatetimeIndex([i+pd.offsets.Minute(n=tdC) for i in idx])
                        
                        if nameOverride != None:
                            if (factor!=None)|(FXconv):
                                if nameOverride in list(self.groupCount.keys()):
                                    groupN = self.groupCount[nameOverride]
                                    addon = '_'+str(groupN+1)
                                    self.groupCount.update({nameOverride:groupN+1})
                                else:
                                    addon = '_'+str(1)
                                    self.groupCount.update({nameOverride:1})
                            else:
                                addon = ''
                            
                            
                            dfO.index.name = nameOverride+addon
                            dfC.index.name = nameOverride+addon
                        else:
                            dfO.index.name = RIC+'_'+ str(CN_YYYYMM)
                            dfC.index.name = RIC+'_'+ str(CN_YYYYMM)
                        
                    
                        dfC['status'] ='CLOSE'
                        
                        self.tradablePrices.append([dfO,retrieveDetails])
                        self.tradablePrices.append([dfC,retrieveDetails])
                        
                        
                    else:
                        if nameOverride != None:
                            if (factor!=None)|(FXconv):
                                if nameOverride in list(self.groupCount.keys()):
                                    groupN = self.groupCount[nameOverride]
                                    addon = '_'+str(groupN+1)
                                    self.groupCount.update({nameOverride:groupN+1})
                                else:
                                    addon = '_'+str(1)
                                    self.groupCount.update({nameOverride:1})
                            else:
                                addon = ''
                            
                            
                            df.index.name = nameOverride+addon
                        else:
                            df.index.name = RIC+'_'+ str(CN_YYYYMM)
                        df['status'] ='CLOSE'
                        self.tradablePrices.append([df,retrieveDetails])
                else:
                    print('No price data added')
            elif dataType[:4]=='IDR_':
                print('not yet implemented')
                pass
            elif dataType[:4]=='IDM_':
                Nmins = int(dataType[4:])
                cont = True
                if self.checkContractType(CN_YYYYMM)==False:
                    print('Price addition failed1.')
                    cont= False
                
                timeWhereClause = ''
                if start_date != None:            
                    if self.checkDateStr(start_date)==False:
                        print('Price addition failed2.')
                        cont= False
                    else:
                        timeWhereClause = timeWhereClause+" and date_time_min_start >= '"+start_date+"'"

                if end_date != None:
                    if self.checkDateStr(end_date)==False:
                        print('Price addition failed3.')
                        cont= False    
                    else:
                        timeWhereClause = timeWhereClause+" and date_time_min_start <= '"+end_date+"'"
                
                if cont:
                    if fromLocal==False:
                        cnxn = pyodbc.connect("DSN={0};Driver=/opt/cloudera/impalaodbc/lib/64/libclouderaimpalaodbc64.so;HOST=peanut-impala.cargill.com;TrustedCerts=C:/Program Files/Cloudera ODBC Driver for Impala/lib/cacerts.pem;CAIssuedCertNamesMismatch=1;AllowSelfSignedServerCert=1;SSL=1;KrbServiceName=impala;PORT=21050;KrbFQDN=_HOST;AuthMech=3;UseSASL=1;UseNativeQuery=0;autocommit=0;KrbFQDN=_HOST;UID=ps538353@EU.CORP.CARGILL.COM;PWD=KpX,Bl=bk9".format(driverName),autocommit=True)

                        cursor = cnxn.cursor()
                        print('requesting data: '+RIC+'_'+str(CN_YYYYMM))

                        R = """select * from prd_product_os_systematic_trading.reuters_fut_ts_min

                        where pseudo_ric='{0}{1}'{2}
                        order by date_time_min_end """.format(RIC,CN_YYYYMM,timeWhereClause)

                        print(R)
                        df = pd.read_sql(R,cnxn)
                        
                        
                        df.set_index(keys=['date_time_min_end'],drop=True,inplace=True)
                        df = df.loc[:,['open','high','low','close','pseudo_roll','contract','status']].copy()
                        df.columns = ['OPEN','HIGH','LOW','CLOSE','Roll','CONTRACTS','status']
                        
                                                                
                        df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']] = df.loc[:,['OPEN','HIGH','LOW','CLOSE','Roll']]/self.tickSizes[RIC]
                    
                    else: # ugly I know, but works for now
                        df = pd.read_parquet('C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/ReutersMinuteDataFix/{1}{2}.gzip'.format(os.getlogin(),RIC,CN_YYYYMM))
                        print('local price loaded')
                        
                        if start_date != None:   
                            if end_date != None:
                                df = df[start_date:end_date]
                            else:
                                df = df[start_date:]
                        elif end_date != None:
                            df = df[:end_date]
                    
                    if Nmins!=1:
                        pass
                    if atTime != None:
                        if (factor!=None)|(FXconv):
                            pass # dealt with in prep to ensure a mathing datapoint for RV
                        else:
                            df = self.atTimeProcess(df,atTime)
                            
                    if nameOverride != None:
                        if (factor!=None)|(FXconv):
                            if nameOverride in list(self.groupCount.keys()):
                                groupN = self.groupCount[nameOverride]
                                addon = '_'+str(groupN+1)
                                self.groupCount.update({nameOverride:groupN+1})
                            else:
                                addon = '_'+str(1)
                                self.groupCount.update({nameOverride:1})
                        else:
                            addon = ''
                        df.index.name = nameOverride+addon
                    else:
                        df.index.name = RIC+'_'+ str(CN_YYYYMM)
                        
                    if self.FX[RIC]!=None:
                        self.FXp.update({self.FX[RIC]:ReutersHub(self.FX[RIC],pathOfData=pathOfData,altName='dummy').CLOSE})
#                    df.to_parquet('C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/standardMinuteData/{1}{2}.gzip'.format(os.getlogin(),RIC,CN_YYYYMM))
                    self.tradablePrices.append([df,retrieveDetails])

    def atTimeProcess(self,df,atTime):
        benchM  = int(atTime[:2])*100+int(atTime[3:])
        HHMM = df.index.hour.values*100+df.index.minute.values
        mask = HHMM>=benchM
        mask[(mask[:-1] & mask[1:]).nonzero()[0]+1] = False
                                    
        rollLocs = (pd.isna(df.Roll)==False).values.nonzero()[0]
        
        if len(rollLocs)>0:
            dfR = df.loc[df.index[rollLocs],['Roll']].copy() # only the datetimes where there is a roll
            df = df.loc[mask,:]
            
            dfR = dfR.loc[dfR.index.isin(df.index)==False,:] # when the roll is discarded from the time selection, it should be included
            if dfR.shape[0]>0:
                dfRR = pd.concat([df.loc[:,['Roll']],dfR],axis=1) # the rolls do not align in the concat, but are 1 further down the line
                dfRR.columns = ['R1','R2']
                dfRR.R2 = dfRR.R2.shift(-1) # move 1 up to align with target date
                dfRR.loc[pd.isna(dfRR.R2)==False,'R1'] = dfRR.loc[pd.isna(dfRR.R2)==False,'R2'].copy() # copy them in
                df.Roll = dfRR.loc[df.index,'R1'].values # move to df
        else:
            df = df.loc[mask,:]
        return df
# ************************************************
# ************************************************
# ************************************************
class orderHandler():
    """Processes all orders and keeps track of positions and PnL."""
    def __init__(self,priceMetaDict,numParams,viz=False,keepOrders=False,aggOrders=False,monitorStart=None,monitorEnd=None):
                
        self.currOrders = {'timeStamp':None,
                            'active':np.zeros([0,numParams]).astype(bool),
                           'id':np.array([]).astype(int),
                          'oType':np.array([]),
                          'BS':np.zeros([0,numParams]).astype(bool),
                          'product':np.array([]),                           
                          'vol':np.zeros([0,numParams]).astype(int),
                          'val':np.zeros([0,numParams]).astype(int)}
        
        self.pList = list(priceMetaDict.keys())

        self.pTypeDict = {pName:priceMetaDict[pName]['dType'] for pName in list(priceMetaDict.keys())}
        self.pLastState = {pName:'CLOSE' for pName in list(priceMetaDict.keys())}
        self.pCurrState = {pName:'CLOSE' for pName in list(priceMetaDict.keys())}
        self.pLast =  {pName:None for pName in list(priceMetaDict.keys())}
        self.pCurrRoll = {pName:np.nan for pName in list(priceMetaDict.keys())}
        self.pCurr =  {pName:None for pName in list(priceMetaDict.keys())}
        self.pCurr_RV = {}
        self.RVpriceCheck = True
        self.pospnlKey = {pName:i for i,pName in enumerate(list(priceMetaDict.keys()))}
        self.pos = [np.zeros([priceMetaDict[pName]['len'],numParams]).astype(int) for pName in list(priceMetaDict.keys())]
        self.pnl = [np.zeros([priceMetaDict[pName]['len'],numParams]).astype(int) for pName in list(priceMetaDict.keys())]
        self.dates = {pName:priceMetaDict[pName]['idx'] for pName in list(priceMetaDict.keys())}
        self.loc = {pName:-1 for pName in list(priceMetaDict.keys())}
        self.numParams = numParams
        self.aggOrders = aggOrders
        
        self.RVdict={}
        for key in list(priceMetaDict.keys()): # check for RVs
            if (priceMetaDict[key]['nameOverride'] is not None) & (key!=priceMetaDict[key]['nameOverride']): #RV identified
                if priceMetaDict[key]['nameOverride'] not in list(self.RVdict.keys()):
                    
                    self.RVdict.update({priceMetaDict[key]['nameOverride']:{'pos':priceMetaDict[key]['nameOverride']+'_1','pnl':[key],'factor':[priceMetaDict[key]['factor']],'FXconv':[priceMetaDict[key]['FXconv']],'RVvolRatio':[priceMetaDict[key]['RVvolRatio']],'tickSize':[priceMetaDict[key]['tickSize']]}})
                else:
                    self.RVdict[priceMetaDict[key]['nameOverride']].update({'pnl':self.RVdict[priceMetaDict[key]['nameOverride']]['pnl']+[key],
                                                                           'factor':self.RVdict[priceMetaDict[key]['nameOverride']]['factor']+[priceMetaDict[key]['factor']],
                                                                           'FXconv':self.RVdict[priceMetaDict[key]['nameOverride']]['FXconv']+[priceMetaDict[key]['FXconv']],
                                                                           'RVvolRatio':self.RVdict[priceMetaDict[key]['nameOverride']]['RVvolRatio']+[priceMetaDict[key]['RVvolRatio']],
                                                                           'tickSize':self.RVdict[priceMetaDict[key]['nameOverride']]['tickSize']+[priceMetaDict[key]['tickSize']]})
                        
        if viz:
            if keepOrders:
                
                if aggOrders:
                    self.keepOrders  = [[]]
                    self.keepOrdersRV  = [[]]
                else:
                    self.keepOrders  = [[] for i in range(numParams)]
                    self.keepOrdersRV  = [[] for i in range(numParams)]
            if aggOrders:
                self.keepTrades  = [[]]
                self.keepTradesRV  = [[]]
            else:
                self.keepTrades  = [[] for i in range(numParams)]
                self.keepTradesRV  = [[] for i in range(numParams)]
            self.monitorStart = monitorStart
            self.monitorEnd = monitorEnd
    def prepareStart(self,priceMetaDict,FXtradePnL,lastFXrate,lastFXpos):       
        """Removes all in memory data in the order handler that will not be used for future continuation runs. Saves the minumum 
        required data for continuing the run with new data.
        """
        self.lastPos = {}        
        keyList = list(self.pospnlKey.keys())
        
        self.lastFXrate_OH = lastFXrate.copy()
        self.lastFXpos_OH  = lastFXpos.copy()
        for pName in keyList:
            self.lastPos.update({pName:self.pos[self.pospnlKey[pName]][-1,:]})
            
            if hasattr(self,'lastPnl'):
                if pName in list(self.lastPnl.keys()):
                    prevLast = self.lastPnl[pName]   
                    self.lastPnl.update({pName:prevLast+np.sum(self.pnl[self.pospnlKey[pName]],axis=0)*priceMetaDict[pName]['tickValues']/1000})               
                else:
                    self.lastPnl.update({pName:np.sum(self.pnl[self.pospnlKey[pName]],axis=0)*priceMetaDict[pName]['tickValues']/1000})               
                
            else:
                self.lastPnl = {}
                self.lastPnl.update({pName:np.sum(self.pnl[self.pospnlKey[pName]],axis=0)*priceMetaDict[pName]['tickValues']/1000})
            
            if pName in list(FXtradePnL.keys()):
                if hasattr(self,'lastPnlFX'):
                    if pName in list(self.lastPnlFX.keys()):
                        prevLast = self.lastPnlFX[pName]   
                        self.lastPnlFX.update({pName:prevLast+np.sum(FXtradePnL[pName].values,axis=0)})        
                    else:
                        self.lastPnlFX.update({pName:np.sum(FXtradePnL[pName].values,axis=0)})        
                else:
                    self.lastPnlFX ={}
                    self.lastPnlFX.update({pName:np.sum(FXtradePnL[pName].values,axis=0)})   
            self.loc.update({pName:-1})
        self.pos = None
        self.pnl = None
        self.dates = None
        
        if self.aggOrders:
            self.keepOrders  = [[]]
            self.keepTrades  = [[]]
        else:
            self.keepOrders  = [[] for i in range(self.numParams)]
            self.keepTrades  = [[] for i in range(self.numParams)]            
        
    def fromStart(self,priceMetaDict):
        """preps the pos, pnl and dates attributes to match the new data and adopts the latest positions."""
        if hasattr(self,'lastPos'):
            self.pos = [np.zeros([priceMetaDict[pName]['len'],self.numParams]).astype(int) for pName in list(priceMetaDict.keys())]
            self.pnl = [np.zeros([priceMetaDict[pName]['len'],self.numParams]).astype(int) for pName in list(priceMetaDict.keys())]
            self.dates = {pName:priceMetaDict[pName]['idx'] for pName in list(priceMetaDict.keys())}
            
            for pName in list(priceMetaDict.keys()):
                self.pos[self.pospnlKey[pName]][-1,:] =  self.lastPos[pName].copy()
        else:
            print('No previous start was recognised. Aborting.')
    def updateOH(self,priceMetaDict):
        """Resizes pos and pnl attributes to match the old combined with new data points."""
        for pName in list(priceMetaDict.keys()):
            loc = self.pospnlKey[pName]
            
            self.pos[loc] = np.append(self.pos[loc],np.zeros([priceMetaDict[pName]['len']-self.pos[loc].shape[0],self.numParams]).astype(int),axis=0)
            self.pnl[loc] = np.append(self.pnl[loc],np.zeros([priceMetaDict[pName]['len']-self.pnl[loc].shape[0],self.numParams]).astype(int),axis=0)
            
        self.dates = {pName:priceMetaDict[pName]['idx'] for pName in list(priceMetaDict.keys())}
        
        
    def newPriceInsert(self,priceSet):
        """Processes new data point."""
        for pName in self.pList:     
            if priceSet['prices'][pName]['info']: # only update when data point contains info
                
                self.pLastState.update({pName:self.pCurrState[pName]})
                if np.isnan(self.pCurrRoll[pName]):
                    self.pLast.update({pName:self.pCurr[pName]})
                else:
                    self.pLast.update({pName:self.pCurr[pName]-self.pCurrRoll[pName]})
                
                self.pCurrState.update({pName:priceSet['prices'][pName]['status']})# adopt status of last data point
                                
                if np.isnan(priceSet['prices'][pName]['CLOSE']): # all prices have CLOSE, if info==True and CLOSE is Nan, -> OPEN
                    self.pCurr.update({pName:priceSet['prices'][pName]['OPEN']})                    
                else:
                    if 'Roll' in list(priceSet['prices'][pName].keys()):
                        if np.isnan(priceSet['prices'][pName]['Roll'])==False:
                            self.pCurrRoll.update({pName: int(priceSet['prices'][pName]['Roll'])})
                            
                        else:
                            self.pCurrRoll.update({pName: np.nan})
                    else:
                        self.pCurrRoll.update({pName: np.nan})
                    self.pCurr.update({pName:priceSet['prices'][pName]['CLOSE']})
                
                # roll pos to new loc and update loc
                
                oldPos = self.pos[self.pospnlKey[pName]][self.loc[pName],:].copy()                
                self.loc.update({pName:self.loc[pName]+1})
                self.pos[self.pospnlKey[pName]][self.loc[pName],:] = oldPos
                                                    
        self.priceSet = priceSet.copy()
        
        if self.RVpriceCheck: # only for Production runner
            rvNames = list(priceSet['RV'].keys())
            if len(rvNames)>0:
                for rvName in rvNames:
                    if priceSet['RV'][rvName]['info']:
                        self.pCurr_RV.update({rvName:priceSet['RV'][rvName]['CLOSE']})
            else:
                self.RVpriceCheck=False
                    
    def replaceRVorder(self,orders):
        """transforms orders on RV values to orders on individual legs when RV order is filled."""
        currN = int(np.max(orders['id']))
        toKeep = []
        toRemove = []
        toAppend = []
        for i,pName in enumerate(list(orders['product'])):
            if pName not in list(self.RVdict.keys()):
                toKeep.append(i)
            else:
                toRemove.append(i)
                if self.priceSet['RV'][pName]['info']:    
                             
                    parts = self.RVdict[pName]['pnl']
                    N = len(parts)
                    
                    
                    ids = []
                    oType = np.array(['MARKET' for j in range(N)])
                    product = []
                    
                    active = np.zeros([N,orders['active'].shape[1]]).astype(bool)
                    BS = np.zeros([N,orders['active'].shape[1]]).astype(bool)
                    vol = np.ones([N,orders['active'].shape[1]])
                    val = np.ones([N,orders['active'].shape[1]])
                    
                    
                    exVal=0
                    for j,part in enumerate(parts):
                        currN +=1
                        ids.append(currN)
                        
                        product.append(part)   
                        val[j,:] = self.priceSet['prices'][part]['CLOSE']
                        vol[j,:] = orders['vol'][i,:]*self.RVdict[pName]['RVvolRatio'][j]
                        BS[j,:] = orders['BS'][i,:]==(np.sign(self.RVdict[pName]['factor'][j])==1)
                        
                        if part in list(self.priceSet['FX'].keys()):
                            FX = self.priceSet['FX'][part]['CLOSE']
                        else:
                            FX = 1
                            
                        exVal = exVal+self.priceSet['prices'][part]['CLOSE']*FX*self.RVdict[pName]['factor'][j]*self.RVdict[pName]['tickSize'][j]
                    if orders['oType'][i]=='MARKET':   
                        active[:,orders['active'][i,:]] = True                    
                    elif orders['oType'][i]=='LIMIT':
                                                
                        loc = (orders['active'][i,:]).nonzero()[0]
                        locB = loc[orders['BS'][i,loc]]
                        locS = loc[orders['BS'][i,loc]==False]
                                                
                        if len(locB)>0:
                            locBa = locB[self.priceSet['RV'][pName]['CLOSE']<=orders['val'][i,locB]]
                            active[:,locBa] = True
                        if len(locS)>0:
                            locSa = locS[self.priceSet['RV'][pName]['CLOSE']>=orders['val'][i,locS]]
                            active[:,locSa] = True                        
                    elif orders['oType'][i]=='STOP':
                        loc = (orders['active'][i,:]).nonzero()[0]
                        locB = loc[orders['BS'][i,loc]]
                        locS = loc[orders['BS'][i,loc]==False]
                                                
                        if len(locB)>0:
                            locBa = locB[self.priceSet['RV'][pName]['CLOSE']>=orders['val'][i,locB]]
                            active[:,locBa] = True
                        if len(locS)>0:
                            locSa = locS[self.priceSet['RV'][pName]['CLOSE']<=orders['val'][i,locS]]
                            active[:,locSa] = True      
                    newOrder = {'active':active,
                           'id':np.array(ids).astype(int),
                          'oType':oType,
                          'BS':BS,
                          'product':np.array(product),                           
                          'vol':vol.astype(int),
                          'val':val.astype(int)}
                    
                    toAppend.append(newOrder)
                    orders['active'][i,active[0,:]] = False # executed RV order
                    
                    if hasattr(self,'keepTradesRV'): # viz is activated            
                        if (self.monitorStart<=self.priceSet['t']) & (self.monitorEnd>=self.priceSet['t']):
                            
                            if hasattr(self,'keepOrdersRV'): # viz is activated        
                                Ot = []
                                x = i
                                y = (orders['active'][i,:]).nonzero()[0]
                                for j in range(len(y)):
                                    O = [self.priceSet['t'],orders['oType'][x],
                                                              orders['product'][x],orders['BS'][x,y[j]],
                                                              orders['vol'][x,y[j]],orders['val'][x,y[j]]]
                                    
                                    if self.aggOrders:
                                        Ot.append(O)
                                    else:
                                        self.keepOrdersRV[y[j]].append(O)
                                if (self.aggOrders) & (len(y)>0):
                                    df = pd.DataFrame(Ot,columns=['t','oType','product','BS','vol','val'])
                                    df.groupby(['t','oType','product','BS','val'])['vol'].sum().reset_index()
                                    O = df.loc[:,['t','oType','product','BS','vol','val']].values.tolist()
                                    self.keepOrdersRV[0] = self.keepOrdersRV[0]+O
                            x = i
                            y = active[0,:].nonzero()[0]
                            Tt = []
                            for j in range(len(y)):                                
                                
                                T = [self.priceSet['t'],orders['oType'][x],
                                                      orders['product'][x],orders['BS'][x,y[j]],
                                                      orders['vol'][x,y[j]],exVal]
                                if self.aggOrders:
                                    Tt.append(T)
                                else:
                                    self.keepTradesRV[y[j]].append(T)
                            if (self.aggOrders) & (len(y)>0):
                                df = pd.DataFrame(Tt,columns=['t','oType','product','BS','vol','val'])
                                df.groupby(['t','oType','product','BS','val'])['vol'].sum().reset_index()
                                T = df.loc[:,['t','oType','product','BS','vol','val']].values.tolist()
                                self.keepTradesRV[0] = self.keepTradesRV[0]+T
                    
                    
               
               
        
        if len(toRemove)>0:

            self.currRVorders = copy.deepcopy(orders)    

            toKeep = np.array(toKeep).astype(int)
            
            for key in ['active','BS','vol','val']:
                self.currRVorders.update({key:orders[key][toRemove,:]})
                orders.update({key:orders[key][toKeep,:]})                
            for key in ['id','oType','product']:
                self.currRVorders.update({key:orders[key][toRemove]})
                orders.update({key:orders[key][toKeep]})                
            for newO in toAppend:
                for key in ['active','BS','vol','val']:
                    orders.update({key:np.append(orders[key],newO[key],axis=0)})
                for key in ['id','oType','product']:
                    orders.update({key:np.append(orders[key],newO[key])})
                
        else:
            self.currRVorders = {'id' : np.array([])}
                    
        return orders
              
    def checkValidity(self,orders,override):
        """Check whether orders are valid given price availability and price type."""

        if len(orders['id'])==0:
            rejected = np.array([])
            
            orders = {'timeStamp':None,
                            'active':np.zeros([0,self.numParams]).astype(bool),
                           'id':np.array([]).astype(int),
                          'oType':np.array([]),
                          'BS':np.zeros([0,self.numParams]).astype(bool),
                          'product':np.array([]),                           
                          'vol':np.zeros([0,self.numParams]).astype(int),
                          'val':np.zeros([0,self.numParams]).astype(int)}
            self.currRVorders = {'id' : np.array([])}
        else:      
            orders = self.replaceRVorder(orders)
            
            rejected = np.zeros(orders['active'].shape).astype(bool)

            for i,pName in enumerate(list(orders['id'])):
                pName = orders['product'][i]
                if (self.priceSet['prices'][pName]['info']==False)&(orders['oType'][i]!='TAS'): # price contains no information, reject order
                    rejected[i,:] = True
                    print('REJECTED '+pName+': no current data point.')
                else:
                    pTypeCurr = self.pTypeDict[pName]

                    if pTypeCurr =='C':                
                        if orders['oType'][i] in ['LIMIT','STOP']: # reject as for this pType only market orders are allowed
                            rejected[i,:] = True
                            print('REJECTED '+pName+ ": 'C' priceType can only process market orders, got orderType: "+orders['oType'][i])
                    elif pTypeCurr =='OHLC': # anything goes
                        pass
                    elif pTypeCurr[:4] == 'IDR_': # 

                        if orders['oType'][i] in ['LIMIT','STOP']: # check for threshold condition
                            N = int(pTypeCurr[4:])
                            if self.priceSet['prices'][pName]['status'] in ['OPEN','CLOSE']: # anything goes
                                pass
                            elif self.priceSet['prices'][pName]['status']=='up': # no limit order within N ticks lower than current, also any limit/stop order through the market is also cancelled
                                idx = orders['active'][i,:].nonzero()[0]
                                if len(idx)!=0:

                                    val = self.priceSet['prices'][pName]['value']
                                    limitSellThroughMarket = (orders['oType'][idx,:]=='LIMIT') & (orders['BS'][idx,:]==False) & (orders['val'][idx,:]<=val).nonzero()[0]
                                    limitBuyThroughTheshold = (orders['oType'][idx,:]=='LIMIT') & (orders['BS'][idx,:]==True) & (orders['val'][idx,:]>val-N*2).nonzero()[0]
                                    stopBuyThroughMarket = (orders['oType'][idx,:]=='STOP') & (orders['BS'][idx,:]==True) & (orders['val'][idx,:]<=val).nonzero()[0]
                                    stopSellThroughTheshold = (orders['oType'][idx,:]=='STOP') & (orders['BS'][idx,:]==False) & (orders['val'][idx,:]>val-N*2).nonzero()[0]

                                    rejected[i,idx[limitSellThroughMarket]] = True
                                    rejected[i,idx[limitBuyThroughTheshold]] = True
                                    rejected[i,idx[stopBuyThroughMarket]] = True
                                    rejected[i,idx[stopSellThroughTheshold]] = True

                                    if np.sum(rejected[i,:])>0:
                                        NNN = np.sum(rejected[i,:])
                                        print('REJECTED '+pName+': '+str(NNN)+ 'out of ' + str(len(rejected[i,:])) +' order(s) do no adhere to the order input boundaries of the priceType: '+pTypeCurr)
                            elif self.priceSet['prices'][pName]['status']=='down':

                                idx = orders['active'][i,:].nonzero()[0]
                                if len(idx)!=0:

                                    val = self.priceSet['prices'][pName]['value']
                                    limitSellThroughTheshold = (orders['oType'][idx,:]=='LIMIT') & (orders['BS'][idx,:]==False) & (orders['val'][idx,:]<val+N*2).nonzero()[0]
                                    limitBuyThroughMarket = (orders['oType'][idx,:]=='LIMIT') & (orders['BS'][idx,:]==True) & (orders['val'][idx,:]>=val).nonzero()[0]
                                    stopBuyThroughTheshold = (orders['oType'][idx,:]=='STOP') & (orders['BS'][idx,:]==True) & (orders['val'][idx,:]<val+N*2).nonzero()[0]
                                    stopSellThroughMarket = (orders['oType'][idx,:]=='STOP') & (orders['BS'][idx,:]==False) & (orders['val'][idx,:]>=val).nonzero()[0]

                                    rejected[i,idx[limitSellThroughTheshold]] = True
                                    rejected[i,idx[limitBuyThroughMarket]] = True
                                    rejected[i,idx[stopBuyThroughTheshold]] = True
                                    rejected[i,idx[stopSellThroughMarket]] = True

                                    if np.sum(rejected[i,:])>0:
                                        NNN = np.sum(rejected[i,:])
                                        print('REJECTED '+pName+': '+str(NNN)+ 'out of ' + str(len(rejected[i,:])) +' order(s) do no adhere to the order input boundaries of the priceType: '+pTypeCurr)
                        else: # reject market orders unless info field shows close or last state shows close (by def open of the market).
                            if (self.pLastState[i]=='CLOSE') | (self.priceSet['prices'][pName]['status']=='CLOSE'):
                                pass
                            else: # reject
                                rejected[i,:] = True
                                print('REJECTED '+pName+': Market orders not allowed for priceType '+pName +' '+pTypeCurr+ ' when the market is in between OPEN or CLOSE')

                    elif pTypeCurr[:4] == 'IDM_': # anything goes
                        pass

            orders['active'][rejected]=False
#         return orders, rejected
        if override:
            self.currOrders = orders
            self.lastRejected = {'timeStamp':orders['timeStamp'],'rejected':rejected}
        else:
            self.mergeOrders(orders,rejected)
    def mergeOrders(self,orders,rejected):
        """Merges new orders with already existing orderbook. Active orders in new orderbook will 
        replace existing orders in the saved orderbook.
        If ID is matching but the ordertype is different between the 2 books, new orders are rejected and original orders remain"""
        
        if len(orders['id'])>0:
            orderID = self.currOrders['id'].copy()
            pNames = self.currOrders['product'].copy()
            oType = self.currOrders['oType'].copy()
            
            for i,oID in enumerate(orders['id']):
                loc = (orderID==oID).nonzero()[0]
                if len(loc)==1: # overwrite with active orders

                    loc = loc[0]

                    if self.currOrders['oType'][loc]==orders['oType'][i]: # orderTypes may not change as subselection is changed

                        active = orders['active'][i,:].nonzero()[0]

                        self.currOrders['active'][loc,active] = True

                        for fld in ['BS','vol','val']:
                            M = self.currOrders[fld].copy()                            
                            M[loc,active] = orders[fld][i,active]
                            self.currOrders.update({fld:M})
                            
                    else:
                        rejected[i,:] = True                
                        print('REJECTED '+pNames[i]+': Order rejectes as orderType is not consistent. Expected '+self.currOrders['oType'][loc]+' got '+orders['oType'][i])

                elif len(loc)==0: # new order line
                    orderID = np.append(orderID,oID)
                    pNames = np.append(pNames,orders['product'][i])
                    oType = np.append(oType,orders['oType'][i])

                    for fld in ['active','BS','vol','val']:
                        M = self.currOrders[fld].copy()
                        M = np.append(M,orders[fld][i,:].reshape([1,-1]),axis=0)
                        self.currOrders.update({fld:M})

                else:
                    print('Orderlist contains non unique order IDs, stop the process!')


            self.currOrders.update({'id':orderID,'product':pNames,'oType':oType,'timeStamp':orders['timeStamp']})
            self.lastRejected = {'timeStamp':orders['timeStamp'],'rejected':rejected}
    def getUnusesID(self):
        """Find first orderID which is available.
        
        Returns
        -------
        
        int
            Free order ID"""

        
        maxID = np.max(self.currOrders['id'])
        for i in range(maxID+1):
            if i not in list(self.currOrders['id']):
                return i
        return maxID+1
    
    def evalOrders(self,instant=False):
        """Evaluate current orders against current price set."""
        fills = np.zeros(self.currOrders['active'].shape).astype(bool)
        exVal = self.currOrders['val'].copy()
        fillDict = {'timeStamp':self.currOrders['timeStamp'],
                 'BS':self.currOrders['BS'].copy(),
                 'product':self.currOrders['product'].copy(),
                 'vol':self.currOrders['vol'].copy()}
        
        for i,oType in enumerate(self.currOrders['oType']):
            
            pName = self.currOrders['product'][i]
            
            if self.priceSet['prices'][pName]['info']: # only evaluate when new data point in provided with info
                
                if self.priceSet['prices'][pName]['dType'] == 'C': # only has market orders and TAS orders
                    evalAs = 'CLOSE'      
                    val = int(self.priceSet['prices'][pName]['CLOSE'])
                    idx = self.currOrders['active'][i,:].nonzero()[0]
                    if len(idx)>0:
                        fills[i,idx] = True
                        exVal[i,idx] = val
                    
                elif self.priceSet['prices'][pName]['dType'][:4] in ['OHLC','IDM_','IDR_']: # 
                    if self.priceSet['prices'][pName]['dType'][:4] == 'OHLC':
                        if (self.pLastState[pName]=='CLOSE'):
                            if (np.isnan(self.priceSet['prices'][pName]['OPEN'])) | (np.isnan(self.priceSet['prices'][pName]['CLOSE'])): # OPEN and CLOSE are split
                                evalAs = 'OPEN'
                            else: # OPEN and close together in one time step
                                evalAs = 'CLOSE'
                        else:
                            evalAs = 'CLOSE'
                    elif self.priceSet['prices'][pName]['dType'][:4] == 'IDM_':
                        if instant:
                            evalAs = 'CLOSE'
                        else:
                            evalAs = 'OPEN'
                    elif self.priceSet['prices'][pName]['dType'][:4] == 'IDR_':
                        evalAs = 'CLOSE'
                        
                    idx = self.currOrders['active'][i,:].nonzero()[0]
                    if len(idx)>0:     
                        B = idx[self.currOrders['BS'][i,idx]]
                        S = idx[self.currOrders['BS'][i,idx]==False]

                        if (evalAs=='OPEN') | (instant): # in this state, no previous market moves are takin into account
                            if (self.priceSet['prices'][pName]['dType'][:4]=='IDR_') | (evalAs=='CLOSE'):
                                val = self.priceSet['prices'][pName]['CLOSE'] # IDR only has close as a field
                            else:
                                val = self.priceSet['prices'][pName]['OPEN']
                            if oType == 'LIMIT':
                                Bex = B[self.currOrders['val'][i,B]>=val]
                            elif oType == 'MARKET':
                                Bex = B
                            elif oType == 'STOP':
                                Bex = B[self.currOrders['val'][i,B]<=val]
                            fills[i,Bex] = True
                            exVal[i,Bex] = val

                            if oType == 'LIMIT':
                                Sex = S[self.currOrders['val'][i,S]<=val]
                            elif oType == 'MARKET':
                                Sex = S
                            elif oType == 'STOP':
                                Sex = S[self.currOrders['val'][i,S]>=val]                                
                            fills[i,Sex] = True
                            exVal[i,Sex] = val
                            
                            if (self.priceSet['prices'][pName]['dType'][:4] in ['OHLC','IDM_']) & (instant==False): # two step approach necessary
                                Hi = self.priceSet['prices'][pName]['HIGH']
                                Lo = self.priceSet['prices'][pName]['LOW']
                                val = self.priceSet['prices'][pName]['CLOSE']
                                
                                if (np.isnan(self.priceSet['prices'][pName]['OPEN'])) | np.isnan(val): # abort two step approach, OPEN and HLC are split
                                    pass
                                else:
                                    active = self.currOrders['active'].copy()
                                    active[fills] = False # update fills for now

                                    idx = active[i,:].nonzero()[0]
                                    
                                    B = idx[self.currOrders['BS'][i,idx]]
                                    S = idx[self.currOrders['BS'][i,idx]==False]
                                    
                                    if oType == 'LIMIT':
                                        Bex = B[self.currOrders['val'][i,B]>=Lo]                            
                                    elif oType == 'STOP':
                                        Bex = B[self.currOrders['val'][i,B]<=Hi]
                                    elif oType == 'MARKET':
                                        Bex = B


                                    if oType == 'LIMIT':
                                        Sex = S[self.currOrders['val'][i,S]<=Hi]
                                    elif oType == 'STOP':
                                        Sex = S[self.currOrders['val'][i,S]>=Lo]
                                    elif oType == 'MARKET':
                                        Sex = S


                                    fills[i,Bex] = True
                                    if oType == 'MARKET':
                                        exVal[i,Bex] = val
                                    else:
                                        exVal[i,Bex] = self.currOrders['val'][i,Bex].copy()


                                    fills[i,Sex] = True
                                    if oType == 'MARKET':
                                        exVal[i,Sex] = val
                                    else:
                                        exVal[i,Sex] = self.currOrders['val'][i,Sex].copy()
                                
                        else: # close value & orders were set before current price timestamp
                            if self.priceSet['prices'][pName]['dType'][:4]=='IDR_':
                                Hi = self.priceSet['prices'][pName]['CLOSE']
                                Lo = self.priceSet['prices'][pName]['CLOSE']
                                val = self.priceSet['prices'][pName]['CLOSE']
                            else:
                                Hi = self.priceSet['prices'][pName]['HIGH']
                                Lo = self.priceSet['prices'][pName]['LOW']
                                val = self.priceSet['prices'][pName]['CLOSE']

                            if oType == 'LIMIT':
                                Bex = B[self.currOrders['val'][i,B]>=Lo]                            
                            elif oType == 'STOP':
                                Bex = B[self.currOrders['val'][i,B]<=Hi]
                            elif oType == 'MARKET':
                                Bex = B


                            if oType == 'LIMIT':
                                Sex = S[self.currOrders['val'][i,S]<=Hi]
                            elif oType == 'STOP':
                                Sex = S[self.currOrders['val'][i,S]>=Lo]
                            elif oType == 'MARKET':
                                Sex = S
                            

                            fills[i,Bex] = True
                            if oType == 'MARKET':
                                exVal[i,Bex] = val
                            else:
                                exVal[i,Bex] = self.currOrders['val'][i,Bex].copy()


                            fills[i,Sex] = True
                            if oType == 'MARKET':
                                exVal[i,Sex] = val
                            else:
                                exVal[i,Sex] = self.currOrders['val'][i,Sex].copy()
        fillDict.update({'fills':fills,
                         'val':exVal}) 
                    
        if hasattr(self,'keepTrades'): # viz is activated
            
            if (self.monitorStart<=self.priceSet['t']) & (self.monitorEnd>=self.priceSet['t']):
                if instant: # insert new orders and fills

                    if hasattr(self,'keepOrders'): # viz is activated        
                        Ot = []
                        x,y = (self.currOrders['active'] | fills).nonzero()
                        for i in range(len(x)):
                            O = [self.priceSet['t'],self.currOrders['oType'][x[i]],
                                                      self.currOrders['product'][x[i]],self.currOrders['BS'][x[i],y[i]],
                                                      self.currOrders['vol'][x[i],y[i]],self.currOrders['val'][x[i],y[i]]]
                            
                            if self.aggOrders:
                                Ot.append(O)
                            else:
                                self.keepOrders[y[i]].append(O)
                        if (self.aggOrders) & (len(x)>0):
                            df = pd.DataFrame(Ot,columns=['t','oType','product','BS','vol','val'])
                            df.groupby(['t','oType','product','BS','val'])['vol'].sum().reset_index()
                            O = df.loc[:,['t','oType','product','BS','vol','val']].values.tolist()
                            self.keepOrders[0] = self.keepOrders[0]+O
                x,y = fills.nonzero()
                Tt = []
                for i in range(len(x)):
                    
                    
                    T = [self.priceSet['t'],self.currOrders['oType'][x[i]],
                                          self.currOrders['product'][x[i]],self.currOrders['BS'][x[i],y[i]],
                                          self.currOrders['vol'][x[i],y[i]],exVal[x[i],y[i]]]
                    if self.aggOrders:
                        Tt.append(T)
                    else:
                        self.keepTrades[y[i]].append(T)
                if (self.aggOrders) & (len(x)>0):
                    df = pd.DataFrame(Tt,columns=['t','oType','product','BS','vol','val'])
                    df.groupby(['t','oType','product','BS','val'])['vol'].sum().reset_index()
                    T = df.loc[:,['t','oType','product','BS','vol','val']].values.tolist()
                    self.keepTrades[0] = self.keepTrades[0]+T

        active = self.currOrders['active'].copy()
        active[fills] = False
        self.currOrders.update({'active':active})
        self.posPnLCalc(fillDict,instant=instant)
        
    def getCurrentPosPnl(self):
        """Returns dictionary of current position and time step pnl."""
        pospnlDict = {}
        for pName in list(self.pospnlKey.keys()):
            idx = self.pospnlKey[pName]
            loc = self.loc[pName]
            pospnlDict.update({pName:{'pos':self.pos[idx][loc,:],'pnl':self.pnl[idx][loc,:]}})
            
        for RV in list(self.RVdict.keys()):
            pospnlDict.update({RV:{'pos':pospnlDict[self.RVdict[RV]['pos']]['pos'],
                                   'pnl':np.nan}}) # does not return the combined PnL as the pnls here are in ticks and I dont value the step Pnl, if wanted, first convert to $ pnl before adding
            
        
        return pospnlDict
        
    def posPnLCalc(self,fillDict,instant=False):
        """Keeps track of position and PnL. Two step process, first current positions are evaluated against price move,
        then new fills are evaluated against final price and the two are added. In instant mode, only position is adjusted as
        only market orders are evaluated and hence have no Pnl impact."""
        
        if instant: # only fills which alter positions   
            for i, pName in enumerate(fillDict['product']):
                idx = fillDict['fills'][i,:].nonzero()[0]
                
                if len(idx)>0:
                    posMut = fillDict['vol'][i,idx].copy()
                    BS = fillDict['BS'][i,idx]                    
                    posMut[BS==False] = posMut[BS==False]*-1                   
                    self.pos[self.pospnlKey[pName]][self.loc[pName],idx] = self.pos[self.pospnlKey[pName]][self.loc[pName],idx]+posMut
                    
        else: # 
            for pName in self.pList: 
                if self.priceSet['prices'][pName]['info']:
                    if self.pLast[pName]!=None:
                        priceDiff = self.pCurr[pName]-self.pLast[pName] 
                        self.pnl[self.pospnlKey[pName]][self.loc[pName],:] = self.pnl[self.pospnlKey[pName]][self.loc[pName],:]+priceDiff*self.pos[self.pospnlKey[pName]][self.loc[pName],:]
                    else:
                        priceDiff = 0
            for i, pName in enumerate(fillDict['product']):
                idx = fillDict['fills'][i,:].nonzero()[0]
                
                if len(idx)>0:
                    posMut = fillDict['vol'][i,idx].copy()
                    BS = fillDict['BS'][i,idx]
                    val = fillDict['val'][i,idx]
                    posMut[BS==False] = posMut[BS==False]*-1
                    pnlMut = (self.pCurr[pName]-val)*posMut
                    
                    self.pnl[self.pospnlKey[pName]][self.loc[pName],idx] = self.pnl[self.pospnlKey[pName]][self.loc[pName],idx]+pnlMut
                    self.pos[self.pospnlKey[pName]][self.loc[pName],idx] = self.pos[self.pospnlKey[pName]][self.loc[pName],idx]+posMut
# ************************************************
# ************************************************
# ************************************************
class tradePack():
    """Core tradePack module, able to perform rolling out of sample, parameterized back-tests and production runs."""
    def __init__(self,priceOrTicks='ticks'):
        self.colors = ['green','cyan','purple','teal', 'darkolivegreen','orangered', 'aqua', 
                          'brown', 'burlywood', 'cadetblue',
                          'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson',  
                          'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen',  
                          'darkkhaki', 'darkmagenta',  'darkorange', 'darkred']
        if priceOrTicks in ['price','ticks']:
            self.DH = dataHander(priceOrTicks=priceOrTicks)            
        else:
            print('priceOrTicks only accepts "price" or "ticks". Got: '+str(priceOrTicks))
        
    def addPrice(self,RIC,
                 CN_YYYYMM,
                 RollDaysPrior=0,
                 dataType='OHLC',
                 start_date=None,
                 end_date=None,
                 fromLocal=False,
                 addTradingHours=True,
                 nameOverride=None,
                 driverName='Peanut-process',
                 pathOfData=None,
                 factor = None,
                 FXconv = False,
                 RVvolRatio=1,
                 atTime = None,
                 skip = False):

        """Duplicate of DH.addPrice"""
        if hasattr(self,'DH'):
            self.DH.addPrice(RIC,
                 CN_YYYYMM,
                 RollDaysPrior=RollDaysPrior,
                 dataType=dataType,
                 start_date=start_date,
                 end_date=end_date,
                 fromLocal=fromLocal,
                 addTradingHours=addTradingHours,
                 nameOverride=nameOverride,
                 driverName=driverName,
                 pathOfData=pathOfData,
                 factor = factor,
                 FXconv = FXconv,
                 RVvolRatio=RVvolRatio,
                 atTime=atTime,
                 skip=skip)
        else:
            print('ERROR: price not added, first set trading type.')
    def addSignal(self,signal_df_in,signal_time=None):
        """Duplicate of dataHandler.addSignal"""
        if hasattr(self,'DH'):
            self.DH.addSignal(signal_df_in,signal_time=signal_time)
        else:
            print('ERROR: signal(s) not added, first set trading type.')
    def checkUniqueParamCombinations(self,params):
        """Checks for uniqueness of input paramter combinations.
        
        Returns
        -------
        
        bool"""
        df = pd.DataFrame(params)
        
        if df.shape[1]==1: # 1 parameter
            if df.shape[0]==1: # 1 value -> pass
                return True
            else:
                Nu = len(np.unique(df.values.flatten()))
                if Nu==df.shape[0]:
                    return True
                else:
                    print('Parameter set contains at least '+str(df.shape[0]-Nu)+' duplicate parameter combinations. Strategy rejected.')
                    return False
        elif df.shape[0]==1: # only one combination so always true
            return True
        else:
            X = df.sort_values(by=list(df.columns)).values
    
            if np.max(np.sum(X[1:,:]==X[:-1,:],axis=1))==len(df.columns):
                Nd = np.sum(np.sum(X[1:,:]==X[:-1,:],axis=1))/len(df.columns)
                print('Parameter set contains at least '+str(int(Nd))+' duplicate parameter combinations. Strategy rejected.')
                return False
            else:
                return True
    def addStrategy(self,strat):
        """Adds instance of strategy object to tradePack and checks compatibility.""" 
        if hasattr(self,'strat'):
            print('Current strategy is overwritten.')
        cont = False
        if 'evaluate' in dir(strat):
            if str(type(strat.evaluate))=="<class 'method'>":
                cont = True
            else:
                print('ERROR: strat input must be an instance of a class containing a evaluate method.')            
        else:
            print('ERROR: strat input must be an instance of a class containing a evaluate method.')
                
        
        if cont:
            if hasattr(strat,'params'):
                for i,key in enumerate(list(strat.params.keys())):
                    if type(strat.params[key])!=type(np.array([])):
                        print('ERROR: strat input must be an instance of a class containing the params attribute as a dict. All keys must contain equal lengt np arrays.')
                        cont = False
                        break
                    if i ==0:                        
                        N = len(strat.params[key])
                    elif strat.params[key].shape!=(N,):
                        print('ERROR: strat input must be an instance of a class containing the params attribute as a dict. All keys must contain equal lengt np arrays.')
                        cont = False
                        break
                if cont:
                    cont =  self.checkUniqueParamCombinations(strat.params)
            else:
                strat.params = {'no params':np.array(['single run'])}
                print('No parameter set passed, expecting a single unparameterised run. ')
        if cont:
            self.strat = copy.deepcopy(strat)
            self.stratParams = copy.deepcopy(strat.params)
    def setupParamSpace(self,pDict):
        """Creates all possible combinations of input parameters and returns new parameter dictionary."""
        numParams = 1
        types = []
        for i in pDict.keys():
            numParams = numParams*len(pDict[i])
            types.append(type(pDict[i][0]))
        numParams = int(numParams)
        print('number of parameter combinations: '+str(numParams))

        paramDict ={}
        counter1=1
        counter2 = numParams

        for idx, paramKey in enumerate(pDict.keys()):
            param = pDict[paramKey]
            counter2 = int(counter2/len(param))
            intData = np.array([])
            for val in param:
                intData = np.append(intData,np.tile(val,counter1))        

            newData = np.tile(intData,counter2)
            newData = newData.astype(types[idx])

            paramDict.update({paramKey:newData})                      
            counter1 = int(counter1*len(param))
        return paramDict, numParams
    def checkDateStr(self,dateSTR):
        """Duplicate of dataHandler.checkDateStr"""
        self.fDict = {10:'%Y-%m-%d',
                16:'%Y-%m-%d %H:%M',
                19:'%Y-%m-%d %H:%M:%S'}
        
        try:
            if dt.datetime.strftime(pd.to_datetime(dateSTR),self.fDict[len(dateSTR)])==dateSTR:                
                return True
            else:
                print('Date format not recognised, expecting YYYY-MM-DD, YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS format, got: '+dateSTR)
                return False
        except:
            print('Date format not recognised, expecting YYYY-MM-DD, YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS format, got: '+str(dateSTR))
            return False
    def addTodaysPrices(self,append=False,info=True):        
        """Adds todays prices to historical up to yesterday. Add today's signals as well after this step and then run vizCheck(first=False)"""
        return self.DH.addTodaysPrices(append=append,info=info)
            
    def createMarketOrder(self,pName,YYYYMM,volume,BS):
        dfO = pd.DataFrame(np.full([1,13],np.nan),columns=['Symbol','Side','Order Size','Display Size','Account','Order Type','Price Type','Price Offset','Stop Price','TIF','Dest','Sub Account','CustomText1'])
        
        RH = ReutersHub('BO',altName='dummy')
        dfO['Symbol'] = self.priceMetaDict[pName]['neovest']+RH.YYYYMMtoLabel(int(YYYYMM))
        dfO['Account'] = 'Cargill Inc 1'
        dfO['Sub Account'] = self.priceMetaDict[pName]['neoAccount']
        dfO['Side'] = 'BUY' if BS else 'SELL'
        dfO['Order Type'] = 'MKT'
        dfO['Price Type'] = 'Static'
        dfO['Order Size'] = int(volume)
        dfO['Dest'] = 'JP-DMA'
        dfO['TIF'] = 'GTC'
        if hasattr(self.strat,'name'):
            dfO['CustomText1'] = self.strat.name
        return dfO
    def orderAggregator(self,maxShowVol=5,
                        prod=False,relaxMin=5,
                        neovestPath='C:/Users/{0}/AppData/Roaming/Neovest/FileOrdersDrop/'.format(os.getlogin()),
                        backUpFolder='C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/orderBackup/'.format(os.getlogin())):
        """Aggregates orders of all parameter combinations for Production runs."""

        agg_orders = []    
        orderConvDict = {'LIMIT':'LMT',
                            'STOP':'SLMT',
                            'MARKET':'MKT'}
            
        RH = ReutersHub('BO',altName='dummy')
        
        for i,ID in enumerate(self.OH.currRVorders['id']):
            pName = self.OH.currRVorders['product'][i]    
            RIC = pName
            tickSize = 1
            npAfac = 2
            currC = self.OH.priceSet['RV'][pName]['CONTRACTS']    
            oType = self.OH.currRVorders['oType'][i]
            active = self.OH.currRVorders['active'][i,:]
            BS = self.OH.currRVorders['BS'][i,:][active]
            val = np.around(self.OH.currRVorders['val'][i,:][active]*tickSize,npAfac)
            vol = self.OH.currRVorders['vol'][i,:][active]
    
            if len(val)>0:
                uVal = np.unique(val)
    
                for uv in uVal:
                    loc = (val==uv).nonzero()[0]
                    TV = np.sum(vol[loc])
                    BuySell = 'BUY' if BS[loc[0]] else 'SELL'
    
                    agg_orders.append([RIC,currC,oType,BuySell,TV,uv,pName])    
    
        df = pd.DataFrame(agg_orders,columns=['Commodity','Contract','Order Type','Side','Order Size','Price Offset','internal'])

        if df.shape[0]>0:
            df = df.groupby(['Commodity','Contract','Order Type','Side','Price Offset','internal'])['Order Size'].sum().reset_index()
            df = df.sort_values(by=['Commodity','Contract','Price Offset'],kind='mergesort',ascending=[True,True,False])
        
            neo = df['Commodity'].copy()
            Cs = RH.YYYYMMtoLabel(df.Contract.values)
        
            df['Symbol']=[neo[i]+Cs[i] for i in range(len(neo))]
            df['Account'] = 'Cargill Inc 1'
            df['Order Type'] = df['Order Type'].map(orderConvDict)
            df['TIF'] ='GTC'
            df['Price Type']= 'Static'
            df['Dest'] = 'JP-DMA'
            df['Sub Account'] = 'RV'
        
            df['Stop Price'] = np.nan
        
            df.loc[df.loc[:,'Order Type']=='SLMT','Stop Price'] = df.loc[df.loc[:,'Order Type']=='SLMT','Price Offset'].copy()
    
            df['Display Size'] = np.nan
                        
            if hasattr(self.strat,'name'):
                df['CustomText1'] = self.strat.name
            else:
                df['CustomText1'] = 'Algo orders: '+str(dt.datetime.now())
        
            dfO_RV = df.loc[:,['Symbol','Side','Order Size','Display Size','Account','Order Type','Price Type','Price Offset','Stop Price','TIF','Dest','Sub Account','CustomText1','internal']]            
        else:        
            dfO_RV = pd.DataFrame([],columns=['Symbol','Side','Order Size','Display Size','Account','Order Type','Price Type','Price Offset','Stop Price','TIF','Dest','Sub Account','CustomText1','internal'])
            
        agg_orders = []
        currPs = []
        currPentry = []

        for i,ID in enumerate(self.OH.currOrders['id']):
            pName = self.OH.currOrders['product'][i]    
            RIC = self.priceMetaDict[pName]['RIC']
            tickSize = self.priceMetaDict[pName]['tickSize']
            npAfac = self.priceMetaDict[pName]['npAroundFac']
            currC = self.OH.priceSet['prices'][pName]['CONTRACTS']    
            oType = self.OH.currOrders['oType'][i]
            active = self.OH.currOrders['active'][i,:]
            BS = self.OH.currOrders['BS'][i,:][active]
            val = np.around(self.OH.currOrders['val'][i,:][active]*tickSize,npAfac)
                          
            sym = self.DH.neovest[RIC]                
            if sym not in currPs:
                currPs.append(sym)
                CC = RH.YYYYMMtoLabel(int(currC))
                curV = np.around(self.OH.pCurr[pName]*tickSize,npAfac)                    
                currPentry.append([sym+CC,'CURRENT',np.nan,'',curV,np.nan])    
    
            vol = self.OH.currOrders['vol'][i,:][active]
    
            if len(val)>0:
                uVal = np.unique(val)
    
                for uv in uVal:
                    loc = (val==uv).nonzero()[0]
                    TV = np.sum(vol[loc])
                    BuySell = 'BUY' if BS[loc[0]] else 'SELL'
    
                    agg_orders.append([RIC,currC,oType,BuySell,TV,uv,pName])    
    
        df = pd.DataFrame(agg_orders,columns=['Commodity','Contract','Order Type','Side','Order Size','Price Offset','internal'])



        if df.shape[0]>0:
            df = df.groupby(['Commodity','Contract','Order Type','Side','Price Offset','internal'])['Order Size'].sum().reset_index()
            df = df.sort_values(by=['Commodity','Contract','Price Offset'],kind='mergesort',ascending=[True,True,False])
        
            neo = df['Commodity'].map(self.DH.neovest)
            Cs = RH.YYYYMMtoLabel(df.Contract.values)
        
            df['Symbol']=[neo[i]+Cs[i] for i in range(len(neo))]
            df['Account'] = 'Cargill Inc 1'
            df['Order Type'] = df['Order Type'].map(orderConvDict)
            df['TIF'] ='GTC'
            df['Price Type']= 'Static'
            df['Dest'] = 'JP-DMA'
            df['Sub Account'] = df['Commodity'].map(self.DH.neoAccount)
        
            df['Stop Price'] = np.nan
        
            df.loc[df.loc[:,'Order Type']=='SLMT','Stop Price'] = df.loc[df.loc[:,'Order Type']=='SLMT','Price Offset'].copy()
            mask = (df.loc[:,'Order Type']=='SLMT')&(df.loc[:,'Side']=='BUY')
            df.loc[mask,'Stop Price'] = np.around(df.loc[mask,'Stop Price'].values-df.loc[mask,'Commodity'].map(self.DH.tickSizes).values,2)
            mask = (df.loc[:,'Order Type']=='SLMT')&(df.loc[:,'Side']=='SELL')
            df.loc[mask,'Stop Price'] = np.around(df.loc[mask,'Stop Price']+df.loc[mask,'Commodity'].map(self.DH.tickSizes).values,2)
        
            df['Display Size'] = np.nan
            while True:
                mask = df['Commodity'].isin(['BO','S','SM','G','COM','RS']) & df.loc[:,'Order Size']>200 # only for neovest orders
                if mask.sum()>0:
                    copyOrders = df.loc[df.loc[:,'Order Size']>200,:].copy() 
                    df.loc[df.loc[:,'Order Size']>200,'Order Size'] = 200
                    copyOrders.loc[:,'Order Size'] -=200 
                
                    df = pd.concat([df,copyOrders],axis=0).reset_index(drop=True)
                else:
                    break
            
            mask = (df.loc[:,'Order Type']=='LMT') & (df.loc[:,'Order Size']>maxShowVol)
            df.loc[mask,'Display Size'] = maxShowVol
        
            if hasattr(self.strat,'name'):
                df['CustomText1'] = self.strat.name
            else:
                df['CustomText1'] = 'Algo orders: '+str(dt.datetime.now())
        
            dfO = df.loc[:,['Symbol','Side','Order Size','Display Size','Account','Order Type','Price Type','Price Offset','Stop Price','TIF','Dest','Sub Account','CustomText1','internal']]
            
            df_currV = pd.DataFrame(currPentry,columns=['Symbol','Side','Order Size','Order Type','Price Offset','Stop Price'])
        else:        
            dfO = pd.DataFrame([],columns=['Symbol','Side','Order Size','Display Size','Account','Order Type','Price Type','Price Offset','Stop Price','TIF','Dest','Sub Account','CustomText1','internal'])
            df_currV = pd.DataFrame(currPentry,columns=['Symbol','Side','Order Size','Order Type','Price Offset','Stop Price'])
        return dfO, df_currV, dfO_RV
    def prepFromEoD(self,endDate=None):
        """Rename of dataHandler.fromStart method."""
        self.DH.fromStart(self,endDate=endDate)
    def prodRun(self,hist=True,pathOfData=None,info=True,addActual=False):
        """Production Runner
        
        Parameters
        ----------
        hist : bool
            When True, it is the first run from the start of the time line in this session, 
            when False, it is expected to continue from a previous run with new data.
        pathOfData : str
            Required for Mac, ask Alex Lefter.
        """
        if (hasattr(self,'DH')) & (hasattr(self,'strat')):   
            if info:    
                print('preparing database')
            if self.DH.prep(first=hist):
                pass
            else:
                if info:
                    print('Aborting prodRun')
                return self.DH.errorMessage
            
            self.priceMetaDict = copy.deepcopy(self.DH.priceMetaDict)
            for key in list(self.priceMetaDict.keys()):
                if self.priceMetaDict[key]['skip']:
                    del self.priceMetaDict[key]
            # initialize order handler
            if hist:
                self.OH.fromStart(self.priceMetaDict)
            else:
                self.OH.updateOH(self.priceMetaDict)
            
                
            t0 = dt.datetime.now()
            if info:
                print('started run: '+str(t0))
            while True:
                if self.DH.getNextDataPoint():
                    if self.DH.lastDataPoint['t']>self.monitorEnd:
                        break
                    
                    self.OH.newPriceInsert(self.DH.lastDataPoint)
                    self.OH.evalOrders()
                    pospnlDict = self.OH.getCurrentPosPnl()
                    
                    if self.DH.priceOrTicks=='ticks':
                        newOrders,override = self.strat.evaluate(psDict=self.DH.lastDataPoint,pospnlDict=pospnlDict)  
                    elif self.DH.priceOrTicks=='price':
                        newOrders,override = self.strat.evaluate(psDict=self.DH.fromTicksToActual(),pospnlDict=pospnlDict)  
                        
                        if 'val' in list(newOrders.keys()):
                            val = newOrders['val'].copy()
                            for i,pName in enumerate(newOrders['product']):
                                val[i,:] = np.around(val[i,:]/self.priceMetaDict[pName]['tickSize'],0)

                            newOrders.update({'val':val.astype(int)})
                     
                    newOrders.update({'timeStamp':self.DH.lastDataPoint['t']})
                    self.OH.checkValidity(newOrders,override)
                    self.OH.evalOrders(instant=True)
                    pospnlDict = self.OH.getCurrentPosPnl()
                else:
                    break
            if info:
                print('Processing PnL')
            self.processPnL()
            if info:
                print('Run finished, preparing plot items')
            self.prepPlotItems(self.monitorStart,self.monitorEnd,pathOfData=pathOfData,aggOrders=True,addActual=addActual)
            if info:
                print('All done')
            return 'run successful'
    def vizCheck(self,monitorStart=None,monitorEnd=None,pathOfData=None,first=True,aggOrders=False,orderCheck_TF=False):
        """Run backtest with order and fills plotting capability
        
        Parameters
        ----------
        monitorStart : None or str
            orders and fills are collected from this timestamp
        monitorEnd : None or str
            orders and fills are collected up to this timestamp
        first : bool
            Always set to True, legacy.
        aggOrders : bool
            Set to True if orders of different parameter combinations need to be combined.
        orderCheck_TF : bool
            Set to True to enable order checks before they go into the system. Will provide readable error messages and returns the 
            rejected order, plus the data that went into the evaluate function. 
        """
        cont = True
        if first:
            if monitorStart != None:
                if type(monitorStart)==type(''):
                    if self.checkDateStr(monitorStart):
                        monitorStart = pd.to_datetime(monitorStart)
                    else:
                        cont = False
            else:
                monitorStart = dt.datetime(1950,1,1)
                
            if monitorEnd != None:
                if type(monitorEnd)==type(''):
                    if self.checkDateStr(monitorEnd):
                        monitorEnd = pd.to_datetime(monitorEnd)
                    else:
                        cont = False
            else:
                monitorEnd = dt.datetime(2050,1,1) 
                
            if cont==False:
                print('vizCheck Failed and aborted.')
                return False
            
            self.monitorStart = monitorStart
            self.monitorEnd = monitorEnd
        
        if (hasattr(self,'DH')) & (hasattr(self,'strat')):       
            print('preparing database')
            
            if self.DH.prep(first=first):
                print('prep done')
            else:
                print('Aborting prodRun')
                return
            
            
            self.priceMetaDict = copy.deepcopy(self.DH.priceMetaDict)
            for key in list(self.priceMetaDict.keys()):
                if self.priceMetaDict[key]['skip']:
                    del self.priceMetaDict[key]
            # initialize order handler
            if first:
                numParams = len(self.stratParams[list(self.stratParams.keys())[0]])
                self.OH = orderHandler(self.priceMetaDict,numParams,
                                       viz=True,keepOrders=True,aggOrders=aggOrders,
                                       monitorStart=monitorStart,monitorEnd=monitorEnd)
            else:
                self.OH.updateOH(self.priceMetaDict)
            
            counter=-1
            if first:
                Ntotal = len(self.DH.master_idx)
                
                parts = 0.1
                partsCount = 1
            
                
            t0 = dt.datetime.now()
            print('started run: '+str(t0))
            while True:
                counter = counter+1
                if first:
                    if counter/Ntotal*100>parts*partsCount:
                        
                        t1 = dt.datetime.now()
                        print(str(parts*partsCount) +'% done, estimated end time: '+str(t0+(t1-t0)/(parts*partsCount/100)))
                        
                        partsCount = partsCount+1
                        parts = 10
                
                if self.DH.getNextDataPoint():
                    if self.DH.lastDataPoint['t']>self.monitorEnd:
                        break
                    
                    self.OH.newPriceInsert(self.DH.lastDataPoint)
                    self.OH.evalOrders()
                    pospnlDict = self.OH.getCurrentPosPnl()
                    
                    if self.DH.priceOrTicks=='ticks':
                        newOrders,override = self.strat.evaluate(psDict=self.DH.lastDataPoint,pospnlDict=pospnlDict)  
                        if orderCheck_TF:
                            if self.orderCheck(newOrders,self.OH.numParams,self.DH.priceOrTicks):
                                pass
                            else:
                                return newOrders,self.DH.lastDataPoint,pospnlDict

                    elif self.DH.priceOrTicks=='price':
                        newOrders,override = self.strat.evaluate(psDict=self.DH.fromTicksToActual(),pospnlDict=pospnlDict)  
                        if orderCheck_TF:
                            if self.orderCheck(newOrders,self.OH.numParams,self.DH.priceOrTicks):
                                pass
                            else:
                                return newOrders,self.DH.lastDataPoint,pospnlDict

                        if 'val' in list(newOrders.keys()):
                            val = newOrders['val'].copy()
                            for i,pName in enumerate(newOrders['product']):
                                val[i,:] = np.around(val[i,:]/self.priceMetaDict[pName]['tickSize'],0)

                            newOrders.update({'val':val.astype(int)})
                     
                    newOrders.update({'timeStamp':self.DH.lastDataPoint['t']})
                    self.OH.checkValidity(newOrders,override)
                    self.OH.evalOrders(instant=True)
                    pospnlDict = self.OH.getCurrentPosPnl()
                else:
                    break
            print('Processing PnL')
            self.processPnL()
            print('Run finished, preparing plot items')
            self.prepPlotItems(self.monitorStart,self.monitorEnd,pathOfData=pathOfData,aggOrders=aggOrders)
            
            print('All done')
    
    def processParamsToNames(self):
        """Combine parameter name/values to a single string per combination."""
        cols = []
        df =pd.DataFrame(self.stratParams)
        for i,j in enumerate(df.columns):
            cols = cols+['qwert'+str(i),j]
            if i ==0:
                df['qwert'+str(i)] = j + ': '
            else:
                df['qwert'+str(i)] = ' | ' +j + ': '
                
        df = df.astype(str)        
        return df.loc[:,cols].sum(axis=1).values.tolist()
    
    def prepareSetForBTviewer(self):
        """Prepare data structure for backTestViewer()"""
        if hasattr(self,'pnlT'):
            if self.pnlT.shape[1]==1:                    
                RM = self.riskmetricsRealBasic(self.pnlT.values.reshape([-1,1]))
            else:
                X = self.groupPnL(mergePrices=True)
                RM = self.riskmetricsRealBasic(X.values)
        
            RMcols = list(RM.columns)
            colNames = list(self.stratParams.keys())
            for paramName in colNames:
                if paramName in RMcols:
                    print('parameter name not allowed. Aborting. Invalid options are: '+str(RMcols))
                    return None
                else:
                    RM[paramName] = self.stratParams[paramName]
                    
            self.BTviewFile = RM.loc[:,colNames+RMcols]
        else:
            print('First run processPnL. Aborted.')
    
    def backTestViewer(self):
        """Plots PnL curves and risk metrics for any parameter combination or any bucket thereof."""
        if hasattr(self,'BTviewFile')==False:
            print('First running prepareSetForBTviewer. Moment.')
            self.prepareSetForBTviewer()
            print('Done running prepareSetForBTviewer.')
        
        def showMetrics(doc):
            def paramOrOutputsel(attr,old,new):
                if doc.callBackTriggered:            
                    choice = paramOrOutput.labels[paramOrOutput.active]
                    doc.allVars.loc[doc.allVars.index[doc.currIndex],'function'] = choice            
                    new_src = ColumnDataSource(doc.allVars)            
                    doc.var_src.data.update(new_src.data)

            def setParamOutputChoice(attr,old,new):
                paramOrOutput.disabled = False

                doc.currIndex = new[0]
                doc.callBackTriggered = False
                currFunctionChoice = doc.allVars.loc[doc.allVars.index[doc.currIndex],'function']

                if currFunctionChoice=="Parameter":
                    paramOrOutput.active = 0            
                else:
                    paramOrOutput.active = 1

                T1.value = str(doc.allVars.loc[doc.allVars.index[doc.currIndex],'start'])
                T2.value = str(doc.allVars.loc[doc.allVars.index[doc.currIndex],'end'])
                T3.value = str(doc.allVars.loc[doc.allVars.index[doc.currIndex],'increment'])

                doc.callBackTriggered = True


            doc.callBackTriggered = True



            def saveDefs():


                if optPlotLayout.value=='':
                    Nx = int(plotNumX.value)
                    Ny = int(plotNumY.value)

                    doc.plotLayout = []
                    counter = 1
                    for i in range(Nx):
                        newRow = []
                        for j in range(Ny):
                            newRow = newRow + [counter]
                            counter = counter+1
                        doc.plotLayout = doc.plotLayout + [newRow]
                    doc.plotLayout = doc.plotLayout + [[-1]]
                    N = Nx*Ny
                else:
                    doc.plotLayout = eval(optPlotLayout.value)
                    N = 1
                    for i in doc.plotLayout:
                        for j in i:
                            if j>N:
                                N = j


                plotVars = list(doc.allVars.loc[doc.allVars.loc[:,'function']=="Output Variable",'Name'].values)            
                controlVars = list(doc.allVars.loc[doc.allVars.loc[:,'function']=="Parameter",'Name'].values)


                doc.plotDefs = column()

                if PlotType.active==0:
                    for i in range(N):
                        if i==0:
                            P1 = Div(text='<h1><b>Plot '+str(i+1)+'</h1>',width=75)
                            P2 = Select(value=controlVars[0], options=controlVars,width=250,title='X axis')
                            P3 = Select(value=controlVars[1], options=controlVars,width=250,title='Y axis')
                            P4 = Select(value=plotVars[0], options=plotVars,width=250,title='Value')
                            doc.plotDefs.children = doc.plotDefs.children+ [row(P1,P2,P3,P4)] 
                        else:
                            P1 = Div(text='<h1><b>Plot '+str(i+1)+'</h1>',width=75)
                            P2 = Select(value='n.a.', options=['n.a.'],width=250,title='X axis',disabled=True)
                            P3 = Select(value='n.a.', options=['n.a.'],width=250,title='Y axis',disabled=True)
                            P4 = Select(value=plotVars[i if i<len(plotVars) else 0], options=plotVars,width=250,title='Value')
                            doc.plotDefs.children = doc.plotDefs.children+ [row(P1,P2,P3,P4)] 
                else:
                    for i in range(N):
                        P1 = Div(text='<h1><b>Plot '+str(i+1)+'</h1>',width=75)
                        P2 = Select(value=plotVars[i if i<len(plotVars) else 0], options=plotVars,width=250,title='X axis')
                        P3 = Select(value=plotVars[i if i<len(plotVars) else 0], options=plotVars,width=250,title='Y axis')
                        doc.plotDefs.children = doc.plotDefs.children+ [row(P1,P2,P3)] 

                createPlots_B = Button(label="Create plots", button_type="success",width=150)
                createPlots_B.on_click(createPlots)

                plotSelect = column(doc.plotDefs,createPlots_B)
                tab2 = Panel(child=plotSelect,title='Plot control')

                if len(tabs.tabs)==1:
                    tabs.tabs = tabs.tabs + [tab2]
                else:
                    tabs.tabs[1] =  tab2


            def make_scatter_plot(src,X,Y):
                loc = (doc.allVars.loc[:,'Name'].values==X).nonzero()[0][0]

                idx = doc.allVars.index[loc]
                xR = (doc.allVars.loc[idx,'start'],doc.allVars.loc[idx,'end'])

                loc = (doc.allVars.loc[:,'Name'].values==Y).nonzero()[0][0]
                idx = doc.allVars.index[loc]
                yR = (doc.allVars.loc[idx,'start'],doc.allVars.loc[idx,'end'])

                p = figure(title=X+" vs "+Y,               
                           plot_width=500, plot_height=500,x_range=xR, y_range=yR,
                          toolbar_location='below')

                p.xaxis[0].axis_label = X
                p.yaxis[0].axis_label = Y
                p.circle(source=src,x=X, y=Y,color='red',size=5) 
                p = style(p)
                return p
            def style(p):
                # Title 
                p.title.align = 'center'
                p.title.text_font_size = '20pt'
                p.title.text_font = 'serif'

                # Axis titles
                p.xaxis.axis_label_text_font_size = '14pt'
                p.xaxis.axis_label_text_font_style = 'bold'
                p.yaxis.axis_label_text_font_size = '14pt'
                p.yaxis.axis_label_text_font_style = 'bold'

                # Tick labels
                p.xaxis.major_label_text_font_size = '12pt'
                p.yaxis.major_label_text_font_size = '12pt'

                return p
            def make_surface_plot(src,X,Y,val,minVal,maxVal):
                print([X,Y,val,minVal,maxVal])
                UX = np.unique(doc.Alldata[X])
                UY = np.unique(doc.Alldata[Y])

        #        if hasattr(doc,'valDictX')==False:
                doc.valDictX = {j:i for i,j in enumerate(UX)}
                doc.valDictY = {j:i for i,j in enumerate(UY)}
                doc.valDict = [X,Y]

                src.data.update({'Xn':np.array([doc.valDictX[i] for i in src.data[X]])+0.5})
                src.data.update({'Yn':np.array([doc.valDictY[i] for i in src.data[Y]])+0.5})

                if type(UX[0])!=type(''):
                    factors1 = [str(i) for i in UX]
                else:
                    factors1 = UX
                if type(UY[0])!=type(''):
                    factors2 = [str(i) for i in UY]    
                else:
                    factors2 = UY

                mapper = LinearColorMapper(palette=doc.colors, low=minVal, high=maxVal)

                TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

                p = figure(title=val,
                           x_range=factors1, y_range=factors2,
                           plot_width=500, plot_height=500,
                           tools=TOOLS, toolbar_location='below',
                           tooltips=[(X, "@"+X),(Y, "@"+Y),(val, "@"+val)])

                p.grid.grid_line_color = None
                p.axis.axis_line_color = None
                p.axis.major_tick_line_color = None
                p.axis.major_label_text_font_size = "10pt"
                p.axis.major_label_standoff = 0
                p.xaxis[0].axis_label = X
                p.yaxis[0].axis_label = Y

        #        if type(UX[0])==type(''):
                p.xaxis.major_label_orientation = np.pi / 2

                p.rect(x='Xn', y='Yn', width=1, height=1,
                   source=src,
                   fill_color={'field': val, 'transform': mapper},
                   line_color=None)
                color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
        #                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                                 ticker=BasicTicker(desired_num_ticks=10),
                                 formatter=PrintfTickFormatter(),
                                 label_standoff=6, border_line_color=None, location=(0, 0))
                p.add_layout(color_bar, 'right')

                return p

            def make_data_set():        
                df = doc.Alldata.copy()

                if hasattr(doc,'meanParameters'):
                    meanParams = [doc.meanParameters[i]  for i in doc.averageLevel.active]            
                else:
                    meanParams = []

                for obj in doc.controls:
                    if obj.name not in meanParams:
                        val = obj.value
                        if obj.tags!=[]:
                            if type(val)==type(()):
                                val = (obj.tags[val[0]] , obj.tags[val[1]])
                            else:
                                val = obj.tags[val]
                            colName = obj.name
                        else:
                            colName = obj.title

                        if type(val)==type(()):
                            df = df.loc[df[colName].values>=val[0]-1E-5,:]
                            df = df.loc[df[colName].values<=val[1]+1E-5,:]
                        elif type(val)==type(""):
                            df = df.loc[df[colName]==val,:]
                        else:                
                            df = df.loc[np.abs(df[colName].values-val)<1E-5,:]

                if len(meanParams)>0:
                    plotVars = list(doc.allVars.loc[doc.allVars.loc[:,'function']=="Output Variable",'Name'].values) 
                    Xaxis = doc.plotDefs.children[0].children[1].value
                    Yaxis = doc.plotDefs.children[0].children[2].value
                    for i in plotVars:

                        dft = df.pivot_table(index=Xaxis,columns=Yaxis,values=i,aggfunc=np.mean)

                        Y = dft.values.flatten().reshape([-1,1])

                        if i ==plotVars[0]:
                            X1 = np.tile(dft.index.values.reshape([-1,1]),[1,len(dft.columns)]).flatten().reshape([-1,1])
                            X2 = np.tile(dft.columns.values.reshape([1,-1]),[len(dft.index),1]).flatten().reshape([-1,1])
                            X = np.concatenate((X1,X2,Y),axis=1)
                            df_new = pd.DataFrame(X,columns=[Xaxis,Yaxis,i])
                        else:
                            try:
                                df_new[i] = Y
                            except:
                                print('went wrong')
                                print([Xaxis,Yaxis,i])
                                print(Y.shape)
                                print(df_new.shape)
                                df_new[i] = Y
                    df = df_new.copy()


                if hasattr(doc,'valDictX'):            
                    df['Xn'] = np.array([doc.valDictX[i] for i in df[doc.valDict[0]]])+0.5
                    df['Yn'] = np.array([doc.valDictY[i] for i in df[doc.valDict[1]]])+0.5

                return ColumnDataSource(df)
            def update(attr,old,new):

                new_src = make_data_set()
                doc.src.data.update(new_src.data)
                grabCurrentVal()
            def grabCurrentVal():
                doc.currStatus = []

                for obj in doc.controls:
                    doc.currStatus = doc.currStatus +[obj.value]

            def updateSpecial(attr,old,new):

                for i,obj in enumerate(doc.controls):
                    if (doc.currStatus[i]==old) & (obj.value==new):
                        if type(new)==type(()):
                            obj.title = obj.name+':  '+str(obj.tags[obj.value[0]]) + '-'+ str(obj.tags[obj.value[1]])
                        else:
                            obj.title = obj.name+':  '+str(obj.tags[obj.value])

                update(attr,old,new)
            def createPlots(): 
                controlVars = list(doc.allVars.loc[doc.allVars.loc[:,'function']=="Parameter",'Name'].values)

                doc.controls = []
                doc.meanParameters = []
                if len(doc.plotDefs.children[0].children)==4: # surface plot

                    for i in controlVars:
                        if i not in [doc.plotDefs.children[0].children[1].value,doc.plotDefs.children[0].children[2].value]:

                            newSlider = Slider(start = 0, end = len(doc.UVals[i])-1, 
                                     step = 1, value = 0,title = i+':  '+str(doc.UVals[i][0]),tags=list(doc.UVals[i]),width=200,callback_policy='mouseup',name=i,show_value=False)    
                            newSlider.on_change("value",updateSpecial)

                            doc.controls = doc.controls + [newSlider]
                            doc.meanParameters.append(i)
                else:
                    for i in controlVars:

                        newSlider = RangeSlider(start = 0, end = len(doc.UVals[i])-1, 
                                     step = 1, value = (0,len(doc.UVals[i])-1),title = i+':  '+str(doc.UVals[i][0])+'-'+str(doc.UVals[i][-1]),width=200,tags=list(doc.UVals[i]),callback_policy='mouseup',name=i,show_value=False)    
                        newSlider.on_change("value",updateSpecial)

                        doc.controls = doc.controls + [newSlider]

                doc.averageLevel = CheckboxGroup(labels=doc.meanParameters, active=[])
                doc.averageLevel.on_change("active",updateSpecial)

                grabCurrentVal()
                plotLay = doc.plotLayout.copy()
                doc.src = make_data_set()
                doc.rangeControl = {}
                for i, plotDef in enumerate(doc.plotDefs.children):
                    if len(plotDef.children)==4: # surface plot
                        if i ==0:
                            X = plotDef.children[1].value
                            Y = plotDef.children[2].value
                        val = plotDef.children[3].value

                        idx = doc.allVars.index[doc.allVars.loc[:,'Name'].values==val][0]   

                        p = make_surface_plot(doc.src,X,Y,val,doc.allVars.loc[idx,'start'],doc.allVars.loc[idx,'end'])
                    else: # scatter plot
                        X = plotDef.children[1].value
                        Y = plotDef.children[2].value
                        p = make_scatter_plot(doc.src,X,Y)
                    for k,rows in enumerate(plotLay):
                        for j,item in enumerate(rows):
                            if item==i+1:
                                plotLay[k][j]=p
                                doc.rangeControl.update({val:[k,j]})
                                break
                controlCol = column()


                doc.plotRangeChangeSelect = Select(value=list(doc.rangeControl.keys())[0], options=list(doc.rangeControl.keys()),width=150,title='Change Color range')


                doc.plotRangeChangeSelect.on_change("value",valChange)
                minLvl,maxLvl = grabCurrentRange(doc.plotRangeChangeSelect.value)

                doc.Ts = TextInput(value=str(minLvl), title="range min",width=80)
                doc.Ts.on_change("value",Tchange)
                doc.Te = TextInput(value=str(maxLvl), title="range max",width=80)
                doc.Te.on_change("value",Tchange)

                controlCol.children.append(row(column(doc.controls),doc.averageLevel))
                controlCol.children.append(row(doc.plotRangeChangeSelect,doc.Ts,doc.Te))
                for k,rows in enumerate(plotLay):
                        for j,item in enumerate(rows):
                            if item==-1:
                                plotLay[k][j]=controlCol
                                break          
                plotLayout = layout(plotLay)

                tab3 = Panel(child=plotLayout,title='Plots')

                if len(tabs.tabs)==2:
                    tabs.tabs = tabs.tabs + [tab3]
                else:
                    tabs.tabs.pop(2)
                    tabs.tabs = tabs.tabs + [tab3]
            def grabCurrentRange(val):
                idx = doc.allVars.index[doc.allVars.loc[:,'Name'].values==val][0]                   
                return doc.allVars.loc[idx,'start'],doc.allVars.loc[idx,'end']
            def setCurrentRange(val,start,end):        
                    idx = doc.allVars.index[doc.allVars.loc[:,'Name'].values==val][0]          
                    doc.allVars.loc[idx,'start'] = start
                    doc.allVars.loc[idx,'end'] = end
                    new_src = ColumnDataSource(doc.allVars)            
                    doc.var_src.data.update(new_src.data)
            def Tchange(attr,old,new):    
                if doc.changeProd==False:
                    LOC = doc.rangeControl[doc.plotRangeChangeSelect.value]

                    setCurrentRange(doc.plotRangeChangeSelect.value,float(doc.Ts.value),float(doc.Te.value))

                    tabs.tabs[2].child.children[LOC[0]].children[LOC[1]].right[0].color_mapper.low=float(doc.Ts.value)
                    tabs.tabs[2].child.children[LOC[0]].children[LOC[1]].right[0].color_mapper.high=float(doc.Te.value)
            def valChange(attr,old,new):
                minLvl,maxLvl = grabCurrentRange(new)
                doc.changeProd = True
                doc.Ts.value=str(minLvl)
                doc.Te.value=str(maxLvl)
                doc.changeProd = False
            def T1change(attr,old,new):
                if doc.callBackTriggered:
                    doc.allVars.loc[doc.allVars.index[doc.currIndex],'start'] = float(new)
                    new_src = ColumnDataSource(doc.allVars)            
                    doc.var_src.data.update(new_src.data)

            def T2change(attr,old,new):
                if doc.callBackTriggered:
                    doc.allVars.loc[doc.allVars.index[doc.currIndex],'end'] = float(new)
                    new_src = ColumnDataSource(doc.allVars)            
                    doc.var_src.data.update(new_src.data)

            def T3change(attr,old,new):
                if doc.callBackTriggered:
                    doc.allVars.loc[doc.allVars.index[doc.currIndex],'increment'] = new
                    new_src = ColumnDataSource(doc.allVars)            
                    doc.var_src.data.update(new_src.data)

            def createTable():
                doc.Alldata = self.BTviewFile.copy()
                setupDone.disabled=False
                opts = list(doc.Alldata.columns)
                doc.allVars = pd.DataFrame(columns=['Name','function','start','end','increment'])
                doc.allVars['Name'] =opts
                doc.allVars['function'] = 'Parameter'

                doc.allVars['start'] = doc.Alldata.min(axis=0).values
                doc.allVars['end'] = doc.Alldata.max(axis=0).values
                doc.allVars['increment'] = ''

                doc.UVals = {}

                for j,i in enumerate(doc.Alldata.columns):
                    if type(doc.allVars.loc[doc.allVars.index[j],'start'])!=type(''): # bokeh table can not deal with inf
                        if doc.allVars.loc[doc.allVars.index[j],'start']<-1E9:
                            doc.allVars.loc[doc.allVars.index[j],'start']=-1E9
                        if doc.allVars.loc[doc.allVars.index[j],'end']>1E9:                    
                            doc.allVars.loc[doc.allVars.index[j],'end']=1E9
                    else: # bokeh plots get distorted when strings larger than 30 letters are present in the axis labels
                        doc.Alldata[i] = [k[:int(np.min([doc.Alldata[i].shape[0],30]))] for k in doc.Alldata[i]]

                    Uvals = doc.Alldata[i].unique()
                    Uvals.sort()

                    doc.UVals.update({i:Uvals})
                    if (len(Uvals)==1) | (i in ['PnL [k$]','SRpos','SR','kalmar','Days Pos [%]','PnlRatio','HitRatio','DDsurf','maxDD','DDratio']):
                        doc.allVars.loc[doc.allVars.index[j],'function'] = "Output Variable"
                    else:
                        if type(Uvals[0])!=type(''):
                            M = np.unique(np.diff(Uvals))
                            if len(M)==1:
                                doc.allVars.loc[doc.allVars.index[j],'increment'] = str(M[0])
                            else:
                                doc.allVars.loc[doc.allVars.index[j],'increment'] = str('varying')

                doc.var_src.data.update(ColumnDataSource(doc.allVars).data)



            doc.colors = ['#00004c', '#00004f', '#000052', '#000054', '#000057', '#00005a', '#00005d', '#000060', '#000062', 
                          '#000065', '#000068', '#00006b', '#00006e', '#000070', '#000073', '#000076', '#000079', '#00007c', 
                          '#00007e', '#000081', '#000084', '#000087', '#00008a', '#00008c', '#00008f', '#000092', '#000095', 
                          '#000098', '#00009a', '#00009d', '#0000a0', '#0000a3', '#0000a6', '#0000a8', '#0000ab', '#0000ae', 
                          '#0000b1', '#0000b4', '#0000b6', '#0000b9', '#0000bc', '#0000bf', '#0000c2', '#0000c4', '#0000c7', 
                          '#0000ca', '#0000cd', '#0000d0', '#0000d2', '#0000d5', '#0000d8', '#0000db', '#0000de', '#0000e0', 
                          '#0000e3', '#0000e6', '#0000e9', '#0000ec', '#0000ee', '#0000f1', '#0000f4', '#0000f7', '#0000fa', 
                          '#0000fc', '#0101ff', '#0505ff', '#0808ff', '#0d0dff', '#1111ff', '#1515ff', '#1919ff', '#1d1dff', 
                          '#2121ff', '#2525ff', '#2828ff', '#2d2dff', '#3131ff', '#3535ff', '#3939ff', '#3d3dff', '#4141ff',
                          '#4545ff', '#4848ff', '#4d4dff', '#5151ff', '#5555ff', '#5959ff', '#5d5dff', '#6161ff', '#6565ff',
                          '#6868ff', '#6d6dff', '#7171ff', '#7575ff', '#7979ff', '#7d7dff', '#8181ff', '#8585ff', '#8888ff', 
                          '#8d8dff', '#9191ff', '#9595ff', '#9999ff', '#9d9dff', '#a1a1ff', '#a5a5ff', '#a8a8ff', '#adadff', 
                          '#b1b1ff', '#b5b5ff', '#b9b9ff', '#bdbdff', '#c1c1ff', '#c5c5ff', '#c8c8ff', '#cdcdff', '#d1d1ff', 
                          '#d5d5ff', '#d9d9ff', '#ddddff', '#e1e1ff', '#e5e5ff', '#e8e8ff', '#ededff', '#f1f1ff', '#f5f5ff', 
                          '#f9f9ff', '#fdfdff', '#fffdfd', '#fff9f9', '#fff5f5', '#fff1f1', '#ffeded', '#ffe9e9', '#ffe5e5', 
                          '#ffe1e1', '#ffdddd', '#ffd9d9', '#ffd5d5', '#ffd1d1', '#ffcdcd', '#ffc9c9', '#ffc5c5', '#ffc1c1', 
                          '#ffbdbd', '#ffb9b9', '#ffb4b4', '#ffb1b1', '#ffadad', '#ffa9a9', '#ffa4a4', '#ffa1a1', '#ff9d9d', 
                          '#ff9999', '#ff9494', '#ff9191', '#ff8d8d', '#ff8989', '#ff8484', '#ff8181', '#ff7d7d', '#ff7979', 
                          '#ff7575', '#ff7171', '#ff6d6d', '#ff6969', '#ff6565', '#ff6161', '#ff5d5d', '#ff5959', '#ff5555', 
                          '#ff5151', '#ff4d4d', '#ff4949', '#ff4545', '#ff4141', '#ff3d3d', '#ff3838', '#ff3535', '#ff3030',
                          '#ff2d2d', '#ff2828', '#ff2525', '#ff2020', '#ff1d1d', '#ff1818', '#ff1515', '#ff1010', '#ff0d0d', 
                          '#ff0808', '#ff0505', '#ff0000', '#fd0000', '#fb0000', '#f90000', '#f70000', '#f50000', '#f30000', 
                          '#f10000', '#ef0000', '#ed0000', '#eb0000', '#e90000', '#e70000', '#e50000', '#e30000', '#e10000', 
                          '#df0000', '#dd0000', '#db0000', '#d90000', '#d70000', '#d50000', '#d30000', '#d10000', '#cf0000', 
                          '#cd0000', '#cb0000', '#c90000', '#c70000', '#c50000', '#c30000', '#c10000', '#bf0000', '#bd0000', 
                          '#bb0000', '#b90000', '#b70000', '#b50000', '#b30000', '#b10000', '#af0000', '#ad0000', '#ab0000', 
                          '#a90000', '#a70000', '#a50000', '#a30000', '#a10000', '#9f0000', '#9d0000', '#9b0000', '#990000', 
                          '#970000', '#950000', '#930000', '#910000', '#8f0000', '#8d0000', '#8b0000', '#890000', '#870000', 
                          '#850000', '#830000', '#810000', '#7f0000'] 

            doc.changeProd = False
            paramOrOutput = RadioButtonGroup(labels=["Parameter", "Output Variable"],active=0,button_type='success',width=200,disabled=True)
            paramOrOutput.on_change("active",paramOrOutputsel)

            T1 = TextInput(value="", title="min",width=200)
            T1.on_change("value",T1change)
            T2 = TextInput(value="", title="max",width=200)
            T2.on_change("value",T2change)
            T3 = TextInput(value="", title="step",width=200)
            T3.on_change("value",T3change)

            setupDone = Button(label="Save and move to tab", button_type="success",width=150,disabled=True)
            setupDone.on_click(saveDefs)

            doc.allVars = pd.DataFrame(columns=['Name','function','start','end','increment'])


            doc.currIndex = None

            plotNumX = Select(value='1', options=[str(i+1) for i in range(4)],width=80,title='# plots X') 
            plotNumY = Select(value='1', options=[str(i+1) for i in range(4)],width=80,title='# plots Y') 
            optPlotLayout = TextInput(value="", title="optional plot layout",width=200)
            PlotType = RadioButtonGroup(labels=["surface", "scatter"],active=0,button_type='success',width=200)
            doc.var_src = ColumnDataSource(doc.allVars)
            doc.var_src.selected.on_change('indices', setParamOutputChoice)
            columns = [TableColumn(field=i, title=i) for j,i in enumerate(doc.allVars.columns)]
            varTable = DataTable(source=doc.var_src, columns=columns, width=1200, editable=False,index_position=None)

            varSetup = column(row(paramOrOutput,plotNumX,plotNumY,optPlotLayout,PlotType),row(T1,T2,T3),setupDone,varTable)

            tab1 = Panel(child=varSetup,title='Setup view')

            tabs = Tabs(tabs=[tab1])  
            createTable()
            doc.add_root(tabs)
        show(Application(FunctionHandler(showMetrics)))
    def processPnL(self):
        """Processes pnl to map it on daily data and include commissions and FX results."""
        paramList = self.processParamsToNames()
        
        allPnLs = []
        self.commissionPnL = {}
        for pName in list(self.OH.pospnlKey.keys()):
            pnl = self.OH.pnl[self.OH.pospnlKey[pName]]
            idx = self.OH.dates[pName]
            
            pos = self.OH.pos[self.OH.pospnlKey[pName]].copy()
            
            pos = np.concatenate((np.zeros([1,pos.shape[1]]),pos),axis=0)
            
            pos = -np.abs(np.diff(pos,axis=0)).astype(int) #posmutations per time step
            
            df = pd.DataFrame(pnl,index=idx,columns=paramList)
            dfPos = pd.DataFrame(pos,index=idx,columns=paramList)
            IDX = df.index.date
            if df.shape[1]<3000:
                df = df.groupby(IDX).sum()
                df.index = pd.DatetimeIndex(df.index)
                
                dfPos = dfPos.groupby(IDX).sum()
                dfPos.index = pd.DatetimeIndex(dfPos.index)
            else:
                
                T = df.loc[:,df.columns[0]].groupby(IDX).sum()
                T.index = pd.DatetimeIndex(T.index)
                dfT = pd.DataFrame(np.nan,index=T.index,columns=df.columns)
                
                
                dfTpos = pd.DataFrame(np.nan,index=T.index,columns=df.columns)
                
                
                for i in range(int(np.ceil(df.shape[1]/1000))):
                    if i == int(np.ceil(df.shape[1]/1000)-1):
                        dfT.loc[:,df.columns[i*1000:]] = df.loc[:,df.columns[i*1000:]].groupby(IDX).sum().values
                        dfTpos.loc[:,dfPos.columns[i*1000:]] = dfPos.loc[:,dfPos.columns[i*1000:]].groupby(IDX).sum().values
                    else:
                        dfT.loc[:,df.columns[i*1000:(i+1)*1000]] = df.loc[:,df.columns[i*1000:(i+1)*1000]].groupby(IDX).sum().values
                        dfTpos.loc[:,dfPos.columns[i*1000:(i+1)*1000]] = dfPos.loc[:,dfPos.columns[i*1000:(i+1)*1000]].groupby(IDX).sum().values
                df = dfT  
                dfPos = dfTpos  
                
            df.index.name = 'Date'
            commissionPnL = (dfPos*self.priceMetaDict[pName]['commissions']/1000).values
            df = df*self.priceMetaDict[pName]['tickValues']/1000+commissionPnL
            
            tdC = self.DH.calcMinOffset('23:59') # dates in PnLT are on whole days, add this time to the day and it will be EoD always
            IDX = pd.DatetimeIndex([i+pd.offsets.Minute(n=tdC) for i in df.index])
            
            self.commissionPnL.update({pName:pd.DataFrame(commissionPnL,index=IDX,columns=paramList)})
            

            if self.priceMetaDict[pName]['FX']!=None: 
                FX = self.DH.FXp[self.priceMetaDict[pName]['FX']]
                dfTemp = pd.concat([df,FX],axis=1).fillna(method='ffill').fillna(method='bfill')
                df = dfTemp.loc[df.index,:]
                if self.priceMetaDict[pName]['FX'] in ['EURUSD']:
                    X = df.loc[:,paramList].values*np.tile(df.loc[:,'c1'].values.reshape([-1,1]),(1,len(paramList)))
                elif self.priceMetaDict[pName]['FX'] in ['MYRUSD','CADUSD']:
                    X = df.loc[:,paramList].values/np.tile(df.loc[:,'c1'].values.reshape([-1,1]),(1,len(paramList)))
                    
                df = df.loc[:,paramList]
                df.loc[:,:] = X

            arrays = [[pName for i in range(len(paramList))],paramList]
            tuples = list(zip(*arrays))
            multiCol = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
            df.columns = multiCol
                    
            allPnLs.append(df)
        
        self.pnlT = pd.concat(allPnLs,axis=1).fillna(0)
        self.FXtradePnL = {}
        self.lastFXpos = {}
        self.lastFXrate = {}
        pnlTdropped = []
        
        for key in list(self.OH.RVdict.keys()): 
            # First add FX result
            
            for i,FXcheck in enumerate(self.OH.RVdict[key]['FXconv']):
                if self.OH.RVdict[key]['FXconv'][i]:
                    pName = self.OH.RVdict[key]['pnl'][i]
                    
                    loc = (np.diff(pd.Index(self.OH.dates[pName]).strftime(date_format='%Y%m%d').values.astype(int))!=0).nonzero()[0]
                    loc = np.append(loc,len(self.OH.dates[pName])-1)
                    EoDpos = self.OH.pos[self.OH.pospnlKey[pName]][loc,:]
                    P = self.DH.master_data['prices'][pName]['CLOSE']['val'][loc]
                    P = np.tile(P.reshape([-1,1]),(1,EoDpos.shape[1]))
                    
                    FX = self.DH.master_data['FX'][pName]['CLOSE']['val'][loc]
                    self.lastFXrate.update({pName:FX[-1]})
                    FX = np.tile(FX.reshape([-1,1]),(1,EoDpos.shape[1]))
                    
                    posMut = np.append(EoDpos[np.array([0]),:],np.diff(EoDpos,axis=0),axis=0)
                    
                    FXpos = np.full(posMut.shape,np.nan)
                    FXpos[posMut!=0] = EoDpos[posMut!=0]
                    FXpos = FXpos*P*self.priceMetaDict[pName]['tickValues']
                    FXpos = pd.DataFrame(FXpos).fillna(method='ffill').fillna(0).values                    
                    FXpnl = np.append(np.zeros([1,EoDpos.shape[1]]),np.diff(FX,axis=0)*FXpos[:-1,:],axis=0)/1000
                    
                    
                    tdC = self.DH.calcMinOffset('23:59') # dates in PnLT are on whole days, add this time to the day and it will be EoD always
                    IDX = pd.DatetimeIndex([i+pd.offsets.Minute(n=tdC) for i in self.pnlT.index])
                    
                    self.FXtradePnL.update({pName:pd.DataFrame(FXpnl,index=IDX,columns=self.pnlT[pName].columns)})
                    self.lastFXpos.update({pName:FXpos[-1,:]})
                    self.pnlT[pName] = self.pnlT[pName].values+FXpnl
                    
            merged = self.groupPnL(mergePrices=True,name=key,df=self.pnlT.loc[:,self.OH.RVdict[key]['pnl']])
            
            pnlTdropped.append(self.pnlT.loc[:,self.OH.RVdict[key]['pnl']].copy())
            
            self.pnlT = pd.concat([self.pnlT.drop(columns=self.OH.RVdict[key]['pnl']),merged],axis=1)
        if len(pnlTdropped)>0:
            self.pnlTdropped = pd.concat(pnlTdropped,axis=1)
        else:
            self.pnlTdropped = None
        self.collectLatestDataPoints()

    def collectLatestDataPoints(self):
        if hasattr(self,'lastDataCollection')==False:
            self.lastDataCollection = {part:{} for part in ['RV','signals','prices','FX']}
        RH = ReutersHub('BO',altName='dummy')
        for part in ['RV','signals','prices','FX']: 
                if part=='signals':
                    L = self.DH.master_data[part]['key']
                else:
                    L = list(self.DH.master_data[part].keys())
                if len(L)>0:                    
                    for pName in L:
                        if part=='signals':
                            if len(self.DH.master_data[part][pName]['val'])>0:
                                val = self.DH.master_data[part][pName]['val'][-1]
                                latest = str(self.DH.master_data[part][pName]['idx'][-1])
                                self.lastDataCollection[part].update({pName:{'idx':latest,'val':val}})
                        else:
                            if len(self.DH.master_data[part][pName]['CONTRACTS']['val'])>0:
                                val = self.DH.master_data[part][pName]['CLOSE']['val'][-1]
                                Contract = RH.YYYYMMtoLabel(int(self.DH.master_data[part][pName]['CONTRACTS']['val'][-1]))
                                latest = str(self.DH.master_data[part][pName]['CONTRACTS']['idx'][-1])
                                self.lastDataCollection[part].update({pName:{'idx':latest,'val':val,'contract':Contract}})
    def getLastDataPointFromCollection(self):
        if hasattr(self,'lastDataCollection'):
            tl = '1950-01-01'

            for part in ['RV','signals','prices','FX']:  
                for pName in list(self.lastDataCollection[part].keys()):                
                    latest = self.lastDataCollection[part][pName]['idx']
                    if tl<latest:
                        tl = latest
            return tl
        else:
            return 'data collection missing'
    def saveRollingOutOfSampleBackTestResult(self,name,prod=False):
        if hasattr(self,'rollingOutofSampleBackTest'):
            if prod:
                basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/prodSave/backTestResult/'.format(os.getlogin())
            else:
                basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/backTestResult/'.format(os.getlogin())

            with open(basePath+name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.rollingOutofSampleBackTest, f)
        else:
            print('ERROR: no back test result attached.')

    def loadRollingOutOfSampleBackTestResult(self,name,prod=False):
        if prod:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/prodSave/backTestResult/'.format(os.getlogin())
        else:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/backTestResult/'.format(os.getlogin())

        try:
            with open(basePath+name + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
                self.rollingOutofSampleBackTest = pickle.load(f)  
            return True
        except:
            return False
        

    def saveResults(self,name):
        """Save pnl, parameter set and price information"""
        if hasattr(self,'pnlT'):
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/simSaves/'.format(os.getlogin())
            
            with open(basePath+name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([self.pnlT,self.stratParams,self.DH.priceMetaDict], f)
            print('file saved')
        else:
            print('First run processPnL. Aborting.')
    def saveEoD(self,name,prod=False):
        """Save data such that it can be used to continue running the next day's data in another session."""
        self.DH.prepareStart()
        self.OH.prepareStart(self.priceMetaDict,self.FXtradePnL,self.lastFXrate,self.lastFXpos)
        self.plotDict = None
        
        if hasattr(self,'histPnL'):
            self.histPnL = pd.concat([self.histPnL,self.pnlT],axis=0)
        else:
            self.histPnL = self.pnlT.copy()
        delattr(self,'pnlT')
        
        if hasattr(self,'histpnlTdropped'):
            self.histpnlTdropped = pd.concat([self.histpnlTdropped,self.pnlTdropped],axis=0)
            delattr(self,'pnlTdropped')
        elif self.pnlTdropped!=None:
            self.histpnlTdropped = self.pnlTdropped.copy()
            delattr(self,'pnlTdropped')
            
        
        if prod:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/prodSave/EoD/'.format(os.getlogin())
        else:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/FullSaves/'.format(os.getlogin())
            
        with open(basePath+name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self, f)
    def loadEoD(self,name,prod=False):
        """Loads EoD save"""
        if prod:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/prodSave/EoD/'.format(os.getlogin())
        else:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/FullSaves/'.format(os.getlogin())
        with open(basePath+name + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
            TP = pickle.load(f)  
        return TP
        
    def saveStatus(self,name,prod=False):
        """Saves full status, which is a complete pickle dump of the object instance"""
        if prod:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/prodSave/status/'.format(os.getlogin())
        else:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/StatusSaves/'.format(os.getlogin())
        
        with open(basePath+name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self, f)
            
    def loadStatus(self,name,prod=False):
        """Loads previously saved status"""
        
        if prod:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/prodSave/status/'.format(os.getlogin())
        else:
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/StatusSaves/'.format(os.getlogin())
            
        with open(basePath+name + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
            TP = pickle.load(f)  
        return TP
    def loadResults(self,name):
        """Loads previously saved results. Overwrites current."""
        basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/simSaves/'.format(os.getlogin())

        with open(basePath+name + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
            self.pnlT,self.stratParams,self.DH.priceMetaDict= pickle.load(f)  
        
        
        print('file loaded')
    def loadMergeResults(self,name):
        """Loads previously saved results. Merges with current."""
        if hasattr(self,'pnlT'):
            basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/simSaves/'.format(os.getlogin())

            with open(basePath+name + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
                pnlT, stratParams,priceMetaDict= pickle.load(f)  
                
            if self.stratParams==stratParams:
                self.pnlT = pd.concat([self.pnlT,pnlT],axis=1).fillna(0)
                self.priceMetaDict.update(priceMetaDict)
                print('Loaded and merged.')
            else:
                print('Parameter set does not match, aborting.')
        else:
            print('Nothing to merge, aborting.')
    def groupPnL(self,mergeParams=False,mergePrices=False,name='Total',df=[]):
        """Groups PnL results on commodity or parameter level.
        
        Parameters
        ----------
        mergeParams : bool
            Parameter results are merged when True
        mergePrices : bool
            All results on different price curves are merged.
        name : str
            Pass a name to the merged result, default='Total'
        df : pandas dataframe
            If left blanc, it operates on the internal self.pnlT attribute, otherwise on input df

        Returns
        -------
            pandas Dataframe
        """
        if type(df)==type([]):
            pnlIn = self.pnlT.copy()
        else:
            pnlIn = df.copy()
            
        if (mergeParams==False) & (mergePrices==False):
            return pnlIn
        elif (mergeParams) & (mergePrices):
            pnlT = pnlIn.sum(axis=1).to_frame()
            pnlT.columns = [np.array([name]),np.array(['merged'])]
            return pnlT
    
        elif mergePrices:
            pnlColnames = list(pnlIn.columns.levels[0])
            for i,colName in enumerate(pnlColnames):
                if i ==0:
                    X = pnlIn[colName].values
                    cols = pnlIn[colName].columns
                    idx = pnlIn[colName].index
                else:
                    X = X+pnlIn[colName].values
                    
            pnlT = pd.DataFrame(X,columns=[np.array([name for i in range(len(cols))]),np.array(cols)],index=idx)
            return pnlT
        elif mergeParams:
            pnls = []
            pnlColnames = list(pnlIn.columns.levels[0])
            for pName in pnlColnames:
                df = pnlIn[pName].sum(axis=1).to_frame()
                df.columns = [np.array([pName]),np.array(['merged'])]
                pnls.append(df)
            pnlT = pd.concat(pnls,axis=1)
            return pnlT
    def riskmetricsRealBasic(self,pnlIn):
        """Calculates risk metrics on the basis of the pnl curve only.
        
        Parameters
        ----------
        pnlIn : numpy array
            Input numpy array need to be of shape N*M, where N is timewise and M are the pnl columns"""
        RMT = []
        
        N1 = int(np.ceil(1E7/pnlIn.shape[0]))
        N2 = int(np.ceil(pnlIn.shape[1]/N1))
        for i in range(N2):
            if i!=N2-1:
                pnl = pnlIn[:,i*N1:(i+1)*N1]
            else:
                pnl = pnlIn[:,i*N1:]
            
            mask = pnl!=0
    
            N = np.sum(mask,axis=0) # days with active risk position on
            percDaysInPos = N/pnl.shape[0]
            nanPnL = pnl.copy()
            nanPnL[mask==False] = np.nan        
            cumPnL = np.cumsum(pnl,axis=0)
    
            endPnL =cumPnL[-1,:].copy()
    
            pnlSTD2 = np.nanstd(pnl,axis=0)        
            pnlnanSTD = np.nanstd(nanPnL,axis=0)
            
            nanPnLup = nanPnL.copy()
            pnlDownDayMask = nanPnLup<0
            nanPnLup[pnlDownDayMask] = np.nan
    
            nanPnLDown = nanPnL.copy()
            pnlUpDayMask = nanPnLDown>=0
            nanPnLDown[pnlUpDayMask] = np.nan
    
            SRpos = endPnL/N*260**0.5/pnlnanSTD
            SR = endPnL/pnl.shape[0]*260**0.5/pnlSTD2
            
            PnlRatio = np.round(np.nanmean(nanPnLup,axis=0)/-np.nanmean(nanPnLDown,axis=0)*100,1)
            HitRatio = np.round(np.sum(pnlUpDayMask,axis=0)/(np.sum(pnlUpDayMask,axis=0)+np.sum(pnlDownDayMask,axis=0))*100,1)
    
            DD = np.maximum.accumulate(cumPnL,axis=0) -cumPnL
    
            DUnew = cumPnL.copy()
            for i in range(1,cumPnL.shape[0]):
                maskDU = cumPnL[i,:]>=cumPnL[i-1,:]
                DUnew[i,maskDU] = DUnew[i-1,maskDU].copy()
    
            DUnew = cumPnL-DUnew
    
            maxDD = np.max(DD,axis=0)
    
            DD[mask==False] = np.nan
            DUnew[mask==False] = np.nan
            DDsurf = np.nansum(DD,axis=0)
            DDratio = np.nansum(DUnew,axis=0)/DDsurf
            DDsurf = DDsurf/np.sum(mask,axis=0)
            kalmar = endPnL/maxDD
    
            C1 = ['PnL [k$]','SRpos','SR','kalmar','Days Pos [%]','PnlRatio','HitRatio','DDsurf','maxDD','DDratio']
            C2 = [endPnL,SRpos,SR,kalmar,percDaysInPos*100,PnlRatio,HitRatio,DDsurf,maxDD,DDratio]     
            C3 = [        1         , 1  ,    2  ,2, 1,            1,   0,     1,     2,     2      ]
            C2 = [np.around(i,C3[j])  for j,i in enumerate(C2)]
            RM = pd.DataFrame({j:C2[i] for i,j in enumerate(C1)})
            RMT.append(RM)
            
        RM = pd.concat(RMT,axis=0)
        return RM
    def createPnLPlotter(self):
        """Produces PnL plotter tool"""
        if hasattr(self,'pnlT'):            
            def pnlCockPit(doc):
                
                def processPnL():
                    
                    mergeParams = True if mergeParam.value=='Yes' else False
                    mergePrices = True if mergePrice.value=='Yes' else False
                    
                    pnlT = self.groupPnL(mergeParams=mergeParams,mergePrices=mergePrices)
                    
                    multiCol = list(pnlT.columns)

                    col1 = [i[0] for i in multiCol]
                    col2 = [i[1] for i in multiCol]
                    
                    if pnlT.shape[1]==1:                    
                        RM = self.riskmetricsRealBasic(pnlT.values.reshape([-1,1]))
                    else:
                        RM = self.riskmetricsRealBasic(pnlT.values)
                    RM['PRD'] = col1
                    RM['Param'] = col2
                    
                    colors = []
                    colorCounter = 0
                    for i in range(RM.shape[0]):
                        colors.append(self.colors[colorCounter])
                        colorCounter = colorCounter+1
                        if colorCounter==len(self.colors):
                            colorCounter = 0
                    RM['colors'] = colors
                    doc.plot_src = ColumnDataSource(pnlT.cumsum())
                    doc.RM = RM.loc[:,['colors','PRD','Param','PnL [k$]','SRpos','SR','kalmar','Days Pos [%]','PnlRatio','HitRatio','DDsurf','maxDD','DDratio']]

                                        
                def selectionMade(attr,old,new):
                    colors = doc.RM['colors'].values
                    prds = doc.RM['PRD'].values
                    params = doc.RM['Param'].values
                    pnlPlot.renderers = []
                    p = figure(plot_width = 800, plot_height = 600,title = 'PnL development',x_axis_type='datetime',y_axis_label='PnL [k$]')
                 
                    for i in table_src.selected.indices:                        
                        p.line(source=doc.plot_src, y=prds[i]+'_'+params[i],x='Date',color=colors[i],line_width=2) # name=
                    pnlPlot.renderers = p.renderers
                    
                    if len(table_src.selected.indices)>1:
                        merge_sel.disabled=False
                    else:
                        if merge_sel.label=='Merge selection':
                            merge_sel.disabled=True
                        
                def MergeSelectionMade():
                    if merge_sel.label=='Merge selection':
                        
                        merge_sel.label = 'Merging'
                        merge_sel.button_type = 'danger'
                        prds = doc.RM['PRD'].values
                        params = doc.RM['Param'].values
                        pnlPlot.renderers = []
                        p = figure(plot_width = 800, plot_height = 600,title = 'PnL development',x_axis_type='datetime',y_axis_label='PnL [k$]')


                        doc.tempIndices = table_src.selected.indices.copy()
                        for j,i in enumerate(table_src.selected.indices): 
                            if j==0:
                                X = doc.plot_src.data[prds[i]+'_'+params[i]]
                            else:
                                X = X+doc.plot_src.data[prds[i]+'_'+params[i]]
                        
                        p.line(y=X,x=doc.plot_src.data['Date'],color='blue',line_width=2) # name=
                        pnlPlot.renderers = p.renderers
                        
                        
                        X = np.append(X[0],np.diff(X))
                        RM = self.riskmetricsRealBasic(X.reshape([-1,1]))
                        RM['PRD'] = 'Merged'
                        RM['Param'] = 'Merged by selection'
                        RM['colors'] = 'blue'
                        
                        
                        table_src.data.update(ColumnDataSource(RM).data)
                        merge_sel.label = 'Switch back'
                        merge_sel.button_type = 'primary'
                    elif merge_sel.label=='Switch back':
                        merge_sel.label = 'Merge selection'
                        merge_sel.button_type = 'success'
                        table_src.data.update(ColumnDataSource(doc.RM).data)
                        table_src.selected.indices=doc.tempIndices.copy()
                        selectionMade(None,None,None)    
                    else:
                        merge_sel.label = 'Merge selection'
                        merge_sel.button_type = 'success'
                        
                        if len(table_src.selected.indices)<2:                        
                            merge_sel.disabled=True
                def process():
                    if run_sel.label=='Run': 
                        run_sel.label = 'Running'
                        run_sel.button_type = 'danger'
                        
                        processPnL()
                                                
                        table_src.data.update(ColumnDataSource(doc.RM).data)
                                    
                        run_sel.label = 'Done'
                        run_sel.button_type = 'primary'
                        
                    else:
                        run_sel.label = 'Run'
                        run_sel.button_type = 'success'
                
                
                def makePlot():        
                    colors = doc.RM['colors'].values
                    prds = doc.RM['PRD'].values
                    params = doc.RM['Param'].values
                    pnlPlot.renderers = []
                    p = figure(plot_width = 800, plot_height = 600,title = 'PnL development',x_axis_type='datetime',y_axis_label='PnL [k$]')
                 
                    for i in range(len(colors)):                        
                        p.line(source=doc.plot_src, y=prds[i]+'_'+params[i],x='Date',color=colors[i],line_width=2,visible=False) # name=
                    pnlPlot.renderers = p.renderers
     
                colorCol = """<b><div style="text-align:right<%=(function colorfromint(){ return(";background:"+value+";color:"+value)}()) %> "> <%= value %></div></b>"""

                
                pnlPlot = figure(plot_width = 800, plot_height = 600,title = 'PnL development',x_axis_type='datetime',y_axis_label='PnL [k$]')
                    
                mergeParam = Select(value='No',options=['Yes','No'],title="Merge Parameters",width=120)     
                mergePrice = Select(value='No',options=['Yes','No'],title="Merge Prices",width=120)     
                
                run_sel = Button(label='Run', button_type="success",width=120)
                run_sel.on_click(process)
                
                merge_sel = Button(label='Merge selection', button_type="success",width=120,disabled=True)
                merge_sel.on_click(MergeSelectionMade)
                
                dummy = pd.DataFrame(columns=['colors','PRD','Param','PnL [k$]','SRpos','SR','kalmar','Days Pos [%]','PnlRatio','HitRatio','DDsurf','maxDD','DDratio'])
                table_src = ColumnDataSource(dummy)
                table_src.selected.on_change('indices', selectionMade) # to see which line is selected
      

                colWidths = [50,80,250]+[60 for i in range(10)]
                columns = [TableColumn(field=i, title=i,width=colWidths[j],formatter=StringFormatter(text_align='right') if i != 'colors' else HTMLTemplateFormatter(template=colorCol)) for j,i in enumerate(dummy.columns)]
                data_table = DataTable(source=table_src, columns=columns, width=1000, editable=False,index_position=None)

                layO = column(pnlPlot,row(mergeParam,mergePrice,run_sel,merge_sel),data_table)
                
                doc.add_root(layO)
            show(Application(FunctionHandler(pnlCockPit)))
            
            
        else:
            print('First run processPnL. Aborted.')
        
    def getNeovestFills(self,fillsFilenameOverride='',info=True):
        """Processes a direct dump from Neovest into excel into fills per product.

        Parameters
        ----------
        fillsFilenameOverride : str
            overrides default path
        info : bool
            If True, prints the file origin it reads from.
        """
        if fillsFilenameOverride=='':
            fileName = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/NeovestFills/fills.xlsx'.format(os.getlogin())
            if info:
                print('Grabbing from /GenericBackTesterFiles/NeovestFills/fills.xlsx:')
        else:
            fileName = fillsFilenameOverride
            if info:
                print('Grabbing from '+fileName)
                    
        df = pd.read_excel(fileName)
        if df.shape[0]>0:
            if 'CompQty [Day]' in list(df.columns):
                for idx in df.index:
                    if np.isnan(df.loc[idx,'CompQty [Day]'])==False:
                        df.loc[idx,'CompQty'] = df.loc[idx,'CompQty [Day]']
            df = df.groupby(['Side', 'Symbol Display','CustText1'])['CompQty'].sum().reset_index()
            
            df1 = df.copy()
        else:
            df1 = pd.DataFrame()

        fileName = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/NeovestFills/fillsTP.xlsx'.format(os.getlogin())
        df = pd.read_excel(fileName)
        if df.shape[0]>0:
            if 'CompQty [Day]' in list(df.columns):
                for idx in df.index:
                    if np.isnan(df.loc[idx,'CompQty [Day]'])==False:
                        df.loc[idx,'CompQty'] = df.loc[idx,'CompQty [Day]']
            df = df.groupby(['Side', 'Symbol Display','CustText1'])['CompQty'].sum().reset_index()
            if df1.shape[0]!=0:
                df = pd.concat([df1,df],axis=0)
            
        elif df1.shape[0]!=0:
            df = df1
        else:
            return None
        return df.pivot_table(index='CustText1',columns=['Symbol Display','Side'],values='CompQty',aggfunc=np.sum)
    
    def createCockpit(self,dateOverride='month',prod=False):
        """Produces cockpit view, where orders/fills/position and pnl can be monitored.
        
        Parameters
        ----------
        dateOverride : str
            Determines the date axis level, options are: 'year','month','day','hour','second' & 'tick'
        prod : bool
            When True, additionally provides current orderbook in a table, today's fills and position and last update time. 
        """
        def btCockPit(doc):
            def rangeChange(attr,old,new):
                launchView.disabled = False
                selN = self.plotDict['paramCombKey'][selParam.value]
                df = self.plotDict['totalpnl'][selN].copy()
                mask = (df.index>=sizeSlider.value[0]) &(df.index<=sizeSlider.value[1])
                df['highlight'] = np.nan
                df.loc[mask,'highlight'] = df.loc[mask,'Total pnl'].copy()            
                
                tpnl_src.data.update(ColumnDataSource(df).data)
            def timeScaleChange(attr,old,new):   
                for i in range(len(layO.children)-1):                    
                    for k in range(3):
                        layO.children[i].children[k].xaxis.major_label_overrides = self.plotDict['dateOverrides'][new]                        
                tp.xaxis.major_label_overrides = self.plotDict['dateOverrides'][new] 
            def updatePlots():
                launchView.label = 'preparing view'
                launchView.button_type = 'danger'
                newChildren = []
                selN = self.plotDict['paramCombKey'][selParam.value]
                
                for i,pName in enumerate(doc.COM):
                    
                    p = self.makeTradePlot(pName,self.plotDict['dateOverrides'][timeScale.value],selN,sizeSlider.value[0],
                                           sizeSlider.value[1],doc.origin[i],sigList=[])
                    newChildren.append(p)
                    
                
                df = self.plotDict['totalpnl'][selN].copy()
                mask = (df.index>=sizeSlider.value[0]) &(df.index<=sizeSlider.value[1])
                df['highlight'] = np.nan
                df.loc[mask,'highlight'] = df.loc[mask,'Total pnl'].copy()            
                
                tpnl_src.data.update(ColumnDataSource(df).data)
                launchView.label = 'Process'
                launchView.button_type = 'success'
                launchView.disabled = True
                if prod:
                    layO.children = newChildren+[column(tp,sizeSlider,row(selParam,timeScale,launchView,switchRV),posText,data_table)]
                else:
                    layO.children = newChildren+[column(tp,sizeSlider,row(selParam,timeScale,launchView,switchRV))]
                
            def changeParamSet(attr,old,new):
                launchView.disabled = False
            def switchRVtoLegs():
                if 'RV' in doc.origin:
                    doc.COM = doc.rvComponent+doc.COMkeep
                    doc.origin = ['prices' for i in doc.COM]                    
                    if prod:
                        src.data.update(ColumnDataSource(dfO).data)                        
                else:
                    doc.COM = doc.COMrv+doc.COMkeep
                    doc.origin = ['RV' for i in doc.COMrv]+['prices' for i in doc.COMkeep]   
                    if prod:
                        src.data.update(ColumnDataSource(dfO_RV).data)
                updatePlots()
            if len(list(self.OH.RVdict.keys()))>0: # contains RVs
                doc.COM = list(self.plotDict['prices'].keys())
                doc.rvComponent = []
                for rvName in list(self.OH.RVdict.keys()): # not all COM is in RV, so identify which needs to be kept apart
                    doc.rvComponent = doc.rvComponent+self.OH.RVdict[rvName]['pnl']
                doc.COMkeep = []
                
                for i in doc.COM:
                    if i not in doc.rvComponent:
                        doc.COMkeep.append(i)
                
                doc.COMrv = list(self.OH.RVdict.keys())
                doc.origin = ['RV' for i in doc.COMrv]
                doc.originKeep = ['prices' for i in doc.COMkeep]    
                
                doc.COM = doc.COMrv+doc.COMkeep
                doc.origin = doc.origin+doc.originKeep
                switchRV = Button(label="Switch RV/Legs", button_type="success",width=100)
                switchRV.on_click(switchRVtoLegs)
            else:
                doc.COM = list(self.plotDict['prices'].keys())
                doc.origin = ['prices' for i in doc.COM]
                switchRV = Button(label="Switch RV/Legs", button_type="success",width=100,disabled=True)
            layO = row()
            opts = ['year','date','month','day','hour','second','tick']
            
            
            
            paramList = list(self.plotDict['paramCombKey'].keys())
            
            selParam = Select(value=paramList[0],options=paramList,title="Parameter combinations",width=300)
            selParam.on_change('value',changeParamSet)
            
            
            
            
            timeScale = Select(value=opts[3],options=opts,title="time scaleType",width=120)                
            timeScale.on_change("value",timeScaleChange)
            
            Nstart = self.plotDict['totalpnl'][0].index[0]
            Nend = self.plotDict['totalpnl'][0].index[-1]
            if prod:
                sizeSlider = RangeSlider(start = Nstart, end = Nend, step = 1, value = (Nend-int((Nend-Nstart)*0.15),Nend),width=600)
            else:
                sizeSlider = RangeSlider(start = Nstart, end = Nend, step = 1, value = (Nstart,Nstart+int((Nend-Nstart)*0.10)),width=600)
            sizeSlider.on_change("value",rangeChange)
            
            df = self.plotDict['totalpnl'][0].copy()
            
            if prod:
                posInfo = column()
                for pName in list(self.OH.pospnlKey.keys()):
                    lastDataPoint = str(self.priceMetaDict[pName]['idx'][-1])
                    loc = (self.DH.master_idx==lastDataPoint).nonzero()[0][0]
                    # loc = int(np.max(np.array(list(self.plotDict['dateOverrides']['month'].keys()))))
                    keepDay = self.plotDict['dateOverrides']['month'][loc]
                    startLoc = 0
                    for i in range(loc+1):
                        if self.plotDict['dateOverrides']['month'][i]==keepDay:
                            startLoc = i
                            break
                    
                    currPos = np.sum(self.OH.pos[self.OH.pospnlKey[pName]][-1,:])
                    if startLoc!=0:
                        dfB = self.plotDict['trades'][0][pName]['BUY'] #  prod always has aggregate orders to True, so zero is the correct index
                        dfS = self.plotDict['trades'][0][pName]['SELL']
                        
                        if np.isnan(dfB.index.max()):
                            sumB = 0
                        else:
                            sumB = dfB.loc[startLoc:,'vol'].sum()
                            
                        if np.isnan(dfS.index.max()):
                            sumS = 0
                        else:
                            sumS = dfS.loc[startLoc:,'vol'].sum()
                        
                        posText = Div(text='<h3>{0} {1} || Curr Pos {2} lots</h3>Today: - Bought {3} lots - Sold {4} lots'.format(pName,lastDataPoint,currPos,sumB,sumS))
                    else:
                        posText = Div(text='<h3>{0} {1} || Curr Pos {2} lots</h3>Today: - Bought {3} lots - Sold {4} lots'.format(pName,lastDataPoint,currPos,0,0))
                    posInfo.children.append(posText)
                
                dfO, df_currV,dfO_RV = self.orderAggregator(maxShowVol=5,prod=False)
                
                dfO = dfO.loc[:,['Symbol','Side','Order Size','Order Type','Price Offset','Stop Price']]
                dfO_RV = dfO_RV.loc[:,['Symbol','Side','Order Size','Order Type','Price Offset','Stop Price']]
                dfO = pd.concat([dfO,df_currV],axis=0)
                dfO = dfO.sort_values(by=['Symbol','Price Offset'],kind='mergesort',ascending=[True,False])
                if 'RV' in doc.origin:
                    src = ColumnDataSource(dfO_RV)
                else:
                    src = ColumnDataSource(dfO)
                
                columns = [TableColumn(field=i, title=i,formatter=StringFormatter(text_align='right')) for j,i in enumerate(dfO.columns)]
                data_table = DataTable(source=src, columns=columns, width=800, editable=False,index_position=None)

            mask = (df.index>=sizeSlider.value[0]) &(df.index<=sizeSlider.value[1])
            df['highlight'] = np.nan
            df.loc[mask,'highlight'] = df.loc[mask,'Total pnl'].copy()
            
            tpnl_src = ColumnDataSource(df)            
            
            tp = figure(plot_width = 800, plot_height = 600, y_axis_label = 'Total PnL [$]',title='Total PnL development')
            tp = self.style(tp)
            tp.xaxis.major_label_overrides = self.plotDict['dateOverrides'][opts[3]]
            tp.line(source=tpnl_src,y='Total pnl',x='Date',color='blue')
            tp.line(source=tpnl_src,y='highlight',x='Date',color='red',line_width=2)
            
            launchView= Button(label="Process", button_type="success",width=100)
            launchView.on_click(updatePlots)
            
            
            
            updatePlots()
            doc.add_root(layO)
        show(Application(FunctionHandler(btCockPit)))
    def transFromNeovestDate(self,timeString):
        if str(type(timeString))=="<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
            currT = timeString-pd.offsets.Hour(2)+pd.offsets.Minute(1)
        else:
            timeHHMMSS = timeString[:8]
            AMPM = timeString[9:11]
            dateSTR = timeString[12:]
            # for i in range(2):
            if dateSTR.find('/')==1:
                dateSTR = '0'+dateSTR

            if dateSTR[3:].find('/')==1:
                dateSTR = dateSTR[:3]+'0'+dateSTR[3:]
            datetimeStr = dateSTR+' '+timeHHMMSS

            currT = dt.datetime.strptime(datetimeStr,'%m/%d/%y %H:%M:%S')-pd.offsets.Hour(2)+pd.offsets.Minute(1)
            if AMPM=='PM':    
                currT = currT+pd.offsets.Hour(12)

        return dt.datetime.strftime(currT,format='%Y-%m-%d %H:%M')    
     
    def addActualTradesToPlotDict(self,key,PRD,dateOverrides):
        
        fileName = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/GenericBackTesterFiles/NeovestFills/fills.xlsx'.format(os.getlogin())
        df = pd.read_excel(fileName)
        
        df.loc[:,'Date(UK)'] = df.loc[:,'Date(UK)'].apply(self.transFromNeovestDate)

        self.TTT = df.copy()
        field = 'date'
        dictKeys = list(dateOverrides['date'])
        dictRev = np.array([[i, dateOverrides['date'][i]+' '+dateOverrides['hour'][i]] for i in dictKeys])


        orderTypeDict = {'LMT':'LIMIT','MKT':'MARKET','SLMT':'STOP'}


        X = df.loc[df['CustText1']==key,['Date(UK)','Order Type','Side','AvgPx','CompQty','CompQty [Day]']]
        mask = pd.isna(X['CompQty [Day]'])==False

        X.loc[mask,'CompQty'] = X.loc[mask,'CompQty [Day]'].copy()
        X['Order Type'] = X['Order Type'].map(orderTypeDict)
        X['Side'] = X['Side']=='Buy'
        X['product'] = PRD

        X = X.loc[:,['Date(UK)','Order Type','product','Side','CompQty','AvgPx']]
        X.columns = ['Date','oType','product','BS','vol','val']

        newDate = []
        for i in X.loc[:,'Date']:
            loc = (dictRev[:,1]>=i).nonzero()[0]
            if len(loc)>0:
                loc = dictRev[loc[0],0]
            else:
                loc = dictRev[-1,0]
            newDate.append(loc)
        X['Date'] = newDate
        X.set_index(drop=True,keys='Date',inplace=True)
        X.index = X.index.astype(int)
        B = X.loc[X['BS'],:]
        S = X.loc[X['BS']==False,:]

        return B,S
        
    def orderCheck(self,orders,N,priceOrTicks):
        if 'id' not in list(orders.keys()):
            print('"id" is missing in order dictionary. Aborting.')
            return False
        else:
            if type(orders['id'])!=type(np.array([])):
                print('"id" is not a required numpy array. Aborting.')
                return False
            else:
                if len(orders['id'])==0: # empty order
                    return True
                else: # contains orders
                    # first check whether all items are there

                    for i in ['id', 'oType', 'product', 'BS', 'active',  'val', 'vol']:
                        if i not in list(orders.keys()):
                            print('"'+i+'" is missing in order dictionary. Aborting.')
                            return False
                        # check whether all are numpy arrays
                        if type(orders[i])!=type(np.array([])):
                            print('"'+i+'" is not a required numpy array. Aborting.')
                            return False

                        if type(orders[i])!=type(np.array([])):
                            print('"'+i+'" is not a required numpy array. Aborting.')
                            return False

                        if i in ['id','oType', 'product']:
                            if len(orders[i].shape)!=1:
                                print('"'+i+'" is not a 1 dimensional array. Aborting.')
                                return False
                            if i =='id':
                                M = len(orders[i])
                                if orders[i].dtype!=np.array([0]).dtype:
                                    print('"'+i+'" is not an integer array Suggest to use .astype(int) to fix. Aborting.')
                                    return False

                            else:
                                if len(orders[i])!=M:
                                    print('"'+i+'" is not of same length as "id" array. Aborting.')
                                    return False
                                if type(orders[i][0])!=type(np.array([''])[0]):
                                    print('"'+i+'" is not an str array. Aborting.')
                                    return False
                        else:
                            if len(orders[i].shape)!=2:
                                print('"'+i+'" is not a 2 dimensional array. Aborting.')
                                return False

                            if (orders[i].shape[0]!=M) | (orders[i].shape[1]!=N):
                                print('"'+i+'" not matching required dimensions as dictated by the number of parameters and lenght of "id" array. Expected: '+str(M)+' by '+str(N)+', got ' +str(orders[i].shape))
                                return False

                            if i in ['BS','active']:
                                if orders[i].dtype!=np.array([0]).astype(bool).dtype:
                                    print('"'+i+'" is not an boolean array Suggest to use .astype(bool) to fix. Aborting.')
                                    return False
                            elif i =='vol':
                                if orders[i].dtype!=np.array([0]).dtype:
                                    print('"'+i+'" is not an integer array Suggest to use .astype(int) to fix. Aborting.')
                                    return False
                            elif priceOrTicks=='ticks': # only val left, make distinction between int and float
                                if orders[i].dtype!=np.array([0]).dtype:
                                    print('"'+i+'" is not an integer array Suggest to use .astype(int) to fix. Aborting.')
                                    return False
                return True
    def prepPlotItems(self,monitorStart,monitorEnd,pathOfData=None,aggOrders=False,addActual=False):
        """Prepares plot items used by cockpit functionality."""
        pList = self.OH.pList
        
        
        plotIDX = self.DH.master_idx.copy()
        plotIDX = plotIDX.append(pd.DatetimeIndex([plotIDX[-1]])+pd.Timedelta(days=(self.DH.master_idx[-1]-self.DH.master_idx[-2]).value/1E9/60/60/24))
        
        minIDX = (plotIDX<monitorStart).nonzero()[0]
        if len(minIDX)>0:
            minIDX = minIDX[-1]+1
        else:
            minIDX = 0
        
        maxIDX = (plotIDX>monitorEnd).nonzero()[0]
        if len(maxIDX)>0:
            maxIDX = maxIDX[0]-1
        else:
            maxIDX = len(plotIDX)
        
        idxMap = {idx:i for i,idx in enumerate(plotIDX)}
        dateOverrides = {'year': {i: date.strftime('%Y') for i, date in enumerate(pd.to_datetime(plotIDX))},
                'date': {i: date.strftime('%Y-%m-%d') for i, date in enumerate(pd.to_datetime(plotIDX))},
                'month':{i: date.strftime('%b %d') for i, date in enumerate(pd.to_datetime(plotIDX))},
                 'day':{i: date.strftime('%a %H' ) for i, date in enumerate(pd.to_datetime(plotIDX))},
                  'hour':{i: date.strftime('%H:%M' ) for i, date in enumerate(pd.to_datetime(plotIDX))},
                 'second':{i: date.strftime('%H:%M:%S' ) for i, date in enumerate(pd.to_datetime(plotIDX))},
               'tick':{i: date.strftime('%H:%M:%S.%f' ) for i, date in enumerate(pd.to_datetime(plotIDX))}}
        realTrades = False
        prices = {}
        for pName in pList:
            idx = plotIDX.isin(self.DH.master_data['prices'][pName]['CLOSE']['idx']).nonzero()[0]
            df = pd.DataFrame(self.DH.master_data['prices'][pName]['CLOSE']['val']*self.priceMetaDict[pName]['tickSize'],
                             index=idx,
                            columns=['CLOSE'])

            mask = np.isnan(self.DH.master_data['prices'][pName]['Roll']['val'])==False
            df['Roll'] = np.nan
            df.loc[mask,'Roll'] = df.loc[mask,'CLOSE'].copy()

            for colName in ['CONTRACTS','status']:
                df[colName] = self.DH.master_data['prices'][pName][colName]['val']

            if self.DH.master_data['prices'][pName]['dtype'][:4] in ['OHLC','IDM_']:        
                for colName in ['OPEN','HIGH','LOW']:
                    if colName=='OPEN':
                        uIdx = self.DH.master_data['prices'][pName]['CLOSE']['idx'].union(self.DH.master_data['prices'][pName]['OPEN']['idx'])
                        idx = plotIDX.isin(uIdx).nonzero()[0]
                        

                    df[colName] = self.DH.master_data['prices'][pName][colName]['val']*self.priceMetaDict[pName]['tickSize']
            df.index.name='Date'
            
            df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
            
            prices.update({pName:df})
            
        RV = {}        
        for rvName in list(self.OH.RVdict.keys()):
            idx = plotIDX.isin(self.DH.master_data['RV'][rvName]['CLOSE']['idx']).nonzero()[0]
            
            df = pd.DataFrame(index=idx)
            for key in list(self.DH.master_data['RV'][rvName].keys()):
                if key in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'Roll', 'CONTRACTS', 'status']:
                    df[key] = self.DH.master_data['RV'][rvName][key]['val']
            df.index.name='Date'            
            df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
            mask = pd.isna(df.Roll)==False
            df.loc[mask,'Roll'] = df.loc[mask,'CLOSE'].copy()
            df['Roll'] = np.nan
            df.loc[mask,'Roll'] = df.loc[mask,'CLOSE'].copy()
            
            RV.update({rvName:df})
        
        signals = {}
        for sigName in self.DH.master_data['signals']['key']:
            idx = plotIDX.isin(self.DH.master_data['signals'][sigName]['idx']).nonzero()[0]
            df = pd.DataFrame(self.DH.master_data['signals'][sigName]['val'],
                                       index=idx,
                                       columns=[sigName])
            df.index.name='Date'
            df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
            signals.update({sigName:df})

        NparamSettings = len(self.OH.keepTrades)


        orders = []
        if hasattr(self.OH,'keepOrders'):        
            for i in range(NparamSettings):
                df = pd.DataFrame(self.OH.keepOrders[i],columns=['datetime','oType','product','BS','vol','val'])        
                df.set_index(keys=['datetime'],drop=True,inplace=True)
                df.index = df.index.map(idxMap)
                
                df['index2'] = df.index+1

                df.index.name='Date'
                df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
                orderDict = {}
                for pName in list(self.priceMetaDict.keys()):
                    dfSet = df.loc[df.loc[:,'product']==pName,:].copy()
                    dfSet.loc[:,'val'] = dfSet.loc[:,'val'].values*self.priceMetaDict[pName]['tickSize']
                    orderDict.update({pName:{'BUY':dfSet.loc[dfSet.BS,:],'SELL':dfSet.loc[dfSet.BS==False,:]}})


                orders.append(orderDict)
        else:
            for i in range(NparamSettings):                
                orderDict = {}
                for pName in list(self.priceMetaDict.keys()):
                    orderDict.update({pName:None})
                orders.append(orderDict)
                
        ordersRV = []
        if hasattr(self.OH,'keepOrdersRV'):        
            for i in range(NparamSettings):
                df = pd.DataFrame(self.OH.keepOrdersRV[i],columns=['datetime','oType','product','BS','vol','val'])        
                df.set_index(keys=['datetime'],drop=True,inplace=True)
                df.index = df.index.map(idxMap)
                df['index2'] = df.index+1

                df.index.name='Date'
                df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
                orderDict = {}
                for pName in list(self.OH.RVdict.keys()):
                    dfSet = df.loc[df.loc[:,'product']==pName,:].copy()
                    orderDict.update({pName:{'BUY':dfSet.loc[dfSet.BS,:],'SELL':dfSet.loc[dfSet.BS==False,:]}})

                ordersRV.append(orderDict)
        else:
            for i in range(NparamSettings):                
                orderDict = {}
                for pName in list(self.OH.RVdict.keys()):
                    orderDict.update({pName:None})
                ordersRV.append(orderDict)
                
        trades = []
        if hasattr(self.OH,'keepTrades'):   
            for i in range(NparamSettings):

                df = pd.DataFrame(self.OH.keepTrades[i],columns=['datetime','oType','product','BS','vol','val'])
                df.set_index(keys=['datetime'],drop=True,inplace=True)
                df.index = df.index.map(idxMap)
                df.index.name='Date'
                df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
                tradesDict = {}
                for pName in list(self.priceMetaDict.keys()):
                    dfSet = df.loc[df.loc[:,'product']==pName,:].copy()
                    dfSet.loc[:,'val'] = dfSet.loc[:,'val'].values*self.priceMetaDict[pName]['tickSize']
                    
                    if addActual:
                        try:
                            B,S = self.addActualTradesToPlotDict(self.strat.name,pName,dateOverrides)
                            tradesDict.update({pName:{'BUY':dfSet.loc[dfSet.BS,:],'SELL':dfSet.loc[dfSet.BS==False,:],'realBUY':B,'realSELL':S}})
                            realTrades = True
                        except:
                            print('Neovest fills into plotDict failed, perhaps unrecognised date format.')
                            tradesDict.update({pName:{'BUY':dfSet.loc[dfSet.BS,:],'SELL':dfSet.loc[dfSet.BS==False,:]}})
                            realTrades = False
                    else:
                        tradesDict.update({pName:{'BUY':dfSet.loc[dfSet.BS,:],'SELL':dfSet.loc[dfSet.BS==False,:]}})
                    

                trades.append(tradesDict)
        else:
            for i in range(NparamSettings):                
                tradesDict = {}
                for pName in list(self.priceMetaDict.keys()):
                    tradesDict.update({pName:None})
                trades.append(tradesDict)
                
        tradesRV = []
        if hasattr(self.OH,'keepTradesRV'):   
            for i in range(NparamSettings):

                df = pd.DataFrame(self.OH.keepTradesRV[i],columns=['datetime','oType','product','BS','vol','val'])
                df.set_index(keys=['datetime'],drop=True,inplace=True)
                df.index = df.index.map(idxMap)
                df.index.name='Date'
                df = df.loc[(df.index>=minIDX) & (df.index<=maxIDX),:]
                tradesDict = {}
                for pName in list(self.OH.RVdict.keys()):
                    dfSet = df.loc[df.loc[:,'product']==pName,:].copy()
                    tradesDict.update({pName:{'BUY':dfSet.loc[dfSet.BS,:],'SELL':dfSet.loc[dfSet.BS==False,:]}})

                tradesRV.append(tradesDict)
        else:
            for i in range(NparamSettings):                
                tradesDict = {}
                for pName in list(self.OH.RVdict.keys()):
                    tradesDict.update({pName:None})
                tradesRV.append(tradesDict)
                
        
        pospnl = []
        pospnlRV = []
        totalPnL = []
        FXrates = {}
        FXmapping = {}
        commissionMappingDict ={}
        
        
        
        for i in range(NparamSettings):                
            pospnlDict = {}
            pospnlDictT = {}
            pnlTint = []
            
            
            for pName in list(self.priceMetaDict.keys()):
                idx = plotIDX.isin(self.OH.dates[pName]).nonzero()[0]        
                
                if aggOrders:
                    df = pd.DataFrame(np.sum(self.OH.pos[self.OH.pospnlKey[pName]],axis=1),index=idx,columns=['pos'])                    
                    df['pnl'] = np.sum(self.OH.pnl[self.OH.pospnlKey[pName]],axis=1)*self.priceMetaDict[pName]['tickValues']/1000
                else:
                    df = pd.DataFrame(self.OH.pos[self.OH.pospnlKey[pName]][:,i],index=idx,columns=['pos'])                    
                    df['pnl'] = self.OH.pnl[self.OH.pospnlKey[pName]][:,i]*self.priceMetaDict[pName]['tickValues']/1000
                
                df.index.name='Date'
                
                dfTpnl = df.loc[:,['pnl']].copy() # have to get a copy before the cumsum.
#                df['pnl'] = np.cumsum(df['pnl'].values)
                
                mask = (df.index>=minIDX) & (df.index<=maxIDX)
                
                firstIDX = df.index[mask.nonzero()[0][0]]
#               
                if i ==0:
                    dfTemp = df.copy()
                    dfTemp.index = self.OH.dates[pName]
                    
                    commissionMapping = []        
                    currIDX = 0
                    Cidx = self.commissionPnL[pName].index       
                    for k,idx in enumerate(dfTemp.index):
                        if idx>Cidx[currIDX]:
                            commissionMapping.append(k-1)
                            currIDX += 1                
                    if len(commissionMapping)+1!=len(Cidx):
                        print('More than 1 index mapping missing, not expected')
                            
                    for leftIdx in range(currIDX,len(Cidx)):
                        commissionMapping.append(len(dfTemp.index)-1)
                    
                    commissionMappingDict.update({pName:np.array(commissionMapping).astype(int)})
                
                commissionPnL = np.zeros(dfTpnl.shape[0])
                if aggOrders:
                    commissionPnL[commissionMappingDict[pName]] = self.commissionPnL[pName].sum(axis=1).values
                else:
                    commissionPnL[commissionMappingDict[pName]] = self.commissionPnL[pName].iloc[:,i].values
                dfTpnl.loc[:,'pnl'] = dfTpnl.loc[:,'pnl'].values+commissionPnL
                
                # FIX FX rates here, index has been transformed and subselected, but the key is in self.OH.dates[pName]
                if self.priceMetaDict[pName]['FX']!=None:
                    if i ==0:
                        
                        FX = ReutersHub(self.priceMetaDict[pName]['FX'],pathOfData=pathOfData,altName='dummy').CLOSE.copy()
                        dfTemp = df.copy()
                        dfTemp.index = self.OH.dates[pName]
                        
                        if pName in list(self.FXtradePnL.keys()): # traded in USD instead of own currency
                            FXtradePnLMapping = []
                            
                            currIDX = 0
                            Fxidx = self.FXtradePnL[pName].index
                            
                            for k,idx in enumerate(dfTemp.index):
                                if idx>Fxidx[currIDX]:
                                    FXtradePnLMapping.append(k-1)
                                    currIDX += 1
                                    
                            if len(FXtradePnLMapping)+1!=len(Fxidx):
                                print('More than 1 index mapping missing, not expected')
                                    
                            for leftIdx in range(currIDX,len(Fxidx)):
                                FXtradePnLMapping.append(len(dfTemp.index)-1)
                            
                            FXmapping.update({pName:np.array(FXtradePnLMapping).astype(int)})
                                                    
                        dfTemp2 = pd.concat([dfTemp,FX],axis=1).fillna(method='ffill').fillna(method='bfill')
                        dfTemp = dfTemp2.loc[dfTemp.index,:]
                        
                        if self.priceMetaDict[pName]['FX'] in ['EURUSD']:
                            FXrates.update({pName:dfTemp.loc[:,'c1'].values})
                        elif self.priceMetaDict[pName]['FX'] in ['MYRUSD','CADUSD']:
                            FXrates.update({pName:(1/dfTemp.loc[:,'c1']).values})
                    
                    dfTpnl.loc[:,'pnl'] = dfTpnl.loc[:,'pnl'].values*FXrates[pName]
                    
                    if pName in list(self.FXtradePnL.keys()):
                        FXtradePnL = np.zeros(dfTpnl.shape[0])
                        if aggOrders:
                            FXtradePnL[FXmapping[pName]] = self.FXtradePnL[pName].sum(axis=1).values
                        else:
                            FXtradePnL[FXmapping[pName]] = self.FXtradePnL[pName].iloc[:,i].values
                        dfTpnl.loc[:,'pnl'] = dfTpnl.loc[:,'pnl'].values+FXtradePnL
                
                    
                if hasattr(self.OH,'lastPnl'):
                    if aggOrders:
                        addonPnLNoFX = np.sum(self.OH.lastPnl[pName])
                    else:
                        addonPnLNoFX = self.OH.lastPnl[pName][i]
                else:
                    addonPnLNoFX = 0
                    
                                     
                if hasattr(self,'histPnL'):
                    if aggOrders:
                        if pName in list(self.histPnL.columns.levels[0]):
                            addonPnLFX = np.sum(np.sum(self.histPnL.loc[:,[pName]].values))
                        else: # part of RV
                            addonPnLFX = np.sum(np.sum(self.histpnlTdropped.loc[:,[pName]].values))
                            
                        if pName in list(self.OH.lastFXpos_OH.keys()):
                            addonPnLFX += (self.DH.master_data['FX'][pName]['CLOSE']['val'][0]-self.OH.lastFXrate_OH[pName])*np.sum(self.OH.lastFXpos_OH[pName])/1000 # From Eod to first timestep the FX move is not taken into account, this fixes it
                    else:
                        if pName in list(self.histPnL.columns.levels[0]):
                            addonPnLFX = np.sum(self.histPnL.loc[:,[pName]].values[:,i])
                        else: # part of RV
                            addonPnLFX = np.sum(self.histpnlTdropped.loc[:,[pName]].values[:,i])
                        if pName in list(self.OH.lastFXpos_OH.keys()):                            
                            addonPnLFX += (self.DH.master_data['FX'][pName]['CLOSE']['val'][0]-self.OH.lastFXrate_OH[pName])*self.OH.lastFXpos_OH[pName][i]/1000 # From Eod to first timestep the FX move is not taken into account, this fixes it
                else:
                    addonPnLFX = 0
                
                df.loc[firstIDX,'pnl'] = df.loc[:firstIDX,'pnl'].sum()+addonPnLNoFX
                dfTpnl.loc[firstIDX,'pnl'] = dfTpnl.loc[:firstIDX,'pnl'].sum()+addonPnLFX
                
                df = df.loc[mask,:]
                
                df['pnl'] = np.cumsum(df['pnl'].values)
                dfTpnl = dfTpnl.loc[mask,:]
                
                pnlTint.append(dfTpnl)
                    
                pospnlDict.update({pName:df})
                pospnlDictT.update({pName:dfTpnl})
            pospnl.append(pospnlDict)
            
            if len(list(self.OH.RVdict.keys()))>0: # contains RV for which pos and pnl needs to be created/merged   
                pospnlDictRV = {}
                for rvName in list(self.OH.RVdict.keys()): 
                    for k,pName in enumerate(self.OH.RVdict[rvName]['pnl']):
                        if k ==0:
                            dfRV = pospnlDict[pName].copy()
                            dfRV['pos'] = dfRV['pos']/self.OH.RVdict[rvName]['RVvolRatio'][k]
                            dfRV['pnl'] *=0
                        
                        dfRV['pnl'] = dfRV['pnl'].values+pospnlDictT[pName]['pnl'].values
                    dfRV['pnl'] = dfRV['pnl'].cumsum()
                    pospnlDictRV.update({rvName:dfRV})
                pospnlRV.append(pospnlDictRV)
                                
            
            pnlTmerge = pd.concat(pnlTint,axis=1).fillna(0).sum(axis=1).to_frame()
            pnlTmerge.columns = ['Total pnl']
            pnlTmerge['Total pnl'] = pnlTmerge['Total pnl'].cumsum()
            totalPnL.append(pnlTmerge)
            
        
        if aggOrders:
            if hasattr(self.strat,'name'):
                name = self.strat.name
            else:
                name = 'Aggregated'
            paramCombKey = {name:0}
        else:
#            pList = [self.pnlT.columns.levels[1][i] for i in list(self.pnlT.columns.labels[1])]
            pList = self.processParamsToNames()
            paramCombKey = {paramCombName:i for i,paramCombName in enumerate(pList)}


        if hasattr(self.strat,'toPlotDict'):
            strategySignals = copy.deepcopy(self.strat.toPlotDict) # hierarchy: toPlotDict['pName']{'data': [ list of df equal to numparams],'plotType': {mapping of df colNames to type ()}}

            for key in list(strategySignals.keys()):
                dfList = strategySignals[key]['data']
                newList = []
                for df in dfList:
                    idx = plotIDX.isin(df.index).nonzero()[0]  
                    df.index = idx
                    newList.append(df)
                strategySignals[key].update({'data':newList})
        else:
            strategySignals = {}


        self.plotDict = {'dateOverrides':dateOverrides,
                        'prices': prices,'RV':RV,
                         'signals':signals,
                         'orders':orders,'ordersRV':ordersRV,
                         'trades':trades,'tradesRV':tradesRV,
                        'pospnl':pospnl,'pospnlRV':pospnlRV,
                        'totalpnl':totalPnL,
                        'paramCombKey':paramCombKey,
                        'realTrades':realTrades,
                        'strategySignals':strategySignals}
    def style(self,p):
        """Styles Bokeh plots into uniform format"""
        # Title 
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'

        # Tick labels
        p.xaxis.major_label_text_font_size = '14pt'
        p.yaxis.major_label_text_font_size = '14pt'        
        return p 
    
    def makeTradePlot(self,pName,dateOverride,paramNum,dateStart,dateEnd,origin,sigList=[]):
        """Creates the trade plots using the the cockpit."""
        if origin=='RV':
            addon = 'RV'
        else:
            addon = ''
        
        df = self.plotDict[origin][pName]
        df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
        srcP = ColumnDataSource(df)

        df = self.plotDict['orders'+addon][paramNum][pName]['BUY']
        df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
        if df.shape[0]>0:
            srcO = {'BUY':ColumnDataSource(df)}
        else:
            srcO = {'BUY':None}
        df = self.plotDict['orders'+addon][paramNum][pName]['SELL']
        df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
        if df.shape[0]>0:
            srcO.update({'SELL':ColumnDataSource(df)})
        else:
            srcO.update({'SELL':None})

        df = self.plotDict['trades'+addon][paramNum][pName]['BUY']
        df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
        if df.shape[0]>0:
            srcT = {'BUY':ColumnDataSource(df)}
        else:
            srcT = {'BUY':None}
        df = self.plotDict['trades'+addon][paramNum][pName]['SELL']
        df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
        if df.shape[0]>0:
            srcT.update({'SELL':ColumnDataSource(df)})
        else:
            srcT.update({'SELL':None})

        if pName in list(self.plotDict['strategySignals'].keys()):
            df = self.plotDict['strategySignals'][pName]['data'][paramNum].copy()    
                    
            df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
            df.index.name = 'Date'
            if df.shape[0]>0:
                
                srcS  = ColumnDataSource(df)
                sigNames = list(df.columns)
                sigPlotTypes = self.plotDict['strategySignals'][pName]['plotType']
            else:
                sigNames = []
        else:
            sigNames = []
        if (addon=='') & (self.plotDict['realTrades']==True):
            df = self.plotDict['trades'+addon][paramNum][pName]['realBUY']
            df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
            if df.shape[0]>0:
                srcRT = {'BUY':ColumnDataSource(df)}
            else:
                srcRT = {'BUY':None}
            df = self.plotDict['trades'+addon][paramNum][pName]['realSELL']
            df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
            if df.shape[0]>0:
                srcRT.update({'SELL':ColumnDataSource(df)})
            else:
                srcRT.update({'SELL':None})
            addRealTrades = True
        else:
            addRealTrades = False

        df = self.plotDict['pospnl'+addon][paramNum][pName]
        df = df.loc[(df.index>=dateStart)&(df.index<=dateEnd),:]
        srcPosPnL = ColumnDataSource(df)
        if origin=='RV':
            p = figure(plot_width = 800, plot_height = 600, y_axis_label = 'RV price units',title=pName)
        else:
            p = figure(plot_width = 800, plot_height = 600, y_axis_label = self.priceMetaDict[pName]['priceUnits'],title=pName)
        p = self.style(p)
        p.xaxis.major_label_overrides = dateOverride
        
        if origin=='RV': # adopting first leg
            pNameSel = self.OH.RVdict[pName]['pos']
        else:
            pNameSel = pName
        
        if self.priceMetaDict[pNameSel]['dType']=='C':
            Candle = False
            EoD = False
        elif self.priceMetaDict[pNameSel]['dType'] == 'OHLC':
            Candle = True
            EoD = False
        elif self.priceMetaDict[pNameSel]['dType'][:4] == 'IDM_':
            Candle = True
            EoD = True
        else: # IDR data
            Candle = False
            EoD = True

        if Candle:
            p.segment(source=srcP,x0='Date', y0='HIGH', x1='Date', y1='LOW', color="black")    
            upMove = np.array(srcP.data['OPEN'])<np.array(srcP.data['CLOSE'])
            w = 0.5

            barSRC = []
            for j in range(2):
                newSRC = {}
                for field in ['Date','OPEN','HIGH','LOW','CLOSE']:   
                    newSRC.update({field:np.array(srcP.data[field])[upMove if j ==0 else upMove==False]})

                newSRC.update({'width':np.full(len(newSRC['CLOSE']),w)})
                barSRC.append(ColumnDataSource(newSRC))

            p.vbar(source=barSRC[0],x='Date', width='width', bottom='OPEN', top='CLOSE', fill_color="#D5E1DD", line_color="black",fill_alpha=0.3)
            p.vbar(source=barSRC[1],x='Date', width='width', bottom='OPEN', top='CLOSE', fill_color="#F2583E", line_color="black",fill_alpha=0.3)
        else:
            p.line(source=srcP,y='CLOSE',x='Date',color='black')

        if EoD:
            mask = np.array(srcP.data['status'])=='CLOSE'
            df = pd.DataFrame(np.array(srcP.data['CLOSE'])[mask],index=np.array(srcP.data['Date'])[mask]+0.5,columns=['EoD'])
            df.index.name = 'Date'
            srcEoD = ColumnDataSource(df)
            p.square_x(source=srcEoD,x="Date", y="EoD", size=10, line_color="yellow", fill_color=None, line_width=2) 

        p.circle(source=srcP,x='Date',y='Roll',color='purple',size=10)

        if srcO['BUY'] != None:    
            p.segment(source=srcO['BUY'],x0='Date', y0='val', x1='index2', y1='val', color="blue") 
        if srcO['SELL'] != None:
            p.segment(source=srcO['SELL'],x0='Date', y0='val', x1='index2', y1='val', color="red") 
        if srcT['BUY'] != None:
            p.triangle(source=srcT['BUY'],y='val',x='Date',color='blue',size=10)
        if srcT['SELL'] != None:
            p.inverted_triangle(source=srcT['SELL'],y='val',x='Date',color='red',size=10)

        if addRealTrades:
            if srcRT['BUY'] != None:
                p.triangle(source=srcRT['BUY'],y='val',x='Date',color='yellow',size=15,fill_color=None, line_width=3)
            if srcRT['SELL'] != None:
                p.inverted_triangle(source=srcRT['SELL'],y='val',x='Date',color='purple',size=15,fill_color=None, line_width=3)
        colorCounter = -1
        legItems = []
        for sigName in sigNames:
            
            colorCounter = colorCounter+1
            
            if sigPlotTypes[sigName]=='line':
                c = p.line(source=srcS,y=sigName,x='Date',color=self.colors[colorCounter])
            elif sigPlotTypes[sigName]=='circle':
                c = p.circle(source=srcS,y=sigName,x='Date',color=self.colors[colorCounter])
            elif sigPlotTypes[sigName]=='dot':
                c = p.dot(source=srcS,y=sigName,x='Date',color=self.colors[colorCounter])
            elif sigPlotTypes[sigName]=='triangle':
                c = p.triangle(source=srcS,y=sigName,x='Date',color=self.colors[colorCounter])
            legItems.append((sigName, [c]))

        if len(legItems)>0:
            legend = Legend(items=legItems)
            legend.click_policy="hide"
            p.add_layout(legend)

        pPos = figure(plot_width = 800, plot_height = 300, y_axis_label = 'lots/tradeable units')
        pPos.line(source=srcPosPnL,y='pos',x='Date',color='blue')
        pPos = self.style(pPos)
        pPos.xaxis.major_label_overrides = dateOverride

        if origin=='RV':
            pPnl = figure(plot_width = 800, plot_height = 300, y_axis_label = 'PnL in RV currency')
        else:
            pPnl = figure(plot_width = 800, plot_height = 300, y_axis_label = 'PnL '+self.priceMetaDict[pName]['currency'])
        pPnl.line(source=srcPosPnL,y='pnl',x='Date',color='blue')
        pPnl = self.style(pPnl)
        pPnl.xaxis.major_label_overrides = dateOverride
        return column(p,pPos,pPnl)

    
    def emptyPrep(self):
        """Preps without run. Legacy/debug function. dataHandler.prep() is always called at the start of a run."""
        if (hasattr(self,'DH')) & (hasattr(self,'strat')):        
            self.DH.prep()
            # initialize order handler
            self.priceMetaDict = copy.deepcopy(self.DH.priceMetaDict)
            for key in list(self.priceMetaDict.keys()):
                if self.priceMetaDict[key]['skip']:
                    del self.priceMetaDict[key]

            numParams = len(self.stratParams[list(self.stratParams.keys())[0]])
            self.OH = orderHandler(self.priceMetaDict,numParams)
        else:
            print('empty prep failed')
    def replicateRollingOutOfSample(self,saveEoD=False,endDate=None):
        """Replicates rolling out of sample result by using the rollingOutofSampleBackTest dictionary loaded by loadRollingOutOfSampleBackTestResult method. 

        Parameters
        ----------
        saveEoD : bool
            Saves prod EoD file on the basis with name self.strat.name after the run. Only use when you're 100% sure of the run being correct.
        endDate : None, str in YYYY-MM-DD format
            Configures the last date for the EoD run. If the strategy needs to run in prod today, set endDate to T-2 for proper prodRunner 
            operation. Default results in T-1 or sometimes T-0 depending on download times of price data.
        """
        if hasattr(self,'rollingOutofSampleBackTest'):
            simParamDict = self.rollingOutofSampleBackTest['parameterSwitches']
            runIDX = []
            for key in list(simParamDict.keys()):
                if key not in ['master','prod_startDate','sim_startDate','leadTime','priceOrTicks','priceMetaDict','sigNames','className']:
                    runIDX.append(key)

            runIDX.sort()
            if endDate!=None:
                runIDX.append(pd.Timestamp(endDate)+pd.offsets.Day(1)) # need to add a day as the day is subtracted in a later stage
            else:
                runIDX.append(endDate)
            rollingPnL = []
            for i in range(len(runIDX)-1):

                if simParamDict['className'] not in list(globals().keys()):

                    basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/'.format(os.getlogin())
                    sys.path.insert(0,basePath+'GenericBackTesterFiles/prodSave/stratCode/')
                    globals().update({simParamDict['className']:getattr(__import__(simParamDict['className']), simParamDict['className'])})

                idxStart = str(runIDX[i]-pd.offsets.BDay(simParamDict['leadTime']))[:10]
                if runIDX[i+1] != None:
                    idxEnd = str(runIDX[i+1]-pd.offsets.Day(1))[:10] # up to, not up and untill, switch happens at the close
                else:
                    idxEnd = None
                TP = tradePack(simParamDict['priceOrTicks'])
                for pName in list(simParamDict['priceMetaDict'].keys()):
                    TP.addPrice(simParamDict['priceMetaDict'][pName]['RIC'],
                                simParamDict['priceMetaDict'][pName]['CN_YYYYMM'],
                                RollDaysPrior=simParamDict['priceMetaDict'][pName]['RollDaysPrior'],
                                dataType=simParamDict['priceMetaDict'][pName]['dType'],
                                start_date=idxStart,
                                end_date=idxEnd,
                                fromLocal=simParamDict['priceMetaDict'][pName]['fromLocal'],
                                addTradingHours=simParamDict['priceMetaDict'][pName]['addTradingHours'],
                                nameOverride=simParamDict['priceMetaDict'][pName]['nameOverride'],
                                driverName=simParamDict['priceMetaDict'][pName]['driverName'],
                                pathOfData=simParamDict['priceMetaDict'][pName]['pathOfData'],
                                factor=simParamDict['priceMetaDict'][pName]['factor'],
                                FXconv=simParamDict['priceMetaDict'][pName]['FXconv'],
                                RVvolRatio=simParamDict['priceMetaDict'][pName]['RVvolRatio'],
                                atTime=simParamDict['priceMetaDict'][pName]['atTime'],
                                skip=simParamDict['priceMetaDict'][pName]['skip'])

                TP.addStrategy(globals()[simParamDict['className']](simParamDict[runIDX[i]]))
                for sigName in simParamDict['sigNames']:
                    dfSig,signal_time = TP.strat.downloadData(sigName,startDate=idxStart)
                    try:
                        dfSig = dfSig[:idxEnd]
                    except: 
                        print('Signal end time truncation failed, possibly no data, continuing')
                        pass
                    TP.addSignal(dfSig,signal_time=signal_time)

                if i <len(runIDX)-2:
                    TP.run()
                    rollingPnL.append(TP.pnlT.sum(axis=1).to_frame('out of sample PnL')[runIDX[i]:])
                else:
                    dates = []
                    for tradeP in TP.DH.tradablePrices:
                        Nidx = len(tradeP[0].index)
                        Ne = -30 if Nidx>30 else 0
                        dates.append(tradeP[0].index[Ne].date())

                    TP.vizCheck(monitorStart=str(max(dates)),aggOrders=True)
                    rollingPnL.append(TP.pnlT.sum(axis=1).to_frame('out of sample PnL')[runIDX[i]:])

                    self.rollingOutofSampleBackTest['pnl'] = pd.concat(rollingPnL,axis=0)

                    self.rollingOutofSampleBackTest['risk metrics'] = self.riskmetricsRealBasic(self.rollingOutofSampleBackTest['pnl'].values.reshape([-1,1]))

                    TP.pnlT = TP.pnlT[runIDX[i]:]
                    TP.rollingOutofSampleBackTest = self.rollingOutofSampleBackTest
                    if saveEoD:
                        TP.saveEoD(TP.strat.name,prod=True) # this saved an EoD file which is ready to use for production
                    else:
                        return TP
        else:
            print('First load rollingOutofSampleBackTest with the loadRollingOutOfSampleBackTestResult method.')

    def overwriteParamSet(self,startDate,paramSet,newProdStartDate=False):
        """Function to include actual parameter set used instead of the default out of sample sets. Overwrites all parameter sets after the startDate with input parameter set.
        
        Parameters:
        -----------
        startDate : str
            Date in YYYY-MM-DD format
        paramSet : dict
            Parameter set of strategy model        
        newProdStartDate : bool
            If True, the prod_startDate is overwritten by the input startDate. From then onwards, the model is assumed to be running live. 
        """
        if hasattr(self,'rollingOutofSampleBackTest'):
            simParamDict = self.rollingOutofSampleBackTest['parameterSwitches']

            runIDX = []
            for key in list(simParamDict.keys()):
                if key not in ['master','prod_startDate','sim_startDate','leadTime','priceOrTicks','priceMetaDict','sigNames','className']:
                    runIDX.append(key)
            runIDX.sort()

            newTimeStamp = pd.Timestamp(startDate)

            for idx in runIDX:
                if idx>=newTimeStamp:
                    del simParamDict[idx]

            simParamDict.update({newTimeStamp:paramSet})
            if newProdStartDate:
                simParamDict['prod_startDate'] = newTimeStamp

        else:
            print('No out of sample backtest is present, either load a set with loadRollingOutOfSampleBackTestResult, or create one with rollingOutofSample method')
    def rollingOutOfSample(self,trainPeriod=750,testPeriod=125,method='bestN',N=50,optimParam=['SR',1,'add','<0.4'],out=False,getParamCombinations=False,leadTime=50,DHdict=None):

        """Constructs rolling out of sample result from full in sample results.
        
        Parameters
        ----------
        trainPeriod : int
            Amount of simulation days used for training to base parameter choices on.
        testPeriod : int
            Amount of simulation days used for testing the chosen parameter settings from the training period.
        method : str
            Sets the method of parameter combination selection. Choices are:
                - 'bestN'       : Selects the best N parameter combinations from the training period.
                - 'likelyhoodN' : Randomly selects the N parameter combinations from the training period, where the score defines the likelyhood of it being picked.
        bestN : int
            Number of parameter combinations to be picked by the method.
        optimParam : list
            Defines how the score is calculated. Rever to Generic backtester introduction.ipynb for explanation of usage.
        out : bool
            If False, a plot with RM table will be the output. If True, the output is dataframes with the PnL data and RM data.
        getParamCombinations : bool
            if True, the full list of the most recent set of parameter combinations are provided, also all parameter sets through time are made available.
        leadTime : int
            Required for saving parameter sets through time, business days required for a model to settle into reproducible operation from any time.
        DHdict : None, dict
            Optional: when loading results to perform rolling out of sample, DH has been deleted. When getParamCombinations is set to True, it requires minimal
            information from the DH object, which can be passed manually with DHdict. Same holds for the strategy class data. The fields required are (passed as keys):
                - priceOrTicks : DH.priceOrTicks
                - priceMetaDict : DH.priceMetaDict ( do note that this is not the same as priceMetaDict attribute of  the tradePack instance, which is an altered version of the original in DH)
                - sigNames : list of signal names as in DH.master_data['signals']['key']
                - className : strat.__class__.__name__ ( name of the class which is used to create an instance.)
                """
        if hasattr(self,'pnlT'):
            
            
            pnlT = self.groupPnL(mergeParams=False,mergePrices=True) # Dont think we will ever need commodity as a parameter
            
            
            X = pnlT.values
            
            IDX = pnlT.index
            if getParamCombinations:
                if DHdict!=None:
                    simParamDict = {'master':copy.deepcopy(self.stratParams),'prod_startDate':IDX[-1]+pd.offsets.Day(1),'sim_startDate':IDX[0],'leadTime':leadTime}
                    simParamDict.update(DHdict)
                else:    
                    simParamDict = {'master':copy.deepcopy(self.stratParams),'prod_startDate':IDX[-1]+pd.offsets.Day(1),'sim_startDate':IDX[0],'leadTime':leadTime,'priceOrTicks':self.DH.priceOrTicks,'priceMetaDict':self.DH.priceMetaDict,'sigNames':self.DH.master_data['signals']['key'],'className':self.strat.__class__.__name__}

            output = np.array([])
            idx = pnlT.index[trainPeriod:]
            for i in range(trainPeriod,X.shape[0],testPeriod):
                train = X[i-trainPeriod:i,:]
                test = X[i:i+testPeriod,:]

                RM = self.riskmetricsRealBasic(train)
              
                if type(optimParam[0])==type(''):
                    optimParam = [optimParam]
                
                M = None
                mask_loc = np.ones(RM.shape[0]).astype(bool)
                for trio in optimParam:
                    if trio[3] != None:
                        exec('mask_loc[RM[trio[0]].values'+trio[3] +'] = False')
                    if trio[2]=='add':
                        if M != None:
                            M = M+RM[trio[0]].values*trio[1]
                        else:
                            M = RM[trio[0]].values*trio[1]
                    elif trio[2]=='multiply':
                        if M != None:
                            M = M*RM[trio[0]].values**trio[1]
                        else:
                            M = RM[trio[0]].values**trio[1]                            
                    else:
                        print('Error, optim param not properly defined. Aborting.')
                        return
                
                mask_loc[np.isnan(M)] = False # clear NaNs
                
                ori_loc = np.arange(len(M))[mask_loc]
                M = M[mask_loc]
    
                if method == 'bestN':
                    if N>len(M):
                        print('N larger than numParams. Picking only ones left')
                        loc = ori_loc
                    else:
                        loc = ori_loc[np.argsort(M)[-N:]]
                    
                elif method =='likelihoodN':
                    if len(M)==0:
                        loc = np.array([])
                    elif len(M)==1:
                        loc = ori_loc
                    else:
                        M = M+np.min(M) # the higher the score, the better, lowest is rebased at zero (no chance to be selected)
                        K = np.cumsum(M)
                        sample = np.random.rand(N)*K[-1]

    #                     can be done faster, but no time now
    #                     KN = len(K)
    #                     loc = np.argsort(np.append(K,sample),kind='mergesort')
    #                     loc = loc-KN

                        loc = ori_loc[np.array([(K>=s).nonzero()[0][0] for s in sample])]
                else:
                    print('Currently implemented options are bestN and likelihoodN. Received: '+method )
                if len(loc)>1:
                    testSel = np.sum(test[:,loc],axis=1)
                    output = np.append(output,testSel/len(loc))
                elif len(loc)==1: 
                    testSel = test[:,loc]
                    output = np.append(output,testSel)
                else:
                    testSel = test[:,0]*0
                    output = np.append(output,testSel)
                if getParamCombinations:
                    paramSel = self.grabSubsetOfParameterSpace(loc)
                    simParamDict.update({IDX[i]:paramSel})


            print('last switch')
            print(pnlT.index[i])
            df  = pd.DataFrame(output,index=idx,columns=['out of sample PnL'])
            

            df_rm = self.riskmetricsRealBasic(df.values.reshape([-1,1]))
            
            if out:                
                if getParamCombinations:

                    self.rollingOutofSampleBackTest = {'simulation_start date':self.pnlT.index[0],
                                                       'risk metrics': df_rm,
                                                       'parameterSwitches':simParamDict,
                                                       'pnl':df,
                                                       'trainPeriod':trainPeriod,
                                                       'testPeriod':testPeriod,
                                                       'method':method,
                                                       'N':N,
                                                       'optimParam':optimParam,
                                                       'last switch':pnlT.index[i],
                                                       'approx next switch': pnlT.index[i]+pd.offsets.BDay(testPeriod)}

                    
                    return paramSel,df,df_rm
                else:
                    return df,df_rm
            else:
                
                src = ColumnDataSource(df_rm)


                colWidths = [60 for i in range(10)]
                columns = [TableColumn(field=i, title=i,width=colWidths[j],formatter=StringFormatter(text_align='right')) for j,i in enumerate(df_rm.columns)]
                data_table = DataTable(source=src, columns=columns, height=100,width=800, editable=False,index_position=None)

                p = figure(plot_width = 800, plot_height = 600,title = 'Out of sample PnL development',x_axis_type='datetime',y_axis_label='PnL [k$]')

                p.line(y=df['out of sample PnL'].cumsum(),x=df.index,color='blue',line_width=2) # name=
                p = self.style(p)
                show(column(data_table,p))

        else:
            print('First run processPnl. Aborting.')
    def grabSubsetOfParameterSpace(self,loc):
        if len(loc)>0:
            if 'volume' in list(self.stratParams.keys()):
                uLoc = np.unique(loc)
                vols = np.zeros(len(uLoc)).astype(int)
                
                for i,j in enumerate(uLoc):
                    vols[i] = len((loc==j).nonzero()[0])
                
                paramSel = copy.deepcopy(self.stratParams)
                for paramName in list(self.stratParams.keys()):
                    if paramName!='volume':
                        paramSel.update({paramName:self.stratParams[paramName][uLoc]})
                    else:
                        paramSel.update({paramName:vols})
            else:
                paramSel = copy.deepcopy(self.stratParams)
                for paramName in list(self.stratParams.keys()):
                    paramSel.update({paramName:self.stratParams[paramName][loc]})
        else:
            paramSel = {}
            for paramName in list(self.stratParams.keys()):
                paramSel.update({paramName:np.array([])})
        return paramSel
    def compareCurrentWithExpected(self,stratnameConversionDict={},nameOverride=None,tradeListCol='stratName',fillsFilenameOverride='',info=True,divOut=True,dictOut=False,fillsFile=None):
        """Function to compare model postion expectation and actual postions. Grabs data from latest tradeCockpit status file and dump of Neovest fills in excel.
        
        Parameters
        ----------
        stratnameConversionDict : dict
            Contains link between strategy name in tradePack (self.strat.name) and the label of the strategy in the online trade cockpit and Neovest trade labels.
        nameOverride : str
            overrides self.strat.name
        tradeListCol : str
            pointer to which column the strategy names column in the trades list in the online cockpit.
        fillsFilenameOverride : str
            overrides default path
        info : bool
            When True, prints out which files are loaded
        divOut : bool
            when True, shows a bokeh Div() object, when false, returns a string
        fillsFile : None or pandas Dataframe
            if dataframe is passed, it needs to be in the exact same format as the tradeCockpit trades list

        """
        # in current form only supports single commodity strategies
        if tradeListCol not in ['stratName','Strategy_ID']:
            print('improper tradeListCol, options are: stratName and Strategy_ID. Aborting')            
            return
        
        if nameOverride!=None:
            useName = nameOverride
        elif len(list(stratnameConversionDict.keys()))>0:
            if hasattr(self.strat,'name'):
                if self.strat.name in list(stratnameConversionDict.keys()):
                    useName = stratnameConversionDict[self.strat.name]
                else:
                    if info:
                        print('Strategy name not in stratnameConversionDict, name is '+self.strat.name+' and keys in dict are: '+str(list(stratnameConversionDict.keys())))
                        print('Aborting')
                    return
            else:
                if info:
                    print('Strategy has no name, aborting')
                return
        elif hasattr(self.strat,'name'):
            useName = self.strat.name
        else:
            'got no strategy name in any form. Aborting'
            return
        
        basePath = 'C:/Users/{0}/Cargill Inc/ReutersHub - HubFiles/'.format(os.getlogin())
        targetFolder = basePath+'tradingDashboard/statusFiles/'  
        
        statusFiles = os.listdir(targetFolder)
        statusFiles.sort()
        with open(targetFolder+statusFiles[-1], 'rb') as f: 
            statusFile = pickle.load(f)
        tradeList = statusFile[0]
        if info:
            print('loaded positions from: '+statusFiles[-1])
        
        currCockpitPos = tradeList.loc[tradeList.loc[:,tradeListCol]==useName,['Neovest','Qty (lots)']]
        
        if currCockpitPos.shape[0]==0:
            if info:
                print('CAUTION: No trades with '+tradeListCol+': '+useName)
            tcp = {}
        else:
            tcp = currCockpitPos.groupby(['Neovest'])['Qty (lots)'].sum().to_frame()
            tcp = tcp.loc[tcp.loc[:,'Qty (lots)']!=0,:]
            tcp = tcp['Qty (lots)'].to_dict()
            for key in list(tcp.keys()):
                if key[:2]=='PO':
                    val = tcp[key]
                    newKey = 'FC'+key
                    del tcp[key]
                    tcp.update({newKey:val})
        if type(fillsFile) == type(None):
            fillsDf = self.getNeovestFills(fillsFilenameOverride=fillsFilenameOverride,info=info)
            try:
                neoFills = fillsDf.loc[self.strat.name,:].to_frame().reset_index().dropna()
                neoFills.columns = [i if i in ['Symbol Display','Side'] else 'vol' for i in neoFills.columns]
                neoFills.loc[neoFills.Side=='Sell',neoFills.columns[-1]] *=-1
                neoFills = neoFills.loc[:,['Symbol Display','vol']]
                neoFills = neoFills.groupby(['Symbol Display'])['vol'].sum().to_frame()
                neoFills = neoFills['vol'].to_dict()            
            except:
                if info:
                    print('no trades in trades file for '+self.strat.name)
                neoFills = {}
        else:
            currCockpitPos = fillsFile.loc[fillsFile.loc[:,tradeListCol]==useName,['Neovest','Qty (lots)']]
        
            if currCockpitPos.shape[0]==0:
                if info:
                    print('CAUTION: No trades with '+tradeListCol+': '+useName)
                neoFills = {}
            else:
                neoFills = currCockpitPos.groupby(['Neovest'])['Qty (lots)'].sum().to_frame()
                neoFills = neoFills.loc[neoFills.loc[:,'Qty (lots)']!=0,:]
                neoFills = neoFills['Qty (lots)'].to_dict()
                for key in list(neoFills.keys()):
                    if key[:2]=='PO':
                        val = neoFills[key]
                        newKey = 'FC'+key
                        del neoFills[key]
                        neoFills.update({newKey:val})
        
        currentPos = {}
        
        for key in list(tcp.keys()):
            if key[:2]=='PO':
                preFix = 'FC'
            else:
                preFix = ''

            if (tcp[key]!=0):
                currentPos.update({preFix+key:tcp[key]})
        
        for key in list(neoFills.keys()):
            if (neoFills[key]!=0):
                if key in list(currentPos.keys()):
                    currentPos.update({key:neoFills[key]+currentPos[key]})
                else:
                    currentPos.update({key:neoFills[key]})
        
        expPos = {}

        RH = ReutersHub('BO',altName='dummy')
        
        for pName in list(self.OH.pospnlKey.keys()):            
            neoC = self.priceMetaDict[pName]['neovest']+RH.YYYYMMtoLabel(int(self.OH.priceSet['prices'][pName]['CONTRACTS']))
            currPos = np.sum(self.OH.pos[self.OH.pospnlKey[pName]][-1,:])
            expPos.update({neoC:currPos})
        
        diffPos = {}
        for key in list(expPos.keys()):            
            diffPos.update({key:expPos[key]})
        for key in list(currentPos.keys()):     
            if key in list(diffPos.keys()):
                diffPos.update({key:diffPos[key]-currentPos[key]})
            else:
                diffPos.update({key:-currentPos[key]})
        if info:
            textT = '<h3>Position overview: </h3>'
        else:
            textT = ''
        if dictOut:
            return {'Model':expPos,'Trade cockpit':tcp,'From file':neoFills,'Posactions':diffPos}
        allExp = list(np.unique(np.array(list(currentPos.keys())+list(expPos.keys())+list(diffPos.keys()))))
        
        for key in allExp:   
            text = ''
            text += '<b>'+key+'</b>: '
            if key in list(currentPos.keys()):
                if currentPos[key]==0:   
                    text += 'actual: FLAT'
                elif currentPos[key]>0:   
                    text += 'actual: '+str(int(np.abs(currentPos[key])))+ ' LONG '
                else:
                    text += 'actual: '+str(int(np.abs(currentPos[key])))+ ' SHORT '
            else:
                text += 'actual: FLAT '
            if key in list(expPos.keys()):
                if expPos[key]==0:   
                    text += '|| expected: FLAT '
                elif expPos[key]>0:   
                    text += '|| expected: '+str(int(np.abs(expPos[key])))+ ' LONG '
                else:
                    text += '|| expected: '+str(int(np.abs(expPos[key])))+ ' SHORT '
            else:
                text += 'expected: FLAT '
            
            if key in list(diffPos.keys()):
                if diffPos[key]==0:   
                    text += '|| action: None '
                elif diffPos[key]>0:   
                    text += '|| action: <b>BUY '+str(int(np.abs(diffPos[key])))+'</b>'
                else:
                    text += '|| action: <b>SELL '+str(int(np.abs(diffPos[key])))+'</b>'
            else:
                text += '|| action: None '
            text += '<br>'
            
            
            if key in list(currentPos.keys()):
                if (currentPos[key]==0) & (diffPos[key]==0): # do not show flat positions which are intended to be flat
                    pass
                else:
                    textT += text
            elif key in list(expPos.keys()):
                if (expPos[key]==0) & (diffPos[key]==0): # do not show flat positions which are intended to be flat
                    pass
                else:
                    textT += text
        if divOut:
            show(Div(text=text,width=800))
        else:
            return textT
        
    def run(self):
        """Runs the backtest and calculates the results. Does not support cockpit functionality as orders and trades are not saved."""
        # initiation
        
        if (hasattr(self,'DH')) & (hasattr(self,'strat')):        
            print('preparing database')
            if self.DH.prep():
                print('prep done')
            else:
                print('Aborting run')
                return
            print('prep done')
            # initialize order handler
            self.priceMetaDict = copy.deepcopy(self.DH.priceMetaDict)
            for key in list(self.priceMetaDict.keys()):
                if self.priceMetaDict[key]['skip']:
                    del self.priceMetaDict[key]
            numParams = len(self.stratParams[list(self.stratParams.keys())[0]])
            self.OH = orderHandler(self.priceMetaDict,numParams)
            
            Ntotal = len(self.DH.master_idx)
            counter=-1
            parts = 0.1
            partsCount = 1
            t0 = dt.datetime.now()
            print('started run: '+str(t0))
            while True:
                counter = counter+1
                if counter/Ntotal*100>parts*partsCount:
                    t1 = dt.datetime.now()
                    print(str(parts*partsCount) +'% done, estimated end time: '+str(t0+(t1-t0)/(parts*partsCount/100)))
                    partsCount = partsCount+1
                    parts = 10
                
                if self.DH.getNextDataPoint():
                    self.OH.newPriceInsert(self.DH.lastDataPoint)
                    self.OH.evalOrders()
                    pospnlDict = self.OH.getCurrentPosPnl()
                    
                    if self.DH.priceOrTicks=='ticks':
                        newOrders,override = self.strat.evaluate(psDict=self.DH.lastDataPoint,pospnlDict=pospnlDict)  
                    elif self.DH.priceOrTicks=='price':
                        newOrders,override = self.strat.evaluate(psDict=self.DH.fromTicksToActual(),pospnlDict=pospnlDict)  
                        
                        if 'val' in list(newOrders.keys()):
                            val = newOrders['val'].copy()
                            for i,pName in enumerate(newOrders['product']):
                                val[i,:] = np.around(val[i,:]/self.priceMetaDict[pName]['tickSize'],0)

                            newOrders.update({'val':val.astype(int)})
                        
                    newOrders.update({'timeStamp':self.DH.lastDataPoint['t']})
                    self.OH.checkValidity(newOrders,override)
                    self.OH.evalOrders(instant=True)
                    pospnlDict = self.OH.getCurrentPosPnl()
                else:
                    break       
            
            print('Run finished, processing PnL')
            self.processPnL()
            print('All done')