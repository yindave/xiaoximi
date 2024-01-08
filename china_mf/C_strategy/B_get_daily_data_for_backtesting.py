# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:39:03 2021

@author: hyin1
"""


import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import numpy as np
import pdb
import os.path
import utilities.mathematics as umath

from backtester.backtester import BACKTESTER
from fql.util import bbg_to_fs,fql_date
from fql.fql import Factset_Query
from blp.bdx import bdh

bt_path=em.path_static_list.replace('static_list.csv','')+'strategy\\'
turnover_path=em.path_static_list.replace('static_list.csv','')+'process_data\\'



# get tickers from signals to use
signals=feather.read_dataframe(bt_path+'signals.feather').set_index(['date','ticker'])[['score']]
turnover=feather.read_dataframe(turnover_path+'turnover.feather').set_index('date')/1000 # in US$ mn
adv_3m_musd=turnover.applymap(lambda x: np.nan if x==0 else x).rolling(63,min_periods=5).mean()
adv_3m_musd=adv_3m_musd.resample('M').last()
adv_3m_musd=adv_3m_musd.stack()
adv_3m_musd.index.names=['date','ticker']
signals_to_use=signals.join(adv_3m_musd.rename('adv_3m_musd')).reset_index()

# we will just download price info for nearly all stocks
top_N=99999
min_adv=0

signals_to_use=signals_to_use[(signals_to_use['adv_3m_musd']>=min_adv)]
signals_to_use['rank_top']=signals_to_use.groupby('date')['score'].rank(ascending=False,method='min')
signals_to_use['rank_bottom']=signals_to_use.groupby('date')['score'].rank(method='min')


signals_quick_load=signals_to_use[
        ((signals_to_use['rank_top']<=top_N)| (signals_to_use['rank_bottom']<=top_N))
            ].set_index(['date','ticker'])['score'].unstack()

tickers_from_signal=signals_quick_load.columns.tolist()


# get tickers from compo
idx_list=['SHSZ300 Index','MXCN1A Index','SH000905 Index','XIN9I Index']
compo_collector=[]
for idx in idx_list:
    compo_collector.append(uc.load_compo(idx))
tickers_from_compo=bbg_to_fs(pd.concat(compo_collector).reset_index().groupby('ticker').last().index.tolist())

# get full ticker list
tickers_all=list(set(tickers_from_signal+tickers_from_compo))

# download the mkt data
fq=Factset_Query()
mkt_data=fq.get_ts(tickers_all,['px_last','vwap'],
          fql_date(signals_to_use['date'].min()),
          fql_date(um.yesterday_date()),
          )

# reload mkt and vwap and add the benchmark level to it
idx_list=['CSIR0300 Index','MSCHANL Index','CSIR0905 Index','TXIN9IC Index']
idx_px=bdh(idx_list,['px_last'],mkt_data.index[0],um.yesterday_date())['px_last'].unstack().T
idx_px=idx_px.reindex(mkt_data.index).fillna(method='ffill')

idx_px=idx_px.rename(columns={'CSIR0300 Index':'CSI300',
                              'MSCHANL Index':'MSCI_ChinaA',
                              'CSIR0905 Index':'CSI500',
                               'TXIN9IC Index':'A50',
                              })


feather.write_dataframe(pd.concat([mkt_data['px_last'],idx_px],axis=1).reset_index(),
                        bt_path+'mkt.feather')

feather.write_dataframe(pd.concat([mkt_data['vwap'],idx_px],axis=1).reset_index(),
                        bt_path+'vwap.feather')

feather.write_dataframe(signals_quick_load.reset_index(),bt_path+'signals_quick_load.feather')













