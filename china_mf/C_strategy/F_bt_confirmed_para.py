# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 17:12:07 2021

@author: hyin1
"""



'''
For LS: we just get the top vs. bottom 50 target weight over time
For L vs. CSI300: we get the target long wgt

We will also output the Axioma friendly wgt matrix for quick attribution analysis
'''

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

from multiprocessing import Process

bt_path=em.path_static_list.replace('static_list.csv','')+'strategy\\'
bt_dump_path=em.path_static_list.replace('static_list.csv','')+'strategy\\axioma\\'
turnover_path=em.path_static_list.replace('static_list.csv','')+'process_data\\'

# bt is used for calculating allocation target
bt=BACKTESTER(bt_path,bps=0)
bt.load_data(signal_name='signals') # a fake load only for mkt and vwap
bt.mkt['cash']=1
bt.vwap['cash']=1
bt.mkt['cash_ls']=1
bt.vwap['cash_ls']=1
bt.vwap=bt.mkt.copy() # just assume last price

bt_actual=BACKTESTER(bt_path,bps=15)
bt_actual.load_data(signal_name='signals')
bt_actual.mkt['cash']=1
bt_actual.vwap['cash']=1
bt_actual.mkt['cash_ls']=1
bt_actual.vwap['cash_ls']=1
bt_actual.vwap=bt_actual.vwap.fillna(bt_actual.mkt)
bt_actual.vwap=bt_actual.mkt.copy() # just assume last price

bt.mkt.loc[um.today_date()+pd.tseries.offsets.DateOffset(months=3)]=np.nan
bt.vwap.loc[um.today_date()+pd.tseries.offsets.DateOffset(months=3)]=np.nan
bt.mkt=bt.mkt.resample('B').last().fillna(method='ffill')
bt.vwap=bt.vwap.resample('B').last().fillna(method='ffill')

signals=feather.read_dataframe(bt_path+'signals.feather').set_index(['date','ticker'])[['score']]
turnover=feather.read_dataframe(turnover_path+'turnover.feather').set_index('date')/1000 # in US$ mn
adv_3m_musd=turnover.applymap(lambda x: np.nan if x==0 else x).rolling(63,min_periods=5).mean()
adv_3m_musd=adv_3m_musd.resample('M').last()
adv_3m_musd=adv_3m_musd.stack()
adv_3m_musd.index.names=['date','ticker']
signals_to_use=signals.join(adv_3m_musd.rename('adv_3m_musd')).reset_index()


def run(size,bench,min_adv):
    collector_perf=[]
    collector_target_shares=[]
    for shift in [0,1,2]:
        print ('worling on shift %s' % (shift))
        bt.signal=signals_to_use[signals_to_use['adv_3m_musd']>=min_adv].set_index(['date','ticker'])['score'].unstack()
        bt.signal.loc[bt.signal.index[-1]+pd.tseries.offsets.DateOffset(months=1)]= bt.signal.loc[bt.signal.index[-1]]
        # fix the signal date
        bt.signal.index=bt.signal.index.map(lambda x: x-2*pd.tseries.offsets.BDay())
        bt.signal=bt.signal.fillna(-999).resample('BM').last().applymap(lambda x: np.nan if x==-999 else x)
        #bt.signal=bt.signal.loc[:bt.mkt.index[-1]]
        bt.signal=bt.signal.iloc[shift:].iloc[::3]
        perf_pct,perf_abs,shares_overtime,to,hlds=bt.run( bt.mkt.index[0],size,bench)
        perf_pct=perf_pct.reset_index()
        perf_pct['size']=size
        perf_pct['shift']=shift
        perf_pct['bench']=bench
        perf_pct['adv']=min_adv
        directions=['l','s']
        shares_collector=[]
        for direction in directions:
            shares_i=shares_overtime['l-s'][direction].shift(-1)
            if direction=='s' :
                shares_i=shares_i*(-1)
            shares_i=shares_i.stack().rename('shares').to_frame()
            shares_i['direction']=direction
            shares_collector.append(shares_i)
        shares_all=pd.concat(shares_collector)
        shares_all['size']=size
        shares_all['shift']=shift
        shares_all['bench']=bench
        shares_all['adv']=min_adv
        collector_perf.append(perf_pct)
        collector_target_shares.append(shares_all)
    perf_all=pd.concat(collector_perf).reset_index()
    shares_target=pd.concat(collector_target_shares).reset_index()
    shares_target_net=shares_target.groupby(['date','ticker'])['shares'].sum()
    shares_target_net=shares_target_net[shares_target_net!=0].unstack()
    wgt_target=umath.normalize_ls_port(shares_target_net.multiply(bt.mkt))
    wgt_target=wgt_target.fillna(0).resample('BM').last().applymap(lambda x: np.nan if x==0 else x)
    if bench!='cash_ls': # then we will calculate long vs. benchmark; otherwise if it's cash_ls then we compute LS return
        wgt_target=wgt_target[wgt_target>0].stack().unstack()
    bt_actual.signal=wgt_target.loc[:bt_actual.mkt.index[-1]].copy()
    perf_pct,perf_abs,shares_overtime,to,hlds=bt_actual.run_with_weight(
                bt_actual.mkt.index[0],#start date
                len(wgt_target.columns),# size
                bench,# benchmark
                wgt_target,# wgt, LS
                normalize_wgt=False)
    # dump results
    dump_name=bt_dump_path+'Size_%s_Bench_%s_ADV_%s_FileType_%s.feather'
    feather.write_dataframe(perf_all,dump_name % (size,bench,min_adv,'GenericPerf'))
    feather.write_dataframe(wgt_target.reset_index(),dump_name % (size,bench,min_adv,'WgtTarget'))
    feather.write_dataframe(perf_pct.reset_index(),dump_name % (size,bench,min_adv,'ActualPerf'))
    feather.write_dataframe(to.reset_index(),dump_name % (size,bench,min_adv,'ActualTO'))



if __name__ == "__main__":


    # parallel run
    sizes=[50] #4
    benches=['cash_ls','cash']#['CSI300','MSCI_ChinaA','A50','cash','cash_ls'] #4
    min_advs=[10] #
    p_collector=[]
    for size in sizes:
        for bench in benches:
            for min_adv in min_advs:
                p=Process(target=run,args=(size,bench,min_adv))
                p_collector.append(p)
                p.start()
    for p in p_collector:
        p.join()


























