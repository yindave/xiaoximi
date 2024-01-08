# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:24:40 2021

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

from multiprocessing import Process



bt_path=em.path_static_list.replace('static_list.csv','')+'strategy\\'
bt_dump_path=em.path_static_list.replace('static_list.csv','')+'strategy\\axioma\\'
turnover_path=em.path_static_list.replace('static_list.csv','')+'process_data\\'

# bt is used for calculating allocation target
bt=BACKTESTER(bt_path,bps=15)
bt.load_data(signal_name='signals') # a fake load only for mkt and vwap
bt.mkt['cash']=1
bt.vwap['cash']=1
bt.mkt['cash_ls']=1
bt.vwap['cash_ls']=1
bt.vwap=bt.mkt.copy()  # using vwap lead to worse perf mainly from 2015 period, just assume last price


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
signals_to_use_raw=signals.join(adv_3m_musd.rename('adv_3m_musd')).reset_index()


bench_map={
            'CSI300':'SHSZ300 Index',
            'A50':'XIN9I Index',
            'MSCI_ChinaA':'MXCN1A Index',
           }


def run(bench,max_deviation_level,top_n):
    print ('working on %s %s %s' % (bench,max_deviation_level, top_n))
    compo=uc.load_compo(bench_map[bench]).map(lambda x: np.nan if abs(x)<=0.000001 else x).dropna().unstack().fillna(0)
    compo.index=compo.index.map(lambda x: x-5*pd.tseries.offsets.BDay())
    compo.loc[um.today_date()+pd.tseries.offsets.DateOffset(months=3)]=compo.iloc[-1]
    compo=compo.resample('M').last().fillna(method='ffill').applymap(lambda x: np.nan if x==0 else x)
    compo.columns=bbg_to_fs(compo.columns)
    compo=compo.stack()
    compo.index.names=['date','ticker']
    signals_to_use=signals_to_use_raw.set_index(['date','ticker']).join(compo.rename('idx_wgt'))
    signals_to_use=signals_to_use.reset_index()
    signals_to_use=signals_to_use[signals_to_use['idx_wgt'].fillna(-1)>0].set_index('date')
    signals_to_use['score_avg']=signals_to_use.reset_index().groupby('date')['score'].mean()
    signals_to_use['deviation']=signals_to_use['score']-signals_to_use['score_avg']
    signals_to_use['deviation_abs']=signals_to_use['deviation'].abs()
    signals_to_use['deviation_max']=signals_to_use.reset_index().groupby('date')['deviation_abs'].max()
    signals_to_use['multiplier']=signals_to_use['deviation_max'].map(lambda x: max_deviation_level/x)
    signals_to_use['deviation_wgt']=signals_to_use['deviation']*signals_to_use['multiplier']
    signals_to_use['wgt']=signals_to_use['idx_wgt']+signals_to_use['deviation_wgt']
    signals_to_use['wgt']=signals_to_use['wgt'].map(lambda x: max(0,x))
    signals_to_use=signals_to_use.reset_index()
    signals_to_use['wgt_rank']=signals_to_use.groupby('date')['wgt'].rank(ascending=False,method='min')
    signals_to_use=signals_to_use[(signals_to_use['wgt']!=0) & (signals_to_use['wgt_rank']<=top_n)]
    signals_to_use['wgt']=signals_to_use.groupby('date')['wgt'].apply(lambda x: x/x.sum())
    wgt_target=signals_to_use.set_index(['date','ticker'])['wgt'].unstack()
    wgt_target.index=wgt_target.index.map(lambda x: x-2*pd.tseries.offsets.BDay())
    wgt_target=wgt_target.fillna(-999).resample('BM').last().applymap(lambda x: np.nan if x==-999 else x)
    bt.signal=wgt_target#.iloc[shift:].iloc[::3]
    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(bt.mkt.index[0],
                                                                 len(bt.signal.columns),
                                                                 bench,
                                                                 bt.signal,normalize_wgt=False)
    perf_pct['deviation']=max_deviation_level
    perf_pct['top_n']=top_n
    perf_pct['bench']=bench
    dump_name='GW_perf_Bench_%s_Deviation_%s_TopN_%s.feather' % (bench,max_deviation_level,top_n)
    feather.write_dataframe(perf_pct.reset_index(),bt_dump_path+dump_name)

    dump_name='GW_wgt_Bench_%s_Deviation_%s_TopN_%s.feather' % (bench,max_deviation_level,top_n)
    compo_wgt=compo.unstack().fillna(0).resample('D').last().fillna(method='ffill').reindex(wgt_target.index)
    compo_wgt=compo_wgt.applymap(lambda x: np.nan if x==0 else -x)
    wgt_target_combined=umath.normalize_ls_port(pd.concat([wgt_target,compo_wgt],axis=1))
    feather.write_dataframe(wgt_target_combined.reset_index(),bt_dump_path+dump_name.replace('GW_wgt','GW_wgt_for_Axioma'))
    feather.write_dataframe(wgt_target.reset_index(),bt_dump_path+dump_name)
    feather.write_dataframe(bt.mkt.reset_index(),bt_dump_path+'mkt.feather')

if __name__ == "__main__":


    # parallel run
    benches=['CSI300','MSCI_ChinaA']
    max_deviation_levels_dict={'CSI300':0.03,'MSCI_ChinaA':0.01}
    top_ns_dict={'CSI300':100,'MSCI_ChinaA':300}
#    run('CSI300',0.03,75)
#
#

    p_collector=[]
    for bench in benches:
        max_deviation_level=max_deviation_levels_dict[bench]
        top_n=top_ns_dict[bench]
        p=Process(target=run,args=(bench,max_deviation_level,top_n))
        p_collector.append(p)
        p.start()

    for p in p_collector:
        p.join()




#collector_perf=[]
#collector_target_shares=[]
#for shift in [0,1,2]:
#
#    print ('working on shift %s' % (shift))
#    bt.signal=wgt_target#.iloc[shift:].iloc[::3]
#    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(bt.mkt.index[0],
#                                                                 len(bt.signal.columns),
#                                                                 bench,
#                                                                 bt.signal,normalize_wgt=False)
#
#    '''
#    Why not taking avg of the 3 tranches but do monthly rebalance directly?
#    because we want to closely track the index compo change
#     (real reaons: b)
#    '''
#
#    asdf
#
#    perf_pct=perf_pct.reset_index()
#    perf_pct['shift']=shift
#    perf_pct['bench']=bench
#
#
#    directions=['l']
#    shares_collector=[]
#    for direction in directions:
#        shares_i=shares_overtime['l-mkt'][direction].shift(-1)
##        if direction=='s' :
##            shares_i=shares_i*(-1)
#        shares_i=shares_i.stack().rename('shares').to_frame()
#        shares_i['direction']=direction
#        shares_collector.append(shares_i)
#    shares_all=pd.concat(shares_collector)
#    shares_all['shift']=shift
#    shares_all['bench']=bench
#
#    collector_perf.append(perf_pct)
#    collector_target_shares.append(shares_all)
#
#
#
#perf_all=pd.concat(collector_perf).reset_index()
#shares_target=pd.concat(collector_target_shares).reset_index()
#
#shares_target_net=shares_target.groupby(['date','ticker'])['shares'].sum()
#shares_target_net=shares_target_net[shares_target_net!=0].unstack()
#wgt_target_actual=umath.normalize_ls_port(shares_target_net.multiply(bt.mkt))
#wgt_target_actual=wgt_target_actual.fillna(0).resample('BM').last().applymap(lambda x: np.nan if x==0 else x)
#
#
#bt_actual.signal=wgt_target_actual.loc[:bt_actual.mkt.index[-1]].copy()
#perf_pct,perf_abs,shares_overtime,to,hlds=bt_actual.run_with_weight(
#            bt_actual.mkt.index[0],#start date
#            len(wgt_target_actual.columns),# size
#            bench,# benchmark
#            wgt_target_actual,# wgt, LS
#            normalize_wgt=False)
#
#
#
#






















