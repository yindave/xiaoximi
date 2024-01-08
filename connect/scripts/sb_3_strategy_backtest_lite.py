# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 13:35:49 2021

@author: davehanzhang
"""
import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
from fql.fql import Factset_Query
import utilities.mathematics as umath
from blp.util import get_bbg_nice_compo_hist_no_limit,load_compo,get_bbg_usual_col
from fql.util import bbg_to_fs,fs_to_bbg,fql_date
from blp.bdx import bdh,bdp,bds
import feather
from datetime import datetime
import matplotlib.pyplot as plt

from excess_return.excess_return import Alpha_Model,Carhart
from connect.connect import STOCK_CONNECT

from joblib import load, dump
from backtester.backtester import BACKTESTER
import os
from multiprocessing import Process


def run(adv_level):
    
    print ('working on adv_level %s' % (adv_level))
    
    #adv_levels=[3,5,7.5,10]
    sizes=[15] # we want to have 30-40 stocks at any time

    shift_no=1
    
    
    sc=STOCK_CONNECT(direction='sb')
    window=20
    tranches=4
    alpha_model_dict={'alpha':'alpha','carhart':'carhart'}
    


    
    # just dummy load one time
    alpha_model_name='alpha'
    path_i=sc.path+'trades_and_models\\%s\\' % (alpha_model_name)
    
    # bt obj for tranches
    bt=BACKTESTER(path_i,bps=0) 
    bt.load_data(signal_name='mkt') 
    # add some extra points for easy shift/aggregate later
    mkt_extra=pd.DataFrame(index=pd.date_range(bt.mkt.index[-1]+1*pd.tseries.offsets.BDay(),bt.mkt.index[-1]+30*pd.tseries.offsets.BDay(),freq='B'),
                                columns=bt.mkt.columns).fillna(bt.mkt.iloc[-1])
    bt.mkt=pd.concat([bt.mkt,mkt_extra],axis=0)
    bt.mkt['cash']=1
    bt.mkt.index.name='date'
    bt.vwap=bt.mkt
    
    # bt obj for actual execution
    bt_actual=BACKTESTER(path_i,bps=20)
    bt_actual.load_data(signal_name='mkt')
    bt_actual.mkt.index.name='date'
    bt_actual.vwap.index.name='date'
    bt_actual.mkt['cash']=1
    bt_actual.vwap['cash']=1
    
    all_strategy_perf_collector=[]
    all_strategy_signal_collector=[]
    
    for alpha_model_name,y in alpha_model_dict.items():
    
        path_i=sc.path+'trades_and_models\\%s\\' % (alpha_model_name)
        signal_raw=feather.read_dataframe(path_i+'prediction_score_and_input_details.feather')
        
        # for adv_level in adv_levels:
        signal_i=signal_raw[signal_raw['adv']>=adv_level].set_index(['date','ticker'])['score_prediction'].unstack()
        signal_extra=pd.DataFrame(index=pd.date_range(signal_i.index[-1]+1*pd.tseries.offsets.BDay(),signal_i.index[-1]+30*pd.tseries.offsets.BDay(),freq='B'),
                        columns=signal_i.columns).fillna(signal_i.iloc[-1])
        signal_i=pd.concat([signal_i,signal_extra],axis=0)
        signal_i.index.name='date'
        signal_i=signal_i.loc[:bt.mkt.index[-1]]
        
        for size in sizes:
            # get each shift first
            shares_collector=[]
            to_collector=[] # for trading date
            shifts=[int(x) for x in np.arange(shift_no,window,window/tranches)]
            for shift in shifts:
                bt.signal=signal_i.iloc[shift:].iloc[::window].copy()
                perf_pct,perf_abs,shares_overtime,to,hlds=bt.run(bt.signal.index[0],size,'HSCEI',)
                shares_overtime=shares_overtime.shift(-1).loc[:signal_i.index[-1]].unstack().rename('shares').reset_index()
                shares_overtime['shift']=shift
                shares_collector.append(shares_overtime)
                to['shift']=shift
                to_collector.append(to)
                
            shares_overtime=pd.concat(shares_collector)
            to_all=pd.concat(to_collector).reset_index()
            
            # then obtain the wgt target
            shares_overtime_s=shares_overtime[shares_overtime['direction']=='s'].copy()
            shares_overtime_s['shares']=shares_overtime_s['shares']*(-1)
            shares_overtime_rest=shares_overtime[shares_overtime['direction']!='s'].copy()
            shares_overtime_combined=pd.concat([shares_overtime_rest,shares_overtime_s],axis=0).groupby(['strategy','direction','ticker','date'])['shares'].sum().unstack().T
        
            port_shares_l=shares_overtime_combined['l-mkt']['l']
            port_l=(port_shares_l.multiply(bt.mkt.reindex(port_shares_l.index,axis=0).reindex(port_shares_l.columns,axis=1))
                    .apply(lambda x: x/x.abs().sum(),axis=1).applymap(lambda x: np.nan if x==0 else x))
        
            port_shares_ls_l=shares_overtime_combined['l-s']['l']
            port_ls_l=(port_shares_ls_l.multiply(bt.mkt.reindex(port_shares_ls_l.index,axis=0).reindex(port_shares_ls_l.columns,axis=1))
                    .apply(lambda x: x/x.abs().sum(),axis=1).applymap(lambda x: np.nan if x==0 else x))
        
            port_shares_ls_s=shares_overtime_combined['l-s']['s']
            port_ls_s=(port_shares_ls_s.multiply(bt.mkt.reindex(port_shares_ls_s.index,axis=0).reindex(port_shares_ls_s.columns,axis=1))
                    .apply(lambda x: x/x.abs().sum(),axis=1).applymap(lambda x: np.nan if x==0 else x))
        
            port_ls=umath.normalize_ls_port(pd.concat([port_ls_l,port_ls_s],axis=1))
        
            rebalance_dates=to_all.groupby('date').last().index
            
            # shift by 1 HK trading day (on the index) to factor in the execution lag
            port_l=port_l.reindex(rebalance_dates).loc[:signal_raw['date'].max()]
            port_ls=port_ls.reindex(rebalance_dates).loc[:signal_raw['date'].max()]
            
            execution_index=bt.mkt.loc[:signal_raw['date'].max()+pd.tseries.offsets.BDay()].index # daily index
            temp=port_l.reindex(execution_index).shift(1)
            port_l_execution=temp.loc[temp.count(1)[temp.count(1)!=0].index]
            temp=port_ls.reindex(execution_index).shift(1)
            port_ls_execution=temp.loc[temp.count(1)[temp.count(1)!=0].index]
            
            
            # use bt_actual with weight target to do the calculation
            strategies_dict={'l_vs_mkt':port_l_execution,'l_vs_s':port_ls_execution}
            for strategy_type,strategy_signal in strategies_dict.items():
                # maybe the below loc[:mkt last date] is necessary
                bt_actual.signal=strategy_signal.loc[:bt_actual.mkt.index[-1]].copy()
                perf_pct,perf_abs,shares_overtime,to,hlds=bt_actual.run_with_weight(bt_actual.signal.index[0],len(bt_actual.signal.columns),
                                                                                    'HSCEI' if strategy_type=='l_vs_mkt' else 'cash',
                                                                                   bt_actual.signal,normalize_wgt=False
                                                                                   )
                # collect perf
                perf_i=perf_pct['l-mkt_net'].rename('perf').to_frame().reset_index()
                perf_i['strategy_type']=strategy_type
                perf_i['size']=size
                perf_i['adv_level']=adv_level
                perf_i['alpha_model']=alpha_model_name
                perf_i['shift_no']=shift_no
                all_strategy_perf_collector.append(perf_i)
                
                # collect signal
                wgt_i=strategy_signal.stack().rename('wgt').reset_index()
                wgt_i['strategy_type']=strategy_type
                wgt_i['size']=size
                wgt_i['adv_level']=adv_level
                wgt_i['alpha_model']=alpha_model_name
                wgt_i['shift_no']=shift_no
                all_strategy_signal_collector.append(wgt_i)
                
                print ('finish %s adv:%s size:%s implementation:%s (shift_no %s)' % (alpha_model_name,adv_level,size,strategy_type,shift_no))
                
            
                 
    perf_all=pd.concat(all_strategy_perf_collector)
    wgt_all=pd.concat(all_strategy_signal_collector)
    
    dump_name='adv_%s' % (adv_level)
    
    dump_path=sc.path+'trades_and_models\\dump_for_bt_lite\\'
    feather.write_dataframe(perf_all, dump_path +dump_name+'_perf.feather')
    feather.write_dataframe(wgt_all, dump_path +dump_name+'_wgt.feather')
    
    

if __name__ == "__main__":
    # parallel run
    adv_levels=[3,5,7.5,10]

    p_collector=[]
    for adv_level in adv_levels:
        p=Process(target=run,args=(adv_level,))
        p_collector.append(p)
        p.start()
    for p in p_collector:
        p.join()














