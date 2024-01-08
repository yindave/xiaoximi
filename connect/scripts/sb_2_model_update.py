# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:05:10 2021

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
from fql.fql import Factset_Query
import pdb

fq=Factset_Query()

def update_model(): # wrap in one func for easy call from the all_in_one script
    #---- load data
    sc=STOCK_CONNECT(direction='sb')
    path=sc.path+"trades_and_models\\ebm_explore\\"
    
    sc.load_db()
    alpha=Alpha_Model.Load_Model_Quick('SB')
    carhart=Carhart.Load_Model_Quick('SB')
    
    
    #---- fixed parameters
    min_adv_universe=3
    window=20
    clean_index=sc.db_clean.index
    data_shift=3
    
    
    #---- prepare X inputs
    holdings=sc.db_clean['holdings_musd'].copy()
    holdings.at[um.today_date()]=holdings.iloc[-1]
    holdings_mask=holdings.fillna(0).applymap(lambda x: True if x==0 else False)
    
    flow=sc.db_clean['flow_musd'].copy()
    flow.at[um.today_date()]=0
    flow=flow.mask(holdings_mask)
    
    turnover=(sc.db_clean['turnover']*sc.db_clean['fx']).applymap(lambda x: np.nan if x==0 else x)/1000
    turnover=turnover.fillna(0)
    turnover.at[um.today_date()]=np.nan
    turnover=turnover.mask(holdings_mask)
    adv=turnover.rolling(63,min_periods=1).mean().shift(1)
    adv_mask=adv.applymap(lambda x: True if x<min_adv_universe else False)
    
    turnover_mask=turnover.applymap(lambda x: True if x==0 else False).copy()
    trade_status_today=bdp(fs_to_bbg(holdings_mask.columns.tolist()),['TRADE_STATUS'])['TRADE_STATUS']
    trade_status_today.index=bbg_to_fs(trade_status_today.index)
    turnover_mask.at[um.today_date()]=trade_status_today.map(lambda x: True if x=='N' else False)
    
    validity_mask=(holdings_mask.stack().rename('holdings').to_frame()
        .join(adv_mask.stack().rename('adv'))
        .join(turnover_mask.stack().rename('turnover'))).applymap(lambda x: 1 if x else 0).sum(1).unstack().applymap(lambda x: True if x>=1 else False)
    
    intensity=flow.rolling(window,min_periods=1).sum().shift(data_shift).mask(validity_mask)
    consistency=(flow.applymap(lambda x: np.nan if x<=0 else x).rolling(window).count()/window).shift(data_shift).mask(validity_mask)
    #rsi=umath.get_RSI(flow,window,method='ema').shift(data_shift).mask(validity_mask)
    
    to_fit_X=(intensity.stack().rename('intensity').to_frame()
        .join(consistency.stack().rename('consistency'),how='outer')
        #.join(rsi.stack().rename('rsi'),how='outer')
             )
    sector=sc.db_clean['sector'].iloc[-1].rename('sector')
    to_fit_X=to_fit_X.reset_index().set_index('ticker').join(sector).reset_index().set_index(['date','ticker'])
    
    #---- prepare Y inputs
    rolling_excess_return_alpha=alpha.get_excess_return_rolling(window,reindex=[True,clean_index]).xs(key='excess',level='factor',axis=1)
    rolling_excess_return_carhart=carhart.get_excess_return_rolling(window,reindex=[True,clean_index]).xs(key='excess',level='factor',axis=1)
    
    to_fit_Y=(rolling_excess_return_alpha.shift(-window).stack().rename('alpha').to_frame()
        .join(rolling_excess_return_carhart.shift(-window).stack().rename('carhart'),how='outer')
    )
    
    #---- combine training set
    to_fit_all=to_fit_X.join(to_fit_Y,how='left').reset_index()
    to_fit_all=to_fit_all.join(to_fit_all.groupby(['date','sector']).rank(pct=True,method='min'),lsuffix='_abs')
    
    to_fit_all['score']=to_fit_all['intensity']*to_fit_all['consistency']
    #to_fit_all['score_new']=to_fit_all['intensity']*to_fit_all['rsi']
    to_fit_all['score']=to_fit_all['score'].fillna(0.5)
    #to_fit_all['score_new']=to_fit_all['score_new'].fillna(0.5)
    to_fit_all=to_fit_all.set_index(['date','ticker']).join(adv.stack().rename('adv'),how='left').reset_index()
    
    feather.write_dataframe(to_fit_all,sc.path+'trades_and_models\\to_fit_all.feather')
    
    
    #---- fitting models
    alpha_model_dict={'alpha':'alpha','carhart':'carhart'}
    Xs=['score']
    
    model_starting_date=datetime(2014,11,30)
    
    for alpha_model_name,y in alpha_model_dict.items():
        
            path_i=sc.path+'trades_and_models\\%s\\' % (alpha_model_name)
            
            to_fit=to_fit_all[['date','ticker',y]+Xs].dropna()
            dates=to_fit.groupby('date').last().index
            for date in dates:
                if date>=model_starting_date:
                    dump_name_i=path_i+'models\\%s.joblib' % (date.strftime('%Y-%m-%d'))
                    if not os.path.isfile(dump_name_i):
                        print ('fitting model for %s on %s' % (alpha_model_name,date.strftime('%Y-%m-%d')))
                        to_fit_i=to_fit[to_fit['date']<=date]                   
                        ebm=umath.get_ebm_initialized(n_jobs=1) # here we have a bug, if n_jobs!=1 then it will eat up all your memory
                        ebm.fit(to_fit_i[Xs],to_fit_i[y])
                        coef_rank_i,shape_i=umath.get_ebm_coef_rank_and_shape(ebm)
                        dump(ebm,dump_name_i)
                        print ('finish fitting model for %s on %s' % (alpha_model_name,date.strftime('%Y-%m-%d')))
                    else:
                        print ('model already dumped for %s on %s' % (alpha_model_name,date.strftime('%Y-%m-%d')))
                    
    
    #---- get prediction score and model shape
    smooth=5 # for shape display only
    columns_to_use=['date','ticker','sector','adv','intensity_abs','consistency_abs','intensity','consistency','score_prediction']
    
    hedge_legs=['HSCEI Index','HSI Index']
    hedge_legs_total_ret=[uc.total_ret_index_map[x] for x in hedge_legs]
    hedge_levels=bdh(hedge_legs_total_ret,['px_last'],datetime(2014,11,1),um.today_date())['px_last'].unstack().T.rename(columns=uc.total_ret_index_map_r)
    hedge_levels.columns=hedge_levels.columns.map(lambda x: x.replace(' Index',''))
    hedge_levels.at[um.today_date()]=np.nan # we don't have today's price for total ret index
    hedge_levels=hedge_levels.fillna(method='ffill')
    
    mkt_data_new=fq.get_ts(sc.db_clean['px_last'].columns.tolist(),
                           ['px_last','vwap'],
                           start=fql_date(sc.db_clean['px_last'].index[0]),
                           end=fql_date(um.today_date()))
    mkt_data_new=mkt_data_new.reindex(hedge_levels.index)
    
    vwap=mkt_data_new['vwap'].join(hedge_levels,how='left').fillna(method='ffill')
    px_last=mkt_data_new['px_last'].join(hedge_levels,how='left').fillna(method='ffill')
    
    Xs=['score']
    for alpha_model_name,y in alpha_model_dict.items():
    
        
        path_i=sc.path+'trades_and_models\\%s\\' % (alpha_model_name)
        path_model_i=path_i+'models\\'
        
        # load individual model dump
        dumped_models=um.iterate_csv(path_model_i,iterate_others=[True,'.joblib'])
        model_series=pd.Series(index=to_fit_all.groupby('date').last().index,dtype=object)
        for dumped_model in dumped_models:
            model_series[pd.to_datetime(dumped_model)]=load(path_model_i+'%s.joblib' % (dumped_model))
        model_series=model_series.sort_index()
        model_series=model_series.shift(window+2).dropna() # +2 should be enough
        
        # get the nice shape
        shape_collector=[]
        for dt_i in model_series.index:
            ebm_i=model_series[dt_i]
            coef_rank,shape=umath.get_ebm_coef_rank_and_shape(ebm_i)
            shape['date']=dt_i
            shape_collector.append(shape)
        shape_all=pd.concat(shape_collector).reset_index().set_index(['x_range','date'])['shape'].unstack().rolling(smooth).mean()
        last_i=shape_all.iloc[:,-1].rename('last')
        std=shape_all.std(axis=1).rename('std')
        shape_nice=pd.concat([
                (last_i-std).rename('lower'),
                last_i,
                (last_i+std).rename('upper')
                ],axis=1)
        feather.write_dataframe(shape_nice.reset_index(),path_i+'model_shape.feather')
        feather.write_dataframe(shape_all.reset_index(),path_i+'model_shape_raw.feather')
        
        # get the prediction (with inputs for transparency)
        res_collector=[]
        for date in model_series.index:
            model_i=model_series.loc[date]
            res_i=to_fit_all[to_fit_all['date']==date].copy()
            score_i=model_i.predict(res_i[model_i.feature_names])
            res_i['score_prediction']=score_i
            res_i=res_i[columns_to_use]
            res_collector.append(res_i)
        res=pd.concat(res_collector)
        feather.write_dataframe(res,path_i+'prediction_score_and_input_details.feather') # note here we should be able to get today's signal
        
        # dump mkt data for backtesting (last price, vwap, and total return index)
        feather.write_dataframe(vwap.reset_index(),path_i+'vwap.feather')
        feather.write_dataframe(px_last.reset_index(),path_i+'mkt.feather')
            
            









    










































