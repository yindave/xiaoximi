# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:34:09 2021

@author: hyin1
"""

import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
import matplotlib.pyplot as plt
import feather
from connect.connect import STOCK_CONNECT
from blp.bdx import bdh,bdp, bds
from fql.fql import Factset_Query

from fql.util import bbg_to_fs,fs_to_bbg,fql_date
from blp.util import get_bbg_nice_compo_hist_no_limit, get_bbg_nice_compo_hist, get_bbg_usual_col,group_marcap, get_region, get_board

import webscraping.eastmoney as em
import itertools

import utilities.mathematics as umath
import plotly.express as px
import plotly.graph_objects as go

from backtester.backtester import BACKTESTER

from ah_arb.ah import AH

import datatable as DT
from datatable import f,by,sort,join

import os


import argparse


### manual switch
dump_summary_compo_only=False
check_existing_dump=False
skip_compo_dump=False


parser = argparse.ArgumentParser()
parser.add_argument('--idx_ticker',dest='idx_ticker',type=str,default='SHSZ300')
parser.add_argument('--min_adv',dest='min_adv',type=float,default=0)
parser.add_argument('--min_feature_level',dest='min_feature_level',type=float,default=0)

args = parser.parse_args()


idx_ticker_short=args.idx_ticker
min_adv=round(float(args.min_adv),1)
min_feature_level=round(float(args.min_feature_level),1)


idx_ticker=idx_ticker_short+' Index'

dump_name='%s_%s_%s' % (idx_ticker_short,min_adv,min_feature_level)

print ('Working on %s' % (dump_name))

ah=AH()
path=ah.path+'\\PROD\\IndexSwitch\\'

def get_over_under(x):
        return x[:x.find('_VS_')]

if check_existing_dump:
    dump_exist=os.path.exists(path+'%s.feather' % (dump_name))
else:
    dump_exist=False


if not dump_exist:
### get basic AH ready
    ah.load_db()
    ha_map=ah.get_ah_map(direction='h_to_a')
    ah_list=ah.ah_list
    all_ah_pairs=list(ah_list['HK'])+list(ah_list['CN'])
    trades=feather.read_dataframe(ah.path+'PROD\\trades_tidyup.feather')
### get nice trades record ready
    trades_to_use_index_switch=trades[(trades['min_adv']>=min_adv) & (trades['feature_level']>=min_feature_level)
                                     & (~trades['is_hedge'])].copy() # We should exclude the hedge here for unique match later. We can add back the 1-to-1 hedge easily later
    trades_to_use_index_switch['ticker_h']=trades_to_use_index_switch['ticker'].copy()
    trades_to_use_index_switch['ticker_a']=trades_to_use_index_switch['ticker_h'].map(lambda x: ha_map[x])
    trades_to_use_index_switch['ticker']=trades_to_use_index_switch.apply(lambda x: x['ticker_a'] if x['direction']=='buy' else x['ticker_h'], axis=1)
    trades_to_use_index_switch['ticker_pair']=trades_to_use_index_switch.apply(lambda x: x['ticker_a'] if x['ticker']==x['ticker_h'] else x['ticker_h'], axis=1)
    trades_to_switch=trades_to_use_index_switch.set_index(['date','ticker'])[['ticker_pair','direction','trade_date','exit_date']].sort_index()
### load index compo
    compo=feather.read_dataframe(uc.compo_daily_path % (idx_ticker))
    compo['shares']=compo['idx_level']*compo['wgt']/compo['px_last']
    compo['value']=compo['shares']*compo['px_last']
    compo=compo[compo['px_last'].fillna(0)!=0].copy()
    compo['ticker']=bbg_to_fs(compo['ticker'])
    ### load price and vwap, unadj
    px_info=feather.read_dataframe(path+'px_info_unadj.feather')
    px_info['ticker']=bbg_to_fs(px_info['ticker'])
    mkt=px_info.set_index(['date','ticker'])['px_last'].unstack().resample('B').last().fillna(method='ffill')
    vwap=px_info.set_index(['date','ticker'])['EQY_WEIGHTED_AVG_PX'].unstack().resample('B').last().fillna(method='ffill')

# get compo for easy summary
    compo_for_summary=compo.set_index(['date','ticker']).join(trades_to_switch,how='left')
    compo_for_summary['direction']=compo_for_summary['direction'].fillna('na')
    compo_for_summary=compo_for_summary.reset_index()
    if not skip_compo_dump:
        feather.write_dataframe(compo_for_summary,path+'SummaryCompo_%s.feather' % (dump_name))


    if not dump_summary_compo_only:
### add HSCEI and A50 price
        hscei=bdh(['HSI 21 Index'],['px_last'],mkt.index[0],mkt.index[-1],currency='USD')['px_last'].unstack().T['HSI 21 Index'].rename('HSCEI')
        a50=bdh(['TXIN9IC Index'],['px_last'],mkt.index[0],mkt.index[-1])['px_last'].unstack().T['TXIN9IC Index'].rename('A50') # quanto USD
        hedge_levels=pd.concat([hscei,a50],axis=1)
        mkt=mkt.join(hedge_levels,how='left').fillna(method='ffill')
        vwap=vwap.join(hedge_levels,how='left').fillna(method='ffill')
###  KEY: start the loop to get the new shares matrix
        shares_matrix=compo.set_index(['date','ticker'])['shares'].unstack()
        for ticker in all_ah_pairs:
            if ticker not in shares_matrix.columns:
                shares_matrix[ticker]=np.nan
        shares_matrix['HSCEI']=0 # for the hedge
        shares_matrix['A50']=0 # for the hedge
        value_matrix=shares_matrix.multiply(mkt)
        shares_matrix_new=shares_matrix.copy()
        total_value_traded=pd.DataFrame(index=shares_matrix_new.index,columns=['value']).fillna(0)
        trades_record=trades_to_switch.reset_index().groupby(['trade_date','ticker']).last()
        trade_dates=trades_record.index.levels[0]
        max_chg_pct=50/100
        for trade_date in trade_dates:
            if trade_date in shares_matrix.index:
                print ('update matrix on %s' % (trade_date))
                trades_record_i=trades_record.loc[trade_date]
                for ticker_to_switch in trades_record_i.index:
                    if ticker_to_switch in shares_matrix.loc[trade_date].dropna().index:
                        pair_ticker=trades_record_i.loc[ticker_to_switch]['ticker_pair']
                        exit_date=trades_record_i.loc[ticker_to_switch]['exit_date']
                        value_traded_i=value_matrix.loc[trade_date][ticker_to_switch]
                        shares_to_trade=value_traded_i/vwap.loc[trade_date][pair_ticker]
                        patch_i=pd.DataFrame(index=shares_matrix.loc[trade_date:exit_date].index,columns=[ticker_to_switch,pair_ticker])
                        patch_i[ticker_to_switch]=0
                        patch_i[pair_ticker]=shares_to_trade
                        patch_i=patch_i.iloc[:-1] # at the end of exit day we will have no shares to hold
                        # record value traded
                        total_value_traded.loc[trade_date,'value']=total_value_traded['value'][trade_date]+value_traded_i
                        # adjust the patch: see if change in the mid is needed
                        adj_shares=shares_matrix.loc[patch_i.index][ticker_to_switch]
                        adj_shares_chg=adj_shares/adj_shares.iloc[0]
                        adj_shares_chg_check=adj_shares_chg[(adj_shares_chg-1).abs()>max_chg_pct].copy()
                        if len(adj_shares_chg_check!=0):
                            adj_date=adj_shares_chg_check.index[0]
                            adj_ratio=adj_shares_chg[adj_date]
                            patch_i=pd.concat([patch_i.loc[:adj_date].iloc[:-1],
                                        patch_i.loc[adj_date:]*adj_ratio],axis=0).sort_index()
                        # add the hedge
                        if ticker_to_switch[-2:]=='CN':
                            patch_i_hedge=pd.DataFrame(index=patch_i.index,columns=['HSCEI','A50'])
                            patch_i_hedge['HSCEI']=(-1)*value_matrix.loc[trade_date][ticker_to_switch]/vwap.loc[trade_date]['HSCEI']
                            patch_i_hedge['A50']=value_matrix.loc[trade_date][ticker_to_switch]/vwap.loc[trade_date]['A50']
                            patch_i_hedge=patch_i_hedge.multiply(patch_i[pair_ticker]/patch_i[pair_ticker].iloc[0],axis='index')
                            shares_matrix_new=shares_matrix_new.add(patch_i_hedge,fill_value=0)
                        # adjust the patch: see if the pair_ticker is also in the index, if so add on top of patch_i
                        pair_exist=shares_matrix.loc[patch_i.index][pair_ticker]
                        if len(pair_exist.dropna())!=0:
                            patch_i[pair_ticker]=patch_i[pair_ticker]+pair_exist.fillna(0)
                        # update new shares matrix
                        shares_matrix_new.update(patch_i)
### tidyup the output and dump
        shares_matrix_new_clean=shares_matrix_new.drop(shares_matrix_new.count()[shares_matrix_new.count()==0].index,1)
        shares_matrix_clean=shares_matrix.drop(shares_matrix.count()[shares_matrix.count()==0].index,1)
        # output outperformance for: 1) hedged or unhedged; 2) add value traded
        old_pnl=shares_matrix_clean.shift(1).multiply(mkt.diff()).sum(1).cumsum()
        new_pnl_hedged=shares_matrix_new_clean.shift(1).multiply(mkt.diff()).sum(1).cumsum()
        new_pnl_unhedged=shares_matrix_new_clean.shift(1).multiply(mkt.diff()).drop(['HSCEI','A50'],1).sum(1).cumsum()
        res=pd.concat([old_pnl.rename('old_pnl'),
                       new_pnl_hedged.rename('new_pnl_hedged'),
                       new_pnl_unhedged.rename('new_pnl_unhedged'),
                       total_value_traded['value'].rename('value_traded'),
                      ],axis=1).reset_index()
        res['index']=idx_ticker
        res['min_adv']=min_adv
        res['min_feature_level']=min_feature_level
        feather.write_dataframe(res,path+'%s.feather' % (dump_name))
        # enhanced
        values_matrix=(shares_matrix_new_clean.resample('D').last().fillna(0)
                    .multiply(mkt.resample('D').last().fillna(method='ffill')))
        feather.write_dataframe(values_matrix.reset_index(),path+'%s_value_matrix.feather' % (dump_name))
        # original
        values_matrix=(shares_matrix_clean.resample('D').last().fillna(0)
                    .multiply(mkt.resample('D').last().fillna(method='ffill')))
        feather.write_dataframe(values_matrix.reset_index(),path+'%s_value_matrix_original.feather' % (dump_name))


















