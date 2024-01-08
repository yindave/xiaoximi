# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:46:54 2021

@author: hyin1
"""

import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.constants as uc
import pdb
from blp.bdx import bdh,bdp,bds
from fql.fql import Factset_Query
from fql.util import bbg_to_fs, fs_to_bbg,fql_date
from blp.util import get_bbg_usual_col, group_marcap
import feather
import os
import utilities.display as ud
import utilities.mathematics as umath

from ah_arb.ah import AH
from backtester.backtester import BACKTESTER_PAIR

'''
Estimated running time is abount 60-70min
'''


import time
start_time=time.time()


### update the index compo
from scripts_misc import index_compo_cache_daily
from ah_arb.scripts import get_price_for_all_compo


### quickly update the basics
ah=AH(quick_mode=True)
ah.refresh_mkt()
ah.refresh_db()

ah.load_db()
ah.dump_pair_distance(include_dtw=False)

### miscellaneous
# dump the FS FX for quick conversion
dummy_ticker_h='1033-HK'
dummy_ticker_a=ah.get_ah_map(direction='h_to_a')[dummy_ticker_h]
px_usd=pd.concat([
                ah.db['px_h'][dummy_ticker_h],
                ah.db['px_a'][dummy_ticker_h].rename(dummy_ticker_a),
                ],axis=1)
fq=Factset_Query()
px_local=fq.get_ts([dummy_ticker_h,dummy_ticker_a],['px_last'],
          start=fql_date(px_usd.index[0]),end=fql_date(px_usd.index[-1]))['px_last']
fx=px_local.divide(px_usd)
fx.columns=['H','A']
feather.write_dataframe(fx.reset_index(),ah.path+'FactSetFX.feather')

# dump tradable dates
ha_map=ah.get_ah_map(direction='h_to_a')
to_h_pct=ah.db['to_h_pct'].fillna(0)
to_both=ah.db['to_both'].fillna(0)
to_h=to_both.multiply(to_h_pct)
to_a=to_both.multiply(1-to_h_pct)
fx=feather.read_dataframe(ah.path+'FactSetFX.feather').set_index('date')['A']
px_a_cny=ah.db['px_a'].multiply(fx,axis='index')
ret_a_cny=px_a_cny.pct_change()

tradable=pd.concat([to_h.stack().rename('to_h'),
                    to_a.stack().rename('to_a'),
                    ret_a_cny.stack().rename('ret_a')],axis=1)
tradable=tradable.reset_index()
tradable['ticker_a']=tradable['ticker'].map(lambda x: ha_map[x])
tradable['ticker_a_bbg']=fs_to_bbg(tradable['ticker_a'])
# bbg util func is too slow, do the board tagging yourself
boards=bdp(tradable.groupby('ticker_a_bbg').last().index.tolist(),['EXCH_MKT_GRP'])
tradable=tradable.set_index('ticker_a_bbg')
tradable['board']=boards['EXCH_MKT_GRP']

cb_levels_new=pd.Series(index=tradable.groupby('board').last().index).fillna(0.1)
cb_levels_old=cb_levels_new.copy()
cb_levels_new['STAR']=0.2
cb_levels_new['3-CHINEXT']=0.2
cb_levels=pd.DataFrame(index=tradable.groupby('date').last().index,columns=cb_levels_new.index)
cb_levels=pd.concat([
                cb_levels.loc[:pd.datetime(2020,8,23)].fillna(cb_levels_old),
                cb_levels.loc[pd.datetime(2020,8,24):].fillna(cb_levels_new),
            ],axis=0).stack()
tradable=tradable.reset_index().set_index(['date','board'])
tradable['cb_level']=cb_levels
# then we use the apply function
def get_tradable(x,direction='buy'):
    if min(x['to_h'],x['to_a'])==0:
        return False
    else:
        if direction=='buy': # sell A
            return False if (x['ret_a']<0 and abs(x['ret_a'])>=x['cb_level']) else True
        elif direction=='sell': # buy A
            return False if (x['ret_a']>0 and abs(x['ret_a'])>=x['cb_level']) else True
tradable['tradable_buy']=tradable.apply(lambda x: get_tradable(x,direction='buy'),axis=1)
tradable['tradable_sell']=tradable.apply(lambda x: get_tradable(x,direction='sell'),axis=1)
feather.write_dataframe(tradable.reset_index(),ah.path+'tradable_date.feather')


### get signal
directions=['buy','sell']

window=63
zlevel=2.5
entry_method='breach'
exit_method='cross_mid'
para_to_use={'buy':['bollinger_adj_simple','simple_adj'],
             'sell':['bollinger','raw']}
hedge_ticker='HSCEI_VS_A50'

band_collector=[]
for direction in directions:
    paras=para_to_use[direction]
    band_type=paras[0]
    fwd_ret_type=paras[1]
    decision_band_para=[band_type,'ha_avg_simple']
    band_to_use=ah.get_band(window,band_type=decision_band_para)
    band_to_use['upper']=band_to_use['mean']+band_to_use['std']*zlevel
    band_to_use['lower']=band_to_use['mean']-band_to_use['std']*zlevel
    band_to_use['ratio_last']=band_to_use['ratio'].unstack().shift(1).stack()
    band_to_use['mean_last']=band_to_use['mean'].unstack().shift(1).stack()
    band_to_use['upper_last']=band_to_use['upper'].unstack().shift(1).stack()
    band_to_use['lower_last']=band_to_use['lower'].unstack().shift(1).stack()

    tradable_date=feather.read_dataframe(ah.path+'tradable_date.feather').set_index(['date','ticker'])[['tradable_buy','tradable_sell']]
    # create the entry signal for breach
    if direction=='buy':
        band_to_use['trade']=band_to_use.apply(lambda x:
            1 if x['ratio']<x['lower'] else 0
            ,axis=1)
    else:
        band_to_use['trade']=band_to_use.apply(lambda x:
            1 if x['ratio']>x['upper'] else 0
            ,axis=1)
    band_to_use=band_to_use.join(tradable_date,how='left')
    band_to_use['can_entry']=band_to_use['tradable_%s' % (direction)].copy()
    band_to_use['can_exit']=band_to_use['tradable_%s' % ('sell' if direction=='buy' else 'buy')].copy()
    band_to_use['can_entry_next']=band_to_use['can_entry'].unstack().fillna(False).shift(-1).fillna(True).stack()
    band_to_use['can_exit_next']=band_to_use['can_exit'].unstack().fillna(False).shift(-1).fillna(True).stack()
    trades=band_to_use[(band_to_use['trade']==1)
                        & (band_to_use['can_entry_next'])][['trade','window','zscore']].copy()
    trades['zscore_trigger']=zlevel
    trades=trades.swaplevel(1,0,0).sort_index()
    band_to_use=band_to_use.reset_index()
    # loop through all tickers to get the actual trade dates
    trades_actual_collector=[]
    all_tickers=trades.reset_index().groupby('ticker').last().index
    for ticker in all_tickers:
        print('working on %s for %s' % (ticker,direction))
        trades_i=trades.loc[ticker].sort_index()
        last_exit_date=pd.datetime(1999,1,1)
        for trade_date in trades_i.index:
            window_i=trades_i['window'][trade_date]
            zscore_i=trades_i['zscore'][trade_date]
            trigger_i=trades_i['zscore_trigger'][trade_date]
            band_i=band_to_use[(band_to_use['ticker']==ticker) & (band_to_use['window']==window_i)].set_index(['date'])#['cross']
            if trade_date>last_exit_date:
                # need to first check if we can exit on the final exit day
                for window_i_i in np.arange(window_i,len(band_i),1):
                    entry_i=band_i.index.get_loc(trade_date)
                    exit_i=entry_i+window_i_i
                    try:
                        exit_i_info=band_i.iloc[exit_i]
                    except IndexError:
                        exit_i_info=band_i.iloc[-1]
                    if exit_i_info['can_exit_next']:
                        break
                    else:
                        continue
                exit_date=exit_i_info.name
                exit_type_i='time'
                if exit_date==um.today_date():
                    exit_date=trade_date+63*pd.tseries.offsets.BDay()
                    exit_type_i='time' # still call it time

                # find the valid cross exit if there is any
                band_i_exit=band_i.loc[trade_date:exit_date][['cross','can_exit_next']].copy()
                found_exit=False
                for dt_i,dt in enumerate(band_i_exit.index):
                    if band_i_exit['cross'][dt]==1 and not found_exit:
                        found_exit=True
                        for cross_exit_i in np.arange(dt_i,len(band_i_exit),1):
                            if band_i_exit.iloc[cross_exit_i]['can_exit_next']:
                                exit_date=band_i_exit.iloc[cross_exit_i].name
                                exit_type_i='cross'
                                break
                            else:
                                continue
                last_exit_date=exit_date

                idx_i=['trade_date','exit_date','pair',
                       'size','direction',
                       'fwd_ret_type','band_type','exit_type','window',
                       'zscore','zscore_trigger','entry_method','exit_method']
                data_i=[trade_date,exit_date,'%s_VS_%s' % (ticker,ha_map[ticker]),
                        1 if direction=='buy' else -1,direction,
                        fwd_ret_type,band_type,exit_type_i,window_i,
                        zscore_i,trigger_i,entry_method,exit_method]
                trade_actual_i=pd.Series(index=idx_i,data=data_i).to_frame().T
                # add the hedge leg if needed
                if fwd_ret_type!='raw':
                    if fwd_ret_type=='simple_adj':
                        hedge_ratio=1
                    elif fwd_ret_type=='beta_adj':
                        hedge_ratio=ah.db['ha_idx_ratio_KF_beta'][ticker][trade_date]
                    else:
                        print ('unknown ret type: %s' % (fwd_ret_type))
                        pdb.set_trace()
                    if direction=='buy':
                        hedge_ratio=hedge_ratio*(-1)
                    trade_actual_i.loc[1]=trade_actual_i.loc[0]
                    trade_actual_i.at[1,'pair']=hedge_ticker
                    trade_actual_i.at[1,'size']=hedge_ratio
                trade_actual_i['ultimate_pair']='%s_VS_%s' % (ticker,ha_map[ticker])
                trades_actual_collector.append(trade_actual_i)
    trades_actual_all=pd.concat(trades_actual_collector)
    feather.write_dataframe(trades_actual_all,ah.path+'PROD\\Signal_%s.feather' % (direction))


### get backtesting results

# dump the mkt info
mkt_raw=feather.read_dataframe(ah.path+'data_raw.feather')
mkt_raw=mkt_raw.set_index(['date','ticker'])[['px_last','vwap']].unstack() # already in B-days

px_last=mkt_raw['px_last']
vwap=mkt_raw['vwap']
vwap=vwap.fillna(px_last)

# get idx level from bbg
hscei=bdh(['HSI 21 Index'],['px_last'],start=px_last.index[0],end=px_last.index[-1],
               currency='USD')['px_last'].unstack().T.rename(columns={'HSI 21 Index':'HSCEI'})
a50=bdh(['TXIN9IC Index'],['px_last'],start=px_last.index[0],end=px_last.index[-1],
               currency='CNY')['px_last'].unstack().T.rename(columns={'TXIN9IC Index':'A50'}) # quanto, so we get CNY price and pretend it's USD
idx_levels=pd.concat([hscei,a50],axis=1).reindex(px_last.index).fillna(method='ffill')

# note that A50 total return only starts in 2009/11
start_date=idx_levels.dropna().index[0]
px_last=pd.concat([px_last,idx_levels],axis=1).loc[start_date:]
vwap=pd.concat([vwap,idx_levels],axis=1).loc[start_date:]

# dump
feather.write_dataframe(px_last.reset_index(),ah.path+'PROD\\mkt.feather')
feather.write_dataframe(vwap.reset_index(),ah.path+'PROD\\vwap.feather')




for direction in directions:
    print ('backtesting %s' % (direction))
    paras=para_to_use[direction]
    fwd_ret_type=paras[1]
    bps=72 if fwd_ret_type=='raw' else 36
    # H leg 13+13, A leg 10, comm etc 20
    bt=BACKTESTER_PAIR(ah.path+'PROD\\',bps=bps)
    bt.load_data(signal_name='Signal_%s' % (direction))
    # we modify the signal such that it's executed on the next day
    bt.signal['trade_date']=bt.signal['trade_date']+pd.tseries.offsets.BDay()
    bt.signal['exit_date']=bt.signal['exit_date']+pd.tseries.offsets.BDay()
    bt.signal['trade_date']=bt.signal['trade_date'].map(lambda x: max(x,bt.mkt.index[0]))
    #bt.signal['exit_date']=bt.signal['exit_date'].map(lambda x: min(x,bt.mkt.index[-1]))
    bt.signal=bt.signal[bt.signal['trade_date']<bt.signal['exit_date']]
    # we ffill to tomorrow
    bt.mkt.loc[um.today_date()+pd.tseries.offsets.BDay()]=np.nan
    bt.vwap.loc[um.today_date()+pd.tseries.offsets.BDay()]=np.nan
    bt.mkt=bt.mkt.fillna(method='ffill')
    bt.vwap=bt.vwap.fillna(method='ffill')
    trades=bt.run(extra_cols=['exit_type','ultimate_pair'],dump_trades=False)
    trades['direction']=direction
    feather.write_dataframe(trades,ah.path+'PROD\\Trades_%s.feather' % (direction))


trades_collector=[]
for direction in directions:
    trades_collector.append(feather.read_dataframe(ah.path+'PROD\\Trades_%s.feather' % (direction)))
trades_all=pd.concat(trades_collector)




### dump an easy to use trades record for dynamic switch
trades=trades_all.copy()
def get_over_under(x):
    return x[:x.find('_VS_')]
# add adv tagging
to_both=ah.db['to_both']
to_h_pct=ah.db['to_h_pct']
to_h=to_both.multiply(to_h_pct).rolling(63,min_periods=1).mean()
to_a=to_both.multiply(to_h_pct.applymap(lambda x: 1-x)).rolling(63,min_periods=1).mean()
min_adv_musd=pd.concat([to_h.stack(),to_a.stack()],axis=1).min(axis=1)/1000
min_adv_musd=min_adv_musd.unstack()
min_adv_musd['HSCEI']=999
min_adv_musd=min_adv_musd.stack()
min_adv_musd.index.names=['trade_date','ticker']

trades['is_hedge']=trades[['pair','ultimate_pair']].apply(lambda x: True if x['pair']!=x['ultimate_pair'] else False, axis=1)
trades['ticker']=trades['ultimate_pair'].map(lambda x: get_over_under(x))
trades=trades.set_index(['trade_date','ticker'])
trades['min_adv']=min_adv_musd
trades=trades.reset_index()

# add feature level
# euclidean distance
trades=trades.set_index(['trade_date','ticker'])
distance=feather.read_dataframe(ah.path+'pair_distance_euclidean_only.feather').set_index(['date','ticker'])['distance'].unstack().apply(lambda x: x.rank(pct=True),axis=1).fillna(0.5)
distance.index.name='trade_date'
distance=distance.stack()
trades=trades.join(distance.rename('distance'),how='left')
# ha ratio: need to use the reversed
ha_ratio=ah.db['ratio'].apply(lambda x: x.rank(pct=True),axis=1).fillna(0.5)
ha_ratio.index.name='trade_date'
ha_ratio=ha_ratio.stack()
ha_ratio=1-ha_ratio
trades=trades.join(ha_ratio.rename('ha_level'),how='left')
# marcap (both)
size=ah.db['marcap_both'].apply(lambda x: x.rank(pct=True),axis=1)
size.index.name='trade_date'
size=size.stack()
trades=trades.join(size.rename('marcap'),how='left')

trades=trades.reset_index()

trades['feature_level']=trades[['distance','ha_level']].max(axis=1)


feather.write_dataframe(trades,ah.path+'PROD\\trades_tidyup.feather')




end_time=time.time()

min_used=round((end_time-start_time)/60,2)
um.quick_auto_notice('AH update lite finished (%s min used)' % (min_used))


### send the auto-email
from ah_arb.scripts import AH_daily_signal








