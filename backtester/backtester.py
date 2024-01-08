# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 08:41:16 2020

@author: hyin1
"""


import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
import utilities.mathematics as umath
import pdb
import feather


'''
In case of tie, we use method=' min ' in rank, so that means for basket with size=5 we can trade more stocks than that
(https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.Series.rank.html)

strategy turnover calculations are now based on wgt for all functions

Inputs: signals (dates frequency will become rebalancing date, higher goes to long)
         px data, vwap data (including the benchmark, BDay frequency)
         size: int (quantile later/never)
Outputs: perf (in pct), turnover etc


The turnover estimation actually has some flaw. The estimation is done on wgt matrix, which does not
include the weigth change casued by relative price move. So if there is no constitutents change, then
the turnover will be 0 if the weight target remains the same.

'''

#TODO
# need to replace perf abs with realized turnover
# (need to have a 3m adv feather dumped, so the realized turnover output needs to be optional)

tie_method='min'


class BACKTESTER_SIMPLE_LO():
    '''
    no transaciton, long only, signal is also th weight
    '''
    def __init__(self,path):
        self.path=path
        return None
    def load_data(self,mkt_freq='D'):
        path=self.path
        self.mkt=(feather.read_dataframe(path+'mkt.feather')
                    .set_index('date').resample(mkt_freq).last().fillna(method='ffill'))

    def run(self,signal_name):
        path=self.path
        self.signal=feather.read_dataframe(path+'%s.feather' % (signal_name)).set_index('date')
        signal=self.signal.copy()
        mkt=self.mkt.copy()

        capital=1
        collector=[]
        for i,date in enumerate(signal.index):
            wgt_i=signal.loc[date].map(lambda x: np.nan if x==0 else x).dropna()
            try:
                date_next=signal.index[i+1]
            except IndexError:
                date_next=mkt.index[-1]
            mkt_i=mkt.loc[date:date_next][wgt_i.index]
            bsk_i=(mkt_i/mkt_i.iloc[0]).multiply(capital*wgt_i).sum(1)
            capital=bsk_i.iloc[-1]
            collector.append(bsk_i.rename(date))
        bsk=pd.concat(collector,axis=1).mean(1)

        return bsk




class BACKTESTER():
    def __init__(self,path,bps=20,bps_mkt=0,notional=100,
                 accurate_transaction_cost=False,divide_notional=[False,'divisor, 2 for example']):
        self.path=path
        self.bps=bps/100/100
        self.bps_mkt=bps_mkt
        self.notional=notional
        self.accurate_transaction_cost=accurate_transaction_cost
        return None
    def load_data(self,signal_name='signal'):
        '''
        can read multiple signal names for the same strategy the shares px and mkt
        all input are row-col matrix
        both mkt and vwap should contain benchmark/index price in there
        
        we can also manually assign the below variable so that we do not need to actually dump the data
        '''
        path=self.path
        self.mkt=(feather.read_dataframe(path+'mkt.feather')
                    .set_index('date').resample('B').last().fillna(method='ffill'))
        self.vwap=(feather.read_dataframe(path+'vwap.feather')
                    .set_index('date').resample('B').last().fillna(method='ffill'))

        self.signal=feather.read_dataframe(path+'%s.feather' % (signal_name)).set_index('date')
        return None
    
    def run(self,start_date,size,benchmark,disable_compounding=False):
        '''
        Outputs are: perf_pct,perf_abs,shares_overtime,to,hlds
        The shares matrix from backtest output on the rebalancing date is referring to the old basket holdings.
        If we need to back out the targeted weight, we need to shift back by 1 period
        To reconcile 1: perf_pct=shares_overtime*mkt.diff(), need to set vwap=mkt and t-cost=0
        To reconcile 2: run with wgt, with wgt being shifted shares times mkt (normalized obviously)
                        this should lead to same as normal run
        '''
        mkt=self.mkt.copy()
        vwap=self.vwap.copy()
        signal=self.signal.copy()
        notional=self.notional
        bps=self.bps
        bps_mkt=self.bps_mkt

        signal=signal.loc[start_date:]
        l=signal.apply(lambda x: x.dropna().rank(ascending=False,method=tie_method),axis=1).applymap(lambda x: x if x<=size else np.nan)
        s=signal.apply(lambda x: x.dropna().rank(ascending=True,method=tie_method),axis=1).applymap(lambda x: x if x<=size else np.nan)


        l_wgt=l.stack().rename('rank').to_frame()
        l_wgt['wgt']=1
        l_wgt=l_wgt['wgt'].unstack().apply(lambda x: x/x.dropna().sum(),axis=1)
        s_wgt=s.stack().rename('rank').to_frame()
        s_wgt['wgt']=1
        s_wgt=s_wgt['wgt'].unstack().apply(lambda x: x/x.dropna().sum(),axis=1)

        to_l=l_wgt.applymap(lambda x: 0 if np.isnan(x) else x).diff().abs().sum(1)/2
        to_s=s_wgt.applymap(lambda x: 0 if np.isnan(x) else x).diff().abs().sum(1)/2
#        to_l=to_l.map(lambda x: min(1,x))
#        to_s=to_s.map(lambda x: min(1,x))
        to_l.iloc[0]=1
        to_s.iloc[0]=1
        to_mkt=to_l.map(lambda x: 0).copy()
        bsk_dict={'l':l,'s':s,'mkt':mkt[[benchmark]]}
        to_dict={'l':to_l,'s':to_s,'mkt':to_mkt}
        # calculate perf
        collector=[]
        collector_shares=[]
        pairs=[['l','s'],['l','mkt']]
        for pair in pairs:
            notional_i=notional
            for i,dt in enumerate(signal.index):
                collector_i=[]
                for bsk_name in pair:
                    try:
                        dt_next=signal.index[i+1]
                    except IndexError:
                        dt_next=vwap.index[-1]
                    bsk_memb=bsk_dict[bsk_name].loc[dt].dropna().index.tolist()
                    if bsk_name!='mkt':
                        wgt=l_wgt.copy() if bsk_name=='l' else s_wgt.copy()
                        wgt_i=wgt.loc[dt][bsk_memb]
                        wgt_i=wgt_i/wgt_i.sum()
                    else:
                        wgt_i=1
#                    if dt==pd.datetime(2021,6,30):
#                        pdb.set_trace()
                    shares=(wgt_i*notional_i)/vwap.loc[dt][bsk_memb]
                    shares_to_output=shares.rename('shares').to_frame()
                    shares_to_output['date_beg']=dt
                    shares_to_output['date']=dt_next # we need to use next date here, as on rebalance date the pnl is still coming from the old bsk
                    shares_to_output['strategy']='-'.join(pair)
                    shares_to_output['direction']=bsk_name
                    shares_to_output.index.name='ticker'
                    collector_shares.append(shares_to_output)
                    #pdb.set_trace()
                    px_ss=pd.concat([
                            vwap[bsk_memb].loc[[dt]],
                            mkt[bsk_memb].loc[dt:dt_next].iloc[1:-1],
                            vwap[bsk_memb].loc[[dt_next]],
                              ],axis=0)
                    bsk_px=px_ss.multiply(shares,axis='columns').sum(1).rename('gross').to_frame()
                    bsk_mtm=bsk_px.diff().fillna(0)
                    bsk_mtm['bsk']=bsk_name
                    if not self.accurate_transaction_cost:
                        levy=-1*to_dict[bsk_name][dt]*notional_i*bps
                        if i!=0:
                            levy=levy*2
                        if bsk_name=='mkt':
                            # we can't actually calculate the levy of the market as we don't calculate the turnover of market
                            levy=bps_mkt
                    else:
                        # a more accurate way to calculate transaction cost
                        try:
                            shares_last=collector_shares[-3]
                            if shares_last['strategy'][0]=='-'.join(pair):
                                shares_last=shares_last['shares']
                                shares_chg_df=pd.concat([shares_last.rename('last'),
                                                         shares.rename('now')],axis=1,sort=True).fillna(0)
                                shares_chg=shares_chg_df['now']-shares_chg_df['last']
                            else:
                                shares_chg=shares
                        except IndexError:
                            shares_chg=shares
                        traded_value=(vwap.loc[dt][shares_chg.index]*shares_chg.abs()).sum()
                        levy=traded_value*self.bps*(-1) if bsk_name!='mkt' else traded_value*self.bps_mkt*(-1)
                    bsk_mtm['net']=bsk_px.diff().fillna(levy)
                    bsk_mtm['rebalance_dt']=dt
                    bsk_mtm['pair']='-'.join(pair)
                    collector_i.append(bsk_mtm)
                    collector.append(bsk_mtm)
                
                notional_chg=pd.concat(collector_i,axis=0).groupby('bsk').sum()
                net_i=(notional_chg['gross']['l']-notional_chg['gross'].drop('l').values[0]
                        -(notional_chg['gross']-notional_chg['net']).sum())
                if not disable_compounding:
                    notional_i=notional_i+net_i

        mtm_df=pd.concat(collector,axis=0).reset_index().groupby(['date','bsk','pair']).sum()
        mtm_gross=mtm_df['gross'].unstack()
        mtm_net=mtm_df['net'].unstack()
        mtm_levy=mtm_gross-mtm_net
        l_mkt_gross=mtm_gross['l-mkt'].unstack()
        l_mkt_gross=l_mkt_gross['l']-l_mkt_gross['mkt']
        l_mkt_net=l_mkt_gross-mtm_levy['l-mkt'].unstack()[['l','mkt']].sum(1)
        l_s_gross=mtm_gross['l-s'].unstack()
        l_s_gross=l_s_gross['l']-l_s_gross['s']
        l_s_net=l_s_gross-mtm_levy['l-s'].unstack()[['l','s']].sum(1)

        perf_pct=pd.concat([
           #(l_mkt_gross.cumsum()/notional+1).rename('l-mkt'),
           (l_mkt_net.cumsum()/notional+1).rename('l-mkt_net'),
           #(l_s_gross.cumsum()/notional+1).rename('l-s'),
           (l_s_net.cumsum()/notional+1).rename('l-s_net'),
                ],axis=1)
        perf_abs=mtm_df.copy()
        #sharpe=umath.get_sharpe(perf_pct.pct_change())
        to=pd.concat([to_l.rename('l'),to_s.rename('s')],axis=1)
        hlds=pd.concat([l.stack().rename('l'),s.stack().rename('s')],axis=1)
        hlds.index.names=['date','ticker']
        shares_overtime=pd.concat(collector_shares,axis=0).reset_index()
        shares_overtime=shares_overtime[shares_overtime['date_beg']!=shares_overtime['date']]
        shares_overtime=shares_overtime.set_index(['strategy','direction','ticker','date'])['shares'].unstack().T
        shares_overtime=shares_overtime.fillna(0).reindex(self.mkt.index).fillna(method='bfill').loc[start_date:]
        return perf_pct,perf_abs,shares_overtime,to,hlds

    def run_with_weight(self,start_date,size,benchmark,
                        weight_matrix,disable_compounding=False,
                        normalize_wgt=True):
        '''
        Outputs are: perf_pct,perf_abs,shares_overtime,to,hlds
        # We no longer calculate both gross and net performance in the same run as transaction cost
        will affect the position value overtime
        # In case we need the gross pnl we just set bps=0 and re-run
        Weight cannot have missing value on the top/bottom baskest for the rebalance date
        Once the top/bottom names are selected we will normalize the wgt so that they sum to 1
        We need to check separately the basket size
        no need to normalize weight
        for LS weighted port, turn off normalize wgt, set market = cash, and get long vs. mkt performance
        '''

        mkt=self.mkt.copy()
        vwap=self.vwap.copy()
        signal=self.signal.copy()
        notional=self.notional
        bps=self.bps
        bps_mkt=self.bps_mkt
        wgt=weight_matrix.copy()

        signal=signal.loc[start_date:]


        l=signal.apply(lambda x: x.dropna().rank(ascending=False,method=tie_method),axis=1).applymap(lambda x: x if x<=size else np.nan)
        s=signal.apply(lambda x: x.dropna().rank(ascending=True,method=tie_method),axis=1).applymap(lambda x: x if x<=size else np.nan)
        l_wgt=l.stack().rename('rank').to_frame()
        l_wgt['wgt']=wgt.stack()
        l_wgt=l_wgt[l_wgt['wgt']>0].copy()
        l_wgt=l_wgt['wgt'].unstack().apply(lambda x: x/x.dropna().sum(),axis=1)
        s_wgt=s.stack().rename('rank').to_frame()
        s_wgt['wgt']=wgt.stack()
        s_wgt=s_wgt[s_wgt['wgt']<0].copy()
        s_wgt=s_wgt['wgt'].unstack().abs().apply(lambda x: x/x.dropna().sum(),axis=1)
        to_l=l_wgt.applymap(lambda x: 0 if np.isnan(x) else x).diff().abs().sum(1)/2
        to_s=s_wgt.applymap(lambda x: 0 if np.isnan(x) else x).diff().abs().sum(1)/2
#        to_l=to_l.map(lambda x: min(1,x))
#        to_s=to_s.map(lambda x: min(1,x))
        to_l.iloc[0]=1
        try:
            to_s.iloc[0]=1
        except IndexError:
            to_s=to_l.copy() # a dirty fix for combined LS wgt input
            to_s.iloc[0]=1
        to_mkt=to_l.map(lambda x: 0).copy()
        bsk_dict={'l':l,'s':s,'mkt':mkt[[benchmark]]}
        to_dict={'l':to_l,'s':to_s,'mkt':to_mkt}
        # calculate perf
        collector=[]
        collector_shares=[]
        pairs=[['l','mkt'],['l','s'],]
        for pair in pairs:
            notional_i=notional
            for i,dt in enumerate(signal.index):
                collector_i=[]
                for bsk_name in pair:
                    try:
                        dt_next=signal.index[i+1]
                    except IndexError:
                        dt_next=vwap.index[-1]

                    bsk_memb=bsk_dict[bsk_name].loc[dt].dropna().index.tolist()

                    if bsk_name!='mkt':
                        wgt_i=wgt.loc[dt][bsk_memb]
                        if normalize_wgt:
                            wgt_i=wgt_i/wgt_i.sum()
                    else:
                        wgt_i=1

                    shares=(wgt_i*notional_i)/vwap.loc[dt][bsk_memb]
                    shares_to_output=shares.rename('shares').to_frame()
                    shares_to_output['date_beg']=dt
                    shares_to_output['date']=dt_next
                    shares_to_output['strategy']='-'.join(pair)
                    shares_to_output['direction']=bsk_name
                    shares_to_output.index.name='ticker'
                    collector_shares.append(shares_to_output)
                    px_ss=pd.concat([
                            vwap[bsk_memb].loc[[dt]],
                            mkt[bsk_memb].loc[dt:dt_next].iloc[1:-1],
                            vwap[bsk_memb].loc[[dt_next]],
                              ],axis=0)
                    bsk_px=px_ss.multiply(shares,axis='columns').sum(1).rename('gross').to_frame()
                    bsk_mtm=bsk_px.diff().fillna(0)
                    bsk_mtm['bsk']=bsk_name
                    if not self.accurate_transaction_cost:
                        levy=-1*to_dict[bsk_name][dt]*notional_i*bps
                        if i!=0:
                            levy=levy*2
                        if bsk_name=='mkt':
                            # we can't actually calculate the levy of the market as we don't calculate the turnover of market
                            levy=bps_mkt
                    else:
                        # a more accurate way to calculate transaction cost
                        try:
                            shares_last=collector_shares[-3]
                            if shares_last['strategy'][0]=='-'.join(pair):
                                shares_last=shares_last['shares']
                                shares_chg_df=pd.concat([shares_last.rename('last'),
                                                         shares.rename('now')],axis=1,sort=True).fillna(0)
                                shares_chg=shares_chg_df['now']-shares_chg_df['last']
                            else:
                                shares_chg=shares
                        except IndexError:
                            shares_chg=shares
                        traded_value=(vwap.loc[dt][shares_chg.index]*shares_chg.abs()).sum()
                        levy=traded_value*self.bps*(-1) if bsk_name!='mkt' else traded_value*self.bps_mkt*(-1)

                    bsk_mtm['net']=bsk_px.diff().fillna(levy)
                    bsk_mtm['rebalance_dt']=dt
                    bsk_mtm['pair']='-'.join(pair)
                    collector_i.append(bsk_mtm)
                    collector.append(bsk_mtm)

                notional_chg=pd.concat(collector_i,axis=0).groupby('bsk').sum()
                net_i=(notional_chg['gross']['l']-notional_chg['gross'].drop('l').values[0]
                        -(notional_chg['gross']-notional_chg['net']).sum())
                if not disable_compounding:
                    notional_i=notional_i+net_i
        #pdb.set_trace()
        mtm_df=pd.concat(collector,axis=0).reset_index().groupby(['date','bsk','pair']).sum()
        mtm_gross=mtm_df['gross'].unstack()
        mtm_net=mtm_df['net'].unstack()
        mtm_levy=mtm_gross-mtm_net
        l_mkt_gross=mtm_gross['l-mkt'].unstack()
        l_mkt_gross=l_mkt_gross['l']-l_mkt_gross['mkt']
        l_mkt_net=l_mkt_gross-mtm_levy['l-mkt'].unstack()[['l','mkt']].sum(1)
        l_s_gross=mtm_gross['l-s'].unstack()
        l_s_gross=l_s_gross['l']-l_s_gross['s']
        l_s_net=l_s_gross-mtm_levy['l-s'].unstack()[['l','s']].sum(1)

        perf_pct=pd.concat([
           #(l_mkt_gross.cumsum()/notional+1).rename('l-mkt'),
           (l_mkt_net.cumsum()/notional+1).rename('l-mkt_net'),
           #(l_s_gross.cumsum()/notional+1).rename('l-s'),
           (l_s_net.cumsum()/notional+1).rename('l-s_net'),
                ],axis=1)
        perf_abs=mtm_df.copy()
        #sharpe=umath.get_sharpe(perf_pct.pct_change())
        to=pd.concat([to_l.rename('l'),to_s.rename('s')],axis=1)
        hlds=pd.concat([l.stack().rename('l'),s.stack().rename('s')],axis=1)
        hlds.index.names=['date','ticker']
        shares_overtime=pd.concat(collector_shares,axis=0).reset_index()
        shares_overtime=shares_overtime[shares_overtime['date_beg']!=shares_overtime['date']]
        shares_overtime=shares_overtime.set_index(['strategy','direction','ticker','date'])['shares'].unstack().T
        shares_overtime=shares_overtime.fillna(0).reindex(self.mkt.index).fillna(method='bfill').loc[start_date:]
        return perf_pct,perf_abs,shares_overtime,to,hlds


    def run_q(self,q,start_date,benchmark,
              manual_q_ls=[False,'Q long','Q short'],
              disable_compounding=False):
        '''
        Outputs are: perf_pct,perf_abs,shares_overtime,to,hlds
        '''
        # here we do qunitle style backtest
        # equal weight only
        # higher Q means higher signal value
        mkt=self.mkt.copy()
        vwap=self.vwap.copy()
        signal=self.signal.copy()
        notional=self.notional
        bps=self.bps
        bps_mkt=self.bps_mkt
        labels=['Q%s' % (x+1) for x in np.arange(0,q,1)]

        signal=signal.loc[start_date:]

        qs=signal.apply(lambda x: pd.Series(index=x.index,data=pd.qcut(x.dropna(),q,labels)),axis=1)
        if not manual_q_ls[0]:
            q_long='Q%s' % (q)
            q_short='Q1'
        else:
            q_long=manual_q_ls[1]
            q_short=manual_q_ls[2]

        l=qs.applymap(lambda x: 1 if x==q_long else np.nan)
        s=qs.applymap(lambda x: 1 if x==q_short else np.nan)
        l_wgt=l.stack().rename('rank').to_frame()
        l_wgt['wgt']=1
        l_wgt=l_wgt['wgt'].unstack().apply(lambda x: x/x.dropna().sum(),axis=1)
        s_wgt=s.stack().rename('rank').to_frame()
        s_wgt['wgt']=1
        s_wgt=s_wgt['wgt'].unstack().apply(lambda x: x/x.dropna().sum(),axis=1)

        to_l=l_wgt.applymap(lambda x: 0 if np.isnan(x) else x).diff().abs().sum(1)/2
        to_s=s_wgt.applymap(lambda x: 0 if np.isnan(x) else x).diff().abs().sum(1)/2

#        to_l=to_l.map(lambda x: min(1,x))
#        to_s=to_s.map(lambda x: min(1,x))
        #pdb.set_trace()
        to_l.iloc[0]=1
        to_s.iloc[0]=1
        to_mkt=to_l.map(lambda x: 0).copy()
        bsk_dict={'l':l,'s':s,'mkt':mkt[[benchmark]]}
        to_dict={'l':to_l,'s':to_s,'mkt':to_mkt}
        # calculate perf
        collector=[]
        collector_shares=[]
        pairs=[['l','s'],['l','mkt']]
        for pair in pairs:
            notional_i=notional
            for i,dt in enumerate(signal.index):
                collector_i=[]
                for bsk_name in pair:
                    try:
                        dt_next=signal.index[i+1]
                    except IndexError:
                        dt_next=vwap.index[-1]
                    bsk_memb=bsk_dict[bsk_name].loc[dt].dropna().index.tolist()
                    if bsk_name!='mkt':
                        wgt=l_wgt.copy() if bsk_name=='l' else s_wgt.copy()
                        wgt_i=wgt.loc[dt][bsk_memb]
                        wgt_i=wgt_i/wgt_i.sum()
                    else:
                        wgt_i=1
                    shares=(wgt_i*notional_i)/vwap.loc[dt][bsk_memb]
                    shares_to_output=shares.rename('shares').to_frame()
                    shares_to_output['date_beg']=dt
                    shares_to_output['date']=dt_next
                    shares_to_output['strategy']='-'.join(pair)
                    shares_to_output['direction']=bsk_name
                    shares_to_output.index.name='ticker'
                    collector_shares.append(shares_to_output)
                    px_ss=pd.concat([
                            vwap[bsk_memb].loc[[dt]],
                            mkt[bsk_memb].loc[dt:dt_next].iloc[1:-1],
                            vwap[bsk_memb].loc[[dt_next]],
                              ],axis=0)
                    bsk_px=px_ss.multiply(shares,axis='columns').sum(1).rename('gross').to_frame()
                    bsk_mtm=bsk_px.diff().fillna(0)
                    bsk_mtm['bsk']=bsk_name
                    if not self.accurate_transaction_cost:
                        levy=-1*to_dict[bsk_name][dt]*notional_i*bps
                        if i!=0:
                            levy=levy*2
                        if bsk_name=='mkt':
                            # we can't actually calculate the levy of the market as we don't calculate the turnover of market
                            levy=bps_mkt
                    else:
                        # a more accurate way to calculate transaction cost
                        try:
                            shares_last=collector_shares[-3]
                            if shares_last['strategy'][0]=='-'.join(pair):
                                shares_last=shares_last['shares']
                                shares_chg_df=pd.concat([shares_last.rename('last'),
                                                         shares.rename('now')],axis=1,sort=True).fillna(0)
                                shares_chg=shares_chg_df['now']-shares_chg_df['last']
                            else:
                                shares_chg=shares
                        except IndexError:
                            shares_chg=shares
                        traded_value=(vwap.loc[dt][shares_chg.index]*shares_chg.abs()).sum()
                        levy=traded_value*self.bps*(-1) if bsk_name!='mkt' else traded_value*self.bps_mkt*(-1)
                    bsk_mtm['net']=bsk_px.diff().fillna(levy)
                    bsk_mtm['rebalance_dt']=dt
                    bsk_mtm['pair']='-'.join(pair)
                    collector_i.append(bsk_mtm)
                    collector.append(bsk_mtm)

                notional_chg=pd.concat(collector_i,axis=0).groupby('bsk').sum()
                net_i=(notional_chg['gross']['l']-notional_chg['gross'].drop('l').values[0]
                        -(notional_chg['gross']-notional_chg['net']).sum())
                if not disable_compounding:
                    notional_i=notional_i+net_i
        mtm_df=pd.concat(collector,axis=0).reset_index().groupby(['date','bsk','pair']).sum()
        mtm_gross=mtm_df['gross'].unstack()
        mtm_net=mtm_df['net'].unstack()
        mtm_levy=mtm_gross-mtm_net
        l_mkt_gross=mtm_gross['l-mkt'].unstack()
        l_mkt_gross=l_mkt_gross['l']-l_mkt_gross['mkt']
        l_mkt_net=l_mkt_gross-mtm_levy['l-mkt'].unstack()[['l','mkt']].sum(1)
        l_s_gross=mtm_gross['l-s'].unstack()
        l_s_gross=l_s_gross['l']-l_s_gross['s']
        l_s_net=l_s_gross-mtm_levy['l-s'].unstack()[['l','s']].sum(1)

        perf_pct=pd.concat([
           #(l_mkt_gross.cumsum()/notional+1).rename('l-mkt'),
           (l_mkt_net.cumsum()/notional+1).rename('l-mkt_net'),
           #(l_s_gross.cumsum()/notional+1).rename('l-s'),
           (l_s_net.cumsum()/notional+1).rename('l-s_net'),
                ],axis=1)
        perf_abs=mtm_df.copy()
        #sharpe=umath.get_sharpe(perf_pct.pct_change())
        to=pd.concat([to_l.rename('l'),to_s.rename('s')],axis=1)
        hlds=pd.concat([l.stack().rename('l'),s.stack().rename('s')],axis=1)
        hlds.index.names=['date','ticker']
        shares_overtime=pd.concat(collector_shares,axis=0).reset_index()
        shares_overtime=shares_overtime[shares_overtime['date_beg']!=shares_overtime['date']]
        shares_overtime=shares_overtime.set_index(['strategy','direction','ticker','date'])['shares'].unstack().T
        shares_overtime=shares_overtime.fillna(0).reindex(self.mkt.index).fillna(method='bfill').loc[start_date:]
        return perf_pct,perf_abs,shares_overtime,to,hlds



class BACKTESTER_PAIR():
    '''
    Similar data input format: matrix of last price and vwap
    Need to make sure about the FX by yourself in the mkt data input
    Let the signal column format be over_VS_under for the moment
    We do dollar flat trade for each pair
    In the signal matrix, instead of using 1,-1 and 0, we can use other float to size the trade
    signals contains columns: trade_date, exit_date, pair, size
    '''
    def __init__(self,path,bps=20,notional=100,
                 verbose=False):
        self.path=path
        self.bps=bps/100/100 # levied on notional, ignore the drift
        self.notional=notional

        self.verbose=verbose
        return None
    def load_data(self,signal_name='signal'):
        '''
        signal needs to be stacked df with 4 columns:
            trade_date, exit_date, pair, size (signed, with unit being 1, self.notional will be applied later)
        We can add extra tagging in the signal input, just set extra_cols list in run func
        '''
        path=self.path
        self.mkt=(feather.read_dataframe(path+'mkt.feather')
                    .set_index('date').resample('B').last().fillna(method='ffill'))
        self.vwap=(feather.read_dataframe(path+'vwap.feather')
                    .set_index('date').resample('B').last().fillna(method='ffill'))
        # We can have vwap=0, we ffill here but take care of it in the signal input section
        self.vwap=self.vwap.applymap(lambda x: np.nan if x==0 else x).fillna(method='ffill')

        self.signal=feather.read_dataframe(path+'%s.feather' % (signal_name))
        self.signal_name=signal_name
        return None
    def get_over_under(self,x):
        return x[:x.find('_VS_')], x[x.find('_VS_')+4:]
    def run(self,extra_cols=[],dump_trades=True):
        '''
        We assume vwap execution
        We book transaction cost in the end of trade
        We don't deal with overlapping/exiting etc here. We just calculate performance
        We output:
            cumulative dollar pnl
            notional exposure
            per trade stats
        we can pass extra col (e.g. exit type, ultimate_pair etc) from signal input
        '''
        mkt=self.mkt.copy()
        vwap=self.vwap.copy()
        signals=self.signal.copy()
        signals['size']=signals['size']*self.notional
        # drop the signal entries where we don't have the date in self.mkt/self.vwap
        signals=signals[signals['trade_date']>=mkt.index.min()]
        #signals=signals[signals['exit_date']<=mkt.index.max()]

        collector=[]

        signals=signals.set_index(['pair','trade_date'])

        for i,pair_dt in enumerate(signals.index):
            pair=pair_dt[0]
            trade_date=pair_dt[1]
            exit_date=signals.iloc[i]['exit_date']
            size=signals.iloc[i]['size']
            over,under=self.get_over_under(pair)
#            if pair=='2218-HK_VS_605198-CN':
#                pdb.set_trace()
#
#            if over=='921-HK' and under =='000921-CN' and trade_date==pd.datetime(2006,8,9):
#                # we have vwap =0
#                pdb.set_trace()
            if len(extra_cols)!=0:
                extra_cols_dict={}
                for col_i in extra_cols:
                    extra_cols_dict[col_i]=signals.iloc[i][col_i]

            # calculate the mtm pnl
            direction=1 if size>0 else -1
            size_abs=abs(size)
            shares=size_abs/vwap[[over,under]].loc[trade_date]
            if direction==1:
                shares[under]=shares[under]*(-1)
            else:
                shares[over]=shares[over]*(-1)
            if exit_date>=um.today_date():
                to_calc=pd.concat([
                    vwap[[over,under]].loc[trade_date:exit_date].iloc[[0]],
                    mkt[[over,under]].loc[trade_date:exit_date].iloc[1:-1],
                    mkt[[over,under]].loc[trade_date:exit_date].iloc[[-1]],
                    ],axis=0)
            else:
                to_calc=pd.concat([
                    vwap[[over,under]].loc[trade_date:exit_date].iloc[[0]],
                    mkt[[over,under]].loc[trade_date:exit_date].iloc[1:-1],
                    vwap[[over,under]].loc[trade_date:exit_date].iloc[[-1]],
                    ],axis=0)

            mtm=to_calc.diff().fillna(0)*shares
            mtm=mtm.sum(1).rename('mtm').to_frame()
            mtm['mtm_net']=mtm['mtm'].copy()
            mtm['mtm_net'].iloc[-1]=mtm['mtm_net'].iloc[-1]-size_abs*self.bps

            mtm['pair']=pair
            mtm['size']=size_abs
            mtm['direction']=direction
            mtm['trade_date']=trade_date
            mtm['exit_date']=exit_date
            if len(extra_cols)!=0:
                for col_i in extra_cols:
                    mtm[col_i]=extra_cols_dict[col_i]
            collector.append(mtm)
            if self.verbose:
                print ('%s done for strategy: %s' % (round(i/len(signals),2)*100,self.signal_name))
        trades=pd.concat(collector,axis=0).reset_index()
        if dump_trades:
            feather.write_dataframe(trades,self.path+'trades_%s.feather' % (self.signal_name))
        self.trades=trades
        return trades

    def get_trades_stats(self):
        '''
        WARNING: do not use the pct return.
        The method of dividing over previous equity value will leads to understated pct return as we keeps wining overtime
        Just use the old method, dollar pnl divided by average number of position
        No need to separate by direction. We can do this in the signal section
        This func only does the summary for the simple pair strategy. For ones with extra_cols, do the grouping yourself
        '''
        # get cumu pct return
        groupby=['date']
        trades=self.trades.copy()

        mtm=trades.groupby(groupby).sum()['mtm'].fillna(0)
        mtm_net=trades.groupby(groupby).sum()['mtm_net'].fillna(0)
        notional=trades.groupby(groupby).sum()['size'].fillna(0)
#        wealth=notional+mtm.cumsum()
#        pct_ret=(mtm.divide(wealth.shift(1).fillna(wealth.iloc[0])).fillna(0)+1).cumprod().rename('gross_pct')
#        pct_ret_net=(mtm_net.divide(wealth.shift(1).fillna(wealth.iloc[0])).fillna(0)+1).cumprod().rename('net_pct')
#
        cumu=pd.concat([
#                        pct_ret,pct_ret_net,
                        notional.rename('notional'),
                        mtm.cumsum().rename('gross_dollar'),
                        mtm_net.cumsum().rename('net_dollar'),
                        ],axis=1)

        # get per-trade stas
        groupby= ['trade_date','pair']
        per_trade_stats=pd.concat([
                trades.groupby(groupby).sum()[['mtm','mtm_net']],
                trades.groupby(groupby).last()['size'],
                trades.groupby(groupby).last()['exit_date'],
                trades.groupby(groupby).count()['mtm'].rename('holding_time'),
                ],axis=1)
        per_trade_stats['ret']=per_trade_stats['mtm']/per_trade_stats['size']
        per_trade_stats['ret_net']=per_trade_stats['mtm_net']/per_trade_stats['size']
        per_trade_stats['result']=per_trade_stats['ret_net'].map(lambda x: 'hit' if x>0 else 'miss')
        per_trade_stats=per_trade_stats.reset_index()
        col=['value']
        quick_summary=pd.DataFrame(columns=['sharpe','sharpe_net','ret','ret_net','hit_ratio','trade_count'],index=col)
        quick_summary['sharpe']=umath.get_sharpe(cumu['gross_dollar'].diff())
        quick_summary['sharpe_net']=umath.get_sharpe(cumu['net_dollar'].diff())
        quick_summary['ret']= per_trade_stats.mean()['ret']
        quick_summary['ret_net']=per_trade_stats.mean()['ret_net']
        quick_summary['trade_count']=len(per_trade_stats)
        quick_summary['hit_ratio']=len(per_trade_stats[per_trade_stats['result']=='hit'])/len(per_trade_stats)
        quick_summary=quick_summary.T
        quick_summary.index.name='stats'
        return quick_summary,per_trade_stats,cumu



if __name__ =='__main__':
    print ('ok')
    
    path="C:\\Users\\davehanzhang\\python_data\\connect\\southbound\\trades_and_models\\alpha_new_test\\"
    bt=BACKTESTER(path)
    bt.load_data(signal_name='signal')
    bt.vwap['cash']=1
    bt.mkt['cash']=1
    
    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(
            bt.signal.index[0],len(bt.signal.columns),
            'cash',
            bt.signal,normalize_wgt=False
        )
    
    
    # # Create the backtested performance, pretending index wgt not changing from 1yr ago
    # attr_path="C:\\Users\\hyin1\\temp_data\\attribution\\"
    # bt=BACKTESTER(attr_path+'pictet_single_stock_hedge\\',bps=0)
    # hedges=['top10','tpx','tpx400']
    # start_date=um.yesterday_date()-pd.tseries.offsets.DateOffset(years=1)
    # for hedge in hedges:
    #     bt.load_data(signal_name=hedge)
    #     bt.mkt['cash']=1
    #     bt.vwap['cash']=1
    #     bt.signal.index=[start_date]
    #     perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(start_date,len(bt.signal.columns),'cash',bt.signal,normalize_wgt=False)
    #     asdf
#    bt_path="C:\\Users\\hyin1\\temp_data\\attribution\\AH_allocation_combine_backtest\\"
#    bt=BACKTESTER(bt_path,bps=0)
#    bt.load_data(signal_name='port_0.8')
#    bt.mkt['cash']=1
#    bt.vwap['cash']=1
#    collector=[]
#    for wgt_i in np.arange(0,1.1,0.1):
#        wgt_i=round(wgt_i,2)
#        bt.signal=feather.read_dataframe(bt_path+'port_%s.feather' % (wgt_i)).set_index('date')
#        perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(pd.datetime(2014,12,31),um.yesterday_date(),'cash',bt.signal)
#        perf_pct['wgt']=wgt_i
#        collector.append(perf_pct)
#    perf_all=pd.concat(collector).reset_index()

#    # run some backtet first
#    path="C:\\Users\\hyin1\\temp_data\\ah_db\\PROD\\IndexSwitch\\"
#    collector=[]
#    bt=BACKTESTER(path,accurate_transaction_cost=True,bps=20)
#    bt.load_data(signal_name='Signal_XIN9I_hedged_Switch')
#    bt.mkt['cash']=1
#    bt.vwap['cash']=1
#
#    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(bt.signal.index[0],len(bt.signal.columns),'cash',bt.signal,
#                                                                 normalize_wgt=False)
#    perf_pct['norm_wgt']='no'
#    collector.append(perf_pct)
#    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(bt.signal.index[0],len(bt.signal.columns),'cash',bt.signal,
#                                                                 normalize_wgt=True)
#    perf_pct['norm_wgt']='yes'
#    collector.append(perf_pct)
#
#    bt.load_data(signal_name='Signal_XIN9I_unhedged_Switch')
#    bt.mkt['cash']=1
#    bt.vwap['cash']=1
#    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(bt.signal.index[0],len(bt.signal.columns),'cash',bt.signal,
#                                                                 normalize_wgt=False)
#    perf_pct['norm_wgt']='unhedged'
#    collector.append(perf_pct)
#
#
#

#    bt_path="Z:\\dave\\data\\backtester\\pair_test_quick_ah\\"
#    bt=BACKTESTER_PAIR(bt_path)
#    bt.load_data(signal_name='signal_lash')
#    trades=bt.run()
#    quick_summary,per_trade_stats,cumu=bt.get_trades_stats()
#    bt_path="C:\\Users\\hyin1\\temp_data\\jcm\\"
#    bt=BACKTESTER_PAIR(bt_path)
#    bt.load_data(signal_name='PT_pre_bb')
#    trades=bt.run()
#    quick_summary,per_trade_stats,cumu=bt.get_trades_stats()

#    path="Z:\\dave\\data\\connect\\southbound\\models_and_signals\\7.5\\"
#    bt=BACKTESTER(path,bps=0)
#    # dump the signal first
#    port=pd.read_csv("Z:\\dave\\data\\connect\\southbound\\quantum\\MaxReturn_CSJACSSM.csv",parse_dates=['date']).set_index('date')
#    bt.load_data(signal_name='TEMP_SB_PORT')
#
#
#    perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(pd.datetime(2015,1,30),len(port.columns),
#                                                             'HSI 21 Index',bt.signal,disable_compounding=False)
#
#
##    port_weight=shares_overtime['l-mkt']['l'].multiply(bt.mkt).applymap(lambda x: np.nan if x==0 else x)
#    port_weight=port_weight.iloc[::10]
#    port_weight=port_weight.apply(lambda x: x/x.sum(),axis=1)
#
#    perf_pct_w,perf_abs_w,shares_overtime_w,to_w,hlds_w=bt.run_with_weight( pd.datetime(2015,1,1),15,'HSI51 Index',port_weight)
#
    #perf_pct,perf_abs,shares_overtime,to,hlds=bt.run_with_weight(pd.datetime(2015,1,1),100,'HSI51 Index',quantum_signal)
    #perf_pct,perf_abs,sharpe,to,hlds=bt.run_with_weight(pd.datetime(2015,1,1),99999,'HSI51 Index',quantum_signal)
    #perf_pct[['l-mkt','l-mkt_net']].iloc[-1].map(lambda x: x**(252/len(perf_pct)))
#
#
##    # testing the pair backtester
#
#    bt_path="Z:\\dave\\data\\backtester\\pair_test\\"
#    bt=BACKTESTER_PAIR(bt_path)
#
#    bt.load_data(signal_name='signal')
#    trades=bt.run()
#    quick_summary,per_trade_stats,cumu=bt.get_trades_statas()
#

#    # dump some dummy data first
#    from fql.fql import Factset_Query
#    fq=Factset_Query(session=6778)
#    tickers=['7203-JP','AAPL-US','5-HK','000725-CN']
#    mkt_data=fq.get_ts(tickers,['px_last','vwap'],fx='USD')
#    feather.write_dataframe(mkt_data['px_last'].reset_index(),bt_path+'mkt.feather')
#    feather.write_dataframe(mkt_data['vwap'].reset_index(),bt_path+'vwap.feather')
#
#
#    import itertools
#    all_pairs=list(itertools.combinations(tickers, 2))
#
#    pair_list=[x[0]+'_VS_'+x[1] for x in all_pairs]
#
#    signals=pd.DataFrame(index=mkt_data.index,columns=pair_list)
#    for col in signals.columns:
#        over=col[:col.find('_VS_')]
#        under=col[col.find('_VS_')+4:]
#        signals[col]=mkt_data['px_last'][over]/mkt_data['px_last'][under]
#
#
#    signals=(signals-signals.rolling(63,min_periods=1).mean()).divide(signals.rolling(63,min_periods=1).std())
#
#    def get_signal(x):
#        if x>=3:
#            return -1
#        elif x<=-3:
#            return 0.5
#        else:
#            return np.nan
#    signals=signals.applymap(lambda x: get_signal(x))
#
#
#    signals=signals.stack().rename('size').reset_index()
#    signals['exit_date']=signals['date'].map(lambda x: x+21*pd.tseries.offsets.BDay())
#    signals=signals.rename(columns={'date':'trade_date','level_1':'pair'})
#
#    feather.write_dataframe(signals,bt_path+'signal.feather')
#
#    bt_path="Z:\\dave\\data\\backtester\\sb\\"
#    bt=BACKTESTER(bt_path)
#    signals=['flow_q','turnover_impact_q','stake_impact_q']
#    qs=['Q1','Q2','Q3','Q4','Q5']
#    signal='flow_q'
#    bt.load_data(signal_name=signal)
#    perf_pct,perf_abs,sharpe,to,hlds=bt.run_q(5,pd.datetime(2014,12,31),'HSI 21 Index',manual_q_ls=[True,'Q5','Q1'])
#
#
#
#    # load data
#    path="Z:\\dave\\data\\backtester\\sb\\"
#    sb_backtest=BACKTESTER(path)
#    start_date=pd.datetime(2015,1,1)
#
#
#    bench='HSI 21 Index'
#    sizes=[1,3,5,10,15,20]
#    shifts=[0]
#    signals=['signal_expanding_uniform_sector']
#
#
#    perf_collector=[]
#    to_collector=[]
#    hlds_collector=[]
#
#
#    for signal in signals:
#        sb_backtest.load_data(signal_name=signal)
#        signal_original=sb_backtest.signal.copy()
#        for shift in shifts:
#            for size in sizes:
#                sb_backtest.signal=signal_original.iloc[shift:].iloc[::21]
#                perf_pct,perf_abs,sharpe,to,hlds=sb_backtest.run(start_date,size,bench)
#                perf_pct['shift']=shift
#                perf_pct['size']=size
#                perf_pct['signal']=signal
#                perf_collector.append(perf_pct)
#
#                to['shift']=shift
#                to['size']=size
#                to['signal']=signal
#                to_collector.append(to)
#
#                hlds['shift']=shift
#                hlds['size']=size
#                hlds['signal']=signal
#                hlds_collector.append(hlds)
#
#        print ('finish %s' % (signal))
#
#    perf_comp=pd.concat(perf_collector,axis=0)
#    perf_check=perf_comp.reset_index().groupby(['date','size','signal']).mean().unstack().unstack()
#
#    hlds_comp=pd.concat(hlds_collector,axis=0)
#    hlds_comp=hlds_comp.reset_index().rename(columns={'level_1':'ticker'})
#
#
#    # test qunitle backtest
#    countries=['KS','JP','HK','CH','AU']
#    factors=['momentum','valuation','quality','overall']
#    periods=[2003,2019]
#    path="Z:\\dave\\data\\backtester\\sars_vs_cov_%s\\"
#    bench='dummy'
#    quantile=5
#
#    for period in periods:
#        for country in countries:
#            for factor in factors:
#                strategy='%s_%s_%s' % (country,factor,period)
#
#                bt=BACKTESTER(path % (period))
#                bt.load_data(signal_name=strategy)
#                perf_pct,perf_abs,sharpe,to,hlds=bt.run_q(quantile,bt.signal.index[0],bench)
#
##
#    # test different weighting method
#    # load data
#    path="Z:\\dave\\data\\backtester\\sb\\"
#    start_date=pd.datetime(2015,1,1)
#    bench='HSI 21 Index'
#
#    signal='signal_expanding_winsor_sector'
#
#    perf_collector=[]
#    to_collector=[]
#    hlds_collector=[]
#
#    sb=BACKTESTER(path)
#    sb.load_data(signal_name=signal)
#    sb.signal=sb.signal.iloc[::21]
#
#    adv_wgt=feather.read_dataframe("Z:\\dave\\data\\connect\\southbound\\models_and_signals_REDO\\adv.feather")
#    adv_wgt=adv_wgt.set_index('date')
#
    #perf_pct,perf_abs,sharpe,to,hlds=sb.run_with_weight(start_date,15,bench,adv_wgt)
    #perf_pct,perf_abs,sharpe,to,hlds=sb.run(start_date,15,bench)
    #perf_pct,perf_abs,sharpe,to,hlds=sb.run_q(3,start_date,15,bench)
    #q=5
    #perf_pct,perf_abs,sharpe,to,hlds=sb.run_q(q,start_date,bench)
#
#
#    path="Z:\\dave\\data\\backtester\\jcm_jd\\"
#    jd=BACKTESTER(path)
#    jd.load_data(signal_name='signal')
#    signal_original=jd.signal.copy()
#
#    start_date=pd.datetime(2013,12,31)
#
#
#    shifts=np.arange(0,6,1)
#    size=9999 #include all
#    for shift in shifts:
#        jd.signal=signal_original.loc[start_date:].iloc[shift:].iloc[::6]
#        perf_pct,perf_abs,sharpe,to,hlds=jd.run(jd.signal.index[0],size,'mkt')
#    path="Z:\\dave\\data\\backtester\\jcm_jd\\"
#    jd=BACKTESTER(path)
#    jd.load_data()
#
#    sizes=[10,20,30,40,50,100,150,200]
#    collector=[]
#
#    for size in sizes:
#        perf_pct,perf_abs,sharpe,to,hlds=jd.run(pd.datetime(2014,12,31),size)
#        print ('finish %s' % (size))
#
#        res=perf_pct[['l-mkt','l-mkt_net']].copy()
#        res['size']=size
#        collector.append(res)
#
#
#    perf=pd.concat(collector,axis=0)

#
#
#
#


#path="Z:\\dave\\data\\backtester\\jcm_jd\\"
#mkt=feather.read_dataframe(path+'mkt.feather').set_index('date').resample('B').last().fillna(method='ffill')
#vwap=feather.read_dataframe(path+'vwap.feather').set_index('date').resample('B').last().fillna(method='ffill')
#vwap['mkt']=mkt['mkt']
#
## signal should be tradable (turnover !-0)
#signal=feather.read_dataframe(path+'signal.feather').set_index('date')
#
#start_date=pd.datetime(2014,12,31)
#signal=signal.loc[start_date:]
#
#bps=20/100/100#transaction cost
#bps_mkt=0/100/100
#notional=100
#
## get perf by abs number
#size=30
#
#l=signal.apply(lambda x: x.dropna().rank(ascending=False),axis=1).applymap(lambda x: x if x<=size else np.nan)
#s=signal.apply(lambda x: x.dropna().rank(ascending=True),axis=1).applymap(lambda x: x if x<=size else np.nan)
#
#to_l=l.applymap(lambda x: 0 if np.isnan(x) else 1).diff().abs().sum(1)/l.count(1)/2
#to_s=s.applymap(lambda x: 0 if np.isnan(x) else 1).diff().abs().sum(1)/s.count(1)/2
#to_l.iloc[0]=1
#to_s.iloc[0]=1
#to_mkt=to_l.map(lambda x: 0).copy()
#
#
#bsk_dict={'l':l,'s':s,'mkt':mkt[['mkt']]}
#to_dict={'l':to_l,'s':to_s,'mkt':to_mkt}
#
##loop through each date
#collector=[]
#for bsk_name in bsk_dict.keys():
#    for i,dt in enumerate(signal.index):
#        try:
#            dt_next=signal.index[i+1]
#        except IndexError:
#            dt_next=vwap.index[-1]
#
#        bsk_memb=bsk_dict[bsk_name].loc[dt].dropna().index.tolist()
#        shares=vwap.loc[dt][bsk_memb].map(lambda x: notional/len(bsk_memb)/x)
#        px_ss=pd.concat([
#                vwap[bsk_memb].loc[[dt]],
#                mkt[bsk_memb].loc[dt:dt_next].iloc[1:-1],
#                vwap[bsk_memb].loc[[dt_next]],
#                  ],axis=0)
#        bsk_px=px_ss.multiply(shares,axis='columns').sum(1).rename('gross').to_frame()
#        bsk_mtm=bsk_px.diff().fillna(0)
#        bsk_mtm['bsk']=bsk_name
#        levy=-1*to_dict[bsk_name][dt]*notional*bps
#        if i!=0:
#            levy=levy*2
#        if bsk_name=='mkt':
#            levy=bps_mkt
#        bsk_mtm['net']=bsk_px.diff().fillna(levy)
#        bsk_mtm['rebalance_dt']=dt
#        collector.append(bsk_mtm)
#
#mtm=pd.concat(collector,axis=0)
#
#perf=mtm.reset_index().groupby(['date','bsk']).sum().unstack()['gross'].cumsum()
#perf['l-s']=perf['l']-perf['s']
#perf['l-mkt']=perf['l']-perf['mkt']
#net=mtm.reset_index().groupby(['date','bsk']).sum().unstack()['net'].cumsum()
#levy=(perf.diff()-net.diff()).fillna(net.iloc[0].abs())
#levy['l-mkt']=levy['l']+levy['mkt']
#levy['l-s']=levy['l']+levy['s']
#
#perf['l-s_net']=perf['l-s']-levy['l-s'].cumsum()
#perf['l-mkt_net']=perf['l-mkt']-levy['l-mkt'].cumsum()
#
#
#gross_pct=umath.dollar_return_to_pct_return(perf[['l-s','l-mkt']],notional=notional)
#net_pct=umath.dollar_return_to_pct_return(perf[['l-s_net','l-mkt_net']],notional=notional)
#pct_daily=pd.concat([gross_pct,net_pct],axis=1)
#
#perf_pct=(pct_daily+1).cumprod()
#
#sharpe=umath.get_sharpe(pct_daily)
#
#
#
#
#
#
#



































