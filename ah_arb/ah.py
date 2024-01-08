# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:02:52 2020

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
from blp.util import get_bbg_usual_col, group_marcap,load_compo
import feather
import os
import utilities.display as ud
import utilities.mathematics as umath


import webscraping.eastmoney as em
from webscraping.ccass import get_h_share_list

from datetime import datetime


'''
We will operate on B-day index
We deal with circuit breaker in backtesting signal separately

We no longer do shift by 1 for beta here. We just refresh everything after market
close and assume execution on the next day

In case trading wants to get intraday fitting, we can set the start date and reduce
the downloading time.
'''

class AH():
    START=datetime(2005,1,1)
    def __init__(self,quick_mode=False):
        self.fq=Factset_Query(session=np.random.randint(1,99999999))
        #self.path="Z:\\dave\\data\\ah\\"
        self.path=uc.root_path_data+"ah_db\\"
        self.ah_list=pd.read_csv(self.path+'AH Pair.csv')
        # some fields to use, all will be in USD except for some macro indicator
        self.mkt_data_fs_fields=['px_last','vwap','marcap_sec']
        self.mkt_data_fs_zfill_fields=['volume','turnover']
        self.fund_data_fs_fields=['roe','roa','ebit_margin','ebitda_margin','debt_to_equity','asset_to_equity']
        self.mkt_data_bbg_fields=['EQY_FREE_FLOAT_PCT','TOT_ANALYST_REC','BEST_EPS']
        self.macro_data_bbg_fields=['VIX Index','HSAHP Index',
                       # risk free. There are 2 HK ones. HKGG10Y has longer history but missing value for recent series
                       'USGG10YR Index','GBFCY10Y Index','HKGG10Y Index','GCNY10YR Index',
                       # fwd points (driven by int differential but also reflects future spot move expectation?)
                       'HKDCNH3M Curncy','HKDCNH6M Curncy','HKDCNH12M Curncy',
                       ]
        self.macro_data_index_bbg_fields=['XIN9I Index','HSCEI Index']
        self.quick_mode=quick_mode
        return None
    def load_db(self):
        self.db_raw=feather.read_dataframe(self.path+'data_raw.feather').set_index(['date','ticker']).unstack()
        self.db=feather.read_dataframe(self.path+'db.feather').set_index(['date','ticker']).unstack()
        print('AH database loaded')
    def refresh_mkt(self,):
        quick_mode=self.quick_mode
        if quick_mode:
            print ('quick update enabled')
        start=self.START
        end=um.today_date()
        tickers_h=self.ah_list['HK'].tolist()
        tickers_a=self.ah_list['CN'].tolist()
        tickers=tickers_h+tickers_a

        # get FS mkt data
        print ('Getting data from FS')
        fields=self.mkt_data_fs_fields
        mkt_data_fs=self.fq.get_ts(tickers,fields,
                                   start=fql_date(start),end=fql_date(end),
                                   fx='USD')
        fields=self.mkt_data_fs_zfill_fields
        mkt_data_fs_zfill=self.fq.get_ts(tickers,fields,
                                   start=fql_date(start),end=fql_date(end),
                                   fx='USD')
        mkt_data_fs_zfill=mkt_data_fs_zfill.resample('B').last().fillna(0)
        # get FS reported fundamental data
        # China strategy report says ROE can distinguish AH cross-sectionally
        # Let's see if this is just related to size
        # The rbasis = auto leads to some A-share with na rpt freq dropped,
        #   This is not an issue as we can use the H-share data
        if not quick_mode:
            fields=self.fund_data_fs_fields.copy()
            fund_data_fs=self.fq.get_ts_reported_fundamental(tickers,fields,
                                                start=fql_date(start),end=fql_date(end),
                                                rbasis='ANN',auto_clean=False)
            # get BBG estimate and FF data
            # Seems no distinguish from FS eps est for A and H
            print ('Getting mkt data from BBG')
            fields=self.mkt_data_bbg_fields
            overrides={'BEST_FPERIOD_OVERRIDE':'1BF','EQY_FUND_CRNCY':'USD'}
            ## sometimes the BBG code may got stuck for unknown reason if we send the query for all tickers
            collector=[]
            for ticker in tickers:
                mkt_data_bbg_i=bdh(fs_to_bbg([ticker]),fields,start,end,
                             overrides=overrides)
                collector.append(mkt_data_bbg_i)
                print ('finish %s' % (ticker))
            mkt_data_bbg=pd.concat(collector,axis=0)
            mkt_data_bbg=mkt_data_bbg.reset_index()
            mkt_data_bbg['ticker']=bbg_to_fs(mkt_data_bbg['ticker'])
            mkt_data_bbg=mkt_data_bbg.set_index(['date','ticker']).unstack()
    #        mkt_data_bbg=bdh(fs_to_bbg(tickers),fields,start,end,overrides=overrides)
    #        mkt_data_bbg=mkt_data_bbg.reset_index()
    #        mkt_data_bbg['ticker']=bbg_to_fs(mkt_data_bbg['ticker'])
    #        mkt_data_bbg=mkt_data_bbg.set_index(['date','ticker']).unstack()
            # get BBG macro index
        print ('Getting macro data from BBG')
        fields=['px_last']
        tickers_macro=self.macro_data_bbg_fields
        macro_data_bbg=bdh(tickers_macro,fields,start,end)
        macro_data_bbg=macro_data_bbg.reset_index()
        macro_data_bbg['ticker']=macro_data_bbg['ticker'].map(lambda x: x.replace(' Index','').replace(' Curncy',''))
        macro_data_bbg=macro_data_bbg.set_index(['date','ticker']).unstack()
        # get BBG equity index
        print ('Getting idx data from BBG')
        fields=['px_last']
        tickers_macro=self.macro_data_index_bbg_fields
        macro_data_index_bbg=bdh(tickers_macro,fields,start,end,currency='USD')
        macro_data_index_bbg=macro_data_index_bbg.reset_index()
        macro_data_index_bbg['ticker']=macro_data_index_bbg['ticker'].map(lambda x: x.replace(' Index',''))
        macro_data_index_bbg=macro_data_index_bbg.set_index(['date','ticker']).unstack()
        # load from loacl the hardcoded implied market risk premium spread
        # erp has incorporated the effect rf differential
        # idr_erp=pd.read_csv(self.path+'idr_erp.csv',parse_dates=['date']).set_index('date')
        # idr_erp.columns.name='ticker'
        print ('Combining all')
        #combine everything
        if quick_mode:
            data=(mkt_data_fs_zfill.stack()
                .join(mkt_data_fs.stack(),how='left')
               # .join(fund_data_fs.stack(),how='left')
                #.join(mkt_data_bbg.stack(),how='left')
                )
            data=data.reset_index().set_index('date')
            data=data.join(macro_data_bbg['px_last'],how='left')
            data=data.join(macro_data_index_bbg['px_last'],how='left')
            # data=data.join(idr_erp,how='left')
        else:
            data=(mkt_data_fs_zfill.stack()
                .join(mkt_data_fs.stack(),how='left')
                .join(fund_data_fs.stack(),how='left')
                .join(mkt_data_bbg.stack(),how='left')
                )
            data=data.reset_index().set_index('date')
            data=data.join(macro_data_bbg['px_last'],how='left')
            data=data.join(macro_data_index_bbg['px_last'],how='left')
            # data=data.join(idr_erp,how='left')
        data=data.reset_index().set_index(['date','ticker']).unstack()
        data=data.resample('B').last().fillna(method='ffill')
        # add the tagging
        tags=get_bbg_usual_col( fs_to_bbg(tickers),add_short_industry=True)
        tags=tags[['Sector','Industry','Name']]
        tags.index.name='ticker'
        tags.index=bbg_to_fs(tags.index)
        data=data.stack().reset_index().set_index('ticker')
        data['sector']=tags['Sector']
        data['industry']=tags['Industry']
        data['name']=tags['Name']
        feather.write_dataframe(data.reset_index(),self.path+'data_raw.feather')

        return None
    def refresh_db(self):
        quick_mode=self.quick_mode
        # convert to pair data
        data=feather.read_dataframe(self.path+'data_raw.feather')
        # add some columns
        if not quick_mode:
            data['ff_sec']=data['marcap_sec']*data['EQY_FREE_FLOAT_PCT']/100
        else:
            data['ff_sec']=data['marcap_sec']
        data=data.set_index(['date','ticker']).unstack().swaplevel(1,0,1)
        tickers_h=self.ah_list['HK'].tolist()
        tickers_a=self.ah_list['CN'].tolist()
        #ha_map=self.get_ah_map(direction='h_to_a')
        ah_map=self.get_ah_map(direction='a_to_h')

        data_h=data[tickers_h].swaplevel(1,0,1).copy()
        data_a=data[tickers_a].rename(columns=ah_map,level=0).swaplevel(1,0,1).copy()
        # some basic inputs
        ratio=data_h['px_last'].divide(data_a['px_last'])
        px_h=data_h['px_last']
        px_a=data_a['px_last']
        adv_h=data_h['turnover'].applymap(lambda x: np.nan if x==0 else x).rolling(63,min_periods=1).mean()
        adv_a=data_a['turnover'].applymap(lambda x: np.nan if x==0 else x).rolling(63,min_periods=1).mean()
        adv_min=pd.concat([adv_h.stack(),adv_a.stack()],axis=1).min(1)
        to_both=data_h['turnover']+data_a['turnover']
        to_h_pct=data_h['turnover'].divide(to_both.applymap(lambda x: np.nan if x==0 else x))
        volume_both=data_h['volume']+data_a['volume']
        volume_h_pct=data_h['volume'].divide(volume_both.applymap(lambda x: np.nan if x==0 else x))
        marcap_both=data_h['marcap_sec']+data_a['marcap_sec']
        marcap_h_pct=data_h['marcap_sec'].divide(marcap_both.applymap(lambda x: np.nan if x==0 else x))
        ff_both=data_h['ff_sec']+data_a['ff_sec']
        ff_h_pct=data_h['ff_sec'].divide(ff_both.applymap(lambda x: np.nan if x==0 else x))
        if not quick_mode:
            anr_both=data_h['TOT_ANALYST_REC'].fillna(0)+data_a['TOT_ANALYST_REC'].fillna(0)
            anr_h_pct=data_h['TOT_ANALYST_REC'].divide(anr_both.applymap(lambda x: np.nan if x==0 else x))
            # we can have negative eps
            best_eps_diff_h=data_h['BEST_EPS'].divide(data_h['BEST_EPS'].abs()+data_a['BEST_EPS'].abs() )
            # for fundamental, we just use H- leg. Seems there are some rbasis difference of AH leg of the same pair
            funda_fix=data_h[self.fund_data_fs_fields].copy()
        # calculate some macro indicators
        ha_avg_simple=ratio.mean(1)
        ha_median=ratio.median(1)
        ha_avg_marcap=(ratio.multiply(marcap_both).sum(1))/(marcap_both.sum(1))
        ha_avg_ff=(ratio.multiply(ff_both).sum(1))/(ff_both.sum(1))
        ha_idx_ratio=data_h['HSCEI'].mean(1)/data_h['XIN9I'].mean(1)
        hsahp_inverse=data_h['HSAHP'].mean(1).map(lambda x: 100/x)
#        ratio_rv_1m=ratio.rolling(21).std()
#        ratio_rv_3m=ratio.rolling(63).std()
        ratio_rv_12m=ratio.rolling(252).std()
#        corr_1m=data_h['px_last'].pct_change().rolling(21).corr(data_a['px_last'].pct_change())
#        corr_3m=data_h['px_last'].pct_change().rolling(63).corr(data_a['px_last'].pct_change())
        corr_12m=data_h['px_last'].pct_change().rolling(252).corr(data_a['px_last'].pct_change())
        # get the time series beta
        # since rolling regression cannot handle missing value we only keep the pair with 1-year data
        tickers_to_calc=ratio.columns[ratio.count()>253].tolist() # because we need to do percentage change!
        to_calc_beta=ratio[tickers_to_calc].stack().rename('ratio').reset_index().set_index('date')
        to_calc_beta['ha_avg_simple']=ha_avg_simple
        to_calc_beta['ha_avg_median']=ha_median
        to_calc_beta['ha_avg_marcap']=ha_avg_marcap
        # to_calc_beta['ha_avg_ff']=ha_avg_ff  # NOTE: no ff data until 2009, which affects KF beta as we drop na in the groupby
        to_calc_beta['ha_idx_ratio']=ha_idx_ratio
        to_calc_beta['hsahp']=hsahp_inverse
        to_calc_beta_level=to_calc_beta.reset_index().copy()
        to_calc_beta=to_calc_beta.reset_index().set_index(['date','ticker']).unstack().pct_change().stack().reset_index()

        # so the beta here is for the hedge ratio
        collector=[]
        for bench in to_calc_beta.drop(['ratio','date','ticker'],1).columns.to_list():
            beta_i=to_calc_beta.groupby('ticker').apply(lambda x: umath.rolling_regression(x.set_index('date').sort_index(),'ratio',[bench],252).beta)
            collector.append(beta_i)
            # add the KF filter beta here
            beta_i_KF=to_calc_beta.groupby('ticker').apply(lambda x: umath.KF_beta(x.dropna().set_index('date').sort_index(),'ratio',bench)['beta'])
            beta_i_KF=beta_i_KF.rename('%s_KF' % (bench)).to_frame()
            collector.append(beta_i_KF)
        betas=pd.concat(collector,axis=1)
        betas=betas.swaplevel(1,0,0).unstack()
        betas=betas.resample('B').last().fillna(method='ffill')#.shift(1)
        betas=betas.stack()
        betas.columns=betas.columns.map(lambda x: x+'_beta')
        # combine everything
        # join (ticker, time series)
        if not quick_mode:
            res=(ratio.stack().rename('ratio').to_frame()
                .join(px_h.stack().rename('px_h').to_frame())
                .join(px_a.stack().rename('px_a').to_frame())
                .join(adv_min.rename('adv_min').to_frame())
                .join(to_both.stack().rename('to_both').to_frame())
                .join(to_h_pct.stack().rename('to_h_pct').to_frame())
                .join(volume_both.stack().rename('volume_both').to_frame())
                .join(volume_h_pct.stack().rename('volume_h_pct').to_frame())
                .join(marcap_both.stack().rename('marcap_both').to_frame())
                .join(marcap_both.stack().map(np.log).rename('marcap_both_log').to_frame())
                .join(marcap_h_pct.stack().rename('marcap_h_pct').to_frame())
                .join(ff_both.stack().rename('ff_both').to_frame())
                .join(ff_both.stack().map(np.log).rename('ff_both_log').to_frame())
                .join(ff_h_pct.stack().rename('ff_h_pct').to_frame())
                .join(anr_both.stack().rename('anr_both').to_frame())
                .join(anr_h_pct.stack().rename('anr_h_pct').to_frame())
                .join(best_eps_diff_h.stack().rename('best_eps_diff_h').to_frame())
                .join(funda_fix.stack())
                .join(ratio_rv_12m.stack().rename('ratio_rv').to_frame())
                .join(corr_12m.stack().rename('corr').to_frame())
                .join(betas)
                )
        else:
            res=(ratio.stack().rename('ratio').to_frame()
                .join(px_h.stack().rename('px_h').to_frame())
                .join(px_a.stack().rename('px_a').to_frame())
                .join(adv_min.rename('adv_min').to_frame())
                .join(to_both.stack().rename('to_both').to_frame())
                .join(to_h_pct.stack().rename('to_h_pct').to_frame())
                .join(volume_both.stack().rename('volume_both').to_frame())
                .join(volume_h_pct.stack().rename('volume_h_pct').to_frame())
                .join(marcap_both.stack().rename('marcap_both').to_frame())
                .join(marcap_both.stack().map(np.log).rename('marcap_both_log').to_frame())
                .join(marcap_h_pct.stack().rename('marcap_h_pct').to_frame())
                .join(ff_both.stack().rename('ff_both').to_frame())
                .join(ff_both.stack().map(np.log).rename('ff_both_log').to_frame())
                .join(ff_h_pct.stack().rename('ff_h_pct').to_frame())
                #.join(anr_both.stack().rename('anr_both').to_frame())
                #.join(anr_h_pct.stack().rename('anr_h_pct').to_frame())
                #.join(best_eps_diff_h.stack().rename('best_eps_diff_h').to_frame())
                #.join(funda_fix.stack())
                .join(ratio_rv_12m.stack().rename('ratio_rv').to_frame())
                .join(corr_12m.stack().rename('corr').to_frame())
                .join(betas)
                )
        # join time series
        res=res.reset_index().set_index('date')
        res=res.join(to_calc_beta_level.groupby('date').last().drop(['ticker','ratio'],1))
        # join ticker
        res=res.reset_index().set_index('ticker')
        res=res.join(data.swaplevel(1,0,1)[['sector','industry','name']].iloc[-1].unstack().T)
        # dump
        feather.write_dataframe(res.reset_index(),self.path+'db.feather')
        return None
    def get_fwd_ratio_return(self,ret_type,bench,window):
        '''
        This fwd ratio move is actually an accurate estimation
        ret_type can be raw, simple_adj, beta_adj
        simple_adj means notional flat move
        bench can be one of the
        'ha_avg_simple', 'ha_avg_median', 'ha_avg_marcap',
        'ha_avg_ff', 'ha_idx_ratio'
        for beta_adj, we use KF beta
        '''
        db=self.db.copy()
        raw_move=db['ratio'].pct_change(window).shift(-window)
        if ret_type=='raw':
            return raw_move
        elif ret_type=='simple_adj':
            bench_move=db[bench].mean(1).pct_change(window).shift(-window)
            return raw_move.subtract(bench_move,axis=0)
        elif ret_type=='beta_adj':
            bench_beta=db[bench+'_KF_beta']
            bench_move=db[bench].mean(1).pct_change()
            beta_move=bench_beta.multiply(bench_move,axis='index')
            beta_move=((beta_move+1).applymap(np.log).rolling(window).sum().applymap(np.exp)-1).shift(-window)
            return raw_move-beta_move

        return None
    def get_band(self, window, band_type=['bollinger','na'],
                 auto_mean_periods=[True,'user defined default is 1/3 of window']):
        '''
        z_type[0] can be 3 types:
            'bollinger','bollinger_adj_simple','bollinger_adj_KF'
            'regression' ('KF' no longer recommended)
        if 'regression' or 'bollinger_adj'  then we need to specify bench_name to be one of the below:
            'ha_avg_simple', 'ha_avg_median', 'ha_avg_marcap',
            'ha_avg_ff', 'ha_idx_ratio'
        (given the similarity we can just choose between 3 bench:
            ha_avg_simple,ha_avg_marcap,and ha_idx_ratio)
        We no longer shift 1 days for the rolling regression, we just assume execution on the next day
        # DO NOT USE KF on levels as the results look bad (moving too fast!)
        if 'KF' then we need to specify bench to use:
            benchmark to use: 'ha_avg_simple', 'ha_avg_median', 'ha_avg_marcap',
            'ha_avg_ff', 'ha_idx_ratio'
            since the bench is at similar scale to the individual ratio, we do not adjust bench level
        '''
        min_periods=int(window/3)
        if band_type[0] in ['bollinger','bollinger_adj_simple','bollinger_adj_KF']:
            if band_type[0]=='bollinger':
                ratios=self.db['ratio']
            elif band_type[0]=='bollinger_adj_simple':
                bench=band_type[1]
                ratios=self.db['ratio'].divide(self.db[bench])
            elif band_type[0]=='bollinger_adj_KF':
                bench=band_type[1]
                ratios=self.db['ratio'].divide(self.db[bench]).divide(self.db['%s_KF_beta' % (bench)])
            mean=ratios.rolling(window,min_periods=min_periods).mean().stack()
            std=ratios.rolling(window,min_periods=min_periods).std().stack()
            zscore=(ratios.stack()-mean)/std
        elif band_type[0]=='regression':
            bench=band_type[1]
            ratios=self.db['ratio']
            data=self.db[['ratio',bench]].stack().reset_index()
            data=data.set_index('ticker')
            data['count']=data.reset_index().groupby('ticker').count()['date']
            data=data[data['count']>window].reset_index()
            reg_res=data.groupby('ticker').apply(lambda x: umath.rolling_regression( x.set_index('date'),'ratio',[bench],window))
            reg_collector=[]
            # we do the extraction
            for ticker in reg_res.index:
                reg_i=pd.concat([
                    reg_res[ticker].alpha.rename('alpha'),
                    reg_res[ticker].beta[bench].rename('beta'),
                    reg_res[ticker].std_err.rename('std_err')],
                    axis=1)#.shift(1)
                reg_i['ticker']=ticker
                reg_collector.append(reg_i)
            reg_res_para=pd.concat(reg_collector).reset_index().set_index(['date','ticker']).sort_index()
            reg_res_para=(reg_res_para
                        .join(ratios.stack().rename('ratio').to_frame())
                        .join(self.db[bench].stack().rename(bench).to_frame())
                        )
            mean=reg_res_para[bench]*reg_res_para['beta']+reg_res_para['alpha']
            std=reg_res_para['std_err']
            zscore=(reg_res_para['ratio']-mean)/std
#        elif band_type[0]=='KF':
#            bench=band_type[1]
#            data=self.db[['ratio',bench]].stack().reset_index()
#            data=data.set_index('ticker')
#            data['count']=data.reset_index().groupby('ticker').count()['date']
#            data=data[data['count']>window].reset_index()
#            reg_res=data.groupby('ticker').apply(lambda x: umath.KF_beta( x.set_index('date'),'ratio',bench))
#            reg_collector=[]
#            # we do the extraction
#            for ticker in reg_res.index.levels[0]:
#                reg_i=reg_res.loc[ticker]
#                reg_i['ticker']=ticker
#                reg_collector.append(reg_i)
#
#            reg_res_para=pd.concat(reg_collector).reset_index().set_index(['date','ticker']).sort_index()
#            reg_res_para=(reg_res_para
#                        .join(self.db['ratio'].stack().rename('ratio').to_frame())
#                        .join(self.db[bench].stack().rename(bench).to_frame())
#                        )
#
#            mean=reg_res_para[bench]*reg_res_para['beta']+reg_res_para['intercept']
#            std=self.db['ratio'].rolling(window,min_periods=min_periods).std().stack()
#            zscore=(reg_res_para['ratio']-mean)/std
        # output the sandardized data
        band=(ratios.stack().rename('ratio').to_frame()
            .join(mean.rename('mean').to_frame())
            .join(std.rename('std').to_frame())
            .join(zscore.rename('zscore').to_frame()))
        band['band_type']=band_type[0]
        band['bench_name']=band_type[1]
        band['window']=window
        # add the mid crossing
        band['ratio_last']=band['ratio'].unstack().shift(1).stack()
        band['mean_last']=band['mean'].unstack().shift(1).stack()
        band['cross_up']=band.apply(lambda x: 1 if x['ratio_last']<x['mean_last'] and x['ratio']>=x['mean'] else 0,axis=1)
        band['cross_down']=band.apply(lambda x: 1 if x['ratio_last']>x['mean_last'] and x['ratio']<=x['mean'] else 0,axis=1)
        band['cross']=band[['cross_up','cross_down']].sum(1).map(lambda x: 1 if x!=0 else 0)
        return band.drop(['ratio_last','mean_last'],axis=1)
    def get_cross_sectional_ratio_deviation(self,adj_by):
        to_calc=self.db[['ratio',adj_by]].stack().reset_index().copy()
        to_calc['residual']=umath.grouped_regression(to_calc,'date','ratio',[adj_by],'residual')
        res=to_calc.set_index(['date','ticker'])['residual'].unstack()
        return res
    def get_overall_ah_level(self):
        fields=['ha_avg_simple','ha_avg_median','ha_avg_marcap','ha_idx_ratio','hsahp']
        return self.db.stack()[fields].reset_index().groupby('date').last().drop('ticker',1)
    def dump_adf(self):
        '''
        we output 9 ADF pvalues, with 3 year lookback
        '''
        from statsmodels.tsa.stattools import adfuller as adf
        def _get_adf(df):
            res=(df.apply(lambda x: x.dropna().rolling(252*3)
                    .apply(lambda y: adf(y,regression='ct',autolag=None)[1],raw=True))
                .reset_index()
                )
            return res
        benches=['ha_avg_simple','ha_avg_marcap','ha_idx_ratio','hsahp']
        # raw adf
        df=self.db['ratio'].copy()
        feather.write_dataframe(_get_adf(df),self.path+'adf_raw.feather')
        # simple adj adf
        for bench in benches:
            df=self.db['ratio'].divide(self.db[bench])
            feather.write_dataframe(_get_adf(df),
                                    self.path+'adf_simple_adj_%s.feather' % (bench))
        # beta adj adf
        for bench in benches:
            df=self.db['ratio'].divide(self.db[bench]).divide(self.db[bench+'_KF_beta'])
            feather.write_dataframe(_get_adf(df),
                                    self.path+'adf_beta_adj_%s.feather' % (bench))
    def dump_pair_distance(self,include_dtw=True):
        '''
        we output with 3 year lookback the rolling distance of all the pairs
        we have 2 types of distance: euclidean and dtw
        dtw seems to be very slow to run?
        '''
        from utilities.mathematics import get_rolling_distance
        if include_dtw:
            distance_types=['euclidean','dtw'] # dtw takes about 3h to run??
        else:
            distance_types=['euclidean']
        lookback=252*3
        to_calc=self.db[['px_h','px_a']].stack().reset_index()
        collector=[]
        for distance_type in distance_types:
            check_distance=to_calc.groupby('ticker').apply(lambda x:
                get_rolling_distance(x.set_index('date'),'px_a','px_h',lookback,method=distance_type) )
            collector.append(check_distance)
        res=pd.concat(collector).reset_index()
        if include_dtw:
            feather.write_dataframe(res,self.path+'pair_distance.feather')
        else:
            feather.write_dataframe(res,self.path+'pair_distance_euclidean_only.feather')
    def get_ah_map(self,direction='a_to_h'):
        if direction=='a_to_h':
            return self.ah_list.set_index('CN')['HK'].to_dict()
        else:
            return self.ah_list.set_index('HK')['CN'].to_dict()
    def update_ah_map(self):
        '''
        We need a different way to screen for AH universe as it's definition is broadened
        The old definition requires company to be incorporated in Mainland
        But recent high profie listing like 981-HK (KY incorporated), which is enabled by the SH STAR board means we need to generalized the AH definition
        We use 2 sources of data to construct the AH list from a list of H tickers
        1. HSCI compo check using MULTIPLE_SHARE_INFO
        2. HKEx H-share list using MULTIPLE_SHARE_INFO
        HSCI one covers 95% of HK marcap, and captures non-conventional high-profile AH like SMIC
        HKEx H-share list covers all the "conventional" AH. Some smaller cap AH may be missed from HSCI
        This way we should be able to capture all the conventional AH
        The only risk is we miss out non-conventional AH from 5% of the HK marcap
        However this is unlikely as only large cap non-conventional AH can list in STAR
        If such cases occurs we modify manually
        We finally cross reference with eastmoney.
        (Eastmoney appears to include only the true AH and can have proxy download issue.)
        '''
        path=self.path
        # run this first so we don't waste time if it has the "need manual trigger" error
        em_ah=em.get_ah_list()

        # get h-tickers from HSCI
        
        compo=load_compo('HSCEI Index')
        h_tickers_hsci=compo.groupby('ticker').last().index.to_list()
        # get h-tickers from HKEx
        h_tickers_hkex=get_h_share_list()['ticker'].tolist()
        tickers_s=pd.Series(index=h_tickers_hsci+h_tickers_hkex).fillna(0)
        tickers=tickers_s.reset_index().groupby('index').last().index.tolist()
        collector=[]
        for ticker in tickers:
            try:
                info_i=bds([ticker],'MULTIPLE_SHARE_INFO')
                print ('Get multi-share infor for %s' % (ticker))
                collector.append(info_i)
            except:
                continue
        info_all=pd.concat(collector)
        info_all['has_CH']=info_all['Ticker'].map(lambda x: True if x.find(' CH')!=-1 else False)
        info_all['is_A']=info_all['Security Description'].map(lambda x: True if x.find('A ')!=-1 else False)
        info_all=info_all[(info_all['has_CH']) & (info_all['is_A'])]
        info_all=info_all.rename(columns={'Ticker':'ticker_a'})
        info_all['ticker_a']=info_all['ticker_a'].map(lambda x: x+' Equity')
        info_all.index.name='ticker_h'
        info_all['a_ticker_length']=info_all['ticker_a'].map(lambda x: len(x))
        info_all=info_all[info_all['a_ticker_length']==16]
        info_all=info_all.reset_index()
        info_all=info_all[info_all['ticker_h'].map(lambda x: len(x)<=14)]
        info_all=info_all.groupby(['ticker_h','ticker_a']).last()
        info_all=info_all.reset_index().set_index('ticker_h')
        # get from east money and compare
        em_ah['ticker_h']=em_ah['ticker_h'].map(lambda x: x+' Equity')
        em_ah['ticker_a']=em_ah['ticker_a'].map(lambda x: x+' Equity')
        # combine the 2
        info_all=info_all.reset_index().set_index(['ticker_h','ticker_a'])
        info_all['source_1']='bbg'
        em_ah=em_ah.reset_index().set_index(['ticker_h','ticker_a'])
        em_ah['source_2']='em'
        full_list=pd.concat([info_all['source_1'],em_ah['source_2']],axis=1)
        full_list.index.names=['HK','CN']
        full_list=full_list.reset_index()
        full_list['HK']=bbg_to_fs(full_list['HK'])
        full_list['CN']=bbg_to_fs(full_list['CN'])
        full_list.set_index('HK')[['CN']].to_csv(path+'AH Pair.csv')
        
        
        
if __name__=='__main__':
    print ('ok')
    #---- test
    # ah=AH(quick_mode=False)
    # ah.refresh_mkt()
    # ah.refresh_db()
    
    
    
#    ah.load_db()
    #ah.get_overall_ah_level()
#    res=umath.rolling_mean_reversion_half_life(ah.db['ratio']['1398-HK'],252)
#
#    check=(ah.db['ratio']
#    .apply(lambda x: umath.rolling_mean_reversion_half_life(x,252),axis=0))
    #ah.dump_adf()
    #res=ah.get_cross_sectional_ratio_deviation('marcap_both_log')

#    check=ah.db[['px_h','px_a']].swaplevel(1,0,1)['1398-HK']
#    check_distance=umath.get_rolling_distance(check.dropna(),'px_h','px_a',252,method='dtw')
#
#    band_kf=ah.get_band(252,band_type=['KF','ha_avg_simple'])
#    band_r=ah.get_band(126,band_type=['regression','ha_avg_simple'])
#    band=pd.concat([band_z,band_r],axis=0)
#
#    ratio_move_raw=ah.get_fwd_ratio_return('raw','ha_avg_simple',63)
#    ratio_move_simple_adj=ah.get_fwd_ratio_return('simple_adj','ha_avg_simple',63)
#    ratio_move_beta_adj=ah.get_fwd_ratio_return('beta_adj','ha_avg_simple',63)
#
#    ratio_move=pd.concat([ratio_move_raw.stack().rename('raw'),
#                          ratio_move_simple_adj.stack().rename('simple_adj'),
#                          ratio_move_beta_adj.stack().rename('beta_adj'),
#                          ],axis=1)

















