# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:53:00 2021

@author: davehanzhang
"""


import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
from fql.fql import Factset_Query
import utilities.mathematics as umath
from blp.util import get_bbg_nice_compo_hist_no_limit,load_compo
from fql.util import bbg_to_fs,fs_to_bbg,fql_date
import os
import pdb
from blp.bdx import bdh
import feather
from datetime import datetime
from backtester.backtester import BACKTESTER
from sklearn.linear_model import LinearRegression # use this for growth factor calculation


'''
Note that for ret_daily on suspended names
'''

# for class method to load quickly
quick_load_dict={
                'NB':[['SHCOMP_L','SZCOMP_L'],'CNY'],
                'SB':[['HSCI'],'HKD']
                }
model_factor_field_map={
    
    'alpha':[
             ['fx','beta','lqdt','mom','size','vol','value','lev','growth','prof','div'], # factors, excluding market
             ['ret_daily','turnover','marcap_sec','pb','marcap_comp',
                'net_income','net_income_est','debt_to_equity','asset_to_equity',
                'sales','sales_est','roe','roa','cfo','asset','div_yield','gross_margin'], # relevant fields
             ['asset', 'asset_to_equity', 'cfo', 'debt_to_equity', 'div_yield',
                  'marcap_comp', 'marcap_sec', 'net_income', 'net_income_est', 'pb',
                  'roa', 'roe', 'sales', 'sales_est','sales_FE','gross_margin'], # ffill fields
             ['ret_daily','turnover'], # zero fill fields
             ],
    
    'carhart':[
              ['mom','size','value'], # factors, excluding market
              ['ret_daily','pb','marcap_comp','net_income','net_income_est'], # relevant fields
              ['marcap_comp', 'net_income', 'net_income_est', 'pb',], # ffill fields
              ['ret_daily'], # zero fill fields
             ]
    }



class Alpha_Model():
    
    MODEL_NAME='alpha'
    
    PATH=uc.root_path_data+"excess_return\\%s\\" % (MODEL_NAME)
    MAX_DAILY_MOVE=0.5
    DAYCOUNT_YEAR=252
    DAYCOUNT_MONTH=21
    
    FACTORS=model_factor_field_map[MODEL_NAME][0]
    FIELDS=model_factor_field_map[MODEL_NAME][1]
    FFILL_FIELDS=model_factor_field_map[MODEL_NAME][2]
    ZFILL_FIELDS=model_factor_field_map[MODEL_NAME][3]
    
    def __init__(self,
                 universe, # a list of index, in BBG index format
                 fx, # currency
                 quantile=3,
                 regression_para=[5,504], # this applies to both stock beta to factors and beta estimation when constructing factor
                 force_start_date=[False,'some date'] # ideally we want to avoid huge change in universe due to MEMB availability issue. We can force a start date for stable universe
                 ):
        self.universe=universe
        self.fx=fx
        self.quantile=quantile
        self.regression_para=regression_para
        self.force_start_date=force_start_date
        
        self.fq=Factset_Query()
        self.bt=BACKTESTER('dummy_path',bps=0,bps_mkt=0)
        
        self.universe_name='_'.join(self.universe)
        
        return None
    
    @classmethod
    def Load_Model_Quick(cls,name):
        '''
        We only load the ret decomp for excess return calculation
        '''
        model_specs=quick_load_dict[name]
        obj=cls(model_specs[0],model_specs[1])
        obj.load_ret_decomp(strategy_name=name)
        return obj

    def update_factor_universe(self):
        
        universe_name=self.universe_name
        universe_tickers=[x+' Index' for x in self.universe]        
        universe_tickers_total_ret=[uc.total_ret_index_map[x] for x in universe_tickers]
        
        #---- load compo
        collector=[]
        for universe_ticker in universe_tickers:
            compo_i=load_compo(universe_ticker)
            collector.append(compo_i)
        compo=pd.concat(collector).reset_index()

        # compo=compo[compo['date']<=datetime(2018,12,31)] # temp for testing
        
        compo['ticker']=bbg_to_fs(compo['ticker'])
        compo=compo.set_index(['date','ticker'])['wgt'].unstack().fillna(0).resample('B').fillna(method='ffill')
        compo=compo.applymap(lambda x: 0 if x==0 else 1)
        start_date=compo.index[0] if not self.force_start_date[0] else self.force_start_date[1]
        
        #---- get index price as mkt
        # In case of multiple indices in universe, we assume daily rebalancing with equal weighting
        mkt_levels=bdh(universe_tickers_total_ret,['px_last'],start_date,um.today_date(),currency=self.fx)['px_last'].unstack().T
        mkt_levels=mkt_levels.resample('B').last().fillna(method='ffill')
        mkt_levels=(mkt_levels.pct_change().mean(1).fillna(0)+1).cumprod()
        feather.write_dataframe(mkt_levels.rename('mkt').reset_index(),self.PATH+'%s_MktReturn.feather' % (universe_name))

        #---- get DXY as proxy for FX factor
        # In case of multiple indices in universe, we assume daily rebalancing with equal weighting
        fx_levels=bdh(['DXY Curncy'],['px_last'],start_date,um.today_date())['px_last'].unstack().T
        fx_levels=fx_levels.resample('B').last().fillna(method='ffill')
        fx_levels=(fx_levels.pct_change().mean(1).fillna(0)+1).cumprod()
        feather.write_dataframe(fx_levels.rename('dxy').reset_index(),self.PATH+'%s_DXYLevel.feather' % (universe_name))
        
        
        #---- get mkt data
        tickers=compo.columns.tolist()
        file_name=self.PATH+'%s_MktData.feather' % (universe_name)
        
        if not os.path.isfile(file_name):    
            print ('No existing mkt data found for %s, start downloading now' % (universe_name))
            mkt_data=self._get_mkt_data(tickers,
                                        start=fql_date(start_date),
                                        end=fql_date(um.today_date()),
                                        # end=fql_date(datetime(2018,12,31)),# temp for testing
                                        mode='new'
                                        )                                                                                
        else:
            print ('Top up updating mkt data for %s' % (universe_name))
            mkt_data_old=feather.read_dataframe(file_name)
            tickers_old=mkt_data_old.groupby('ticker').last().index.tolist()
            start_old=max(mkt_data_old['date'].min(),self.force_start_date[1]) if self.force_start_date[0] else mkt_data_old['date'].min()
            end_old=mkt_data_old['date'].max()            
            
            if end_old!=um.today_date():
                tickers_to_update=[x for x in tickers if x not in tickers_old]
                mkt_data_old=mkt_data_old.set_index(['date','ticker']).unstack()

                if len(tickers_to_update)!=0:
                    print ('New tickers found') # we may repeatedly pick up new tickers that are no longer existing, so we use validate_input to control for this
                    mkt_data_new_tickers=self._get_mkt_data(tickers_to_update,
                                        start=fql_date(start_old),end=fql_date(end_old),
                                        mode='new',
                                        validate_input_ticker=True
                                        )
                    
                    mkt_data_old=pd.concat([mkt_data_old,mkt_data_new_tickers],axis=1).sort_index(axis=1)
                
                mkt_data_new_topup=self._get_mkt_data(tickers,
                                        start=fql_date(end_old),end=fql_date(um.today_date()),
                                        mode='topup'
                                        )
                
                mkt_data=pd.concat([mkt_data_old,mkt_data_new_topup],axis=0)
                mkt_data=mkt_data.stack().reset_index().rename(columns={'level_1':'ticker'}).groupby(['date','ticker']).last().unstack()
            else:
                print ('Mkt data already updated')
                mkt_data=mkt_data_old.set_index(['date','ticker']).unstack().copy() # do nothing
            
        
        feather.write_dataframe(mkt_data.stack().reset_index(),file_name)
        feather.write_dataframe(compo.reset_index(),self.PATH+'%s_Compo.feather' % (universe_name))
        
    
    def _get_mkt_data(self,tickers,start,end,mode='new',validate_input_ticker=False):
        '''
        wrapper for downloading all the needed market data fields
        for reported fundamental it is always latest ANN
        for estimated fundamental it is always FY1 ANN
        need fql date format for start and end as input
        
        mode can be 'new' or 'topup', and it only affects rpt fundamental data download
            as we need extra data for growth estimation. So if we choose mode='new' we will force
            start_date to be ba
            
        add a .loc[:end] as the fundamental data download sometimes get extra data for unknown reason
        
        About the B-day index: the output from this function here is Bday index, however when we save 
            as feather, we stack the multi-index and for the extra fundamental data prior to start date
            the corresponding index is no longer in Bday
            
        For sales of financials, we need to use fq.get_ts_estimates_rolling with fp=0 to get the actual sales
        '''
        if validate_input_ticker:
            check=self.fq.get_snap(tickers,['short_name'])['short_name'] # for cases like 0905737D-HK etc 
            check=check[check!='@NA']
            if len(check)==0:
                return pd.DataFrame()
            else:
                tickers=check.index.tolist()
        
        fields=self.FIELDS
        
        fields_all=self.fq.ts.loc[fields]['field_type'].sort_values()
        
        # get the mkt type (we will always get mkt fields)
        fields_mkt=fields_all[fields_all=='mkt'].index.tolist()
        data_mkt=self.fq.get_ts(tickers,fields_mkt,start=start,end=end,fx=self.fx).loc[:end]
        
        # get the reported fundamental type
        fields_rpt=fields_all[fields_all=='reported'].index.tolist()
        if len(fields_rpt)!=0:
            start_funda=start if mode!='new' else fql_date(pd.to_datetime(start,format='%m/%d/%Y')-pd.tseries.offsets.DateOffset(years=7))
            data_rpt_no_fill=self.fq.get_ts_reported_fundamental(tickers,fields_rpt,start=start_funda,end=end,
                                                         rbasis='ANN',fx=self.fx,
                                                         auto_clean=False).loc[:end]
            data_rpt=data_rpt_no_fill.fillna(method='ffill')
            data_rpt_no_fill=data_rpt_no_fill.stack()
            data_rpt_no_fill.columns=data_rpt_no_fill.columns.map(lambda x: x+'_NoFill')
            data_rpt_no_fill=data_rpt_no_fill.unstack()
        else:
            data_rpt_no_fill=pd.DataFrame()
        
        
        # get the estimated fundamental type
        fields_est=fields_all[fields_all=='estimate'].index.tolist()
        fields_est=[x.replace('_est','') for x in fields_est]
        if len(fields_est)!=0:
            data_est=self.fq.get_ts_estimates_rolling(tickers,fields_est,'ANN_ROLL','+1',
                                                  start=start_funda,end=end,fx=self.fx,
                                                  ).loc[:end].fillna(method='ffill')
        else:
            data_est=pd.DataFrame()
        
        # Get the reprted sales from FE field for consitency between reported and estimated number for financials (e.g. interest expense difference etc.)
        if 'sales' in self.FIELDS:
            fields_rpt_from_FE=['sales']
            data_rpt_FE=self.fq.get_ts_estimates_rolling(tickers,fields_rpt_from_FE,'ANN_ROLL','0',
                                                      start=start_funda,end=end,fx=self.fx,
                                                      ).loc[:end].fillna(method='ffill')
            data_rpt_FE=data_rpt_FE.stack().rename(columns={'sales_est':'sales_FE'}).unstack()
        else:
            data_rpt_FE=pd.DataFrame()
            
            
        data=pd.concat([data_mkt,data_rpt,data_est,data_rpt_FE],axis=1)
        
        # resample to B-days and then mask using marcap (using marcap_comp)
        data=pd.concat([
                data[self.FFILL_FIELDS].fillna(method='ffill').resample('B').last().fillna(method='ffill'),
                data[self.ZFILL_FIELDS].fillna(0).resample('B').last().fillna(0),
                ],axis=1)
        marcap_mask=data['marcap_comp'].fillna(-1).applymap(lambda x: True if x==-1 else False)
        data=data.stack()
        for col in data.columns:
            # we only mask non-fundamental data
            if col in fields_all[fields_all=='mkt'].index:
                data[col]=data[col].unstack().mask(marcap_mask).stack()
        data=data.unstack()
        
        if len(data_rpt_no_fill)!=0:
            data=data.join(data_rpt_no_fill,how='left')
        
        return data
    
    
        
    def _get_factor_signal(self,factor_name,mkt_data,compo):
        '''
        Please refer to the PDF for detailed definition
        mkt_data already in BDay, with marcap mask applied
        '''
        if factor_name not in self.FACTORS:
            print ('factor_name %s not available yet' % (factor_name))
            pdb.set_trace()
        
        regression_para=self.regression_para
        step=regression_para[0]
        window=regression_para[1]
        
        mkt_perf=feather.read_dataframe(self.PATH+self.universe_name+'_MktReturn.feather').set_index('date')
        fx_perf=feather.read_dataframe(self.PATH+self.universe_name+'_DXYLevel.feather').set_index('date')
        mkt_fx_ret=pd.concat([mkt_perf,fx_perf],axis=1).dropna().pct_change(step) # data already in BDay
        
        marcap_mask=mkt_data['marcap_comp'].fillna(-1).applymap(lambda x: True if x==-1 else False)
        daily_ret_clean=(mkt_data['ret_daily']/100).applymap(lambda x: 0 if abs(x)>=self.MAX_DAILY_MOVE else x)
        ret_cumu=(daily_ret_clean.fillna(0)+1).cumprod()
        ret_clean=ret_cumu.pct_change(step).mask(marcap_mask)
        
        #---- fx
        if factor_name=='fx':
            # just use a loop for rolling regression
            res=pd.DataFrame(index=ret_clean.index,columns=ret_clean.columns)
            for ticker in ret_clean.columns:
                to_calc=pd.concat([ret_clean[ticker],mkt_fx_ret],axis=1).dropna()
                reg_res_i=umath.rolling_regression(to_calc, ticker, ['mkt','dxy'],window)
                if reg_res_i is not False:
                    res[ticker]=reg_res_i.beta['dxy']
        #---- beta
        if factor_name=='beta':
            # just use a loop for rolling regression
            res=pd.DataFrame(index=ret_clean.index,columns=ret_clean.columns)
            for ticker in ret_clean.columns:
                to_calc=pd.concat([ret_clean[ticker],mkt_fx_ret],axis=1).dropna()
                reg_res_i=umath.rolling_regression(to_calc, ticker, ['mkt'],window)
                if reg_res_i is not False:
                    res[ticker]=reg_res_i.beta['mkt']
        #---- lqdt
        # this is the excess liquidity so we won't end up with just large cap names
        if factor_name=='lqdt':
            to_calc=mkt_data[['marcap_sec','turnover','ret_daily']]
            volume_to_marcap=(to_calc['turnover'].rolling(3*self.DAYCOUNT_MONTH,min_periods=1).mean()
                              .divide(to_calc['marcap_sec'].rolling(1*self.DAYCOUNT_MONTH,min_periods=1).mean())
                              )
            volume_to_marcap=volume_to_marcap.applymap(lambda x: np.nan if x==0 else x).apply(np.log)
            
            amihud=to_calc['ret_daily'].abs().divide(to_calc['turnover'].divide(to_calc['marcap_sec']))
            amihud=amihud.rolling(6*self.DAYCOUNT_MONTH,min_periods=1).mean()
            amihud=1/amihud
            
            temp_mask=to_calc['marcap_sec'].fillna(method='ffill').fillna(-1).applymap(lambda x: True if x==-1 else False)
            to_clean=to_calc['turnover'].mask(temp_mask)
            pct_of_ret=(to_clean.applymap(lambda x: np.nan if x==0 else x).rolling(self.DAYCOUNT_YEAR,min_periods=1).count()
                        .divide(to_clean.rolling(self.DAYCOUNT_YEAR,min_periods=1).count())
                        )
            
            res=(volume_to_marcap+amihud+pct_of_ret)/3
        #---- mom
        if factor_name=='mom':
            log_ret_daily=(daily_ret_clean+1).applymap(np.log)
            ret_year=log_ret_daily.rolling(self.DAYCOUNT_YEAR).sum().applymap(np.exp)-1
            ret_month=log_ret_daily.rolling(self.DAYCOUNT_MONTH).sum().applymap(np.exp)-1
            res=ret_year-ret_month
        #---- size
        if factor_name=='size':
            # size is based on marcap at company level
            res=mkt_data['marcap_comp'].rolling(self.DAYCOUNT_MONTH,min_periods=1).mean().applymap(np.log)
        #---- vol
        if factor_name=='vol':
            cs_vol=daily_ret_clean.std(axis=1).map(lambda x: np.nan if x==0 else x)
            res=(daily_ret_clean.abs().divide(cs_vol,axis='index')).rolling(6*self.DAYCOUNT_MONTH,min_periods=1).mean().applymap(np.sqrt)
        #---- value
        if factor_name=='value':
            to_calc=mkt_data[['pb','marcap_comp','net_income','net_income_est']].stack()
            to_calc['ey_rpt']=to_calc['net_income']/to_calc['marcap_comp']
            to_calc['ey_est']=to_calc['net_income_est']/to_calc['marcap_comp']
            to_calc['ey_est']=to_calc['ey_est'].fillna(to_calc['ey_rpt'])
            to_calc['bps']=1/to_calc['pb']
            to_calc['ey']=0.75*to_calc['ey_rpt']+0.25*to_calc['ey_est']
            res=to_calc[['bps','ey']].mean(1).unstack()
        #---- lev
        if factor_name=='lev':
            # debt to equity: debt to commom equity
            # debt to asset: debt to total aset
            # documentation says standarize, I prefer just use rank
            to_calc=mkt_data[['asset_to_equity','debt_to_equity']].stack()
            to_calc['debt_to_asset']=1-(1/to_calc['asset_to_equity'])
            res=(0.5*to_calc['debt_to_equity'].unstack().rank(pct=True,axis=1,method='min')
                 +0.5*to_calc['debt_to_asset'].unstack().rank(pct=True,axis=1,method='min')
                 )
        #---- growth (bit slow to calculate so we do top-up update)
        if factor_name=='growth':
            cache_path=self.PATH+self.universe_name+'_GrowthScore.feather'
            def _get_growth_score(tickers,top_up_mode=[False,'last_update_date']):
                # looping through tickers, fields and dates
                to_calc=mkt_data[['sales_NoFill','net_income_NoFill','sales_est','net_income_est','sales_FE']].swaplevel(1,0,1)
                collector=[]
                for field in ['sales','net_income']:
                    f_rpt='%s_NoFill' % (field)
                    f_est='%s_est' % (field)
                    for ticker in tickers:
                        to_calc_i=to_calc[ticker][[f_rpt,f_rpt.replace('NoFill','FE')]].dropna() if field=='sales' else to_calc[ticker][[f_rpt]].dropna()
                        print ('working on %s for %s' % (ticker,field))
                        if len(to_calc_i)>=5:
                            collector_ticker=[]
                            for i,dt_from in enumerate(to_calc_i.index):
                                if i>=4: # we count from 0
                                    dt_next=to_calc_i.index[i+1] if i+1<len(to_calc_i) else um.today_date()
                                    to_reg_rpt=to_calc_i.loc[:dt_from].iloc[-5:].copy()
                                    to_reg_est=to_calc[ticker][[f_est]].loc[dt_from:dt_next][f_est].rename(5).to_frame()
                                    to_reg_rpt=to_reg_rpt.reset_index()['sales_FE' if field=='sales' else f_rpt]
                                    to_reg=pd.DataFrame(index=to_reg_est.index,columns=np.arange(0,6))
                                    to_reg[5]=to_reg_est
                                    to_reg=to_reg.fillna(to_reg_rpt)
                                    
                                    if top_up_mode[0]:
                                        to_reg=to_reg.loc[top_up_mode[1]:]
                                        
                                    if len(to_reg)!=0:
                                        to_reg['beta']=to_reg.apply(lambda x: LinearRegression().fit(np.resize(np.arange(0,len(x.dropna())),(len(x.dropna()),1)),x.dropna().values).coef_[0],axis=1)
                                        to_reg['avg_abs_lvl']=to_reg.drop('beta',1).abs().mean(1)
                                        to_reg['growth']=to_reg['beta']/to_reg['avg_abs_lvl']
                                        collector_ticker.append(to_reg['growth'])
                                        
                            if len(collector_ticker)!=0:
                                growth_i=pd.concat(collector_ticker,axis=0).rename('growth').to_frame()
                                growth_i['ticker']=ticker
                                growth_i['field']=field
                                collector.append(growth_i)
                            else:
                                print ('we should not reach here unless something very odd happens')
                                pdb.set_trace()
                 
                if len(collector)!=0:
                    growth_score=pd.concat(collector).reset_index()
                else:
                    growth_score=pd.DataFrame()
                    
                return growth_score
            
            tickers=ret_clean.columns
            if not os.path.isfile(cache_path):
                print ('No existing data found for growth score, re-run now')
                growth_score=_get_growth_score(tickers)
            else:
                # we can just keep concacting as the output is in stack format
                print ('Top-up updating growth score')

                old=feather.read_dataframe(cache_path)
                if old['date'].max()!=um.today_date():            
                    tickers_old=old.groupby('ticker').last().index.to_list()
                    tickers_to_update=[x for x in tickers if x not in tickers_old] # we also pick up tickers with insufficient data from last run here
                    if len(tickers_to_update)!=0:
                        print ('new tickers found start updating')
                        growth_score_new_tickers=_get_growth_score(tickers_to_update)
                    last_update_date=old['date'].max()
                    new=_get_growth_score(tickers,top_up_mode=[True,last_update_date])
                    growth_score=pd.concat([old,growth_score_new_tickers,new],axis=0)
                    growth_score=growth_score.groupby(['date','ticker','field',]).last().reset_index()
                else:
                    print ('growth score already updated')
                    growth_score=old.groupby(['date','ticker','field',]).last().reset_index()
                
            feather.write_dataframe(growth_score,cache_path)
            res=growth_score.groupby(['date','ticker','field'])['growth'].last().unstack().unstack()
            res=res.rolling(10).mean() # to smooth out the occasional jumpy level caused by 1 or 2 day mis-alignment of the estimate update date
            res=0.5*(res['sales'].rank(axis=1,pct=True,method='min')+res['net_income'].rank(axis=1,pct=True,method='min'))
            res=res.reindex(columns=ret_clean.columns)

        #---- prof (quality)
        if factor_name=='prof':
            to_calc=mkt_data[['roe','roa','cfo','cfo_NoFill','asset','asset_NoFill',
                              'net_income','net_income_NoFill','gross_margin','sales']].stack()
            to_calc['asset_avg']=to_calc['asset_NoFill'].unstack().apply(lambda x:x.dropna().rolling(2).mean(), axis=0).reindex(ret_clean.index).fillna(method='ffill').stack()
            to_calc['cfo_avg']=to_calc['cfo_NoFill'].unstack().apply(lambda x:x.dropna().rolling(2).mean(), axis=0).reindex(ret_clean.index).fillna(method='ffill').stack()
            to_calc['ni_avg']=to_calc['net_income_NoFill'].unstack().apply(lambda x:x.dropna().rolling(2).mean(), axis=0).reindex(ret_clean.index).fillna(method='ffill').stack()
            to_calc['gross_margin']=to_calc['gross_margin']/100
            to_calc['roe']=to_calc['roe']/100
            to_calc['roa']=to_calc['roa']/100
            to_calc['cfta']=to_calc['cfo']/to_calc['asset_avg']
            to_calc['cfti']=to_calc['cfo_avg']/to_calc['ni_avg']
            to_calc['sta']=to_calc['sales']*to_calc['gross_margin']/to_calc['asset_avg']
            
            res=to_calc[['roe','roa','cfta','cfti','gross_margin','sta']].mean(1).unstack()
        #---- div
        if factor_name=='div':
            res=mkt_data['div_yield']
        
        # mask the results
        compo_mask=compo.reindex(res.index,axis=0).fillna(method='ffill').reindex(res.columns,axis=1).applymap(lambda x: True if x==0 else False)
        res_clean=res.mask(compo_mask)
        if self.force_start_date[0]:
            res_clean=um.drop_zero_row(res_clean).loc[self.force_start_date[1]:]
        else:
            res_clean=um.drop_zero_row(res_clean)
        res_clean=res_clean.resample('BM').last().loc[:um.today_date()]
        
        return res_clean
        

    def update_factor_perf(self):
        '''
        A bit slow to run for full factors, so we do topup update as well
        '''
                
        universe_name=self.universe_name
        
        if os.path.isfile(self.PATH+self.universe_name+'_FactorReturn.feather'):
            factor_all_old=feather.read_dataframe(self.PATH+self.universe_name+'_FactorReturn.feather')
            if factor_all_old['date'].max()==um.today_date():
                print ('factor performance already updated')
                return None
        
        
        mkt_data=feather.read_dataframe(self.PATH+'%s_MktData.feather' % (universe_name)).set_index(['date','ticker']).unstack().resample('B').last()
        # we need to resampe the above mkt_data for the extra fundamental data prior to the starting date
        
        compo=feather.read_dataframe(self.PATH+'%s_Compo.feather' % (universe_name)).set_index('date')
        ret_clean=(mkt_data['ret_daily'].resample('B').last().fillna(0)/100).applymap(lambda x: 0 if abs(x)>=self.MAX_DAILY_MOVE else x)
        # initiate backtester
        mkt_for_bt=(ret_clean.fillna(0)+1).cumprod()
        mkt_for_bt['cash']=1
        self.bt.mkt=mkt_for_bt.copy()
        self.bt.vwap=mkt_for_bt.copy()

        #--- get factor signal
        factors=self.FACTORS
        factor_dict={}
        for factor_name in factors:
            print ('getting factor level for %s' % (factor_name))
            factor_dict[factor_name]=self._get_factor_signal(factor_name, mkt_data, compo)
        
        
        def _bt_factor(factor_dict,manual_start=[False,'some start date']):
            collector=[]
            for factor_name, signal in factor_dict.items():
                print ('backtesting %s' % (factor_name))
                
                # A temp fix for qcut duplication, can happen to dividend yield where we may have a lot of 0
                # If the universe is not big enough or it's too back in time then it's more likely to have this problem
                # Solution is to start backtest later by dropping such dateS!
                # So we may have a delayed starting date, and also may skip certain rebalance date (less likely thougn)
                dates_to_drop=[]
                for dt_i in signal.index:
                    try:
                        pd.qcut(signal.loc[dt_i],self.quantile)
                    except ValueError:
                        dates_to_drop.append(dt_i)
                if len(dates_to_drop)!=0:
                    dates_to_drop_display=';'.join([x.strftime('%Y-%m-%d') for x in dates_to_drop])
                    print ('Dropping %s for %s due to bin duplication issue' % (dates_to_drop_display,factor_name))
                
                self.bt.signal=signal.drop(dates_to_drop,axis=0).copy()
                start_date=self.bt.signal.index[0] if not manual_start[0] else manual_start[1]
                if start_date>signal.index[-1]:
                    start_date=signal.index[-1]
                
                perf_pct_ls,perf_abs,shares_overtime,to,hlds=self.bt.run_q(self.quantile, start_date, 'cash')
                perf_pct_l,perf_abs,shares_overtime,to,hlds=self.bt.run_q(self.quantile, start_date, 'cash',manual_q_ls=[True,'Q%s' % (int(self.quantile)),'Q1'])
                perf_pct_s,perf_abs,shares_overtime,to,hlds=self.bt.run_q(self.quantile, start_date, 'cash',manual_q_ls=[True,'Q1','Q1'])
                
                perf_pct_ls=perf_pct_ls['l-s_net'].rename('perf').pct_change().to_frame()
                perf_pct_l=perf_pct_l['l-mkt_net'].rename('perf').pct_change().to_frame()
                perf_pct_s=perf_pct_s['l-mkt_net'].rename('perf').pct_change().to_frame()
                
                perf_pct_ls['type']='LS'
                perf_pct_l['type']='L'
                perf_pct_s['type']='S'
                
                factor_i=pd.concat([perf_pct_ls,perf_pct_l,perf_pct_s],axis=0)
                factor_i['factor']=factor_name
                collector.append(factor_i)
                
            factor_all=pd.concat(collector).reset_index()
            return factor_all
        
        #---- get factor performance backtest        
        if os.path.isfile(self.PATH+self.universe_name+'_FactorReturn.feather'):
            factor_all_old=feather.read_dataframe(self.PATH+self.universe_name+'_FactorReturn.feather')
            manual_start=[True,factor_all_old['date'].max()]
            factor_all_new=_bt_factor(factor_dict,manual_start=manual_start)
            factor_all=pd.concat([factor_all_old,factor_all_new],axis=0)
            factor_all=factor_all.groupby(['date','factor','type']).last().reset_index()
        else:
            factor_all=_bt_factor(factor_dict)
            
        
        feather.write_dataframe(factor_all, self.PATH+'%s_FactorReturn.feather' % (universe_name))

        
        print ('Finish updating factor universe: %s' % (universe_name))
    
        return None
    
    def get_factor_perf_details(self):
        res=feather.read_dataframe(self.PATH+'%s_FactorReturn.feather' % (self.universe_name)).set_index(['date','factor','type'])['perf'].unstack().unstack()
        res=(res.fillna(0)+1).cumprod()
        res=res/res.iloc[0]
        return res
        
    def get_factor_perf_ls(self):
        factor=self.get_factor_perf_details()['LS']
        mkt=feather.read_dataframe(self.PATH+'%s_MktReturn.feather' % (self.universe_name)).set_index(['date'])[['mkt']]
        res=pd.concat([factor,mkt],axis=1).dropna()
        
        res=res.loc[res.index[0]+pd.tseries.offsets.DateOffset(years=2):] # so that we have effective data for fx and beta factor
        
        res=res/res.iloc[0]
        return res
    
    
    def update_strategy_universe(self,strategy_specs=['name','ticker_list']):

        universe_name=self.universe_name
        strategy_name=strategy_specs[0]
        strategy_universe=strategy_specs[1]

        #---- get ret data for the strategy universe
        file_name=self.PATH+'%s_%s_DailyReturn.feather' % (universe_name,strategy_name)
        
        if not os.path.isfile(file_name):    
            print ('No existing return data found for %s %s, start downloading now' % (universe_name,strategy_name))
            existing_ret=feather.read_dataframe(self.PATH+'%s_MktData.feather' % (universe_name)).set_index(['date','ticker']).unstack()['ret_daily']
            additional_ret=self.fq.get_ts(
                            [x for x in strategy_universe if x not in existing_ret],# note some of the existing_ret tickers may not be in the strategy universe, which is totally fine
                            ['ret_daily'],
                           start=fql_date(existing_ret.index[0]),end=fql_date(um.today_date(),),
                           fx=self.fx
                           )['ret_daily']
            ret_for_strategy=pd.concat([existing_ret,additional_ret],axis=1) # so here we may have more tickers than we need for the strategy as some ticker in the factor universe does not exist in the strategy universe
            
        else:
            print ('Top up updating strategy mkt data for %s %s' % (universe_name,strategy_name))
            
            ret_for_strategy_old=feather.read_dataframe(file_name).set_index('date')
            if ret_for_strategy_old.index[-1]!=um.today_date():
                #ret_for_strategy_old=ret_for_strategy_old.loc[:datetime(2021,9,20)].iloc[:,:-10] # temp adjuitment for topup update testing
                tickers_to_update=[x for x in strategy_universe if x not in ret_for_strategy_old.columns]
                if len(tickers_to_update)!=0:
                    print ('New tickers found')
                    ret_for_strategy_new_tickers=self.fq.get_ts(tickers_to_update,['ret_daily'],
                               fql_date(ret_for_strategy_old.index[0]),fql_date(ret_for_strategy_old.index[-1]),
                               fx=self.fx
                               )['ret_daily']
                    ret_for_strategy_old=pd.concat([ret_for_strategy_old,ret_for_strategy_new_tickers],axis=1)
                
                ret_for_strategy_new=self.fq.get_ts(ret_for_strategy_old.columns.tolist(),['ret_daily'],
                               fql_date(ret_for_strategy_old.index[-1]),fql_date(um.today_date()),
                               fx=self.fx
                               )['ret_daily']
                ret_for_strategy=pd.concat([ret_for_strategy_old,ret_for_strategy_new],axis=0).reset_index().groupby('date').last()
            else:
                print ('Mkt data already updated for strategy %s' % (strategy_name))
                ret_for_strategy=ret_for_strategy_old.copy()
            
        feather.write_dataframe(ret_for_strategy.reset_index(),file_name)
        return None
    
    def update_strategy_beta(self,strategy_name=''):
        '''
        The expanding window regression with less than 2 years data slows down the calculation
        So we do top up update here.
            - for beta series longer than 2 years, we run rolling regression directly on the last 2 year data block
            - for beta series shorter than 2 years, we just re-run
        '''
        
        universe_name=self.universe_name
                
        file_name=self.PATH+'%s_%s_DailyReturn.feather' % (universe_name,strategy_name)
        ret_for_strategy=feather.read_dataframe(file_name).set_index('date')/100
        ret_for_strategy=ret_for_strategy.applymap(lambda x: 0 if abs(x)>=self.MAX_DAILY_MOVE else x)
        ret_for_strategy=ret_for_strategy.resample('B').last()
        
        factor_perf=self.get_factor_perf_ls()
        
        step=self.regression_para[0]
        window=self.regression_para[1]
        
        factor_ret=factor_perf.pct_change(step)
        stock_ret=(ret_for_strategy+1).applymap(np.log).rolling(step).sum().applymap(np.exp)-1
        
        factor_ret_daily=factor_perf.pct_change()
        stock_ret_daily=ret_for_strategy.copy()
        
        # check if we have beta already
        has_existing_beta_dump=os.path.isfile(file_name.replace('DailyReturn','Beta'))
        if has_existing_beta_dump:
            print ('found existing beta dump, topup update beta')
            beta_old=feather.read_dataframe(file_name.replace('DailyReturn','Beta')).set_index(['date','factor','ticker'])['beta'].unstack().unstack()
            if beta_old.index.max()==um.today_date():
                print ('beta and ret decompo already udpated')
                return None
            
        
        def _get_beta_new(to_calc):
            reg_res=umath.rolling_regression(to_calc, 'stock', self.FACTORS+['mkt'], window,expanding_starting_mode=[True,6*self.DAYCOUNT_MONTH])
            if reg_res is not False:
                betas=reg_res.reindex(to_calc.index).fillna(0)
            else:
                betas=pd.DataFrame(index=to_calc.index,columns=to_calc.drop('stock',1).columns).fillna(0)
            return betas
        
        
        # calculate betas and daily return decompo by looping through tickers 
        collector_beta=[]
        collector_ret_decomp=[]
        for i,ticker in enumerate(stock_ret.columns):
            
            to_calc=pd.concat([stock_ret[ticker].rename('stock'),factor_ret],axis=1).dropna()
            
            if not has_existing_beta_dump or ticker not in beta_old.columns.levels[0]:
                print ('working on %s (%s/%s): new' % (ticker,i+1,len(stock_ret.columns)))
                betas=_get_beta_new(to_calc)
            else:
                beta_old_i=beta_old[ticker].dropna()
                if len(beta_old_i)>window:
                    print ('working on %s (%s/%s): topup' % (ticker,i+1,len(stock_ret.columns)))
                    top_up_beg_date=beta_old_i.index[-window-1]
                    reg_res_new=umath.rolling_regression(to_calc.loc[top_up_beg_date:], 'stock', self.FACTORS+['mkt'], window,)
                    betas_new=reg_res_new.beta
                    betas=pd.concat([beta_old_i,betas_new]).reset_index().groupby('date').last()
                else:
                    print ('working on %s (%s/%s): topup but still note enough data' % (ticker,i+1,len(stock_ret.columns)))
                    betas=_get_beta_new(to_calc)
                    
    
            ret_decomp=pd.concat([factor_ret_daily.multiply(betas),stock_ret_daily[ticker].rename('stock')],axis=1).dropna()
            
            betas=betas.stack().rename('beta').reset_index().rename(columns={'level_1':'factor'})
            betas['ticker']=ticker
            collector_beta.append(betas)
            
            ret_decomp=ret_decomp.stack().rename('ret').reset_index().rename(columns={'level_1':'factor'})
            ret_decomp['ticker']=ticker
            collector_ret_decomp.append(ret_decomp)
                        
        beta_all=pd.concat(collector_beta)
        ret_decomp_all=pd.concat(collector_ret_decomp)
        
        feather.write_dataframe(beta_all, file_name.replace('DailyReturn','Beta'))
        feather.write_dataframe(ret_decomp_all, file_name.replace('DailyReturn','DailyRetDecomp'))
        
        
        return None

    
    def load_ret_decomp(self,strategy_name=''):
        universe_name=self.universe_name
        file_name=self.PATH+'%s_%s_DailyRetDecomp.feather' % (universe_name,strategy_name)
        self.ret_decomp=feather.read_dataframe(file_name)
        self.ret_decomp=self.ret_decomp.set_index(['date','ticker','factor',])['ret'].unstack().unstack()
        
        return None
    
    def get_excess_return_cumu(self,start,end):
        ret=self.ret_decomp.loc[start:end].copy()
        ret.iloc[0]=0
        ret=((ret+1).cumprod()-1).stack()
        ret['excess']=ret['stock']-ret[self.FACTORS+['mkt']].sum(1)
        return ret
        
    def get_excess_return_around_event(self,ticker,event_date,window_one_side):
        ret=self.ret_decomp.swaplevel(1,0,1)[ticker]
        evt_i=ret.index.get_loc(event_date)
        i_beg=evt_i-window_one_side
        i_end=evt_i+window_one_side+1
        ret_evt=ret.iloc[i_beg:i_end].copy()
        ret_evt.at[ret_evt.index[0]]=0
        ret_evt_cumu=(ret_evt+1).cumprod()
        ret_evt_cumu_rebase=ret_evt_cumu.divide(ret_evt_cumu.loc[event_date])-1
        ret_evt_cumu_rebase['excess']=ret_evt_cumu_rebase['stock']-ret_evt_cumu_rebase[self.FACTORS+['mkt']].sum(1)
        ret_evt_cumu_rebase['event_date']=event_date
        ret_evt_cumu_rebase=ret_evt_cumu_rebase.reset_index()
        ret_evt_cumu_rebase.index=ret_evt_cumu_rebase.index-window_one_side
        ret_evt_cumu_rebase.index.name='event_day'
        ret_evt_cumu_rebase['ticker']=ticker
        return ret_evt_cumu_rebase
    
    def get_excess_return_rolling(self,window,reindex=[False,'pass a clean index']):
        ret=self.ret_decomp.copy()
        if reindex[0]:
            ret=ret.reindex(reindex[1])
    
        ret_rolling=(ret+1).apply(np.log).rolling(window).sum().apply(np.exp)-1
        ret_rolling=ret_rolling.stack()
        ret_rolling['excess']=ret_rolling['stock']-ret_rolling[self.FACTORS+['mkt']].sum(1)
        return ret_rolling.unstack().swaplevel(1,0,1).sort_index(1)


class Carhart(Alpha_Model):
    
    MODEL_NAME='carhart'
    
    PATH=uc.root_path_data+"excess_return\\%s\\" % (MODEL_NAME)
    MAX_DAILY_MOVE=0.5
    DAYCOUNT_YEAR=252
    DAYCOUNT_MONTH=21
    
    FACTORS=model_factor_field_map[MODEL_NAME][0]
    FIELDS=model_factor_field_map[MODEL_NAME][1]
    FFILL_FIELDS=model_factor_field_map[MODEL_NAME][2]
    ZFILL_FIELDS=model_factor_field_map[MODEL_NAME][3]
    
    def __init__(self,
             universe, # a list of index, in BBG index format
             fx, # currency
             quantile=3,
             regression_para=[5,504], # this applies to both stock beta to factors and beta estimation when constructing factor
             force_start_date=[False,'some date'] # ideally we want to avoid huge change in universe due to MEMB availability issue. We can force a start date for stable universe
             ):
        Alpha_Model.__init__(self,universe,fx,quantile=quantile,
                             regression_para=regression_para,force_start_date=force_start_date)







if __name__=='__main__':
    print ('ok')
    #---- test
    universe=['HSCI']#,'SH000905']
    fx='HKD'
    
    # universe=['TPX_L']
    # fx='JPY'
    
    # universe=['SHCOMP_L','SZCOMP_L']
    # fx='CNY'
    
    force_start_date=[True,datetime(2005,12,31)] 
    # alpha=Alpha_Model(universe,fx,force_start_date=force_start_date)
    alpha=Carhart(universe,fx,force_start_date=force_start_date)
    
    # udpate universe
    alpha.update_factor_universe()
    alpha.update_factor_perf()
    # factor_perf_details=alpha.get_factor_perf_details()
    # factor_perf=alpha.get_factor_perf_ls()
    # for col in factor_perf.columns:
    #     fig,ax=ud.easy_plot_quick_subplots((1,1), col)
    #     factor_perf[col].plot(ax=ax)
    
    # update strategy
    from connect.connect import STOCK_CONNECT
    sc=STOCK_CONNECT(direction='sb')
    sc.load_db()
    
    strategy_specs=['SB',sc.db_clean.columns.levels[1].tolist()]
    alpha.update_strategy_universe(strategy_specs=strategy_specs)
    alpha.update_strategy_beta(strategy_name='SB')
    
    
    
    
    
    
    
    # # get excess return
    # alpha_sb=alpha.Load_Model_Quick('SB')


    # start=datetime(2021,8,1)
    # end=um.today_date()
    
    # event_date=datetime(2021,8,16)
    # ticker='600519-CN'
    # window_one_side=21
    
    # window=63

    # excess_period=carhart_nb.get_excess_return_cumu(start, end)
    # excess_evt=carhart_nb.get_excess_return_around_event(ticker, event_date, window_one_side)
    # excess_rolling=carhart_nb.get_excess_return_rolling(window)


    # check=feather.read_dataframe("C:\\Users\\davehanzhang\\python_data\\excess_return\\HSCI_SB_Beta.feather")
    # check=check[check['date']<=datetime(2020,12,31)]
    # feather.write_dataframe(check, "C:\\Users\\davehanzhang\\python_data\\excess_return\\HSCI_SB_Beta.feather")




