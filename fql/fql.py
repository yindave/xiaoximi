# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:21:37 2019

@author: hyin1
"""

import pandas as pd
import pdb
import numpy as np
import win32com.client as wc
import os
import pdb
import time
import sys
from datetime import datetime
from utilities.constants import root_path_data
import utilities.constants as uc


today_string=datetime.today().strftime('%Y-%m-%d')
today_date=datetime.strptime(today_string,'%Y-%m-%d')

fql_temp_path=root_path_data
python_path=uc.get_python_path()

class Factset_Query():
    '''
    Regarding dates, if we want to force all dates to be calendar date then
    When you are in the FactSet workstation and type @SWP in the search bar
    In the dropdown beneath categories, you can then click on regional settings.
    In the preferences section, you can then click on default calendar
    More on dates: using _eps and _guid will have date alignment issue.
    even if the date field is "date"
    To avoid date alignment from failing, make sure query the same type of fields
    We enable auto_fill: volume and turnover fill by 0 others ffill
    Free float data vs BBG: some tickers match some do not
    (use get_special_field function to get float)
    generic mkt data are basically the same as BBG
    (except mkt cap which BBG incorrectly adjust for dividend. you should not adjust div when calculating marcap)

    '''
    
    def __init__(self,session='dummy variable no longer needed'):
        
        self.freq_map={'4':'ANN','2':'SEMI','1':'QTR',
                       '@NA':'QTR' # this one is a temp fix for JP
                       }
        self.batch_size=350
        #for formula map
        self.PATH_static={
          'snap_map':python_path+'fql\\fql_map_snapshot.xlsx',
          'ts_map':python_path+'fql\\fql_map_ts.xlsx'}

        #load function map
        self.snap=pd.read_excel(self.PATH_static['snap_map'],sheet_name='Sheet1').set_index('my_field')
        self.ts=pd.read_excel(self.PATH_static['ts_map'],sheet_name='Sheet1').set_index('my_field')


    # def _update_session(self):
    #     self.session=str(hex(int(np.random.uniform(0,1000000))*int(np.random.uniform(0,1000000))))

    def get_mkt_fields(self,mode='ts'):
        if mode=='ts':
            return list(self.ts[self.ts['tag']==1].index)
        elif mode=='snap':
            return list(self.ts[self.snap['tag']==1].index)
    def get_matrix(self):
        #BDS equivalent
        return None
    def get_snap(self,tickers,fields,_check_fields=True):
        '''
        Quick snapshot view, no parameter needed
        For intra-day stats use Bloomberg
        '''
        tickers=self._remove_duplicated_input(tickers)
        #wrapper
        #check if the fields are "well-defined"
        if _check_fields and (not self._fields_check(fields,'snap')):
            return None

        if len(tickers)<=self.batch_size:
            res=self._get_snap(tickers,fields)
        else:
            batches=int(len(tickers)/self.batch_size)+1
            print ('%s tickers input, breakdown by %s batches' % (len(tickers),batches))
            res_collector=[]
            for i in np.arange(0,batches):
                batch=tickers[i*self.batch_size:(i+1)*self.batch_size]
                if len(batch)!=0:
                    res_batch=self._get_snap(batch,fields)
                    res_collector.append(res_batch)
                    print ('batch %s finished' % (i+1))
            res=pd.concat(res_collector,axis=0) #concat direction is different from ts

        return res

    def get_ts(self,tickers,fields,
           start='-10AY',end='NOW',freq='D',
           fx='', #different type of ts may require different fx input
           adj=True,
           fill='na', #can choose 0 or na but this controls all fields
           auto_fill=True,
           #if tag is 1 (mkt data) we can directly use get_ts. For tag 0 (float, fundamental) use separate function
           _check_fields=True,
           #usually should be true. date field varies depending on type of data
           _automatic_date=[True,'date'],
           #rbasis control for fundamental/estimate data ts
           #For fundamental we can have auto,ANN,SEMI,QTR
           #For estimate we can have ANN, QTR, SEMI, ANN_ROLL, NTM4_ROLL,NTMA
           _rbasis=[False,'auto'],
           # fp is for estimate data only. Should match rbasis
           # fp can be +1 (current unreported),+2,0,-1,-1,
           # or just empty '' for next 12M rbasis
           # 2019/3F for fixed fiscal period reference
           _fp='some fiscal period',
           _window=100, #lookback window for non-stale estimate
           _method='MEAN', #estimate aggregaet method: MEAN, MEDIAN, NEST, UP, DOWN, TOTAL
           _force_date_alignement=False,
           _is_os=False,
           _empty_check=True,
           _os_holder_type='FS',
           _os_group_name='INVESTOR',
           _skip_convert_to_numeric=False,
           ):
        
        
        fields_original=list(fields)
        #---- IMPORTANT: if we download ret_daily then we need to mask it with turnover. Otherwise for suspended days the return will be fwd filled
        if 'ret_daily' in fields and 'turnover' not in fields:
            fields.append('turnover')
        
        tickers=self._remove_duplicated_input(tickers)

        #check if the fields are "well-defined"
        if _check_fields and (not self._fields_check(fields,'ts')):
            return None
        #check if fundamental data request, get the rbasis match if needed
        if _rbasis[0]:
            if _rbasis[1]=='auto':
                freq_map=self.freq_map
                rbasis_map=(self.get_snap(tickers,['reporting_freq'])['reporting_freq']
                            .map(lambda x: freq_map[str(x)]).to_dict())
            else:
                rbasis_map=dict(zip(tickers,[_rbasis[1] for i in np.arange(0,len(tickers))]))
        else:
            rbasis_map=dict(zip(tickers,['dummy' for t in tickers]))
        #automatically append "date"
        if _automatic_date[0]:
            date_field=_automatic_date[1]
            if date_field not in fields:
                fields.append(date_field)
        else:
            date_field=''
        #a wrapper to deal with ticker batch (500 max per batch), let's do self.batch_size for safety
        if len(tickers)<=self.batch_size:
            res=self._get_ts(tickers,fields,start=start,end=end,freq=freq,fx=fx,adj=adj,fill=fill,
                             date_field=date_field,rbasis=rbasis_map,_fp=_fp,_window=_window,_method=_method,
                             _force_date_alignement=_force_date_alignement,
                             _is_os=_is_os,
                            _os_holder_type=_os_holder_type,
                            _os_group_name=_os_group_name,
                            _skip_convert_to_numeric=_skip_convert_to_numeric,
                            )
            if auto_fill:
                res=self._auto_fill(res)
        else:
            batches=int(len(tickers)/self.batch_size)+1
            print ('%s tickers input, breakdown by %s batches' % (len(tickers),batches))
            res_collector=[]
            for i in np.arange(0,batches):
                batch=tickers[i*self.batch_size:(i+1)*self.batch_size]
                if len(batch)!=0:
                    res_batch=self._get_ts(batch,fields,start=start,end=end,freq=freq,fx=fx,adj=adj,fill=fill,
                             date_field=date_field,rbasis=rbasis_map,_fp=_fp,_window=_window,_method=_method,
                             _force_date_alignement=_force_date_alignement,
                             _is_os=_is_os,
                             _skip_convert_to_numeric=_skip_convert_to_numeric,
                             )
                    if auto_fill:
                        res_batch=self._auto_fill(res_batch)
                    res_collector.append(res_batch)
                    print ('batch %s finished' % (i+1))
            res=pd.concat(res_collector,axis=1)
        #check for tickers that returns nothing
        if _empty_check:
            self._check_for_tickers_with_empty_return(res,tickers)
            
            
        # mask ret_daily with turnover
        if 'ret_daily' in fields:
            res=res.stack()
            res['to_mask']=res['turnover'].map(lambda x: True if x==0 else False)
            res['ret_daily']=res['ret_daily'].mask(res['to_mask']).fillna(0)
            if 'turnover' not in fields_original:
                res=res.drop(['turnover','to_mask'],1)
            else:
                res=res.drop(['to_mask'],1)
            res=res.unstack()
                
            res
        
        return res

    def get_ts_float(self, tickers, start='-10AY',end='NOW',freq='D',fx='',fill='na'):
        '''
        Since data format of float related field is different, we match the free float data
            with price data ourselves
        We output 3 float metrics: shares, value, pct
        All at security level
        Calculation: we get marcap (no control over adj, adjuted by default ),
                    shout_sec (need to fixed to adjusted),
                    ff_shares_sec (no control over adj, adjuted by default )
        '''

        ff=self.get_ts(tickers,['ff_shares_sec'],
                       start=start,end=end,freq=freq,fx=fx,adj=True, #irrelevant here
                       fill=fill,_automatic_date=[True,'os_date'],
                       _check_fields=False
                       )
        px=self.get_ts(tickers,['marcap_sec','shout_sec'],
                       start=start,end=end,freq=freq,fx=fx,adj=True,
                       fill=fill,_automatic_date=[True,'date'])
        ff=ff/1000000
        #pdb.set_trace()
        data=pd.concat([px,ff],axis=1).fillna(method='ffill').sort_index(1)
        res=pd.concat([
                data['ff_shares_sec'].stack().rename('ff_shout_sec'),
                (data['ff_shares_sec']/data['shout_sec']).stack().rename('ff_pct_sec'),
                (data['ff_shares_sec']/data['shout_sec']*data['marcap_sec']).stack().rename('ff_marcap_sec')
                ],axis=1)
        res.columns.name='field'
        #do a resample here as the monthly timestamp may be on weekend
        res=res.unstack().resample('B').last().fillna(method='ffill')
        return res
    def get_ts_reported_fundamental(self,tickers,fields,start='-10AY',end='NOW',rbasis='auto',
                                    auto_clean=True,fx=''):
        '''
        Please make sure you put reported fundamental fields into fields
        rbasis can be: auto, ANN, SEMI, QTR, LTM_SEMI, LTM_QTR
        fx here can be:
        local ('', same as trading fx), reported ('"RPT"', need double quotation), others ('USD')
        ***
        Note that FactSet LTM calculation is funny if company trading FX is different from reporting FX
        It will do the FX conversion before the rolling sum.
        This leads to level mismatch between ANN and LTM_SEMI/LTM_QTR
        If calling SEMI on QTR-reporting company we will have na value
        rbasis='auto' is not the perfect solution.
        e.g. FactSet says 600012 keeps changing reporting frequency and this will lead to empty output
        A brutal force way to get reported fundamental is to iterate through all possible rbasis
        (ANN,SEMI,QTR,LTM_SEMI,LTM_QTR)
        '''
        if auto_clean:
            tickers=self._clean_tickers_with_valid_rpt_freq(tickers)
        return self.get_ts(tickers,fields,start=start,end=end,
                          fx=fx,
                         auto_fill=False, # we want to maintain individual report date by simply drop na
                        _check_fields=False,_automatic_date=[True,'rpt_date'],
                        _rbasis=[True,rbasis])
    def get_ts_estimates_rolling(self,tickers,fields,rbasis,fp,
                         start='-10AY',end='NOW',fx='LOCAL',freq='D',
                         method='MEAN',window=100,
                         _is_guidance=False,
                         ):
        '''
        fields input can just be roe, dps, eps, the function will add _est
        method: MEAN, MEDIAN, NEST, UP (number of revision), DOWN, TOTAL (others please refer to factset)
        window: look back period
        fx input: LOCAL (trading fx), ESTCUR (reporting fx), USD, EUR etc
        Most of the time we just use the following 2 combinations:
        - for next 12m estimate it's always NTM4_ROLL or NTMA/STMA with fp being ''
            NTMA-BBG's BF1, STMA-BBG's BF2 , LTMA is blended between next FY and reported
            *** (however its time weighting is not by days but by month so you can't see the beat/miss effect)
            NTM4_ROLL requires quarterly reporting and is not smooth
        - for FY x estimate it's ANN_ROLL with fp being +1,+2,0,-1,-2
        rbasis can be rolling or non-rolling e.g. ANN vs. ANN_ROLL, QTR vs QTR_ROLL
        rolling means point in time
        non-rolling is useful to get exsimates for a fixed fiscal period
        For fixed fiscal period data we use a separate function
        (because factset ffill beyond announcement date, which is not good)
        '''
        if not _is_guidance:
            fields=[f+'_est' for f in fields]
        else:
            fields=[f+'_guid' for f in fields]
        #check if _ROLL in rbasis
        if '_ROLL' not in rbasis and rbasis not in ['NTMA','LTMA','STMA']:
            print ('please make sure rbasis contains _ROLL for this function (xTMA exempted)')
            return
        #factset will auto ffill anyway
        return self.get_ts(tickers,fields,start=start,end=end,
                fx=fx,_check_fields=False,_rbasis=[True,rbasis],_fp=fp,_window=window,freq=freq,
                _force_date_alignement=True)

    def get_ts_estimates_abs_and_dates(self,tickers,fields,rbasis, #'auto' or 'ANN'
                             fiscal_years,
                             start='-10AY',end='NOW',fx='LOCAL',freq='D',
                             method='MEAN',window=100,
                             auto_clean=True,
                              _is_guidance=False,
                             ):
        '''
        fields can just be roe, eps, dps
        Outputs are 2 df: ts of estimates and reporting date schedule
        To get full history of a range of fiscal year requires using a loop to run factset query
        rbasis can be auto, ANN, SEMI and QTR.
        If auto is chosen, we will break them down into ANN, SEMI and QTR outside the self.get_ts
        To avoid funny jump on reporting date, we use < not <= to mask the estimate ts
        To still have estimate ts right on reporting date we do ffill with limit=1 in the final output
        '''
        if not  _is_guidance:
            fields=[f+'_est' for f in fields]
        else:
            fields=[f+'_guid' for f in fields]
        if auto_clean:
            tickers=self._clean_tickers_with_valid_rpt_freq(tickers)
        #get the reporting date with "fiscal period end" as the index for matching
        rpt_dates=self.get_ts(tickers,['rpt_date'],start=start,end=end,
                   auto_fill=False,
                   _check_fields=False,
                   _automatic_date=[True,'rpt_date_fp_end'],
                   _rbasis=[True,'auto'],
                      )
        rpt_dates=(rpt_dates.apply(lambda x: x.dropna().map(str)
                .map(lambda x:pd.to_datetime(x,format='%Y%m%d.0'))))
        rpt_dates=rpt_dates.stack().reset_index().set_index(['ticker','date']).sort_index()['rpt_date']
        #the fake but useful date index is forced to be BQE
        #resample to Q end to match est dates
        rpt_dates=rpt_dates.swaplevel(1,0,0).unstack().resample('Q').last().T.stack()
        rpt_freq=self.get_snap(tickers,['reporting_freq'])
        if rbasis=='auto':
            rpt_freq=rpt_freq.applymap(lambda x: self.freq_map[str(x)])
            all_fy_freq=rpt_freq.groupby('reporting_freq').last().index
        else:
            rpt_freq=rpt_freq.applymap(lambda x: rbasis)
            all_fy_freq=[rbasis]
        collector=[]
        collector_dates=[]
        for fy_freq in all_fy_freq:
            collector_fy_freq=[]
            for fy in fiscal_years:
                #build fp code
                if fy_freq=='ANN':
                    fp_codes=[fy]
                elif fy_freq=='SEMI':
                    fp_codes=['%s/%sF' % (fy,i) for i in np.arange(1,3,1)]
                elif fy_freq=='QTR':
                    fp_codes=['%s/%sF' % (fy,i) for i in np.arange(1,5,1)]
                else:
                    print ('unknown reporting frequency')
                    pdb.set_trace()
                #build the ticker list
                tickers_current_fy_freq=rpt_freq[rpt_freq['reporting_freq']==fy_freq].index.tolist()
                #iterate through all fiscal periods
                for fp_code in fp_codes:
                    print ('Getting %s reporting comp for %s' % (fy_freq,fp_code))
                    #get the estimates ts
                    res=self.get_ts(tickers_current_fy_freq,fields,start=start,end=end,
                            fx=fx,_check_fields=False,_rbasis=[True,fy_freq],
                            _fp=fp_code,_window=window,freq=freq,_force_date_alignement=True)

                    #get the estimated report date and fp-end-date
                    res_est_dates=self.get_ts(tickers_current_fy_freq,['fp_end_est','rpt_date_est'],start='NOW',end='NOW',
                            fx=fx,_check_fields=False,_rbasis=[True,fy_freq],
                            _fp=fp_code,_window=window,freq=freq)
                    res_est_dates=res_est_dates.applymap(lambda x: pd.to_datetime(str(x),format='%Y%m%d'))
                    #get the correct estimate dates by fp
                    res_est_dates=res_est_dates.stack().reset_index().set_index(['ticker','fp_end_est']).drop('date',1)
                    res_est_dates['rpt_date']=rpt_dates
                    #fill date if no rpt_date for future dates
                    res_est_dates['rpt_date']=(res_est_dates[['rpt_date','rpt_date_est']]
                        .fillna(False)
                        .apply(lambda x: x['rpt_date_est'] if x['rpt_date'] is False else x['rpt_date'],axis=1))
                    res_est_dates=res_est_dates.reset_index().set_index('ticker')
                    #add the estimate date and fp to estimate series
                    res=res.stack().reset_index().set_index('ticker')
                    res['rpt_date']=res_est_dates['rpt_date']
                    #missing date is only possible for case where no estimate for the tickers & fp
                    res['date']=res['date'].fillna(today_date)
                    '''
                    #drop the data point that's beyond reporting
                    #if we use <= we will see funny jump on the actual reporting date
                    #however we do want the end of each fp estimate to be the reporting date
                    #we can do a ffill with limit=1 in the final output
                    '''
                    res=res[res['date']<res['rpt_date']]
                    res=res.drop('rpt_date',1).reset_index()
                    res['fp']=fp_code
                    #collect estimate
                    collector_fy_freq.append(res)
                    #collect dates
                    res_est_dates=res_est_dates.reset_index()
                    res_est_dates['freq']=fy_freq
                    res_est_dates['fp']=fp_code
                    collector_dates.append(res_est_dates)
            output_fy_freq=pd.concat(collector_fy_freq,axis=0)
            output_fy_freq=output_fy_freq.set_index(['date','fp','ticker']).unstack().unstack().sort_index(1)
            collector.append(output_fy_freq)
        output=pd.concat(collector,axis=1).sort_index(1)
        output=output.apply(lambda x: x.fillna(method='ffill',limit=1),axis=0)
        output_dates=pd.concat(collector_dates,axis=0)
        return output,output_dates

    def get_ts_guidance_rolling(self,tickers,fields,start='-10AY',end='NOW',fx='LOCAL',freq='D'):
        '''
        fields can be just dps and eps
        Basically exactly the same as ts_estimate rolling
        In Japan only (for Asia)
        force rbasis to be ANN
        force fp to be +1
        '''
        res=self.get_ts_estimates_rolling(tickers,fields,'ANN_ROLL','+1',
                                          start=start,end=end,fx=fx,freq=freq,
                                          _is_guidance=True)
        return res


    def get_ts_guidance_abs_and_dates(self,tickers,fields,fiscal_years,
                                      start='-10AY',end='NOW',fx='LOCAL',freq='D',
                                      auto_clean=True):
        '''
        fields can be just dps and eps
        '''
        output,output_dates=self.get_ts_estimates_abs_and_dates(tickers,fields,'ANN',fiscal_years,
                        start=start,end=end,fx=fx,freq=freq,auto_clean=auto_clean,
                        _is_guidance=True)
        return output,output_dates



    def get_ts_estimates_guidance_beat_miss(self,tickers,fields,fiscal_years,
                            mode='est', #can be est or guid
                             start='-10AY',end='NOW',fx='LOCAL',freq='D',
                             method='MEAN',window=100,
                             auto_clean=True):
        '''
        fields can be just dps and eps (if you are using mode='guid')
        This function returns (in order) bm(including rpt-ed), est, est_dates
        (So do not double call other function for getting duplicated data)
        Due to data availability issue we lock the rbasis to be ANN
        Fields input need to be eps/dps. The function will add _est or _guid automatically
        '''
        if auto_clean:
            tickers=self._clean_tickers_with_valid_rpt_freq(tickers)

        if mode=='est':
            est,est_dates=self.get_ts_estimates_abs_and_dates(tickers,
                            list(fields), # a copy of fields
                            'ANN',fiscal_years,
                            start=start,end=end,fx=fx,freq=freq,
                             method=method,window=window,auto_clean=False)
        else: #guid
            est,est_dates=self.get_ts_guidance_abs_and_dates(tickers,
                            list(fields),
                            fiscal_years,
                            start=start,end=end,fx=fx,freq=freq,
                            auto_clean=False)
        rpt=self.get_ts_reported_fundamental(tickers,list(fields),
                            start=start,end=end,rbasis='ANN',auto_clean=False,fx=fx)
        bm=pd.concat([
                est.stack().stack().reset_index().groupby(['date','ticker']).first().drop('fp',1),
                rpt.stack()],axis=1)
        for field in fields:
            bm['%s_bm' % field]=bm['%s' % (field)]-bm['%s_%s' % (field,mode)]
        fields_output=['%s_bm' % (f) for f in fields]
        bm=bm[fields_output]
        bm=pd.concat([bm.stack().unstack(),rpt.stack()],axis=1)
        return bm,est,est_dates


    def get_fund_holdings(self,ticker,start='-12M',end='-0D',
                          use_primary_ticker=True,
                          drop_cash=True,
                          force_normalize=True,
                          stacked=True):
        '''
        We need to download ticker by ticker
        Always month end snapshot
        '''
        dt=self.get_ts([ticker],['os_date'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[True,'os_date'],
           _is_os=False,auto_fill=False,_empty_check=False
                        )

        wgt_id=self.get_ts([ticker],['fund_hlds_id'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        wgt=self.get_ts([ticker],['fund_hlds'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # combine everything and stack
        res=pd.DataFrame(index=wgt_id['value'].values,columns=dt.index,data=wgt.values)
        # drop cash
        if drop_cash:
            try:
                res=res.drop('CASH',0)
            except KeyError:
                print ('no CASH in %s' % (ticker))
        # convert to nice ticker name
        if use_primary_ticker:
            nice_ticker_map=self.get_snap(res.index.tolist(),['primary_ticker'])['primary_ticker'].to_dict()
            res=res.rename(index=nice_ticker_map)
            try:
                res=res.drop('@NA',0)
            except KeyError:
                pass
        # force normalization
        if force_normalize:
            res=res.apply(lambda x: x/x.sum(),axis=0)
        res.columns=pd.to_datetime(res.columns)
        res.columns.name='asof'
        res.index.name='ticker'
        if not stacked:
            return res
        else:
            res=res.stack().rename('wgt').reset_index()
            res['fund_ticker']=ticker
            return res

    def get_holders(self,ticker,start='-12M',end='-0D',                          ):
        '''
        We need to download ticker by ticker
        Always month end snapshot
        We tag holders by type and region
        start and end can only be relative reference
        For now at security level only, make it a parameter later when necessary
        '''
        # core stuff
        dt=self.get_ts([ticker],['os_date'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[True,'os_date'],
           _is_os=False,auto_fill=False,_empty_check=False
                        )
        wgt=self.get_ts([ticker],['holder_stake'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # extra tagging
        wgt_id=self.get_ts([ticker],['holder_name'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        wgt_type=self.get_ts([ticker],['holder_type'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        wgt_country=self.get_ts([ticker],['holder_country'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # NEW: holder style
        wgt_style=self.get_ts([ticker],['holder_style'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # combine all
        stake=pd.DataFrame(index=wgt_id['value'].values,columns=dt.index,data=wgt.values)
        holder_tags=pd.DataFrame(data=[wgt_id['value'].values,
                           wgt_type['value'].values,
                           wgt_country['value'].values,
                           wgt_style['value'].values,
                           ],index=['name','type','country','style']).T.set_index('name')
        stake.index.name='holder_name'
        stake=stake.stack().rename('stake').reset_index().set_index('holder_name')
        holder_tags=holder_tags.reset_index().groupby('name').last()
        for col in holder_tags.columns:
            stake[col]=holder_tags[col]
        stake['ticker']=ticker
        return stake


    def get_holders_type_only(self,ticker,start='-12M',end='-0D',):
        '''
        we only download os_date, holder_stake, holder_name and holder_type
        '''
        # core stuff
        dt=self.get_ts([ticker],['os_date'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[True,'os_date'],
           _is_os=False,auto_fill=False,_empty_check=False
                        )
        wgt=self.get_ts([ticker],['holder_stake'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        wgt_id=self.get_ts([ticker],['holder_name'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                       )
        wgt_type=self.get_ts([ticker],['holder_type'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )

        # combine all
        stake=pd.DataFrame(index=wgt_id['value'].values,columns=dt.index,data=wgt.values)
        holder_tags=pd.DataFrame(data=[
                            wgt_id['value'].values,
                           wgt_type['value'].values,
                           ],index=['name','type']).T.set_index('name')
        stake.index.name='holder_name'
        stake=stake.stack().rename('stake').reset_index().set_index('holder_name')
        holder_tags=holder_tags.reset_index().groupby('name').last()
        for col in holder_tags.columns:
            stake[col]=holder_tags[col]
        stake['ticker']=ticker
        return stake


    def get_holders_snapshot(self,ticker):
        '''
        only the latest snapshot
        '''
        start='0D'
        end=''
        # core stuff
        dt=self.get_ts([ticker],['os_date'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[True,'os_date'],
           _is_os=False,auto_fill=False,_empty_check=False
                        )
        wgt=self.get_ts([ticker],['holder_stake'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # extra tagging
        wgt_id=self.get_ts([ticker],['holder_name'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        wgt_type=self.get_ts([ticker],['holder_type'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        wgt_country=self.get_ts([ticker],['holder_country'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # NEW: holder style
        wgt_style=self.get_ts([ticker],['holder_style'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False
                        )
        # combine all
        stake=pd.DataFrame(index=wgt_id['value'].values,columns=dt.index,data=wgt.values)
        holder_tags=pd.DataFrame(data=[wgt_id['value'].values,
                           wgt_type['value'].values,
                           wgt_country['value'].values,
                           wgt_style['value'].values,
                           ],index=['name','type','country','style']).T.set_index('name')
        stake.index.name='holder_name'
        stake=stake.stack().rename('stake').reset_index().set_index('holder_name')
        holder_tags=holder_tags.reset_index().groupby('name').last()
        for col in holder_tags.columns:
            stake[col]=holder_tags[col]
        stake['ticker']=ticker
        return stake
    def get_holder_stake_grouped(self,ticker,holder_type,group_name,
                            start='-12M',end='-0D'):
        '''
        Faster download for 1-level grouped analysis
        MTD can capture the latest intra-month movement
        Need to download one by one
        holder_type: fund (M), inst (F),
                    insider (S, including individual and institution),
                    inst_insider (FS, the rest will be unidentified)
        group_name: ULTPARENT (actual entity name, e.g. BlackRock, Schrolder etc),
                INVESTOR (investor type),
                ACTPASS (ONLY USE WITH holder_type=fund),
                STYLE (ONLY USE WITH holder_type=fund)
        '''
        group_name_input='"%s"' % (group_name)
        holder_type_map={'fund':'M','inst':'F','insider':'S','inst_insider':'FS',}
        holder_type_input=holder_type_map[holder_type]
        dt=self.get_ts([ticker],['os_date'],start=start,end=end,
            _check_fields=False,
           _automatic_date=[True,'os_date'],
           _is_os=False,auto_fill=False,_empty_check=False
                        )
        wgt_chg=self.get_ts([ticker],['holder_stake_grouped'],
            start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False,
           _os_holder_type=holder_type_input,
           _os_group_name=group_name_input
                        )
        name=self.get_ts([ticker],['holder_name_grouped'],
            start=start,end=end,
            _check_fields=False,
           _automatic_date=[False,'os_date'],
           _is_os=True,auto_fill=False,_empty_check=False,
           _os_holder_type=holder_type_input,
           _os_group_name=group_name_input
                        )
        res=wgt_chg.copy()
        res.index=name['value'].values.tolist()
        res.columns=dt.index
        res.index.name='holder_group'
        res=res.stack().rename('stake').reset_index()
        res['holder_type']=holder_type
        res['group_name']=group_name
        res['ticker']=ticker
        return res

    def get_sedol_hist(self,tickers,start='-10AY',end='NOW'):
        fields=['sedol_hist_id','sedol_hist_start','sedol_hist_end']
        res=self.get_ts(tickers,fields,
                        start=start,end=end,
                        freq='M',
                        _skip_convert_to_numeric=True).stack().swaplevel(1,0,0)
        res['sedol_hist_end']=res['sedol_hist_end'].map(lambda x: today_date.strftime('%Y%m%d') if x=='@NA' else x)
        res['sedol_hist_start']=res['sedol_hist_start'].map(lambda x: res.index.levels[1].min().strftime('%Y%m%d') if x=='@NA' else x)
        res['sedol_hist_end']=pd.to_datetime(res['sedol_hist_end'],format='%Y%m%d')
        res['sedol_hist_start']=pd.to_datetime(res['sedol_hist_start'],format='%Y%m%d')
        res=res.reset_index()
        res_short=res.groupby(['ticker','sedol_hist_start','sedol_hist_end'])['sedol_hist_id'].last()
        res_short=res_short.map(lambda x: np.nan if x=='@NA' else x)
        res_short=res_short.dropna()
        return res_short


    def _fields_check(self,fields,mode):
        if mode=='ts':
            available_fields=self.ts[self.ts['tag']==1].index
        elif mode=='snap':
            available_fields=self.snap[self.snap['tag']==1].index
        fields_index=pd.Index(fields)
        check=fields_index[~fields_index.isin(available_fields)]
        if len(check)!=0:
            print ('Fields %s are not defined by self.ts please remove them.' % (check))
            return False
        return True
    def _auto_fill(self,df_input,zero_fields=['volume','turnover']):
        #zero fill the zero_fileds and ffill the rest
        all_fields=df_input.columns.levels[0].tolist()
        df=df_input.copy()
        for field in all_fields:
            df[field]=df[field].fillna(0) if field in zero_fields else df[field].fillna(method='ffill')
        return df
    #core function for time series
    def _get_ts(self,tickers,fields,
           start='-10AY',end='NOW',freq='D',fx='',adj=True,
           fill='na', #can choose 0 or na
           date_field='date',
           rbasis='pass the rbasis map dict here',
           _fp='pass the fiscal period here',
           _window='pass the estimate lookback window here',
           _method='pass the estimate aggregate method here',
           _force_date_alignement=False,
           _is_os=False, # whether it's ownership data
           _os_holder_type='FS',
           _os_group_name='INVESTOR',
           _skip_convert_to_numeric=False,
           ):
        '''
        fx can be replaced by USD EUR etcs
        adj either no adj or adj everything
        can expand to more granular type if necessary
        '''
        #build formula
        fill_dict={'na':' R','0':' Z R'}
        para_dict={'start':start,'end':end,'freq':freq,'fx':fx,
                   'adj': '4' if adj else '9',
                   'fill':fill_dict[fill],
                   #here we do not replace rbasis as it needs ticker mapping info
                   'fp':str(_fp),
                   'window':str(_window),
                   'method':_method,
                   'holder_type':_os_holder_type,
                   'group_name':_os_group_name,
                   }
        formulas=[]
        for field in fields:
            formula=self.ts['fql'][field].replace('field',self.ts['field'][field])
            for k,v in para_dict.items():
                formula=formula.replace(k,v)
            formulas.append(formula)
        field_formula_map=pd.Series(dict(zip(fields,formulas)))

        fql=self._build_fql(tickers,fields,field_formula_map,
                            rbasis=[True,rbasis])
        res=self._execute_fql(fql)

        if not _is_os:
            data=self._tidy_up_fql(res,'ts',date_field=date_field,
                               _force_date_alignement=_force_date_alignement,
                               _skip_convert_to_numeric=_skip_convert_to_numeric,
                               )
        else:
            #ownership data
            data=res.iloc[:,3:]
        return data

    #core function for snapshot
    def _get_snap(self,tickers,fields):
        formulas=[]
        for field in fields:
            formula=self.snap['fql'][field]
            formulas.append(formula)
        field_formula_map=pd.Series(dict(zip(fields,formulas)))

        fql=self._build_fql(tickers,fields,field_formula_map,rbasis=[False,'pass rbasis dict here'])
        res=self._execute_fql(fql)
        res=self._tidy_up_fql(res,'snap')
        return res
    def _build_fql(self,tickers,fields,field_formula_map,rbasis=[False,'pass rbasis dict here']):
        index=pd.MultiIndex.from_product([tickers,fields],names=['ticker','field'])
        fql=pd.DataFrame(index=index,columns=['value']).reset_index().set_index('field')
        fql['value']=field_formula_map
        fql=fql.reset_index()
        fql['ticker_fql']=fql['ticker'].map(lambda x: x+'^')
        fql=fql.set_index(['ticker','field','ticker_fql']).sort_index().reset_index()
        if rbasis[0]:
            rbasis_map=rbasis[1]
            fql['value']=(
                        fql[['ticker','value']]
                        .apply(lambda x: x['value'].replace('rbasis',rbasis_map[x['ticker']]),axis=1)
                        )
        return fql
    def _execute_fql(self,fql):
        # we need to redo the dispatch if we have com error
        '''
        common com error:
            com_error: (-2147352567, 'Exception occurred.', (0, None, None, None, 0, -2147467259), None)
            com_error: (-2147023174, 'The RPC server is unavailable.', None, None)
        '''
        self._fs=wc.Dispatch("FactSet.FactSet_API")
        self.session=str(hex(int(np.random.uniform(0,1000000))*int(np.random.uniform(0,1000000))))
        #for data
        self.PATH={
            'temp':fql_temp_path+'fql\\fql_temp_%s.xlsx' % (self.session),
            'temp_res':fql_temp_path+'fql\\D_fql_temp_%s.xlsx' % (self.session),
          }

        #make sure no existing temp file
        if os.path.isfile(self.PATH['temp']):
            os.remove(self.PATH['temp'])
            print ('existing temp file deleted')
        if os.path.isfile(self.PATH['temp_res']):
            os.remove(self.PATH['temp_res'])
            print ('existing D_temp file deleted')
        temp=self.PATH['temp']
        fql.to_excel(temp)
        para="%s, Batch=TRUE" % (temp)
        self._fs.RunApplication('Downloading',para)
        res=pd.read_excel(self.PATH['temp_res'],sheet_name='Sheet1',index_col=0)
        os.remove(self.PATH['temp'])
        os.remove(self.PATH['temp_res'])
        return res
    def _tidy_up_fql(self,res,mode,date_field='date',_force_date_alignement=False,
                     _skip_convert_to_numeric=False):
        if mode=='snap':
            res=res.drop('ticker_fql',axis=1)
            res=res.set_index(['ticker','field'])['value'].unstack()
            return res
        elif mode=='ts':
            if (date_field=='date' or date_field=='os_date') and not _force_date_alignement:
                dates=res[res['field']==date_field].drop(['field','ticker_fql','ticker'],1).iloc[0].values
                dates=pd.to_datetime(dates)
                data=res[res['field']!=date_field].drop(['ticker_fql'],1).set_index(['field','ticker'])
                data.columns=dates
                data=data.T
                data.index.name='date'
                #force all to numeric
                if not _skip_convert_to_numeric:
                    data=data.applymap(lambda x: np.nan if x=='@NA' else x).apply(pd.to_numeric)
                data=data.sort_index().sort_index(1)
                return data
            elif (date_field=='rpt_date' or date_field=='rpt_date_fp_end') or _force_date_alignement:
                '''
                since reporting date is different by stock we need a different way to tidy up
                '''
                rpt_dates=res.drop('ticker_fql',1).set_index(['field','ticker']).loc[[date_field]]
                if not _force_date_alignement:
                    rpt_dates=rpt_dates.applymap(lambda x: np.nan if x in ['@NA','nan'] else x).applymap(float)
                rpt_values=res.drop('ticker_fql',1).set_index(['field','ticker']).drop(date_field,level=0)
                rpt_values=rpt_values.applymap(lambda x: np.nan if x=='@NA' else x).apply(pd.to_numeric)
                data_temp=pd.concat([rpt_dates,rpt_values],axis=0).swaplevel(1,0,0).sort_index()
                temp_collector=[]
                for ticker in data_temp.index.levels[0]:
                    df=data_temp.loc[ticker].T
                    if not _force_date_alignement:
                        df=df[df[date_field].map(lambda x: True if ~np.isnan(x) else False)]
                    else:
                        df=df[df[date_field].fillna(0).map(lambda x: True if x!=0 else False)]
                    if not _force_date_alignement:
                        df[date_field]=df[date_field].map(lambda x: pd.to_datetime(str(int(x)),format='%Y%m%d'))
                    else:
                        df[date_field]=df[date_field].map(lambda x: pd.to_datetime(x))
                    df['ticker']=ticker
                    #we use groupby here because sometimes we have duplicatged rpt_date (factset data error)
                    temp_collector.append(df.groupby([date_field,'ticker']).last().unstack())
                data=pd.concat(temp_collector,1).sort_index(1)
                data.index.name='date'
                return data

    def _check_for_tickers_with_empty_return(self,res,tickers):
        #for ts output only, for now
        returned_tickers=res.columns.levels[1]
        input_tickers=pd.Series(tickers)
        missing_tickers=input_tickers[~input_tickers.isin(returned_tickers)]
        if len(missing_tickers)!=0:
            print ('Below %s tickers returned nothing for the selected fields, please check manually' % (len(missing_tickers)))
            print (missing_tickers)
        else:
            print ('All tickers have returned value (nan included) for the selected fields')
    def _clean_tickers_with_valid_rpt_freq(self,tickers):
        '''
        Check if the latest reporting_freq(FF_FREQ_CODE) returns @NA
        return clean ticker list with no @NA freq
        '''
        print ('checking ticker list for valid fundamental reporting date')
        res=self.get_snap(tickers,['reporting_freq']).applymap(str)
        check=res[res['reporting_freq']=='@NA']
        if len(check)!=0:
            print('Following tickers contain @NA reporting frequency, drop them automatically')
            print (check.index)
            tickers=res.drop(check.index,0).index.tolist()
        else:
            print ('All tickers have valid reporting frequncy')
        return tickers
    def _remove_duplicated_input(self,tickers):
        return list(set(tickers))
    
    
    
if __name__=='__main__':
    print ('ok')
    
    # fq=Factset_Query()
    # tickers=['2662-HK']
    # fields=['ret_daily','turnover']
    # data=fq.get_ts(tickers,fields,start='-10AY')
    
    
    
    # fq=Factset_Query()
    # tickers=['700-HK','5-HK']
    # fields=['px_last',
    #         'px_open',
    #         'vwap',]
    # fs_data=fq.get_ts(tickers,fields)
    
    
    
#     ticker='700-HK'
# #             ,'601398-CN','883-HK','9983-JP','2413-JP',
# #             '241-HK','3333-HK']
#     res=fq.get_holder_stake_grouped(ticker,'inst_insider','CNTRY', start='-3M',end='-0D')

#    fields=['sedol_hist_id','sedol_hist_start','sedol_hist_end']
#    res=fq.get_ts(tickers,fields,freq='M',
#                    _skip_convert_to_numeric=True).stack().swaplevel(1,0,0)
#    res['sedol_hist_end']=res['sedol_hist_end'].map(lambda x: today_date.strftime('%Y%m%d') if x=='@NA' else x)
#    res['sedol_hist_start']=res['sedol_hist_start'].map(lambda x: res.index.levels[1].min().strftime('%Y%m%d') if x=='@NA' else x)
#    res['sedol_hist_end']=pd.to_datetime(res['sedol_hist_end'],format='%Y%m%d')
#    res['sedol_hist_start']=pd.to_datetime(res['sedol_hist_start'],format='%Y%m%d')
#    res=res.reset_index()
#
#    res_short=res.groupby(['ticker','sedol_hist_start','sedol_hist_end'])['sedol_hist_id'].last()
#    res_short=res_short.map(lambda x: np.nan if x=='@NA' else x)
#    res_short=res_short.dropna()
#
#    fq=Factset_Query(session=1234)
##    res=fq.get_ts_reported_fundamental(['1834-JP','1839-JP','2052-JP',
##                                        '2261-JP','2536-JP','2572-JP'],['roe'],start='-10Y',auto_clean=False)
#    stake_grouped=fq.get_holder_stake_grouped(
#                                '017670-KR',
#                                'inst_insider','INVESTOR',
#                            start='-0D',end='-0D')
#    check=fq.get_holders_snapshot()
#    # get reporting date history
#    fq=Factset_Query(session=1234)
#    tags=fq.get_holders_type_only('601398-CN',start='01/01/2009',
#                                  end='01/01/2010')
#    #double check float
#    fq=Factset_Query(session=1234)
#    start=datetime(2014,6,1)
#    res=fq.get_holders('1398-HK',start=start.strftime('%m/%d/%Y'),end='-0D')
#
#
#    fq=Factset_Query(session=1)
#    fields=['dps','eps']
#    tickers=['8306-JP','2687-JP']
#    fiscal_years=np.arange(2015,2021,1)
#    bm_est,est,est_dates=fq.get_ts_estimates_guidance_beat_miss(tickers,fields,fiscal_years,mode='est')
#    bm_guid,guid,guid_dates=fq.get_ts_estimates_guidance_beat_miss(tickers,fields,fiscal_years,mode='guid')
#
#    fq=Factset_Query(session=1)
#    '''Get guidance data, rolling'''
#    tickers=['2687-JP','8306-JP']
#    fields=['dps','eps']
#    guid_roll=fq.get_ts_guidance_rolling(tickers,fields)

#    ### test the beat/miss wrapper
#    fq=Factset_Query(session=1)
#    fields=['eps','dps']
#    tickers=['883-HK','8306-JP','2914-JP']
#    fiscal_years=np.arange(2015,2021,1)
#    mode='guid'
#    bm,est,est_dates=fq.get_ts_estimates_guidance_beat_miss(tickers,fields,fiscal_years,mode=mode)
#

#    ### test guidance abs
#    fq=Factset_Query(session=1)
#    fields=['dps_guid']
#    tickers=['2687-JP','8306-JP']
#    fiscal_years=np.arange(2010,2021,1)
#    res,res_dates=fq.get_ts_guidance_abs_and_dates( tickers,fields,fiscal_years)
#    res_rpt_div=fq.get_ts_reported_fundamental(tickers,['dps'],rbasis='ANN')
#
#    res_roll=fq.get_ts_guidance_rolling(tickers,fields)
#
#    #with the above we should be able to calculate the beat/miss
#    #we should also be able to reconcile the FY1 rolling from the abs data
#    guid_abs_stacked=res.stack().stack().swaplevel(1,2,0).reset_index().set_index(['date','ticker'])
#    rpt_stacked=res_rpt_div.stack()
#    roll_stacked=res_roll.stack().rename(columns={'dps_guid':'dps_guid_roll'})
#
#    #reconcile fy1
#    check=pd.concat([
#            guid_abs_stacked.reset_index().groupby(['date','ticker']).last(),
#            #with .last we put the next guidance on the current reporting date
#            roll_stacked],axis=1)
#    #create the beat and miss
#    beat_miss=pd.concat([
#            guid_abs_stacked.reset_index().groupby(['date','ticker']).first(),
#            #with .first we put the current guidance on the current reporting date
#            rpt_stacked],axis=1)
#    ### test guidance rolling
#    fq=Factset_Query(session=1)
#    fields=['dps_guid']
#    tickers=['2687-JP','8306-JP']
#    res=fq.get_ts_guidance_rolling(tickers,fields)
#    res_est=fq.get_ts_estimates_rolling(tickers,['dps_est'],'ANN_ROLL','+1')
#
#    check=pd.concat([res['dps_guid']['8306-JP'].rename('guidance'),
#                     res_est['dps_est']['8306-JP'].rename('estimate'),
#                     ],axis=1)
#    ### test estimate date -- fixed
#    fq=Factset_Query(session=1)
#    fields=['eps_est']#,'roe_est']
#    tickers=['1398-HK','700-HK','883-HK','7203-JP']
#    fiscal_years=np.arange(2015,2020,1)
#    res,res_dates=fq.get_ts_estimates_abs_and_dates(tickers,fields,'auto',fiscal_years)
#    res_reported=fq.get_ts_reported_fundamental(tickers,['eps'],rbasis='auto')
#
#
#    ticker='700-HK'
#    est=res['eps_est'].stack()[ticker].dropna().unstack()
#    rpt=res_reported['eps'][ticker].dropna().rename('reported')
#    eps_check=pd.concat([est,rpt],axis=1)
#    eps_check['reported']=eps_check['reported'].fillna(method='ffill')
#
#    eps_check.loc[datetime(2014,1,1):].plot(figsize=(16,9))
#
#    eps_check[['2017/4F','reported']].loc[datetime(2014,1,1):].plot(figsize=(16,9))
#
#    ### test estimate data -- rolling
#
#    fq=Factset_Query()
#    fields=['roe_est']
#    tickers=['1398-HK','700-HK','883-HK']
#    fy1=fq.get_ts_rolling_estimates(tickers,fields,'ANN_ROLL','+1')
#    fy2=fq.get_ts_rolling_estimates(tickers,fields,'ANN_ROLL','+2')
#    fy3=fq.get_ts_rolling_estimates(tickers,fields,'ANN_ROLL','+3')
#    bf1_qtr=fq.get_ts_rolling_estimates(tickers,fields,'NTM4_ROLL','')
#    bf1_ntma=fq.get_ts_rolling_estimates(tickers,fields,'NTMA','')
#    bf1_ltma=fq.get_ts_rolling_estimates(tickers,fields,'LTMA','')
#    bf1_stma=fq.get_ts_rolling_estimates(tickers,fields,'STMA','')
#    reported_eps=fq.get_ts_reported_fundamental(tickers,['roe'],rbasis='ANN')
#
#    fy=pd.concat([
#            fy1.stack()['roe_est'].rename('fy1'),
#            fy2.stack()['roe_est'].rename('fy2'),
#            fy3.stack()['roe_est'].rename('fy3'),
#            bf1_qtr.stack()['roe_est'].rename('NTM4_ROLL'),
#            bf1_ntma.stack()['roe_est'].rename('NTMA'),
#             bf1_ltma.stack()['roe_est'].rename('LTMA'),
#             bf1_stma.stack()['roe_est'].rename('STMA'),
#             reported_eps.stack()['roe'].rename('reported'),
#            ],axis=1)
#
#    fy.swaplevel(1,0,0).loc['700-HK'][['fy1','NTMA','fy2']].plot(figsize=(10,8))
#    fy.swaplevel(1,0,0).loc['700-HK'][['reported','LTMA','fy1']].fillna(method='ffill').plot(figsize=(10,8))
#

#
#    def bbg_to_fs(tickers):
#        '''
#        This should work for China A, HK, JP
#        '''
#        new_tickers=[]
#        for t in tickers:
#            t=t.replace(' CH Equity','-CN')
#            t=t.replace(' HK Equity','-HK')
#            t=t.replace(' JP Equity','-JP')
#            t=t.replace(' JT Equity','-JP')
#            t=t.replace(' KS Equity','-KR')
#
#            new_tickers.append(t)
#
#        return new_tickers
#
#    ### get both generic field and special field
#    session=1
#    fq=Factset_Query(session=session)
#    #tickers=['1398-HK','601398-CN','700-HK','883-HK']
#
#    fields=[
##            'px_last',
##            'px_open',
##            'vwap',
#
##        'roe',
#        'eps',
##        'net_income',
##        'sales',
##        'ebit',
##        'ebit_margin',
##        'ebitda',
##        'ebitda_margin',
##        'roa',
##        'debt_to_equity',
##        'cash_st_inv',
##        'cash',
##        'net_debt',
##        'asset_turnover',
##        'asset_to_equity',
#            ]
#
##    fields=fq.get_all_defined_fields()
##    tickers_all=bbg_to_fs(pd.read_csv("Y:\\Dave Yin\\index_compo.csv")['tickers'].values)[:1008]
#
#    tickers_all=['600012-CN','600115-CN','883-HK','700-HK','600005-CN']
#
#    #tickers_all=fq.clean_tickers_with_valid_rpt_freq(tickers_all)
#    data=fq.get_ts_reported_fundamental(tickers_all,fields,start='-10AY',rbasis='auto',fx='"RPT"')
    #data=fq.get_ts(tickers_all,fields,start='-1AY')
#    #check for missing tickers
#    output=data.columns.levels[1]
#    input_tickers=pd.Series(tickers_all)
#    missing_tickers=input_tickers[~input_tickers.isin(output)]
    #data=fq.get_ts_reported_fundamental(tickers,fields,start='-10AY')

    #data=fq.get_ts_reported_fundamental(tickers,fields)
#    ff_stats=fq.get_special_field_float(tickers,start=start,fx=fx)
#
#    res=pd.concat([data,ff_stats],axis=1).sort_index(1).dropna()
#
#
#    import feather
#    path = 'Z:\\dave\\data\\Misc\\data.feather'
#    feather.write_dataframe(res.stack().stack().rename('value').reset_index(), path)

#    ### test batch wraper
#
#    def bbg_to_fs(tickers):
#        '''
#        This should work for China A, HK, JP
#        '''
#        new_tickers=[]
#        for t in tickers:
#            t=t.replace(' CH Equity','-CN')
#            t=t.replace(' HK Equity','-HK')
#            t=t.replace(' JP Equity','-JP')
#            t=t.replace(' JT Equity','-JP')
#            t=t.replace(' KS Equity','-KR')
#
#            new_tickers.append(t)
#
#        return new_tickers
#
#    fq=Factset_Query()
#    tickers=pd.read_csv("Y:\\Dave Yin\\index_compo.csv")['tickers']
#
#    tickers=bbg_to_fs(tickers)
#
#    fields=['px_open','px_high','px_low','px_last']#fq.get_all_defined_ts_fields()
#    start='-6M'
#    df=fq.get_ts(tickers,fields,start=start)


#    ### test get snap
#
#    def bbg_to_fs(tickers):
#        '''
#        This should work for China A, HK, JP
#        '''
#        new_tickers=[]
#        for t in tickers:
#            t=t.replace(' CH Equity','-CN')
#            t=t.replace(' HK Equity','-HK')
#            t=t.replace(' JP Equity','-JP')
#            t=t.replace(' JT Equity','-JP')
#            t=t.replace(' KS Equity','-KR')
#
#            new_tickers.append(t)
#
#        return new_tickers
#
#    fq=Factset_Query()
#    tickers=pd.read_csv("Y:\\Dave Yin\\index_compo.csv")['tickers']
#    tickers=bbg_to_fs(tickers)
#    fields=['name','short_name','country','exchange','factset_sector','factset_industry']
#    res=fq.get_snap(tickers,fields)





































