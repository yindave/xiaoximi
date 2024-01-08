# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:50:54 2021

@author: davehanzhang
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 08:33:38 2020

@author: hyin1
"""

'''
In this lite version we update JD and JB
- for JD we only keep the guidance change, no beat and miss
- for market data we need adjusted-shareout, px, px-unadj, vwap, to

- we update event window using carhart module

- for other fundamental inputs, we download separately

Notes:
Dividend split adjustment: note that even factset data can fail for split adjustment sometimes
        (e.g. 1301-JP, FY1 guidance data from 16-18 messed up)
        
        Some tickers have guidance in BBG but not in Factset (hopefully only a few there)
        (e.g. 1887-JP)

Guid timing: in most of the cases the guidance change happens on announcement date
    however there are few cases where guid happens on non-announcement date
    e.g. 1419-JP guid upgrade on 2020/02/04 but reporting dates are in 01/14, 1H and 04/14, 3Q (same FY period)
    And this unexpected guid-up leads to very positive price return
    one more example (1430-JP guid upgrade on 2015/12/24)
    
    However sometimes it's just a data issue (e.g. 1605-JP on 2018/11/08)
    
    Guidance can also happen a few days BEFORE the closet reporting date
    e.g. 1377-JP on 2017/07/06 guid is before the 07/13 reporting date
    after checking BBG it shows that the 07/06 guidance is for previous reporting FY on 2017/04/07
    and the upgrade to 28jpy is only for that FY, and the next FY dps guid is lower
    
    
    For now let's just tag div and bb event date with the closest reporting date before the event
    
        
'''

import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.constants as uc
import feather
from excess_return.excess_return import CARHART
from fql.util import bbg_to_fs,fs_to_bbg,fql_date
import pdb

from fql.fql import Factset_Query
from blp.bdx import bdp,bdh

from blp.util import tidy_up_buyback_dump

import os,time


class JCM():
    GUIDANCE_PCT_CHG_MAX=0.8
    BB_PCT_MAX=0.5
    LAST_UPDATE=8 # so when updating backend etc, if existing files are updated 8 hours ago we will skip the new update
    
    def __init__(self):
        self.path="Z:\\dave\\data\\jcm\\"
        #self.path="C:\\Users\\hyin1\\temp_data\jcm\\"
        
        print ('current path is %s' % (self.path))
        self.bb_path="Z:\\dave\\data\\buyback\\japan\\raw\\"
        self.start=pd.datetime(2004,12,31)
        self.end=um.yesterday_date()
        
        mkt_data_path="Z:\\dave\\data\\excess_return\\carhart-TPX-JPY\\"
        compo=feather.read_dataframe(mkt_data_path+'compo.feather').groupby('ticker').last()
        
        self.universe=bbg_to_fs([x for x in compo.index if len(x)==14])
        #self.universe=self.universe[30:60]
        print ('%s stocks in the universe' % (len(self.universe)))
        
        
        self.fql=Factset_Query(session=np.random.uniform(0,1000000))
        
        return None

    
    
    def load_data(self):
       
        #market data
        mkt_to_usd=feather.read_dataframe(self.path+'turnover_usd.feather').set_index(['date','ticker'])
        mkt_px_unadj_usd=feather.read_dataframe(self.path+'px_unadj_usd.feather').set_index(['date','ticker'])
        mkt_px_unadj_jpy=feather.read_dataframe(self.path+'px_unadj_jpy.feather').set_index(['date','ticker'])
        mkt_px_vwap=feather.read_dataframe(self.path+'vwap_last_jpy.feather').set_index(['date','ticker'])
        mkt=(mkt_px_vwap
        .join(mkt_to_usd.rename(columns={'turnover':'turnover_usd'})
        .join(mkt_px_unadj_usd.rename(columns={'px_last':'px_last_unadj_usd'}))
        .join(mkt_px_unadj_jpy.rename(columns={'px_last':'px_last_unadj_jpy'}))
        ))
        
        self.mkt=mkt.copy()
        mkt=mkt.unstack().resample('D').last().fillna(method='ffill').stack()
        
        #fundamental data
        data_p=feather.read_dataframe(self.path+'funda_p.feather').set_index(['date','ticker'])
        gics=feather.read_dataframe(self.path+'gics.feather').set_index('index')
        funda=data_p.copy()
        funda=funda.reset_index().set_index('ticker')
        funda['sector']=gics['sector']
        self.funda=funda.reset_index().set_index(['date','ticker']).copy()
        funda=funda.reset_index().set_index(['date','ticker']).unstack().resample('D').last().fillna(method='ffill').stack()
        
        #guidance data
        self.guid_roll_level=feather.read_dataframe(self.path+'dps_guid_roll.feather').set_index(['date','ticker'])
        guid_roll_chg=self.guid_roll_level['dps_guid'].unstack().fillna(method='ffill').fillna(method='bfill').pct_change()
        guid_roll_diff=self.guid_roll_level['dps_guid'].unstack().fillna(method='ffill').fillna(method='bfill').diff().stack().rename('guid_diff_chg').to_frame()
        guid_mask=guid_roll_chg.applymap(lambda x: False if abs(x)<=self.GUIDANCE_PCT_CHG_MAX else True)
        guid_roll_chg_clean=guid_roll_chg.mask(guid_mask).applymap(lambda x: np.nan if x==0 else x).stack()
        guid_chg=guid_roll_chg_clean.rename('guid_pct_chg').to_frame()
        guid_chg=guid_chg.join(guid_roll_diff).join(funda[['sector']])
        guid_chg=guid_chg.join(mkt[['px_last_unadj_jpy','px_last_unadj_usd','shout_sec']])
        guid_chg['guid_chg_amount_musd']=guid_chg['guid_diff_chg']*guid_chg['shout_sec']*guid_chg['px_last_unadj_usd']/guid_chg['px_last_unadj_jpy']
        guid_chg=guid_chg[['guid_pct_chg','guid_chg_amount_musd','sector']].copy()
        self.div=guid_chg.copy()
        
        #buyback data
        bb=feather.read_dataframe(self.path+'buyback.feather').set_index(['date','ticker'])
        bb=bb.join(mkt[['px_last_unadj_jpy','px_last_unadj_usd','marcap_sec']])
        bb['bb_size_mn_usd']=(bb[['bb_amount_mn','bb_amount_unit','px_last_unadj_usd','px_last_unadj_jpy']]
            .apply(lambda x: x['bb_amount_mn']*x['px_last_unadj_usd'] if x['bb_amount_unit']=='shares' else x['bb_amount_mn']/x['px_last_unadj_jpy']*x['px_last_unadj_usd'],axis=1)
            )
        bb['bb_size_pct']=bb['bb_size_mn_usd']/bb['px_last_unadj_usd']*bb['px_last_unadj_jpy']/bb['marcap_sec']
        bb=bb.dropna()[['bb_size_mn_usd','bb_size_pct']].copy()
        bb=bb[bb['bb_size_pct']<=self.BB_PCT_MAX]
        bb=bb.join(funda[['sector']])
        self.bb=bb.copy()
        
        #Add the closest reporting date on or before the buyback and dividend event
        div=self.div.reset_index().copy()
        bb=self.bb.reset_index().copy()
        funda=self.funda.reset_index().copy()
        div['year']=div['date'].map(lambda x: x.year)
        div['month']=div['date'].map(lambda x: x.month)
        bb['year']=bb['date'].map(lambda x: x.year)
        bb['month']=bb['date'].map(lambda x: x.month)

        div=div.set_index(['ticker','date']).sort_index()
        bb=bb.set_index(['ticker','date']).sort_index()
        funda=funda.set_index(['ticker','date']).sort_index()


        jcm_data_all=[div,bb]
        for jcm_data_i in jcm_data_all:
            for ticker_dt in jcm_data_i.index:
                ticker_i=ticker_dt[0]
                dt_i=ticker_dt[1]
                if ticker_i in funda.index:
                    funda_i=funda.loc[ticker_i]                    
                    for i,funda_dt_i in enumerate(funda_i.index):
                        try:
                            funda_dt_i_next=funda_i.index[i+1]
                        except IndexError:
                            funda_dt_i_next=um.today_date()
                        if dt_i>=funda_dt_i and dt_i<funda_dt_i_next:
                            jcm_data_i.at[ticker_dt,'latest_rpt_dt']=funda_dt_i
                            break
            
        self.div=div.copy()
        self.bb=bb.copy()            

        print ('database loaded')
        
        
    def load_evt_window(self):
        self.evt_window_simple=feather.read_dataframe(self.path+'event_window_simple.feather')
        self.evt_window_carhart=feather.read_dataframe(self.path+'event_window_carhart.feather')
    
    
    
    def update_data(self):
        
        self._update_mkt()
        self._update_fundamental()
        self._update_guidance()
        self._update_buyback()
    
    
    def _is_fresh(self,file_name):        
        return um.is_fresh(file_name,level_h=self.LAST_UPDATE)
    
    def _update_mkt(self):
              
        
        tickers=self.universe

        file_name='turnover_usd'
        if not self._is_fresh(self.path+'%s.feather' % (file_name)):
            turnover=self.fql.get_ts(tickers,['turnover'],start=fql_date(self.start),end='NOW',freq='D',fx='USD')
            feather.write_dataframe(turnover.stack().reset_index(),self.path+'%s.feather' % (file_name))
        
        file_name='vwap_last_jpy'
        if not self._is_fresh(self.path+'%s.feather' % (file_name)):
            idx=bdh(['TPXDDVD Index'],['px_last'],self.start,um.today_date())['px_last'].loc['TPXDDVD Index']
            vwap_last_jpy=self.fql.get_ts(tickers,['vwap','px_last','marcap_sec','shout_sec'],start=fql_date(self.start),end='NOW',freq='D')
            vwap_last_jpy[('px_last','tpxddvd')]=idx
            vwap_last_jpy[('vwap','tpxddvd')]=idx
            feather.write_dataframe(vwap_last_jpy.sort_index(axis=1).stack().reset_index(),self.path+'%s.feather' % (file_name))
        
        file_name='px_unadj_usd'
        if not self._is_fresh(self.path+'%s.feather' % (file_name)):
            px_unadj=self.fql.get_ts(tickers,['px_last'],start=fql_date(self.start),end='NOW',freq='D',adj=False,fx='USD')
            feather.write_dataframe(px_unadj.stack().reset_index(),self.path+'%s.feather' % (file_name))
        
        
        file_name='px_unadj_jpy'
        if not self._is_fresh(self.path+'%s.feather' % (file_name)):
            px_unadj_jpy=self.fql.get_ts(tickers,['px_last'],start=fql_date(self.start),end='NOW',freq='D',adj=False)
            feather.write_dataframe(px_unadj_jpy.stack().reset_index(),self.path+'%s.feather' % (file_name))
        
        print ('finish updating market')


    def _update_fundamental(self):
        '''
        From FQL we just get eps here for reporting date record
        '''
        tickers=self.universe
        
        fields_p=['eps']
        
        file_name='funda_p'
        if not self._is_fresh(self.path+'%s.feather' % (file_name)):
            data_p=self.fql.get_ts_reported_fundamental(tickers,fields_p,
                                                        fql_date(self.start),'NOW',
                                                        rbasis='auto',
                                                        auto_clean=False)
    
            tickers_bbg=fs_to_bbg(data_p.columns.levels[1])
            
            gics=bdp(tickers_bbg,['gics_sector_name'])
            gics=gics['gics_sector_name'].dropna().rename('sector').map(lambda x: uc.short_sector_name[x])
            gics.index=bbg_to_fs(gics.index)
    
            feather.write_dataframe(data_p.stack().reset_index(),self.path+'%s.feather' % (file_name))
            feather.write_dataframe(gics.to_frame().reset_index(),self.path+'gics.feather')

        print ('finish updating fundamental')
        
        return None

    def _update_guidance(self):
        '''
        We need:rolling FY1 guidance          
        '''
        #get the tickers with valid reporting frequency
        tickers=self.universe

        #dump guidance
        file_name='dps_guid_roll'
        if not self._is_fresh(self.path+'%s.feather' % (file_name)):
            guid_dps_roll=self.fql.get_ts_guidance_rolling(tickers,['dps'],
                                                           start=fql_date(self.start),
                                                           end='NOW')   
            feather.write_dataframe(guid_dps_roll.stack().reset_index(),self.path+'%s.feather' % (file_name))
        
        print ('finish updating guidance')
        return None
    
    
    def _update_buyback(self):
        bb_data=tidy_up_buyback_dump(self.bb_path).reset_index()
        
        bb_data_clean=bb_data[['Ticker','declared_date','effective_date','bb_amount_mn','bb_amount_unit','share_out_m']].copy()
        bb_data_clean['ticker']=bbg_to_fs(bb_data_clean['Ticker'])
        bb_data_clean=bb_data_clean[bb_data_clean['declared_date']!=bb_data_clean['effective_date']]
        bb_data_clean=bb_data_clean[['ticker','declared_date','bb_amount_mn','bb_amount_unit','share_out_m']]
        bb_data_clean=bb_data_clean.rename(columns={'declared_date':'date'})
        feather.write_dataframe(bb_data_clean,self.path+'buyback.feather')
        
        print ('finish updating buyback')
        
        
        return None

    
    
    def load_carhart(self):
        self.carhart=CARHART.Load_Model_Quick('JP')
    
    def update_event_window(self):
        self.load_carhart()
        
        
        # need to run the update data first (which loads the old event window)
        
        px=self.mkt['px_last'].unstack()
       
        px=px.resample('B').last().fillna(method='ffill')
        guid_roll_change=self.div.copy()
        guid_roll_change['value']=guid_roll_change['guid_pct_chg']
        bb_announcement=self.bb.copy()
        bb_announcement['value']=bb_announcement['bb_size_pct']

        
        windows=[126]
        event_dict={'div':guid_roll_change,'bb':bb_announcement}
        collector=[]
        

        for window in windows:
            for event_type, event_df in event_dict.items():
                for ticker_date in event_df.index:
                    
                    try:
                        event_date=ticker_date[1]
                        ticker=ticker_date[0]
                        chg=event_df.loc[ticker_date]['value']
                        
                        i=px.index.get_loc(event_date)
                        i_beg=i-window
                        i_end=i+window+1
                        
                        px_i=px.iloc[i_beg:i_end][['tpxddvd',ticker]]
                        px_i=px_i/px_i.loc[event_date]
                        px_i=px_i.rename(columns={ticker:'stock'})
                        px_i['rel']=px_i['stock']-px_i['tpxddvd']
                        px_i['event_date']=event_date
                        px_i['event_year']=event_date.year
                        px_i['ticker']=ticker
                        px_i['event_type']=event_type
                        px_i['chg']=chg
                        px_i['window']=window
                        try:
                            px_i['event_day']=np.arange(0,window*2+1)-window
                        except ValueError:
                            px_i['event_day']=np.arange(0,len(px_i))-window
        
                        
                        collector.append(px_i)
                        
                        print ('finish simple %s for %s, window size %s' % (event_type,ticker_date,window))
                    except KeyError:
                        print ('key error for %s for %s, window size %s' % (event_type,ticker_date,window))
         
        res=pd.concat(collector,axis=0)
        res['chg_type']=res['chg'].map(lambda x: '+ve' if x>0 else '-ve')
        
        #dump
        feather.write_dataframe(res,self.path+'event_window_simple.feather')
    
    
        ### get carhart event window
        windows=[126]
        collector_carhart=[]
        for window in windows:
            for event_type, event_df in event_dict.items():
                for ticker_date in event_df.index:
                    try:
                        event_date=ticker_date[1]
                        ticker=ticker_date[0]
                        chg=event_df.loc[ticker_date]['value']
                        
                        px_i=self.carhart.get_excess_return_around_event(ticker,event_date,window)
                        px_i['event_date']=event_date
                        px_i['event_year']=event_date.year
                        px_i['ticker']=ticker
                        px_i['event_type']=event_type
                        px_i['chg']=chg
                        px_i['window']=window
                        try:
                            px_i['event_day']=np.arange(0,window*2+1)-window
                        except ValueError:
                            px_i['event_day']=np.arange(0,len(px_i))-window
                        
                        collector_carhart.append(px_i)
                        
                        print ('finish carhart %s for %s, window size %s' % (event_type,ticker_date,window))
                    except:
                        print ('key or index error for %s for %s, window size %s' % (event_type,ticker_date,window))
                        
                        
        res_carhart=pd.concat(collector_carhart,axis=0)
        res_carhart['chg_type']=res_carhart['chg'].map(lambda x: '+ve' if x>0 else '-ve')
        
        #dump
        feather.write_dataframe(res_carhart,self.path+'event_window_carhart.feather')              
   
    
    
    
    
    
if __name__=="__main__":
    print ("ok")

    jcm=JCM()
    #jcm.update_data()
    jcm.load_data()
#    jcm.update_event_window()
#    
    