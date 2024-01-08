# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:19:24 2021

@author: hyin1
"""

from multiprocessing import Process
import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import argparse #optparse deprecated
import numpy as np
import pdb


'''
For this scripts, we want to run after all the funds have published their latest report
If we run with only a proportion of stocks reported (e.g. mid of month), then the data will be restated later

We can run this monthly to keep populating month end MF position snapshot
'''

path=em.path_static_list.replace('static_list.csv','')+'process_data\\'
hlds_to_use=feather.read_dataframe(path+'hlds_to_use.feather')
hlds_to_use['type']=hlds_to_use['ticker'].map(lambda x: x[-2:])
hlds_to_use=hlds_to_use[hlds_to_use['type']=='CN'].copy()
fund_tickers=hlds_to_use.groupby('ticker_fund').last().sort_index()

def run(batch,batch_total):
    all_funds=fund_tickers.iloc[batch:].iloc[::batch_total].index
    collector=[]
    for i,ticker_fund in enumerate(all_funds):
        print ('working on %s (%s/%s)' % (ticker_fund,i+1,len(all_funds)))
    #    try:
        hlds_fund_i=hlds_to_use[hlds_to_use['ticker_fund']==ticker_fund].set_index(['date','ticker'])[['shares','top_n','asof','values']].sort_index().loc[pd.datetime(2001,12,31):]
        top10=hlds_fund_i[hlds_fund_i['top_n']<=10]['shares'].unstack().fillna(0).resample('M').last().fillna(method='ffill') # we know this part in 1,4,7,10, and it will remain the same in 3 and 8
        rest=hlds_fund_i[hlds_fund_i['top_n']>10]['shares'].unstack().fillna(0).resample('M').last().fillna(method='ffill') # we know this part only in 3 and 8
        if len(top10)!=0:
            # so the logic is that we first fill "top" (in 3/8 month) with "rest", then we ffill "top" columns. This way we assume no sudden unwind and we are using the latest information
            if len(rest)!=0:
                rest=rest.reindex(top10.index).fillna(method='ffill')
                rebuilt=pd.concat([top10.stack().rename('top'),rest.stack().rename('rest')],axis=1).reset_index().fillna(0)
            else:
                rebuilt=pd.concat([top10.stack().rename('top'),top10.stack().rename('rest')],axis=1).reset_index().fillna(0)
            #rebuilt['shares']=rebuilt.apply(lambda x: x['rest'] if (x['top']==0 and x['date'].month in [3,8]) else x['top'],axis=1)
            rebuilt['shares']=rebuilt.apply(lambda x: x['rest'] if x['top']==0 else x['top'],axis=1)
            #rebuilt['shares']=rebuilt['shares'].map(lambda x: np.nan if x==0 else x)
        #    # to prevent the situation where we ffill indefinitely (leading to stake in some "legacy" stocks e.g. 000001 for 110011, we set some ffill limit below (2 years))
        #    rebuilt=rebuilt.set_index(['date','ticker'])['shares'].unstack().fillna(method='ffill',limit=12).fillna(0)
            rebuilt=rebuilt.set_index(['date','ticker']).unstack()
            rebuilt.loc[um.today_date()]=np.nan
            rebuilt=rebuilt.resample('M').last().fillna(method='ffill')
            rebuilt=rebuilt.stack().reset_index()
        #    rebuilt=rebuilt.stack().rename('shares').reset_index()
            rebuilt['ticker_fund']=ticker_fund
            # create a "date-ticker" index for easier datatable join later
            rebuilt['date_ticker']=rebuilt.apply(lambda x: '%s-%s' % (x['date'].strftime('%Y%m%d'),x['ticker']),axis=1)
            rebuilt['date_ticker_fund']=rebuilt.apply(lambda x: '%s-%s' % (x['date'].strftime('%Y%m%d'),x['ticker_fund']),axis=1)
            collector.append(rebuilt)
        #    except:
        #        print ('error on %s, skipping' % (ticker_fund))
    rebuilt_all=pd.concat(collector)
    dump_name=path+'dump_rebuilt\\rebuilt_holdings_shares_%s.feather' % (batch)
    feather.write_dataframe(rebuilt_all,dump_name)


if __name__ == "__main__":
    # parallel run
    job_N=15
    p_collector=[]
    for job_i in np.arange(0,job_N,1):
        p=Process(target=run,args=(job_i,job_N,))
        p_collector.append(p)
        p.start()
    for p in p_collector:
        p.join()




























