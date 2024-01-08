# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:02:22 2021

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

import datatable as dt
from datatable import f,by,sort,join,shift


print ('working on A_d_process_filled_holdings')

path=em.path_static_list.replace('static_list.csv','')+'process_data\\'
# load from multiple dumps
files=um.iterate_csv(path+'dump_rebuilt\\',iterate_others=[True, '.feather'])
collector=[]
for file in files:
    rebuilt_i=feather.read_dataframe(path+'dump_rebuilt\\'+file+'.feather')
    collector.append(rebuilt_i)
rebuilt_raw=pd.concat(collector)



# join with unadjusted price data
px_unadj=feather.read_dataframe(path+'px_unadj.feather').set_index('date').resample('D').last().fillna(method='ffill')
px_unadj.columns.name='ticker'
clean_index=rebuilt_raw.groupby('date').last().index
px_unadj_to_join=px_unadj.reindex(clean_index).fillna(method='ffill').stack()
px_unadj_to_join=px_unadj_to_join.rename('px_unadj').reset_index()
px_unadj_to_join['date_ticker']=px_unadj_to_join.apply(lambda x: '%s-%s' % (x['date'].strftime('%Y%m%d'),x['ticker']),axis=1)


# use data table to speed things up
rebuilt_dt=dt.Frame(rebuilt_raw)
px_unadj_to_join_dt=dt.Frame(px_unadj_to_join[['px_unadj','date_ticker']])

px_unadj_to_join_dt.key='date_ticker'
rebuilt_dt=rebuilt_dt[:,:,join(px_unadj_to_join_dt)]
rebuilt_dt[:,'value']=rebuilt_dt[:,f['shares']*f['px_unadj']]
fund_value_sum=rebuilt_dt[:,dt.sum(f['value']),by('date_ticker_fund')]
fund_value_sum.key='date_ticker_fund'
fund_value_sum.names={'value':'value_sum_by_fund'}
rebuilt_dt=rebuilt_dt[:,:,join(fund_value_sum)]
rebuilt_dt[:,'wgt']=rebuilt_dt[:,f['value']/f['value_sum_by_fund']]
rebuilt=rebuilt_dt.to_pandas()
rebuilt['date']=pd.to_datetime(rebuilt['date']) # bit slow but tolerable

# dump the rebuilt results for centrality stats calculation
feather.write_dataframe(rebuilt.drop(['date_ticker','date_ticker_fund'],1),path+'rebuilt_results.feather')


# dump the holdings of all mutual funds for further calculation
mf_hlds_shares_mn=rebuilt_raw.groupby(['date','ticker'])['shares'].sum().unstack()/1000000
shout_unadj=feather.read_dataframe(path+'shout_sec_unadj.feather').set_index('date').resample('D').last().fillna(method='ffill')
shout_unadj=shout_unadj.reindex(mf_hlds_shares_mn.index).fillna(method='ffill')

shout_to_join=shout_unadj.stack().rename('shout')
shout_to_join.index.names=['date','ticker']
res=mf_hlds_shares_mn.stack().rename('mf_holdings_shares').to_frame().join(shout_to_join)
res['stake']=res['mf_holdings_shares']/res['shout']


feather.write_dataframe(res.reset_index(),path+'mf_stake.feather')


print ('finished A_d_process_filled_holdings')




