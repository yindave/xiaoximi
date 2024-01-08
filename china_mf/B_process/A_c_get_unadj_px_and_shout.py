# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:44:48 2021

@author: hyin1
"""

import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import argparse #optparse deprecated
import numpy as np
import pdb
from fql.fql import Factset_Query
from fql.util import fql_date

import os.path

'''
Run this daily to have the most updated unadjusted price
'''

print ('working on A_c_get_unadj_px_and_shout')

fq=Factset_Query()
start=pd.datetime(2005,12,31)
end=um.yesterday_date()

path=em.path_static_list.replace('static_list.csv','')
hlds=feather.read_dataframe(path+'TidyUp_hlds.feather')


tickers_a=hlds[(hlds['location'].isin(['A']))].groupby('ticker').last().index.tolist()


# get the effective tickers
check=fq.get_snap(tickers_a,['name'])
tickers_a_raw=check[check['name']!='@NA'].index.tolist()

dump_path=path+'process_data\\'


# market based metrics
fields=['px_last','shout_sec','turnover','ret_daily']
field_stats_dict={'px_last':[False,'px_unadj','CNY'],
                  'shout_sec':[False,'shout_sec_unadj','CNY'],
                  'turnover':[False,'turnover','USD'],
                  'ret_daily':[False,'ret_daily','CNY'],
                  }

for field in fields:
    adj=field_stats_dict[field][0]
    file_name=field_stats_dict[field][1]
    fx=field_stats_dict[field][2]
    # unadjusted price
    if not os.path.isfile(dump_path+'%s.feather' % (file_name)):
        print ('no existing dump found for %s' % (field))
        print ('getting %s' % (field))
        data=fq.get_ts(tickers_a,[field],start=fql_date(start),end=fql_date(end),adj=adj,fx=fx)[field]
        feather.write_dataframe(data.reset_index(),dump_path+'%s.feather' % (file_name))
    else:
        print ('find existing dump for %s, top-up updating' % (field))
        data_old=feather.read_dataframe(dump_path+'%s.feather' % (file_name)).set_index('date')
        tickers_old=data_old.columns.tolist()
        last_date=data_old.index[-1]
        tickers_a=list(set(tickers_a_raw+tickers_old))
        data_new=fq.get_ts(tickers_a,[field],start=fql_date(last_date),end=fql_date(end),adj=adj,fx=fx)[field]
        data=pd.concat([data_old,data_new],sort=True).reset_index().groupby('date').last()
        feather.write_dataframe(data.reset_index(),dump_path+'%s.feather' % (file_name))

# fundamental based metrics
fields=['pe','pb','roe']
for field in fields:
    file_name=field
    if not os.path.isfile(dump_path+'%s.feather' % (file_name)):
        print ('no existing dump found for %s' % (field))
        print ('getting %s' % (field))
        data=fq.get_ts_reported_fundamental(tickers_a,[field],start=fql_date(start),end=fql_date(end),auto_clean=False)[field]
        feather.write_dataframe(data.reset_index(),dump_path+'%s.feather' % (file_name))
    else:
        print ('find existing dump for %s, top-up updating' % (field))
        data_old=feather.read_dataframe(dump_path+'%s.feather' % (file_name)).set_index('date')
        tickers_old=data_old.columns.tolist()
        last_date=data_old.index[-1]
        tickers_a=list(set(tickers_a_raw+tickers_old))
        data_new=fq.get_ts_reported_fundamental(tickers_a,[field],start=fql_date(last_date),end=fql_date(end),auto_clean=False)[field]
        data=pd.concat([data_old,data_new],sort=True).reset_index().groupby('date').last()
        feather.write_dataframe(data.reset_index(),dump_path+'%s.feather' % (file_name))



print ('finished on A_c_get_unadj_px_and_shout')



