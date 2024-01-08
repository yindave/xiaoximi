# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:23:19 2021

@author: hyin1
"""

from multiprocessing import Process
import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import numpy as np
import pdb


'''
Not part of the strategy but can run daily
'''

path=em.path_static_list.replace('static_list.csv','')
dump_path=path+'\\process_data\\dump_manager\\'

nav_quick=feather.read_dataframe(path+'\\TidyUp_nav_quick.feather').set_index('date')
nav_quick.columns=nav_quick.columns.map(int)
daily_ret=nav_quick.pct_change()
managers=feather.read_dataframe(path+'TidyUp_%s.feather' % ('managers'))

managers=managers[['manager','ticker','from','to','manager_name']]

mf_stake=feather.read_dataframe(path+'process_data\\mf_stake.feather')
all_managers=managers.groupby('manager').last().index

def run(batch, batch_total):
    print ('working on batch %s/%s' % (batch,batch_total))
    all_dates=mf_stake.groupby('date').last().iloc[batch:].iloc[::batch_total].index
    min_periods=126
    collector=[]
    for date_i in all_dates:
        print ('working on %s' % (date_i))
        since_date_i=date_i-pd.tseries.offsets.DateOffset(years=1)
        for manager_i in all_managers:
            #print ('working on %s %s' % (date_i,manager_i))
            record_i=pd.Series(index=['date','manager','exp','return','sharpe',])
            # currently what fund(s) are managed by the manager
            manager_i_record_current=managers[(managers['manager']==manager_i)
                                     & (date_i>managers['from'])
                                     & (date_i<=managers['to'])
                                     ].copy()
            # the full history of manager (till date_i)
            manager_i_record_full=managers[(managers['manager']==manager_i)
                                    & (date_i>managers['from'])
                                          ].copy()
            # the last 1yr history of manager
            manager_i_record_trailing=managers[(managers['manager']==manager_i)
                                    & (date_i>managers['from'])
                                    & (since_date_i<managers['to'])
                                          ].copy()
            if len(manager_i_record_current)!=0:
                career_start=manager_i_record_full['from'].min()
                exp=(date_i-career_start).days/365
                for idx_i in manager_i_record_trailing.index:
                    fund_i=manager_i_record_trailing.loc[idx_i]['ticker']
                    if fund_i in nav_quick.columns:
                        from_i=max(since_date_i,manager_i_record_trailing.loc[idx_i]['from'])
                        to_i=date_i
                        nav_i=nav_quick[fund_i].loc[from_i:to_i]
                        ret_i=daily_ret[fund_i].loc[from_i:to_i]
                        if len(nav_i)>=min_periods and ret_i.std()!=0:
                            return_i=nav_i.iloc[-1]/nav_i.iloc[0]-1
                            sharpe_i=ret_i.mean()/ret_i.std()*np.sqrt(252)
                            manager_i_record_trailing.at[idx_i,'return']=return_i
                            manager_i_record_trailing.at[idx_i,'sharpe']=sharpe_i
                        else:
                            manager_i_record_trailing.at[idx_i,'return']=np.nan
                            manager_i_record_trailing.at[idx_i,'sharpe']=np.nan
                    else: # this happens when the funds managed are money mkt funds
                        manager_i_record_trailing.at[idx_i,'return']=np.nan
                        manager_i_record_trailing.at[idx_i,'sharpe']=np.nan
                record_i['date']=date_i
                record_i['manager']=manager_i
                record_i['exp']=exp
                record_i['return']=manager_i_record_trailing['return'].mean()
                record_i['sharpe']=manager_i_record_trailing['sharpe'].mean()
                record_i['fund_count']=len(set(manager_i_record_trailing.dropna()['ticker'].values.tolist()))
                collector.append(record_i.to_frame().T)
    res=pd.concat(collector)
    res['date']=res['date'].map(pd.to_datetime)
    res['exp']=res['exp'].map(pd.to_numeric)
    res['return']=res['return'].map(pd.to_numeric)
    res['sharpe']=res['sharpe'].map(pd.to_numeric)
    res['fund_count']=res['fund_count'].map(pd.to_numeric)
    feather.write_dataframe(res,dump_path+'%s.feather' % (batch))




if __name__ == "__main__":
    # parallel run
    job_N=20
    p_collector=[]
    for job_i in np.arange(0,job_N,1):
        p=Process(target=run,args=(job_i,job_N,))
        p_collector.append(p)
        p.start()
    for p in p_collector:
        p.join()









