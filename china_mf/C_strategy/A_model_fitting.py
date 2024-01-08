# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:01:02 2021

@author: hyin1
"""

import webscraping.eastmoney as em
import utilities.misc as um
import pandas as pd
import utilities.constants as uc
import feather
import numpy as np
import pdb

from excess_return.excess_return import CARHART
from joblib import dump, load
import os.path
import utilities.mathematics as umath


'''
note that the alpha model only goes back to 2008/11/30
and for fair comparison we used 2-3yr data for fund rating model
and we have 1 yr minimum requirement for initial gam fitting
this means the begining date of the model should be arund 2012/06/30
'''

carhart=CARHART.Load_Model_Quick('MF')

bt_path=em.path_static_list.replace('static_list.csv','')+'strategy\\'
turnover_path=em.path_static_list.replace('static_list.csv','')+'process_data\\'

turnover=feather.read_dataframe(turnover_path+'turnover.feather').set_index('date')/1000 # in US$ mn
adv_3m_musd=turnover.applymap(lambda x: np.nan if x==0 else x).rolling(63,min_periods=5).mean()
adv_3m_musd=adv_3m_musd.resample('M').last()
adv_3m_musd=adv_3m_musd.stack()
adv_3m_musd.index.names=['date','ticker']

# get alphas
alpha_collector=[]
for window in [63]:
    alphas=carhart.get_excess_return_rolling_REVISED(window).swaplevel(1,0,1)['excess'].shift(-1*window).resample('M').last().stack().rename('alpha').to_frame()
    alphas=alphas.reset_index()
    alphas['alpha_rank']=alphas.groupby(['date'])['alpha'].rank(pct=True,method='min')
    alphas=alphas.set_index(['date','ticker'])
    alphas['window']=window
    alpha_collector.append(alphas)
alpha_all=pd.concat(alpha_collector)
feather.write_dataframe(alpha_all.reset_index(),bt_path+'alphas.feather')


# get mf stats
temp_path=em.path_static_list.replace('static_list.csv','')+'process_data\\'
mf_stake=feather.read_dataframe(temp_path+'mf_stake.feather')
mf_stake=mf_stake[mf_stake['ticker'].map(lambda x: True if '-CN' in x else False)]
mf_stake=mf_stake.set_index(['date','ticker'])['stake'].map(lambda x: np.nan if x<=0.0001 else x).map(lambda x: np.nan if np.isinf(x) else x).dropna().unstack()
#mf_stake_chg=mf_stake.fillna(0).diff(3).mask(mf_stake.fillna(0).applymap(lambda x: True if x==0 else False))
mf_stake_chg=mf_stake.diff(3)


mf_stats=pd.concat([mf_stake.stack().rename('mf_stake'),
                    mf_stake_chg.stack().rename('mf_stake_chg'),
                    ],axis=1).reset_index()

ranked=mf_stats.groupby('date')[['mf_stake','mf_stake_chg']].rank(pct=True,method='min')
ranked.columns=ranked.columns.map(lambda x: x+'_rank')
mf_stats=pd.concat([mf_stats,ranked],axis=1)
feather.write_dataframe(mf_stats,bt_path+'mf_stats.feather')


#get top20 crowding universe (KEY)
'''
This top20 filter is actually quite important, it basically defines the fitting universe
    - without the filter the relationship curve becomes bit messy
    - a filter with more constraint (e.g. top10 or 5pct etc) leads to worse performance
'''
path_network=em.path_static_list.replace('static_list.csv','')+'process_data\\dump_network\\'
files=um.iterate_csv(path_network,iterate_others=[True,'.feather'])
crowding_collector=[]
density_collector=[]
for file in files:
    if 'centrality' in file:
        crowding_collector.append(feather.read_dataframe(path_network+file+'.feather'))
crowding=pd.concat(crowding_collector)
crowding_to_show=crowding[crowding['id_type']=='ticker'].set_index(['date','id'])['centrality'].unstack().apply(lambda x: x.dropna().rank(pct=True),axis=1)
crowding_to_show=crowding_to_show.stack().rename('crowding_names')
crowding_to_show.index.names=['date','ticker']
feather.write_dataframe(crowding_to_show.reset_index(),bt_path+'crowding_mask.feather')


#get data to fit
alpha_all=feather.read_dataframe(bt_path+'alphas.feather').set_index(['date','ticker'])
mf_signals=feather.read_dataframe(bt_path+'mf_stats.feather').set_index(['date','ticker'])
crowding_mask=feather.read_dataframe(bt_path+'crowding_mask.feather').set_index(['date','ticker'])


data_to_fit=mf_signals.join(alpha_all).join(crowding_mask).join(adv_3m_musd.rename('adv'))[['alpha_rank','mf_stake_rank','mf_stake_chg_rank','crowding_names','adv']]
data_to_fit=data_to_fit[data_to_fit['crowding_names'].fillna(-100)!=-100]
data_to_fit=data_to_fit[data_to_fit['alpha_rank'].fillna(-100)!=-100]

data_to_fit['StakeLevel_OR_StakeChg']=data_to_fit[['mf_stake_rank','mf_stake_chg_rank']].mean(1)
data_to_fit=data_to_fit[data_to_fit['adv']>=1] # we fit on liquid names

data_to_fit=data_to_fit.fillna(0.5)

data_to_fit=data_to_fit.reset_index()
cols=['alpha_rank','StakeLevel_OR_StakeChg'] # re-rank the columns
for col in cols:
    data_to_fit[col]=data_to_fit.groupby('date')[col].rank(pct=True,method='min')

'''
The input mf signal level may change even on non-updating month
This is likely because share out changed
If we see signal level changed but underlying stake no change on updateing month
this could be because some other stocks are dropped from the crowding20 universe
'''

#fit and dump the model
min_points=12
y='alpha_rank'
xs=['StakeLevel_OR_StakeChg']
to_fit=data_to_fit[['date','ticker',y]+xs].copy()
all_dates=to_fit.groupby('date').last().loc[pd.datetime(2011,6,30):].index



for i,date in enumerate(all_dates):
    if i>=min_points-1:
        print ('fitting %s' % (date.strftime('%Y%m%d')))
        date_beg=all_dates[0]
        date_end=date
        if not os.path.isfile(bt_path+'models\\%s.joblib' % (date.strftime('%Y%m%d'))):
            to_fit_i=to_fit[(to_fit['date']>=date_beg) & (to_fit['date']<=date_end)].copy()
            ebm=umath.get_ebm_initialized()
            ebm.fit(to_fit_i[xs],to_fit_i[y],)
            dump(ebm,bt_path+'models\\%s.joblib' % (date.strftime('%Y%m%d')))

#load model and make predictions
all_dates=mf_signals.reset_index().groupby('date').last().index
model_to_use=pd.Series(index=all_dates)
for date in all_dates:
    if  os.path.isfile(bt_path+'models\\%s.joblib' % (date.strftime('%Y%m%d'))):
        ebm=load(bt_path+'models\\%s.joblib' % (date.strftime('%Y%m%d')))
        model_to_use[date]=ebm
model_to_use=model_to_use.shift(4) # to be conservative

# re-run data to fit without joining alpha
data_to_predict=mf_signals.join(crowding_mask)[['mf_stake_rank','mf_stake_chg_rank','crowding_names','mf_stake','mf_stake_chg']]
data_to_predict=data_to_predict[data_to_predict['crowding_names'].fillna(-100)!=-100]
data_to_predict['mf_stake_rank']=data_to_predict['mf_stake_rank'].fillna(0.5)
data_to_predict['mf_stake_chg_rank']=data_to_predict['mf_stake_chg_rank'].fillna(0.5)
data_to_predict['StakeLevel_OR_StakeChg']=data_to_predict[['mf_stake_rank','mf_stake_chg_rank']].mean(1)
data_to_predict=data_to_predict.reset_index()
cols=['StakeLevel_OR_StakeChg'] # re-rank the columns
for col in cols:
    data_to_predict[col]=data_to_predict.groupby('date')[col].rank(pct=True,method='min')

data_to_predict=data_to_predict.set_index(['date','ticker'])[['mf_stake','mf_stake_chg','StakeLevel_OR_StakeChg']].reset_index()

signal_collector=[]
for date in model_to_use.dropna().index:
    ebm=model_to_use[date]
    input_i=data_to_predict[data_to_predict['date']==date].copy()
    score_i=ebm.predict(input_i[ebm.feature_names])
    input_i['score']=score_i
    signal_collector.append(input_i)

signals=pd.concat(signal_collector)
feather.write_dataframe(signals,bt_path+'signals.feather')


















