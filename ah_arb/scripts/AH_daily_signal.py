# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:54:16 2021

@author: hyin1
"""

from connect.connect import STOCK_CONNECT; import pandas as pd; import numpy as np; import plotly.express as px; import plotly.graph_objects as go; from plotly.subplots import make_subplots; import utilities.misc as um; import utilities.display as ud; import utilities.constants as uc;from excess_return.excess_return import CARHART;import feather; from fql.util import bbg_to_fs,fs_to_bbg;from blp.util import get_bbg_usual_col,get_bbg_nice_compo_hist;import utilities.mathematics as umath
import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
import matplotlib.pyplot as plt
import feather
from connect.connect import STOCK_CONNECT
from blp.bdx import bdh,bdp, bds
from fql.fql import Factset_Query

from fql.util import bbg_to_fs,fs_to_bbg,fql_date
from blp.util import get_bbg_nice_compo_hist_no_limit, get_bbg_nice_compo_hist, get_bbg_usual_col,group_marcap, get_region, get_board

import webscraping.eastmoney as em
import itertools

import utilities.mathematics as umath
import plotly.express as px
import plotly.graph_objects as go

from backtester.backtester import BACKTESTER

from ah_arb.ah import AH


ah=AH();ah.load_db()


buy_ha=ah.get_band(63,band_type=['bollinger_adj_simple','ha_avg_simple'],)
sell_ha=ah.get_band(63,band_type=['bollinger', 'na'])


dump_path="Z:\\dave\\report\\Authoring\\AH\\auto_email_ah\\"

band_dict={'buy':buy_ha,'sell':sell_ha}

dump_path="Z:\\dave\\report\\Authoring\\AH\\auto_email_ah\\"

directions=['buy','sell']
direction_nice={'buy':'buy HA ratio (premium contraction trade)',
                'sell':'sell HA ratio (premium expansion trade)'}
collector_history=[]
collector_last=[]

for direction in directions:
    band_i=band_dict[direction][['ratio','mean','std']].copy()
    band_i['upper']=band_i['mean']+2.5*band_i['std']
    band_i['lower']=band_i['mean']-2.5*band_i['std']
    band_i=band_i.drop('std',1)
    #band_i=band_i.unstack().loc[um.today_date()-pd.tseries.offsets.DateOffset(years=10):].resample('W-FRI').last().stack()
    band_i=band_i.unstack().iloc[-252*1:].stack() # we only show for the last 2 years
    band_i=band_i.reset_index().set_index('ticker')
    band_i['ticker_a']=pd.Series(ah.get_ah_map(direction='h_to_a'))
    band_i['name_h']=ah.db['name'].iloc[-1]
    band_i.index.name='ticker_h'
    band_i=band_i.reset_index().set_index(['name_h','ticker_h','ticker_a','date']).reset_index()
    last_data=band_i.set_index(['date','ticker_h']).unstack().iloc[-1]
    band_i_last=last_data.unstack().T

    if direction=='buy':
        band_i_last['breached']=band_i_last.apply(lambda x: 'yes' if x['ratio']<x['lower'] else 'no',axis=1)
    else:
        band_i_last['breached']=band_i_last.apply(lambda x: 'yes' if x['ratio']>x['upper'] else 'no',axis=1)
    band_i_last['date']=last_data.name
    band_i_last['direction']=direction_nice[direction]
    band_i_last['direction']=direction_nice[direction]
    band_i['direction']=direction_nice[direction]
    band_i_last=band_i_last.reset_index().set_index(['direction','ticker_h','ticker_a','name_h']).reset_index()
    collector_last.append(band_i_last)
    collector_history.append(band_i)

summary_last=pd.concat(collector_last)
summary_history=pd.concat(collector_history)

### add sedol
all_tickers=fs_to_bbg(list(ah.get_ah_map().keys())+list(ah.get_ah_map().values()))
sedols=bdp(all_tickers,['ID_SEDOL1'])
sedols.index=bbg_to_fs(sedols.index)
sedols=sedols['ID_SEDOL1'].rename('sedol')
sedols.index.name='ticker'

summary_last=summary_last.set_index('ticker_h')
summary_last['sedol_h']=sedols
summary_last=summary_last.reset_index().set_index('ticker_a')
summary_last['sedol_a']=sedols
summary_last=summary_last.reset_index()


summary_history=summary_history.set_index('ticker_h')
summary_history['sedol_h']=sedols
summary_history=summary_history.reset_index().set_index('ticker_a')
summary_history['sedol_a']=sedols
summary_history=summary_history.reset_index()


summary_last=summary_last.set_index(['direction','ticker_h','ticker_a','sedol_h','sedol_a'])
summary_history=summary_history.set_index(['direction','ticker_h','ticker_a','sedol_h','sedol_a'])
summary_history=summary_history.sort_values(by=['date','name_h'],ascending=[False,True])
signal_highlight=summary_last[summary_last['breached']=='yes']





summary_last.dropna().to_csv(dump_path+'summary_last.csv')
summary_history.dropna().to_csv(dump_path+'summary_1yr_history.csv')
signal_highlight.dropna().to_csv(dump_path+'signal_highlight.csv')



### write the email

html=ud.HTML_Builder()



html.insert_body('Dear client,',bold=False)
html.insert_body('' ,bold=False)


html.insert_body('Please find attached the AH strategy signal.',bold=False)
html.insert_body('Buy HA ratio (premium contraction trade): buy H sell A.',bold=False)
html.insert_body('Sell HA ratio (premium expansion trade): sell H buy A.',bold=False)
html.insert_body('' ,bold=False)

if len(signal_highlight)!=0:
    html.insert_body('Summary of the latest signals',bold=False)
    html.insert_table(signal_highlight,'')
else:
    html.insert_body('Currently there is no signal',bold=False)
    html.insert_body('' ,bold=False)


html.insert_body('Details of the strategy/signal construction can be found here: https://plus.credit-suisse.com/s/V7qml54AF-ZsV3',bold=False)
html.insert_body('' ,bold=False)


html.insert_body('Regards,',bold=False)
html.insert_body('',bold=False)
html.insert_body('Credit Suisse Quantitative & Systematic Strategy',bold=False)
html.insert_body('',bold=False)


subject='Credit Suisse Quantitative and Systematic Strategy daily AH signal as of %s' % (um.today_date().strftime('%Y-%m-%d'))


dl=pd.read_excel(dump_path+'dl.xlsx',index_col=[0])['email'].tolist()

sendto=[
        'dave.yin@credit-suisse.com'
        ]
bcc=';'.join(dl)

attachment=[dump_path+'summary_last.csv',
            dump_path+'summary_1yr_history.csv',
            ]

um.send_mail_comobj( ';'.join(sendto),subject,html.body,attachment=attachment,Bcc=bcc)










