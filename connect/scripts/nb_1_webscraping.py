# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:00:03 2021

@author: davehanzhang
"""

import utilities.misc as um
import pandas as pd
from connect.connect import STOCK_CONNECT
import numpy as np
from datetime import datetime


direction='nb'

scraper=STOCK_CONNECT(direction=direction).get_scraper()
get_nb_universe=STOCK_CONNECT(direction=direction).get_nb_universe()
update_bbg_ccass_id_map=STOCK_CONNECT(direction=direction).update_bbg_ccass_id_map()

def run_update_connect(start,end):
	
    dates=pd.date_range(start,end,freq='B')
    for date in dates:
        scraper(date,mode='shanghai')
        scraper(date,mode='shenzhen')
    get_nb_universe()
    update_bbg_ccass_id_map()
    um.quick_auto_notice('Connect NB ccass download finished')




start=um.yesterday_date()-10*pd.tseries.offsets.BDay()
end=um.yesterday_date()

retries=5
for retry in np.arange(0,retries):
    try:
        run_update_connect(start,end)
        break
    except:
        print ('Download error.')
        um.trigger_internet() #drigger internet may not solve the problem
    #failure after many retires
    if retry+1==retries:
        msg='Connect update failed after %s retries' % (retries)
        um.quick_auto_notice(msg)







