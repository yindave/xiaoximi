# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:57:38 2021

@author: davehanzhang
"""

import utilities.misc as um
import pandas as pd
from connect.connect import STOCK_CONNECT
import numpy as np
from datetime import datetime

import time
import random



direction='sb'

scraper=STOCK_CONNECT(direction=direction).get_scraper()

def run_update_connect(start,end):
	
    dates=pd.date_range(start,end,freq='B')
    for date in dates:
        scraper(date,mode='shanghai')
        scraper(date,mode='shenzhen')

        random_number = random.randint(1, 10)
        if random_number<=3:
            print ('sleep for %s sec' % (random_number))
            time.sleep(random_number)
            
        
    um.quick_auto_notice('Connect SB ccass download finished')






# start=um.yesterday_date()-10*pd.tseries.offsets.BDay()
# end=um.yesterday_date()

start=datetime(2023,9,14)
end=um.yesterday_date()

run_update_connect(start,end)

# retries=1
# for retry in np.arange(0,retries):
#     try:
#         run_update_connect(start,end)
#         break
#     except:
#         print ('Download error.')
#         um.trigger_internet() #drigger internet may not solve the problem
#     #failure after many retires
#     if retry+1==retries:
#         msg='Connect update failed after %s retries' % (retries)
#         um.quick_auto_notice(msg)









