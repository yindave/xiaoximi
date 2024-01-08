# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:24:43 2021

@author: davehanzhang
"""

import utilities.misc as um
import subprocess
from connect.connect import STOCK_CONNECT
from connect.scripts.sb_2_model_update import update_model
import utilities.constants as uc
python_path=uc.get_python_path()


#---- update ccass dump (in the morning only)
from connect.scripts import sb_1_webscraping

#---- update SB database, with re-try for FS disconnection (in the morning only)
sc=STOCK_CONNECT(direction='sb')
res_1=um.retry_func(sc.update_connect_data,retries=5,error_msg_email='SB update connect data error')
res_2=um.retry_func(sc.update_mkt_data,retries=5,error_msg_email='SB update market data error')
sc.update_db()

# ---- update_model 
# in both morning and afternoon (need updated mkt data)
# also since we shift model by 2+window so we use yesterday's model (fit in the morning) for today's after close run
res_3=um.retry_func(update_model,retries=5,error_msg_email='SB update model error')


print ('start backtesting')
subprocess.call(['python.exe', python_path+'connect\\scripts\\sb_3_strategy_backtest_lite.py'])












