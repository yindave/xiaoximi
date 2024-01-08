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
from connect.scripts import nb_1_webscraping

#---- update SB database, with re-try for FS disconnection (in the morning only)
sc=STOCK_CONNECT(direction='nb')
res_1=um.retry_func(sc.update_connect_data,retries=5,error_msg_email='NB update connect data error')
res_2=um.retry_func(sc.update_mkt_data,retries=5,error_msg_email='NB update market data error')
sc.update_db()










