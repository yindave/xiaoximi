
from webscraping.ccass import download_ccass_hkex
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process
from datetime import datetime


path=uc.root_path_data+'\\connect\ccass_all\\'
tickers=pd.read_excel(path+'full_eqs_hk_static.xlsx').set_index('ticker').index.tolist()

start_date=datetime(2023,5,5)#um.yesterday_date()-pd.tseries.offsets.DateOffset(years=1)
end_date=um.yesterday_date()
dates=pd.date_range(start_date,end_date,freq='W-Fri')


parallel_n=20
loop_n=int(len(tickers)/parallel_n)
jobs_to_loop=[tickers[i:i+parallel_n] for i in range (0, len (tickers), parallel_n)]



if __name__ == "__main__":
    
    p_collector=[]
    for i,date in enumerate(dates):
        
        print ('working on %s (%s/%s)' % (date,i,len(dates)))
        for j,jobs in enumerate(jobs_to_loop):
            print ('jobs %s/%s' % (j,len(jobs_to_loop)))
            
            for ticker in jobs:
                p=Process(target=download_ccass_hkex,args=(ticker,date,True,True,))
                p_collector.append(p)
                p.start()
            for p in p_collector:
                p.join()










