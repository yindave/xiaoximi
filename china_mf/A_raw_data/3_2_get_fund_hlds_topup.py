# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:43:35 2021

@author: hyin1
"""

from webscraping.eastmoney import update_all_fund_info,path_static_list
import utilities.misc as um
import numpy as np
import pandas as pd
import utilities.constants as uc

from multiprocessing import Process

'''
delete all the individual dumps for top-up from fund_details folder before running this script
Use must contain to make sure we get the latest hlds data (3rd element is for exception names (usually the terminated fund))


set job_N to be some lower number e.g. 3 (or even 1!)
Not sure why but if we set the job_N too high we tend to skip some fund ticker (not sure how to fix that in the script)
'''

data_type='hlds_TopUp'
job_N=3

must_contain=[False,pd.datetime(2021,6,30),[]]
              # the terminated fund, keep these names for the future run
              #[9158,7115,519667,100053,291007,10102,5376,3000,7883,70031,8088,710301,8332,9415,10218,2823]
              #]

finished=[]

if __name__ == "__main__":
    # parallel run
    p_collector=[]
    for job_i in np.arange(0,job_N,1):
        if job_i not in finished:
            p=Process(target=update_all_fund_info,args=(data_type,job_N,job_i,must_contain))
            p_collector.append(p)
            p.start()
    for p in p_collector:
        p.join()


