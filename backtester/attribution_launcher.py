# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:20:44 2021

@author: hyin1
"""



import pandas as pd
import numpy as np
import utilities.misc as um
import utilities.display as ud
import utilities.constants as uc
import utilities.mathematics as umath
import pdb
import feather

from blp.bdx import bdh,bdp,bds
from fql.fql import Factset_Query
from fql.util import bbg_to_fs, fs_to_bbg,fql_date
from blp.util import get_bbg_usual_col, group_marcap

from attribution import ATTRIBUTION
import argparse


# this remains unchanged and we do the "run" in local drive to speed things up (loading)
master_path="C:\\Users\\hyin1\\temp_data\\attribution\\"
dump_sedol_hist=True # set to True with just 1 date if run for the 1st time


parser = argparse.ArgumentParser()
parser.add_argument('--start',dest='start',type=str,default='aaa')
parser.add_argument('--end',dest='end',type=str,default='aaa')
parser.add_argument('--method',dest='method',type=str,default='custom')

parser.add_argument('--name',dest='name',type=str,default='aaa')
parser.add_argument('--region',dest='region',type=str,default='aaa')
parser.add_argument('--model',dest='model',type=str,default='aaa')


args = parser.parse_args()

name=args.name
region=args.region
model=args.model

start=pd.to_datetime(args.start,format='%m/%d/%Y')
end=pd.to_datetime(args.end,format='%m/%d/%Y')
method=args.method

## manual
#name='sb_l_vs_BalanceRisk'
#region='AXCN4'
#model='AXCN4-MH'
#start=pd.datetime(2021,3,18)
#end=pd.datetime(2021,5,31)
#method='custom'


attr=ATTRIBUTION(name,region,model,MASTER_PATH=master_path)
attr.load(dump_sedol_hist=dump_sedol_hist)
attr.run(start,end,method=method)

















