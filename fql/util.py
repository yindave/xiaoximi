# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:29:00 2019

@author: hyin1
"""

import utilities.constants as uc
import pandas as pd
import numpy as np
import utilities.misc as um

euro_match={'SE':'CH','GY':'DE','NA':'NL',
            'LN':'GB','FP':'FR','DC':'DK',
            'SQ':'ES','IM':'IT','BB':'BE',
            'SS':'SE','FH':'FI',
            'ID':'IE','NO':'NO','PL':'PT',
            'AV':'AT'}

euro_match_bbg_to_fs={}
for k,v in euro_match.items():
    euro_match_bbg_to_fs[' %s Equity' % (k)]='-%s' % (v)

euro_match_fs_to_bbg={}
for k,v in euro_match.items():
    euro_match_fs_to_bbg['-%s' % (v)]=' %s Equity' % (k)

def bbg_to_fs(tickers):
    '''
    This should work for China A, HK, JP
    '''
    new_tickers=[]
    US_exs=['UN','UW','UQ','UR', 'UA', 'UF','US']
    for t in tickers:
        t=t.replace(' CH Equity','-CN')
        t=t.replace(' HK Equity','-HK')
        t=t.replace(' JP Equity','-JP')
        t=t.replace(' JT Equity','-JP')
        t=t.replace(' KS Equity','-KR')
        t=t.replace(' AT Equity','-AU')
        t=t.replace(' AU Equity','-AU')
        # for US
        for US_ex in US_exs:
            if US_ex in t:
                t=t.replace(' %s Equity' % (US_ex),'-US')
                t=t.replace('/','.')
        # for europe
        for k,v in euro_match_bbg_to_fs.items():
            t=t.replace(k,v).replace('/','')
        new_tickers.append(t)

    return new_tickers


def fs_to_bbg(tickers):
    '''
    This should work for China A, HK, JP
    '''
    new_tickers=[]
    for t in tickers:
        t=t.replace('-CN',' CH Equity')
        t=t.replace('-HK',' HK Equity')
        t=t.replace('-JP',' JP Equity')
        t=t.replace('-KR',' KS Equity')
        t=t.replace('-AU',' AU Equity')
        t=t.replace('-US',' US Equity').replace('.','/')
        # for europe
        for k,v in euro_match_fs_to_bbg.items():
            t=t.replace(k,v).replace('/','')
        new_tickers.append(t)

    return new_tickers


def fql_date(dt):
    return dt.strftime('%m/%d/%Y')


if __name__=='__main__':
    print ('ok')
    df=pd.read_csv("Z:\\dave\\data\\Misc\\ray.csv")

    df['ticker_fs']=bbg_to_fs(df['Ticker'])

























