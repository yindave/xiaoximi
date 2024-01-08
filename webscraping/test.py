import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request, urllib.error, urllib.parse
import pdb
import utilities.misc as um
import requests
import utilities.constants as uc
import json
import feather

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request, urllib.error, urllib.parse
import pdb
import utilities.misc as um
import requests
import utilities.constants as uc
import json
import feather




use_proxy=uc.use_proxy
proxy_to_use=uc.proxy_to_use
dump_path={'sb':uc.root_path_data+"connect\\southbound",
            'nb':uc.root_path_data+"connect\\northbound",
            # 'circulation':"Z:\\dave\\data\\connect\\circulation",
            }


def download_SB(date,mode='shanghai'):
    if mode=='shanghai':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=1323&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
    elif mode=='shenzhen':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=1456&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
    elif mode=='h_circulation':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=1296&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
        
        
    elif mode=='ib':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=243&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
        
    else:
        print ('Wrong mode type. Currently only shanghai and shenzhen are valid inputs')
        return None
    if use_proxy:
        proxy = urllib.request.ProxyHandler(proxy_to_use)# 'nj02ga03cmp01.us.db.com'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
    #pdb.set_trace()
    request = urllib.request.urlopen(download)
    html_doc=request.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    table = soup.findAll('table', {'class': 'optable'})
    #prepare heading
    heading_row=[row.text for row in table[0].find_all('th') if row.text]
    #get table columns
    data={}
    data_holdingNrow=[row.text for row in table[0].find_all('td',{'class':'colHide1'}) if row.text] #contains both the row counter and holding number
    data['Row']=data_holdingNrow[::2]
    data['Holding']=data_holdingNrow[1::2]
    data['Date']=[row.text for row in table[0].find_all('td',{'class':'colHide2'}) if row.text]
    data['Value']=[row.text for row in table[0].find_all('td',{'class':'colHide3'}) if row.text]
    data['Issue']=[row.text for row in table[0].find_all('td',{'class':'left'}) if row.text]
    data_lastcodeNholding=[row.text for row in table[0].find_all('td',{'class':''}) if row.text and row.text!='*'] #* is a mark for suspended stock. We will still have a static holding number though
    data['Lastcode']=data_lastcodeNholding[::2]
    data['Stake%']=data_lastcodeNholding[1::2]
    output=pd.DataFrame(data)[heading_row]
    output['Row']=output['Row'].map(lambda x: int(x))
    output['BBG Ticker']=output['Lastcode'].map(lambda x: str(int(x))+' HK Equity') #new column for BBG ticker
    output['Short Name']=output['Issue'].map(lambda x: str(x)) #new column for short name ('Issue' doesn't sound like short name)
    output['Holding']=output['Holding'].map(lambda x: float(x.replace(',','')))
    output['Value']=output['Value'].map(lambda x: float(x.replace(',','')))
    output['Stake%']=output['Stake%'].map(lambda x: float(x))
    output['Last Holding Change Date']=output['Date'].map(lambda x: pd.to_datetime(x)) #new column for last holding change date
    #rearrange column order
    new_columns=['Row','BBG Ticker','Short Name','Holding','Value','Stake%','Last Holding Change Date']
    output=output[new_columns]
    output=output.set_index('Row')

    if len(output)!=0:
        if mode=='shanghai':
            output.to_csv("%s\\sh\\%s.csv" % (dump_path['sb'],date.strftime('%Y-%m-%d')))
        elif mode=='shenzhen':
            output.to_csv("%s\\sz\\%s.csv" % (dump_path['sb'],date.strftime('%Y-%m-%d')))
        elif mode=='h_circulation':
            output.to_csv("%s\\%s.csv" % (dump_path['circulation'],date.strftime('%Y-%m-%d')))
        else:
            print ('Wrong mode type. Currently only shanghai and shenzhen are valid inputs')
            return None
        print ('%s connect -- Finish downloading Webb CCASS %s' % (mode,date.strftime('%Y-%m-%d')))
    else:
        print ('No data on %s' % (date.strftime('%Y-%m-%d')))




download_SB(um.yesterday_date(),'ib')