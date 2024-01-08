# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:00:45 2021

This is helpful for codec and csv parsing etc

@author: davehanzhang
"""



import pandas as pd
from bs4 import BeautifulSoup
import pdb
import requests
import csv
from io import StringIO
import re
import utilities.constants as uc

from datetime import datetime


dump_path=uc.root_path_data+"\\misc\\"

use_proxy=uc.use_proxy
proxy_to_use=uc.proxy_to_use



def today_string(strf='%Y-%m-%d'):
    return datetime.today().strftime(strf)
def today_date():
    return datetime.strptime(today_string(),'%Y-%m-%d')


### 1. get cbbc and dw csv
url_dict={'cbbc':"http://www.hkex.com.hk/eng/cbbc/search/cbbcFullList.csv",
          'dw':"http://www.hkex.com.hk/eng/dwrc/search/dwFullList.csv"}


for k,url_i in url_dict.items():
    r=requests.get(url_i, proxies=proxy_to_use if use_proxy else None)
    path_i=dump_path+'%s_%s.csv' % (k,today_date().strftime('%Y%m%d'))
    open(path_i, 'wb').write(r.content)
    
    # tidy up the csv (in byte)
    
    import chardet    
    result = chardet.detect(r.content)
    charenc = result['encoding']
    #pdb.set_trace()
    print ('encoding estimated by chardet is %s' % (charenc))
    
    scsv=r.content.decode(charenc) # can use chardet to determine encoding
    f = StringIO(scsv)
    reader = csv.reader(f)
    collector=[]
    for i,row in enumerate(reader):
        collector.append(pd.Series(''.join(row).split('\t')))
    res=pd.concat(collector,axis=1).T.iloc[1:-2 if k=='cbbc' else -3]
    columns=res.iloc[0]
    data=res.iloc[1:]
    res_nice=pd.DataFrame(data=data.values,columns=columns)
    res_nice.columns=res_nice.columns.fillna('na')
    res_nice=res_nice.drop('na',1)
    
    res_nice.to_csv(path_i)
    
    

### 2. get the table
#url="https://www.hkex.com.hk/eng/prod/drprod/hkifo/RTlist_stockoptions.htm"
url="https://www.hkex.com.hk/products/listed-derivatives/single-stock/stock-options?sc_lang=en"

session = requests.Session()
response = session.post(url,  proxies=proxy_to_use if use_proxy else None)
soup = BeautifulSoup(response.content)
session.close()



# first table
table = soup.findAll('table', {'class': 'table migrate'})
table_1=table[0].find('tbody')
contents=table_1.findAll('tr')
collector=[]
for i,content_i in enumerate(contents):

    collector_row=pd.Series(index=['index','ticker','name','hkats_code','contract_size_shares','number_of_board_lot','tier_no','position_limit','approved_by_sfc_tw'],
                            dtype=object)
    content_i=content_i.findAll('td')

    collector_row['index']=int(content_i[0].text)
    collector_row['ticker']=int(content_i[1].text)
    collector_row['name']=content_i[2].text.replace('/n','').replace('\n','').replace(' ','').replace('\xa0','').replace("'A'",'')
    collector_row['hkats_code']=content_i[3].text.replace('/n','').replace('\n','').replace(' ','').replace('\xa0','').replace("'A'",'')
    collector_row['contract_size_shares']=int(content_i[4].text.replace(',','').replace('/n','').replace('\n','').replace(' ','').replace('\xa0','').replace("'A'",''))
    collector_row['number_of_board_lot']=int(content_i[5].text)
    collector_row['tier_no']=int(content_i[6].text)
    collector_row['position_limit']=int(content_i[7].text.replace(',',''))
    collector_row['approved_by_sfc_tw']='yes' if "✓" in content_i[8].text else ''

    collector.append(pd.Series(collector_row))
res=pd.concat(collector,axis=1).T
    
res['contract_size_shares']=pd.to_numeric(res['contract_size_shares'])
res['position_limit']=pd.to_numeric(res['position_limit'])

res.to_csv(dump_path+'table_1_%s.csv' % (today_date().strftime('%Y%m%d')))


# second table
table = soup.findAll('table', {'class': 'table migrate;'}) # the ; matters!!
table_1=table[0].find('tbody')
contents=table_1.findAll('tr')
collector=[]
for i,content_i in enumerate(contents):
    collector_row=pd.Series(index=['index','ticker','name','hkats_code','contract_size_shares','number_of_board_lot','tier_no','position_limit','approved_by_sfc_tw'],
                             dtype=object)
    content_i=content_i.findAll('td')

    collector_row['index']=int(content_i[0].text)
    collector_row['ticker']=int(content_i[1].text)
    collector_row['name']=content_i[2].text.replace('/n','').replace('\n','').replace(' ','').replace('\xa0','').replace("'A'",'')
    collector_row['hkats_code']=content_i[3].text.replace('/n','').replace('\n','').replace(' ','').replace('\xa0','').replace("'A'",'')
    collector_row['contract_size_shares']=int(content_i[4].text.replace(',','').replace('/n','').replace('\n','').replace(' ','').replace('\xa0','').replace("'A'",''))
    collector_row['number_of_board_lot']=1 # by definition
    collector_row['tier_no']=int(content_i[5].text)
    collector_row['position_limit']=int(content_i[6].text.replace(',',''))
    collector_row['approved_by_sfc_tw']='yes' if "✓" in content_i[7].text else ''

    collector.append(pd.Series(collector_row))
res_2=pd.concat(collector,axis=1).T
res_2.to_csv(dump_path+'table_2_%s.csv' % (today_date().strftime('%Y%m%d')))


# 3rd table
table = soup.findAll('table', {'class': 'table migrate'})
table_1=table[5].find('tbody')
contents=table_1.findAll('tr')
collector=[]

headers=table[5].find('thead').findAll('td')
header_collector=[]
for header_i in headers:
    header_collector.append(header_i.text)    
header_collector=header_collector[-11:]

for i,content_i in enumerate(contents):
    collector_row=pd.Series(index=header_collector, dtype=object)
    content_i=content_i.findAll('td')
    for j,col in enumerate(header_collector):
        collector_row[col]=content_i[j+1].text
    
    collector.append(pd.Series(collector_row))
    
res_3=pd.concat(collector,axis=1).T
res_3=res_3.applymap(lambda x: x.replace('√','tick'))
res_2=res_2.replace()
res_3.to_csv(dump_path+'table_3_%s.csv' % (today_date().strftime('%Y%m%d')))











