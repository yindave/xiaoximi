# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:23:31 2016

@author: yindave
"""
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

from datetime import datetime
import os

use_proxy=uc.use_proxy
proxy_to_use=uc.proxy_to_use
dump_path={'sb':uc.root_path_data+"connect\\southbound",
            'nb':uc.root_path_data+"connect\\northbound",
            # 'circulation':"Z:\\dave\\data\\connect\\circulation",
            }


def download_ccass_hkex(ticker,date,auto_dump=False,check_existing_dump=False):
    
    dump_path=uc.root_path_data+'connect\\ccass_all\\data_raw\\%s_%s.feather' % (ticker,date.strftime('%Y%m%d'))
    if check_existing_dump:
        if os.path.isfile(dump_path):
            print ('%s on %s dumped already' % (ticker,date))
            return False
    
    
    url = 'https://www3.hkexnews.hk/sdw/search/searchsdw.aspx'
    https_proxy = proxy_to_use
    headers = {'Referer': url,}
    session = requests.Session()
    try:
        response = session.get(url,proxies=https_proxy if use_proxy else None)# headers=headers)
    except:
        print ('Something wrong with verification (HKEx problem), set verify to false and retry')
        response = session.get(url,proxies=https_proxy if use_proxy else None,verify = False)# headers=headers)
    soup = BeautifulSoup(response.content,features='lxml')

    try:
        viewstate = soup.select("#__VIEWSTATE")[0]['value']
    except IndexError:
        pdb.set_trace()
        print ('someting odd happened')


    # eventvalidation = soup.select("#__EVENTVALIDATION")[0]['value']
    viewstate_generator = soup.select("#__VIEWSTATEGENERATOR")[0]['value']

    post_data = {
        '__EVENTTARGET': 'btnSearch',
        '__EVENTARGUMENT':'',
        '__VIEWSTATE': viewstate,
        '__VIEWSTATEGENERATOR':viewstate_generator,
        'alertMsg': '',
        'today': um.today_date().strftime('%Y%m%d'),
        'txtShareholdingDate':date.strftime('%Y/%m/%d'),
        'txtStockCode': ticker.replace(' HK Equity','').zfill(5),    

    }
    response = session.post(url, data=post_data, headers=headers,proxies=https_proxy if use_proxy else None)
    soup = BeautifulSoup(response.content,features='lxml')
    session.close()


    #table=soup.findAll('td',{'valign': 'top'})
    table=soup.findAll('div',{'class': 'mobile-list-body'})
    line_num=len(table)/5
    if line_num==0:
        print ('no data found for %s on %s' % (ticker,date))
        return False

    data_collector=[]
    for my_j in np.arange(1,line_num+1):
        temp_data=pd.DataFrame([table[i+int((my_j-1)*5)].text for i in np.arange(0,5,1)]).T
        data_collector.append(temp_data)
    data=pd.concat(data_collector,axis=0)

    #process the data
    if len(data)!=0:
        data.columns=['ccass_id','ccass_name','address','shares','stake']
        data=data.drop(['address'],axis=1)
        data['shares']=data['shares'].map(lambda x: float(x.replace(',','')))
        data['stake']=data['stake'].map(lambda x: '0.00%' if x=='' else x)
        data['stake']=data['stake'].map(lambda x: float(x.replace('%',''))/100)
        data['ticker']=ticker
        data['date']=date
    else:
        print ('no data found for %s on %s' % (ticker,date))
        return False
    
    print ('finish downloading %s on %s' % (ticker,date))
    
    if auto_dump:
        # dump_path=uc.root_path_data+'connect\\ccass_all\\data_raw\\%s_%s.feather' % (ticker,date.strftime('%Y%m%d'))
        feather.write_dataframe(data,dump_path)
    
    return data



def download_ccass(date,ccass_id=''):
    
    path=uc.root_path_data+'connect\\ccass\\data_raw\\'

    ccass_id_to_webb_id={'B01590':'243'}
    
    if ccass_id not in ccass_id_to_webb_id.keys():
        print ('No mapping found between CCASS and Webb')
        return None
    else:
        download= 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=%s&d=%s&z=1' % (ccass_id_to_webb_id[ccass_id],date.strftime('%Y-%m-%d'))
        
        
    if use_proxy:
        proxy = urllib.request.ProxyHandler(proxy_to_use)# 'nj02ga03cmp01.us.db.com'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
    #()
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
    try:
        output['Stake%']=output['Stake%'].map(lambda x: float(x))
    except:
        output['Stake%']=np.nan
    output['Last Holding Change Date']=output['Date'].map(lambda x: pd.to_datetime(x)) #new column for last holding change date
    #rearrange column order
    new_columns=['Row','BBG Ticker','Short Name','Holding','Value','Stake%','Last Holding Change Date']
    output=output[new_columns]
    output=output.set_index('Row')

    output.columns=['ticker','name','holding','value','stake','date']
    output=output[['ticker','name','holding','stake','date']]
    output['stake']=output['stake']/100
    if len(output)!=0:
        feather.write_dataframe(output,path+'%s_%s.feather' % (ccass_id,date.strftime('%Y%m%d')))
        print ('Finish downloading %s on %s from Webb' % (ccass_id,date.strftime('%Y-%m-%d')))
    else:
        print ('No data on %s' % (date.strftime('%Y-%m-%d')))




def download_SB(date,mode='shanghai'):
    if mode=='shanghai':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=1323&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
    elif mode=='shenzhen':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=1456&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
    elif mode=='h_circulation':
        download = 'https://webb-site.com/ccass/cholder.asp?sort=valndn&part=1296&d=%s&z=1' % (date.strftime('%Y-%m-%d'))
    else:
        print ('Wrong mode type. Currently only shanghai and shenzhen are valid inputs')
        return None
    if use_proxy:
        proxy = urllib.request.ProxyHandler(proxy_to_use)# 'nj02ga03cmp01.us.db.com'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
    
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



def download_NB(date,mode='sh'):
    '''
    exchange can be sh or sz
    UPDATE on Oct 30 2018: HKEx changed the ccass website, it's not impacting the Webb Site but the NB scraping needs to change
    Just the date input format in the post form is changed, not a big deal
    html tags are changed slightly as well
    '''
    mode_dict={'shanghai':'sh','shenzhen':'sz'}	
    exchange=mode_dict[mode]
	
	
    dump_path_local={'sh':'%s\\sh\\' % (dump_path['nb']),
                     'sz':'%s\\sz\\' % (dump_path['nb']),}
    ### hkex
    url = 'https://www.hkexnews.hk/sdw/search/mutualmarket.aspx?t=%s' % (exchange)
    https_proxy = proxy_to_use
    headers = {
        #'Accept':'text/html, application/xhtml+xml, image/jxr, */*',
        #'Host': 'http://www.hkexnews.hk',
        #'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0;rv:11.0) like Gecko',
        #'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'https://www.hkexnews.hk/sdw/search/mutualmarket.aspx?t=%s' % (exchange),
                    #https://www.hkexnews.hk/sdw/search/mutualmarket.aspx?t=sh

        #'Accept-Encoding': 'gzip, deflate',
        #'Accept-Language': 'en-US, en; q=0.8, zh-Hans-CN; q=0.5, zh-Hans; q=0.3',
        #'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3'
    }
    session = requests.Session()
    try:
        response = session.get(url,proxies=https_proxy if use_proxy else None)# headers=headers)
    except:
        print ('Something wrong with verification (HKEx problem), set verify to false and retry')
        response = session.get(url,proxies=https_proxy if use_proxy else None,verify = False)# headers=headers)
    soup = BeautifulSoup(response.content)

    try:
        viewstate = soup.select("#__VIEWSTATE")[0]['value']
    except IndexError:
        pdb.set_trace()
        um.send_mail('NB webscraping failed for unknown reason, try to re-run','',['dave.yin@credit-suisse.com'])
    eventvalidation = soup.select("#__EVENTVALIDATION")[0]['value']
    viewstate_generator = soup.select("#__VIEWSTATEGENERATOR")[0]['value']
    post_data = {
        '__EVENTVALIDATION': eventvalidation,
        '__VIEWSTATE': viewstate,
        '__VIEWSTATEGENERATOR':viewstate_generator,
        'alertMsg': '',
        'btnSearch':'Search',
#        'btnSearch.x': '99999',
#        'btnSearch.y': '99999',
        'today': um.today_date().strftime('%Y%m%d'),
        'txtShareholdingDate':date.strftime('%Y/%m/%d'),
#        'ddlShareholdingDay': date.strftime('%d'),
#        'ddlShareholdingMonth': date.strftime('%m'),
#        'ddlShareholdingYear': date.strftime('%Y'),
    }
    response = session.post(url, data=post_data, headers=headers,proxies=https_proxy if use_proxy else None)
    soup = BeautifulSoup(response.content)
    session.close()

    #table=soup.findAll('td',{'valign': 'top'})
    table=soup.findAll('div',{'class': 'mobile-list-body'})
    line_num=len(table)/4
    if line_num==0:
        print ('no data found for %s on %s' % (exchange,date))
        return None

    data_collector=[]
    for my_j in np.arange(1,line_num+1):
        temp_data=pd.DataFrame([table[i+int((my_j-1)*4)].text for i in np.arange(0,4,1)]).T
        data_collector.append(temp_data)
    data=pd.concat(data_collector,axis=0)

    #process the data

    if len(data)!=0:
        data=data.reset_index()
        data.drop('index',1,inplace=True)
        data.rename(columns={0:'ccass_code',1:'name',2:'holdings_shares',3:'holdings_pct'},inplace=True)
        data['bbg_ticker']=data['name'].map(lambda x: x[x.find('#')+1:-1] + ' CH Equity')
        data['bbg_ticker']=data['bbg_ticker'].map(lambda x: x[-16:])
        data['bbg_ticker'].map(lambda x: len(x))
        data['date']=date
        data['holdings_shares']=data['holdings_shares'].map(lambda x: float(x.replace(',','')))
        #fix new ticker issue (e.g. Sinotrans A shares listed on 1/18/2019 and becom tradable immediately due to dual listing.)
        #this creates empty value instead of 0%
        data['holdings_pct']=data['holdings_pct'].map(lambda x: '0.00%' if x=='' else x)
        data['holdings_pct']=data['holdings_pct'].map(lambda x: float(x.replace('%',''))/100)

        data.to_csv(dump_path_local[exchange]+'%s.csv' % (date.strftime('%Y-%m-%d')))
        print ('finish downloading NB for %s on %s' % (exchange,date.strftime('%Y-%m-%d')))
    else:
        print ('no data on %s for %s' % (date.strftime('%Y-%m-%d'),exchange) )
        return False
    return None


def holdings_not_in_ccass(ticker):
    '''
    the link gives the holdings that are NOT in CCASS overtime
    https://webb-site.com/articles/CCASSanalysis.asp
    A sudden apperance of a large chunk of shares in CCASS (drop in stake% in the output of this function) could indicate stock being pledge
    6863 Huishan http://www.reuters.com/article/us-huishan-stocks-idUSKBN16Z08T gives a good example
    #let's use session method to access the link and close the session afterwards.
    '''

    download="https://webb-site.com/ccass/reghist.asp?s=&sc=%s" % (ticker)
    https_proxy = proxy_to_use
    session=requests.Session()
    response=session.post(download,proxies=https_proxy if use_proxy else None)
    soup = BeautifulSoup(response.content)
    table = soup.findAll('table', {'class': 'numtable'})
    session.close()
    try:
        heading_row=[row.text for row in table[1].find_all('th') if row.text]
    except IndexError:
        print ('Index error on %s' % (ticker))
        return False
    data_row=[row.text for row in table[1].find_all('td') if row.text]
    current_df=pd.DataFrame(index=np.arange(1,(len(data_row)-6)/7+1,1),columns=heading_row)
    for row in current_df.index:
        current_col=int((row-1)*7)
        for j,col in enumerate(current_df.columns):
            current_df.set_value(row,col,data_row[current_col+j])
    #tidy things up
    try:
        current_df['Row']=current_df['Row'].map(lambda x: float(x))
        current_df['Holdingdate']=pd.to_datetime(current_df['Holdingdate'])
        current_df['Holding']=current_df['Holding'].map(lambda x: float(x.replace(',','')))
        current_df['Change']=current_df['Change'].map(lambda x: float(x.replace(',','')))
        current_df['Issuedshares']=current_df['Issuedshares'].map(lambda x: float(x.replace(',','')))
        current_df['As at date']=pd.to_datetime(current_df['As at date'])
        current_df['Stake%']=current_df['Stake%'].map(lambda x: float(x))
    except ValueError:
        print ('value error for %s' % (ticker))
    current_df.set_index('Row',inplace=True)
    current_df.rename(columns={'Holdingdate':'Date','Issuedshares':'Issued Shares'},inplace=True)
    if len(heading_row)!=0:
        print ('%s successful' % (ticker))
    else:
        print ('%s failed' % (ticker))
        return False
    return current_df

def get_nb_universe():
    dump_path=uc.root_path_data+"connect\\northbound\\eligible_list\\"
    #  SH change
    sh_chg_link="https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/View-All-Eligible-Securities_xls/Change_of_SSE_Securities_Lists.xls?la=en"
    r=requests.get(sh_chg_link, proxies=proxy_to_use) if use_proxy else requests.get(sh_chg_link)
    open(dump_path+'Change_of_SSE_Securities_Lists.xls', 'wb').write(r.content)
    #  SZ change
    sh_chg_link="https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/View-All-Eligible-Securities_xls/Change_of_SZSE_Securities_Lists.xls?la=en"
    r=requests.get(sh_chg_link, proxies=proxy_to_use) if use_proxy else requests.get(sh_chg_link)
    open(dump_path+'Change_of_SZSE_Securities_Lists.xls', 'wb').write(r.content)
    #  SH last
    sh_chg_link="https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/View-All-Eligible-Securities_xls/SSE_Securities.xls?la=en"
    r=requests.get(sh_chg_link, proxies=proxy_to_use) if use_proxy else requests.get(sh_chg_link)
    open(dump_path+'SSE_Securities.xls', 'wb').write(r.content)
    #  SZ change
    sh_chg_link="https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Eligible-Stocks/View-All-Eligible-Securities_xls/SZSE_Securities.xls?la=en"
    r=requests.get(sh_chg_link, proxies=proxy_to_use) if use_proxy else requests.get(sh_chg_link)
    open(dump_path+'SZSE_Securities.xls', 'wb').write(r.content)
    return None




def get_h_share_list():

    url = 'https://www.hkex.com.hk/Market-Data/Statistics/Consolidated-Reports/China-Dimension?sc_lang=en#select1=0&select2=0'
    url='https://www.hkex.com.hk/eng/stat/smstat/mthbull/rpt_data_ChinaDimension_MB_ListOfHShare.json?_=1595240601888'
    if use_proxy:
        proxy = urllib.request.ProxyHandler(proxy_to_use)# 'nj02ga03cmp01.us.db.com'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
    #()
    request = urllib.request.urlopen(url)
    tables=json.loads(request.read().decode())['tables'][0]['body']
    collector=[]

    for i in np.arange(0,int((len(tables)-3)/4)): # last 3 rows are not tables
        # iterate row by row, each row contains 4 columns
        info_i=pd.Series(index=['date','ticker','name','marcap'],
                           data=[tables[i*4+j]['text'] for j in np.arange(0,4)])
        collector.append(info_i.rename(i).to_frame().T)
    res=pd.concat(collector,axis=0)
    # tidy up the tables
    res['date']=pd.to_datetime(res['date'])
    res['ticker']=res['ticker'].map(lambda x: str(int(pd.to_numeric(x)))+' HK Equity')
    return res


def download_nb_single_stock_all_participant(stock_code,dates):

    path=dump_path['nb']
    id_map=pd.read_csv(path+'\\bbg_ccass_id_map.csv').set_index('bbg_ticker')['ccass_code']

    collector=[]

    for date in dates:

        tickers=[id_map[stock_code]]

        url="https://www.hkexnews.hk/sdw/search/searchsdw.aspx"
        https_proxy = proxy_to_use
        headers = {
            #'Accept':'text/html, application/xhtml+xml, image/jxr, */*',
            #'Host': 'http://www.hkexnews.hk',
            #'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0;rv:11.0) like Gecko',
            #'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': "https://www.hkexnews.hk/sdw/search/searchsdw.aspx",
            #'Accept-Encoding': 'gzip, deflate',
            #'Accept-Language': 'en-US, en; q=0.8, zh-Hans-CN; q=0.5, zh-Hans; q=0.3',
            #'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3'
        }
        session = requests.Session()
        response = session.get(url,proxies=https_proxy)# headers=headers)
        soup = BeautifulSoup(response.content)
        viewstate = soup.select("#__VIEWSTATE")[0]['value']
        #()
        #eventvalidation = soup.select("#__EVENTVALIDATION")[0]['value']
        viewstate_generator = soup.select("#__VIEWSTATEGENERATOR")[0]['value']
        for ticker in tickers:
            ccass_code=ticker#ticker_map.loc[ticker]['ccass_code']
            post_data = {
            #'__EVENTVALIDATION': eventvalidation,
            '__EVENTTARGET':'btnSearch',
            '__EVENTARGUMENT':'',
            '__VIEWSTATE': viewstate,#
            '__VIEWSTATEGENERATOR':viewstate_generator,#
            'alertMsg': '',
#            'btnSearch.x': '99999',
#            'btnSearch.y': '99999',
            'today': um.today_date().strftime('%Y%m%d'),
            'txtShareholdingDate': date.strftime('%Y/%m/%d'),
#            'ddlShareholdingDay': date.strftime('%d'),
#            'ddlShareholdingMonth': date.strftime('%m'),
#            'ddlShareholdingYear': date.strftime('%Y'),
            'txtParticipantID':'',#'A00003' if exchange=='shanghai' else 'A00004',
            'txtStockCode':ccass_code
            }
            response = session.post(url, data=post_data, headers=headers,proxies=https_proxy)
            soup = BeautifulSoup(response.content)

            table=soup.findAll('div',{'class': 'mobile-list-body'})
            rows=len(table)/5
            res_i=pd.DataFrame(index=np.arange(0,rows),
                               columns=['participant_id','participant_name',
                                        #'participant_address',
                                        'shares','stake'])
            for row in np.arange(0,rows):
                row=int(row)
                entity_id=table[row*5].text
                entity_name=table[row*5+1].text
                #entity_address=table[row*5+2].text
                shares_i=int(table[row*5+3].text.replace(',',''))
                stake_i=float(table[row*5+4].text.replace('%',''))/100
                res_i.at[row,'participant_id']=entity_id
                res_i.at[row,'participant_name']=entity_name
                #res_i.at[row,'participant_address']=entity_address
                res_i.at[row,'shares']=shares_i
                res_i.at[row,'stake']=stake_i
            res_i['date']=date
            res_i['ticker_ccass']=ticker
            res_i['ticker_bbg']=stock_code
            collector.append(res_i)

            print ('finish %s on %s' % (stock_code,date.strftime('%Y-%m-%d')))
        res=pd.concat(collector)

    return res



def update_bbg_ccass_id_map():

    path=dump_path['nb']
    directions=['sh','sz']
    collector=[]
    for direction in directions:
        path_i=path+'\\%s\\' % (direction)
        files_all=um.iterate_csv(path_i)
        for file_i in files_all:
            res_i=pd.read_csv(path_i+'%s.csv' % (file_i))
            res_i['exchange']=direction
            collector.append(res_i)
    res_all=pd.concat(collector)
    res_all.groupby('bbg_ticker').last()[['ccass_code']].to_csv(path+'\\bbg_ccass_id_map.csv')
    #um.quick_auto_notice('bbg ccass id map refreshed')



def download_sc_daily_top10(date=um.yesterday_date()):
    '''
    Just get the json file
    This script downloads the most actively traded stocks by channel and direction
    '''
    url="https://www.hkex.com.hk/eng/csm/DailyStat/data_tab_daily_%se.js" % (date.strftime('%Y%m%d'))
    if use_proxy:
        proxy = urllib.request.ProxyHandler(proxy_to_use)# 'nj02ga03cmp01.us.db.com'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)

    request = urllib.request.urlopen(url)
    json_str=request.read().decode().replace('tabData = ','')
    info=json.loads(json_str)

    res_collector=[]
    for i in np.arange(0,4):
        info_i=info[i]
        mkt_i=info_i['market']
        table_i=info_i['content'][1]['table']
        cols=table_i['schema'][0]
        contents=table_i['tr']
        collector=[]
        for i in np.arange(0,len(contents)):
            row_i=contents[i]['td'][0]
            collector.append(row_i)
        res_i=pd.DataFrame(index=np.arange(0,len(contents)),columns=cols,data=collector).applymap(lambda x: x.replace('-','0'))
        res_i['Buy Turnover']=res_i['Buy Turnover'].map(lambda x: x.replace(',','')).map(pd.to_numeric)
        res_i['Sell Turnover']=res_i['Sell Turnover'].map(lambda x: x.replace(',','')).map(pd.to_numeric)
        res_i['Total Turnover']=res_i['Total Turnover'].map(lambda x: x.replace(',','')).map(pd.to_numeric)
        res_i['mkt']=mkt_i
        res_i['Stock Code']=res_i['Stock Code'].map(lambda x: str(int(x)))
        res_i['Buy Turnover']=res_i['Buy Turnover']/1000000/7.8
        res_i['Sell Turnover']=res_i['Sell Turnover']/1000000/7.8
        res_i['Total Turnover']=res_i['Total Turnover']/1000000/7.8
        res_collector.append(res_i)
    return pd.concat(res_collector)







if __name__ == "__main__":
    print ('all good')

    from datetime import datetime
    dates=pd.date_range(datetime(2023,7,17),um.yesterday_date(),freq='B')
    for date in dates:
        download_ccass(date,ccass_id='B01590')
    
    # res=download_sc_daily_top10()
#    update_bbg_ccass_id_map()
#
#    use_proxy=uc.use_proxy
#    proxy_to_use=uc.proxy_to_use
#
#    proxy = urllib.request.ProxyHandler(proxy_to_use)# 'nj02ga03cmp01.us.db.com'})
#    opener = urllib.request.build_opener(proxy)
#    urllib.request.install_opener(opener)
#
#    url = "https://webb-site.com/ccass/cholder.asp?part=1323&d=2021-01-21"
#
#    request = urllib.request.urlopen(url)
    # parse html
    #page = str(BeautifulSoup(request.content))
    #h_shares=get_h_share_list()
    #get_nb_universe()

#    #df=holdings_not_in_ccass('0700')
#
#    start=pd.datetime(2021,3,20)
#    end=pd.datetime(2021,3,25)
#    dates=pd.date_range(start,end,freq='B')
#
#
#    check=download_nb_single_stock_all_participant('688002 CH Equity',dates)
#
#    for date in dates:
#        download_SB(date,mode='shanghai')
#        #download_SB(date,mode='shenzhen')
#
#        #download_NB(date,exchange='shanghai')
#        #download_NB(date,exchange='shenzhen')
#        asdf
#        #download_SB(date,mode='h_circulation')





