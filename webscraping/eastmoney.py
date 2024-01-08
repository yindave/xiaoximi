# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:59:46 2019

@author: hyin1

eastmoney outputs can be in JSON or HTML
eastmoney build params into url

outputs are in Chinese so it's useful to install dateparser

useful link for some encode/decode related issue (e.g. 乱码)
https://blog.csdn.net/ARPOSPF/article/details/95536418

useful tips for extracting url:
e.g. info[2].findAll('a')[0]['href']
https://blog.csdn.net/suibianshen2012/article/details/62040460


Monitor for mutual fund actual disclosure dates:
http://fund.eastmoney.com/gonggao/dingqibaogao.html

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
import re
import os

import feather
import zlib
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime

from nlp import baidu_nltk as bltk

### constants
use_proxy=uc.use_proxy
proxy_to_use=uc.proxy_to_use
python_path=uc.get_python_path()

tagging_mapping=pd.read_excel(python_path+'webscraping'+'\\eastmoney_type_match.xlsx')
path=uc.root_path_data+'eastmoney\\fund_details\\%s.csv'

path_static_list=path.replace('\\fund_details','').replace('%s','static_list')



translation_type=tagging_mapping.set_index('type_chinese')['type_english'].to_dict()

style_map=tagging_mapping.set_index('type_english')['style'].to_dict()

type_map=tagging_mapping.set_index('type_english')['type'].to_dict()


# translation for NAV data columns
buy_status={
        '场内买入':'exchange_buy',
        '封闭期':'closed_period',
        '开放申购':'sub_normal',
        '暂停交易':'suspension',
        '暂停申购':'sub_paused',
        '认购期':'initial_subscription_period',
        '限制大额申购':'sub_small_only',
        None: 'unknown',
        np.nan:'unknown',
        }
sell_status={
        '场内卖出':'exchange_sell',
        '封闭期':'closed_period',
        '开放赎回':'red_normal',
        '暂停交易':'suspension',
        '暂停赎回':'red_paused',
        '认购期':'initial_subscription_period',
        None: 'unknown',
        np.nan:'unknown',
        }


### functions for AH and starboard

def _eastmoney_static_table(rename,nice_order,url):
    if use_proxy:
        proxy = urllib.request.ProxyHandler(proxy_to_use)
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
    request = urllib.request.urlopen(url)
    df=pd.DataFrame.from_dict(json.loads(request.read().decode())['data']['diff'])
    df=df.rename(columns=rename)
    df=df[nice_order]
    df.index=df.index+1
    return df


def get_ah_list(retries=5):
    '''
    Add a retry function
    '''
    def run():
        rename={'f12':'ticker_h',
                'f14':'name',
                'f15':'price_h',
                'f191':'ticker_a',
                'f186':'price_a',
                }
        nice_order=['name','ticker_h','ticker_a','price_h','price_a']
        url=('http://push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=500&po=1&np=1&fltt=2&invt=2&fid=f3&fs=b:MK0101'
            +'&fields=f12,f14,f15,f191,f186')
        df=_eastmoney_static_table(rename,nice_order,url)
        df[['price_h','price_a']]=(df[['price_h','price_a']]
                            .applymap(lambda x: np.nan if x=='-' else x)
                            .apply(pd.to_numeric))
        df['ticker_h']=df['ticker_h'].map(lambda x: str(int(str(x)))+' HK')
        df['ticker_a']=df['ticker_a'].map(lambda x: (str(x)+' CH'))
        return df

    for retry in np.arange(0,retries):
        try:
            df=run()
            break
        except:
            print ('Download error.')
            um.trigger_internet() #drigger internet may not solve the problem
        #failure after many retires
        if retry+1==retries:
            msg='EM get ah list failed after %s retries' % (retries)
            um.quick_auto_notice(msg)
            return None
    return df

def get_star_board():
    rename={'f12':'ticker',
            'f14':'name',
            'f2':'price',
            'f3':'pct_chg',
            'f6':'turnover_mCNY', #unit yuan, can convert it to USD
            }
    nice_order=['ticker','name','price','pct_chg','turnover_mCNY']
    url=('http://push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=1000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:1+t:23&'
         +'fields=f12,f14,f2,f3,f6')
    df=_eastmoney_static_table(rename,nice_order,url)
    df['ticker']=df['ticker'].map(lambda x: str(x)+' CH')
    df[['price','pct_chg','turnover_mCNY']]=(df[['price','pct_chg','turnover_mCNY']]
                            .applymap(lambda x: np.nan if x=='-' else x)
                            .apply(pd.to_numeric))
    df['turnover_mCNY']=df['turnover_mCNY']/1000000
    df=df.sort_values(by='pct_chg',ascending=False)
    df['pct_chg']=df['pct_chg']/100
    return df


### individual functions for fund info
def get_fund_list(use_static=True):
    '''
    All the fund tickers are here
    http://fund.eastmoney.com/allfund.html
    '''
    if not use_static:
        url='http://fund.eastmoney.com/js/fundcode_search.js'
        r=requests.get(url,proxies=proxy_to_use if use_proxy else None)
        cont = re.findall('var r = (.*])', r.text)[0]
        ls = json.loads(cont) #json can load list as well!
        fund_list = pd.DataFrame(ls, columns=['ticker', 'short_name', 'full_name', 'type_chinese', 'short_pinin'])
        fund_list=fund_list[['ticker','type_chinese','full_name','short_pinin']]
        #translate the type
        fund_list=fund_list[fund_list['type_chinese']!=''].copy()
        fund_list['type']=fund_list['type_chinese'].map(lambda x: translation_type[x])
        fund_list=fund_list.set_index('ticker')
        fund_list.index=fund_list.index.map(int)
    else:
        fund_list=pd.read_csv(path_static_list,index_col=[0])

    return fund_list.sort_index()



def get_manager_list(build_from_dump=False):
    '''
    Just load from existing manager dump and create the manager_idx to manager_name map
    '''
    manager_details=load_fund_manager(build_from_dump=build_from_dump)

    manager_details=manager_details.groupby(['manager']).last()['manager_name'].reset_index()
    manager_details=manager_details.rename(columns={'manager':'manager_idx','manager_name':'manager'})
    manager_details['manager_idx']=manager_details['manager_idx'].map(int)
    feather.write_dataframe(manager_details,path_static_list.replace('static_list.csv','manager_list.feather'))
    return manager_details.set_index('manager_idx').sort_index()

def get_fund_basic_info(ticker):

    url='http://fundf10.eastmoney.com/jbgk_%s.html' % (str(ticker).zfill(6))
    r=requests.get(url,proxies=proxy_to_use if use_proxy else None)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.findAll('table', {'class': 'info w790'})
    fields=[row.text for row in table[0].find_all('th')] #if row.text] #adding if row.text will ignore missing value
    values=[row.text for row in table[0].find_all('td')] #if row.text]
    res=pd.Series(index=fields,data=values).rename(ticker).to_frame().T
    res.index.name='ticker'
    return res



def get_fund_equity_holdings_TopUp(ticker,must_contain=[False,'some date','list of tickers for exceptions','some list']):
    return get_fund_equity_holdings(ticker,customize_years=[True,[um.today_date().year]],
                                                                  must_contain=must_contain)

def get_fund_equity_holdings(ticker,customize_years=[False,['year_0','year_1','year_2']],
                             must_contain=[False,'some date',['list of tickers for exceptions']]):
    '''
    The same fund can have 2 tickers, one front-end one back-end.
    For holdings, usually (not 100% sure) only one -end will show holding data
    So this could reduce the risk of double counting
    Values are in CNY even for HK or US stocks
    We use must_contain variable to make sure we download the holdings from the key date, especially for the topup update
    '''
    ticker_input=str(ticker).zfill(6)
    if customize_years[0]:
        years=customize_years[1]
    else:
        year_test=9999
        url="http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code=%s&topline=10&year=%s&month=3,6,9,12" % (ticker_input,year_test)
        r=requests.get(url,proxies=proxy_to_use if use_proxy else None)
        soup = BeautifulSoup(r.text, 'html.parser')
        try:
            years=json.loads(soup.text.replace('content','"content"').replace('arryear','"arryear"').replace('curyear','"curyear"').replace('var apidata=','').replace(';',''))['arryear']
            if len(years)==0:
                print ('No equity hlds data for %s' % (ticker))
                return pd.DataFrame()
        except:
            if  "You will not be permitted access until your credentials can be verified" in soup.text:
                print ('Disconnected!')
                raise Exception ('Connection error')

    collector_all_years=[]
    for year in years:
        url="http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code=%s&topline=10&year=%s&month=3,6,9,12" % (ticker_input,year)
        r=requests.get(url,proxies=proxy_to_use if use_proxy else None)
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.findAll('table')
        labels=soup.findAll('label')
        collector_all=[]
        for i in np.arange(0,len(tables),1):
            asof=labels[i*2+1].findAll('font')[0].text
            table=tables[i]
            contents=table.findAll('td')
            collector=[]
            table_col_len=len(table.findAll('th'))
            row_count=int(len(contents)/table_col_len)
            for row in np.arange(0,row_count,1):
                # we do relative reference for col 1~2 and -3~-1 as the "schema" sometimes changes in the middle
                is_top10ShareHolder='*' in contents[0+table_col_len*row].text # but not in the top10 holding
                idx=int(contents[0+table_col_len*row].text.replace('*',''))
                ticker_ss=contents[1+table_col_len*row].text
    #            name_chinese=contents[2+7*row].text
    #            relevant_info=contents[3+7*row].text
                try:
                    wgt=float(contents[-3+table_col_len*(row+1)].text.replace('%',''))
                except ValueError: # '---'
                    wgt=np.nan
                try:
                    hlds_num=float(contents[-2+table_col_len*(row+1)].text.replace(',',''))*10*1000 # original scale in 10k shares
                except ValueError: # '---'
                    hlds_num=np.nan
                try:
                    hlds_val=float(contents[-1+table_col_len*(row+1)].text.replace(',',''))*10*1000 # original scale in 10k CNY
                except ValueError: # '---'
                    hlds_val=np.nan

                data_i=pd.Series(index=['index','ticker','wgt','shares','values','IsTop10HolderButNotTop10Holding'],
                                 data=[idx,ticker_ss,wgt,hlds_num,hlds_val,is_top10ShareHolder]).to_frame().T
                collector.append(data_i)
            data_q=pd.concat(collector,axis=0)
            data_q['asof']=asof
            data_q['asof']=pd.to_datetime(data_q['asof'],format='%Y-%m-%d')
            data_q['index']=pd.to_numeric(data_q['index'])
            data_q['wgt']=pd.to_numeric(data_q['wgt'])
            data_q['shares']=pd.to_numeric(data_q['shares'])
            data_q['values']=pd.to_numeric(data_q['values'])
            #data_q['ticker']=data_q['ticker']

            # the sorting order is wrong, fix it in collector function later
            data_q['top_n']=data_q['values'].rank()
            collector_all.append(data_q)

        if len(collector_all)!=0:
            data=pd.concat(collector_all,axis=0)
            data['ticker_fund']=int(ticker)
            collector_all_years.append(data)
    if len(collector_all_years)!=0:
        output=pd.concat(collector_all_years,axis=0)
        output['location']=output['ticker'].map(lambda x: _helper_get_ticker_location(x))
        output['ticker']=output[['ticker','location']].apply(lambda x: _helper_clean_ss_ticker(x),axis=1)
        output=output.set_index('ticker_fund')
        print ('finish downloading holdings for %s' % (ticker))

        if must_contain[0]:
            must_contain_dt=must_contain[1]
            exception_names=must_contain[2]
            if ticker not in exception_names:
                if must_contain_dt in pd.to_datetime(output['asof'].values):
                    return output
                else:
                    print ('%s failed to pass the must_contain test for %s, returing empty data' % (ticker,must_contain_dt))
                    # then we raise an exception to force stop the downloading
                    raise Exception ('must contain error')
        return output
    else:
        print ('No holdings for %s' % (ticker))
        return pd.DataFrame()


def get_fund_nav_TopUp(ticker):
    return get_fund_nav(ticker,TopUpMode=True)


def get_fund_nav(ticker,start=datetime(2005,1,1),end=um.yesterday_date(),
                 TopUpMode=False):

    '''
    The data structure for some money fund is different
    '''
    if TopUpMode:
        nav_record=feather.read_dataframe(path_static_list.replace('static_list.csv','nav_record.feather')).set_index('ticker')['date']
        if ticker in nav_record.index:
            start=nav_record[ticker]
    print ('getting nav for %s from %s to %s' % (ticker,start,end))
    ticker_input=str(ticker).zfill(6)

    url=("https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz"+
            "&code=%s&page=1&sdate=%s&edate=%s&per=40" # we fix page number to be 40 here
            % (ticker_input,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))
        )

    r=requests.get(url,proxies=proxy_to_use if use_proxy else None,verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')

    total_page_number=int(soup.text[soup.text.find('pages:')+6:soup.text.find(',curpage:')])
    collector=[]
    for page_i in np.arange(1,total_page_number+1,1):
        url_i=url.replace('page=1','page=%s' % (int(page_i)))
        r=requests.get(url_i,proxies=proxy_to_use if use_proxy else None,verify=False)
        soup = BeautifulSoup(r.text, 'html.parser')
        contents=soup.findAll('tbody')[0].findAll('td')
        for row in np.arange(0,40,1):
            try:
                collector.append([contents[row*7+0].text,
                    contents[row*7+1].text,
                    contents[row*7+2].text,
                    contents[row*7+3].text,
                    contents[row*7+4].text,
                    contents[row*7+5].text,
                    #contents[row*7+6].text,
                    ])
            except IndexError:
                pass
    res=pd.DataFrame(index=np.arange(0,len(collector)),
                     columns=['date','nav_net','nav_gross','daily_chg','buy_status','sell_status'],
                     data=collector)
    res=res.drop('daily_chg',axis=1)

    res['date']=pd.to_datetime(res['date'])
    res['nav_net']=pd.to_numeric(res['nav_net'])
    res['nav_gross']=pd.to_numeric(res['nav_gross'])
    res['ticker']=ticker
    return res.set_index('ticker') # this is for the master func to start from last time


def get_fund_manager(ticker):
    '''
    manager track record change can be found here: http://fund.eastmoney.com/gonggao/110011,AN201209280005517504.html
    "其他公告"
    '''
    ticker_input=str(ticker).zfill(6)
    url='http://fundf10.eastmoney.com/jjjl_%s.html' % (ticker_input)

    r=requests.get(url,proxies=proxy_to_use if use_proxy else None,verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')
    info=soup.findAll('table',{'class': 'w782 comm jloff'})[0].findAll('td')
    row_num=int(len(info)/5)
    collector=[]

    for row in np.arange(0,row_num,1):
        data_i=[
                info[row*5+0].text,
                info[row*5+1].text,
                info[row*5+2].text,
                info[row*5+3].text,
                info[row*5+4].text,
            ]
        # obtain the url
        all_urls=info[row*5+2].findAll('a')
        idx_collector=[]
        for url_i in all_urls:
            x=url_i['href']
            idx_i=x[x.find('.html')-8:x.find('.html')]
            idx_collector.append(idx_i)
        data_i.append(' '.join(idx_collector))
        collector.append(data_i)
    res=pd.DataFrame(index=np.arange(0,len(collector)),
                    columns=['from','to','manager_name','tenor','return','manager'],
                    data=collector
                    )
    res['ticker']=ticker
    return res.set_index('ticker')


def get_filings(ticker):

    '''
    This func works on backend fund. It will just return empty df
    '''
    ticker_input=str(ticker).zfill(6)
    url=("https://fundf10.eastmoney.com/F10DataApi.aspx?type=jjgg"+
            "&code=%s&page=1&per=1000"
            % (ticker_input)
        )

    r=requests.get(url,proxies=proxy_to_use if use_proxy else None,verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')

    total_page_number=int(soup.text[soup.text.find('pages:')+6:soup.text.find(',curpage:')])
    collector=[]
    for page_i in np.arange(1,total_page_number+1,1):
        url_i=url.replace('page=1','page=%s' % (int(page_i)))
        r=requests.get(url_i,proxies=proxy_to_use if use_proxy else None,verify=False)
        soup = BeautifulSoup(r.text, 'html.parser')
        contents=soup.findAll('tbody')[0].findAll('td')
        for row in np.arange(0,1000,1):
            try:
                collector.append([contents[row*3+0].text,
                    contents[row*3+1].text,
                    contents[row*3+2].text,
                    #contents[row*7+6].text,
                    ])
            except IndexError:
                pass

    res=pd.DataFrame(index=np.arange(0,len(collector)),
                    columns=['title','type','date'],
                    data=collector
                    )
    res['ticker']=ticker
    return res.set_index('ticker')


def get_holder_split(ticker):
    '''
    This func doesn't work on backend fund
    '''
    ticker_input=str(ticker).zfill(6)
    #print ('working on %s' % (ticker_input))
    url="https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=cyrjg&code=%s"% (ticker_input)
    r=requests.get(url,proxies=proxy_to_use if use_proxy else None,verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')
# TODO: this treatment is not good, need to revise as we may skip tickers when there is internet error instead of empty return
    try:
        contents=soup.findAll('tbody')[0].findAll('td')
    except IndexError:
        return pd.DataFrame()
    collector=[]
    for row in np.arange(0,1000,1): # 1000 is more than enough
        try:
            collector.append([contents[row*5+0].text,
                contents[row*5+1].text,
                contents[row*5+2].text,
                contents[row*5+3].text,
                contents[row*5+4].text,
                ])
        except IndexError:
            pass
    res=pd.DataFrame(index=np.arange(0,len(collector)),
                    columns=['date','institution','individual','internal','total_shares_out_100mn'],
                    data=collector
                    )
    res['ticker']=ticker

    return res.set_index('ticker')



def get_share_out(ticker):
    ticker_input=str(ticker).zfill(6)
    #print ('working on %s' % (ticker_input))
    url="http://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=gmbd&code=%s" % (ticker_input)
    r=requests.get(url,proxies=proxy_to_use if use_proxy else None,verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')

    if 'var' in soup.text:
        try:
            contents=soup.findAll('tbody')[0].findAll('td')
        except IndexError:
            # for sub-tickers like tranched or backend fee version
            return pd.DataFrame()
    else:
        # internet connection issue
        raise NameError('Connection issue')

    collector=[]
    for row in np.arange(0,200,1):
        try:
            collector.append([contents[row*6+0].text,
                contents[row*6+1].text,
                contents[row*6+2].text,
                contents[row*6+3].text,
                contents[row*6+4].text,
                contents[row*6+5].text,
                ])
        except IndexError:
            pass
    res=pd.DataFrame(index=np.arange(0,len(collector)),
                    columns=['date','shares_sub_mn','shares_red_mn','share_out_mn','nav_CNYmn','nav_chg'],
                    data=collector
                    )
    res['date']=pd.to_datetime(res['date'])
    res=res.set_index('date').resample('Q').last().reset_index()
    cols=['shares_sub_mn','shares_red_mn','share_out_mn','nav_CNYmn']

    for col in cols:
        res[col]=pd.to_numeric(res[col].fillna('0').map(lambda x: x.replace('---','').replace('*','').replace(',','')))*100
    res['ticker']=ticker
    return res.set_index('ticker')



def get_manager_bio(manager_idx):
    url="http://fund.eastmoney.com/manager/%s.html" % (str(manager_idx))
    res=requests.get(url,proxies=proxy_to_use if use_proxy else None,verify=False)
    '''
    # important!! use res.content not res.text to fix the decoding issue!!!
    '''
    soup = BeautifulSoup(res.content, 'html.parser')
    if '您访问的页面不存在' in soup.text:
        print ('no data for index %s' % (manager_idx))
        return pd.DataFrame()
    bio=soup.findAll('div',{'class':'right ms'})[0].findAll('p')[0].text.replace('\n','').replace('\r','')
    df=pd.DataFrame(index=[manager_idx],columns=['bio'],data=[[bio]])
    df.index.name='ticker'
    return df





### the master functions for download fund info
def _helper_get_ticker_location(x):
    # some HK ticker has .HK some does not
    x=str(x).replace('.HK','').replace('.SZ','').replace('.SS','')
    #str(x['ticker']).replace('.HK','')
    if re.search('[a-zA-Z]', x) is not None: # US
        return 'F'
    elif len(re.findall(r'[\u4e00-\u9fff]+', x))!=0:
        return 'unknown'
    else:
        if len(x)==6:
            return 'A'
        else:
            return 'H'


def _helper_check_dump_and_combine(data_type,batch,collector):
    file_name='%s_%s' % (data_type,batch)
    file_exist=os.path.isfile(path % (file_name))
    if not file_exist:
        pd.concat(collector,axis=0).to_csv(path % (file_name))#,encoding='utf_8_sig')
    else:
        print ('Found existing dump, prepare to concat')
        old=pd.read_csv(path % (file_name),index_col=[0])
        new=pd.concat(collector,axis=0)
        pd.concat([old,new],axis=0).to_csv(path % (file_name))#,encoding='utf_8_sig')
    print ('%s dumped' % (file_name))
    return None

def _helper_check_for_existing_dump(data_type,batch):
    file_name='%s_%s' % (data_type,batch)
    file_exist=os.path.isfile(path % (file_name))
    if not file_exist:
        return None
    else:
        old=pd.read_csv(path % (file_name),index_col=[0])
        last_ticker=old.sort_index().index[-1]
        return last_ticker

def _helper_email_notice(data_type,batch,batch_total):
    print ('-%s -%s finished' % (data_type,batch))
    msg='eastmoney download finished: datatype %s batch %s/%s' % (data_type,batch,batch_total)
    print(msg)
    um.send_mail(msg,msg,uc.dl['self'])



function_mapper={'basics':get_fund_basic_info,
                 'hlds':get_fund_equity_holdings,
                 'hlds_TopUp':get_fund_equity_holdings_TopUp,
                 'hlds_MakeUp':get_fund_equity_holdings, # we will get the full history for missed names
                 'nav':get_fund_nav,
                 'nav_TopUp':get_fund_nav_TopUp,
                 'manager':get_fund_manager,
                 'filings':get_filings,
                 'split':get_holder_split,
                 'shout':get_share_out,
                 'bio':get_manager_bio,
                 }

def update_all_fund_info(data_type,batch_total=10,batch=0,
                         must_contain=[False,datetime(2021,6,30),'some list']):
    '''
    It appears that the full fund list changes slightly everyday for some reason (new launch?)
    Solution: read from a static fund list that is already dumped
    We need to delete all existing file for a clean refresh
    Use excel launcher to pass all the params
    We dump batch not individual files. Loading thousands of small csv is not good
    we use must_contain when downloading hlds_TopUp
    '''
    ### define split
    email_notice=False
    parts=batch_total
    parts_dict={}
    if data_type not in ['hlds_MakeUp']:
        df_all=get_fund_list() if data_type not in ['bio'] else get_manager_list()
    else:
        df_all=pd.read_csv(path_static_list.replace('static_list','static_list_hlds_makeup')).set_index('ticker')
    for part in np.arange(0,parts):
        df=df_all.iloc[part:].iloc[::parts]
        parts_dict[part]=df
    fund_list_input=parts_dict[batch]
    full_length=len(fund_list_input)

    ### check for existing load
    start_since=_helper_check_for_existing_dump(data_type,batch)
    if start_since is not None:
        print ('Find existing dump for -%s -%s with last ticker -%s' % (data_type,batch,start_since))
        fund_list_input=fund_list_input.loc[start_since:]
        new_length=len(fund_list_input)
        pct=new_length/full_length
        print ('%s pct to do vs. full data' % (pct*100))
    else:
        print ('Cannot find existing dump for -%s -%s, creating new dump' % (data_type,batch))

    tickers=fund_list_input.index.tolist()

    # not sure if the 2 lines below may create discoutinued fund holdings if disconnection happens.
    # safe way is to separately check holdings
#    if start_since is not None:
#        tickers.remove(start_since) #so we don't need to drop duplicate later
#
    collector=[]
    if len(tickers)==0:
        msg='%s %s already finisehd' % (data_type, batch)
        print (msg)
        um.send_mail('EM fund downloading finished',msg,uc.dl['self'])
        return None

    for ticker in tickers:

        # since we do ticker by ticker, as long as in the existing dump we have the ticker there
        # it means that ticker if fully downloaded
        print ('working on %s for %s' % (data_type,ticker))
        try:
            fund_type=df_all.loc[ticker]['type'] if data_type not in ['bio','hlds_MakeUp'] else 'dummy'
            if data_type in ['nav','nav_TopUp'] and fund_type in ['money_mkt','wealth_mgmt','fof_equity']:
                print ('pass %s nav for unwanted fund type' % (ticker))
            else:
                if data_type!='hlds_TopUp':
                    collector.append(function_mapper[data_type](ticker))#.to_csv(path[data_type] % (ticker),encoding='utf_8_sig')
                else:
                    collector.append(function_mapper[data_type](ticker,must_contain=must_contain))
            print ('finished %s for %s' % (data_type,ticker))
        except:
            #report error
            msg='Error when downloading %s on batch %s' % (ticker,batch)
            print (msg)
            if email_notice:
                um.send_mail('eastmoney download error: %s' % (msg),msg,uc.dl['self'])
            _helper_check_dump_and_combine(data_type,batch,collector)
            return None
    _helper_check_dump_and_combine(data_type,batch,collector)
    #always email notice if success
    _helper_email_notice(data_type,batch,batch_total)
    return None


### functions to load from dump
def load_fund_list_summary(examples=3,step=10):
    fund_list=get_fund_list().reset_index()
    summary=fund_list.groupby('type').count()['ticker'].sort_values(ascending=False).rename('count').to_frame()
    summary['type_chinese']=fund_list.groupby('type').last()['type_chinese']
    for i in np.arange(0,examples*step,step):
        summary['example_%s' % (i+1)]=fund_list.groupby('type').head(i+1).groupby('type').last()['ticker']

    return summary


def load_fund_specs(build_from_dump=False):
    data_type='basics'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        files_to_use=[x for x in files_all if data_type in x]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        res['issuance_date']=pd.to_datetime(res['发行日期'],format='%Y年%m月%d日')
        res['establish_date']=res['成立日期/规模'].map(lambda x: x[:x.find('/')]).map(lambda x: np.nan if x==' ' else x)
        res['establish_date']=pd.to_datetime(res['establish_date'],format='%Y年%m月%d日 ')
        res['aum_mCNY']=res['资产规模'].map(lambda x: x[:x.find('亿元')].replace(',','')).map(str).replace('--',np.nan).map(float)*100
        res['type_detail']=res['基金类型'].map(lambda x: translation_type[x])
        res['type']=res['type_detail'].map(lambda x: type_map[x])
        res['style']=res['type_detail'].map(lambda x: style_map[x])
        res['manager']=res['基金经理人'].copy()
        res['fund_parent']=res['基金管理人'].copy()
        res['benchmark']=res['业绩比较基准'].copy()
        res['establish_size_mnCNY']=res['成立日期/规模'].map(lambda x: x[x.find('/'):])
        res['establish_size_mnCNY']=(res['establish_size_mnCNY']
                                        .map(lambda x: x.replace('亿份','')
                                            .replace('/','')
                                            .replace(' ','')
                                            .replace('--','')
                                            .replace(',','')
                                            )
                                        )
        res['establish_size_mnCNY']=pd.to_numeric(res['establish_size_mnCNY'])*100
        to_keep=['ticker','issuance_date','establish_date','aum_mCNY','style','type','type_detail',
                 'manager','fund_parent','benchmark','establish_size_mnCNY']
        res=res[to_keep]
        feather.write_dataframe(res,feather_path)
    else:
        res=feather.read_dataframe(feather_path)
    return res


def load_fund_holdings(build_from_dump=False,for_top_up=False, for_make_up=False):
    '''
    Equity only, we will also add the fund type to the data block
    '''
    if not for_top_up:
        data_type='hlds'
    else:
        data_type='hlds_TopUp'
    if for_make_up:
        data_type='hlds_MakeUp'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        fund_tag=get_fund_list()
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        if data_type=='hlds_TopUp':
            files_to_use=[x for x in files_all if data_type in x]
        elif data_type=='hlds_MakeUp':
            files_to_use=[x for x in files_all if data_type in x]
        else:
            files_to_use=[x for x in files_all if (data_type in x and ('TopUp' not in x and 'MakeUp' not in x))]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        hlds=pd.concat(collector,axis=0)
        hlds=hlds.set_index('ticker_fund')
        hlds['fund_type']=fund_tag['type']
        hlds=hlds.drop('index',1)

        # we do the location tagging in the get_fund_equity_holdings func to avoid format messing up from csv
        #hlds['location']=hlds['ticker'].map(lambda x: _helper_get_ticker_location(x))
        #hlds['ticker']=hlds[['ticker','location']].apply(lambda x: _helper_clean_ss_ticker(x),axis=1)

        #hlds=hlds[hlds['ticker'].map(lambda x: False if type(x) is not str else True)]
        hlds['asof']=pd.to_datetime(hlds['asof'])
        hlds=hlds.reset_index()
        #hlds=hlds[hlds['ticker'].map(lambda x: True if type(x) is str else False)]
        feather.write_dataframe(hlds,feather_path)
    else:
        hlds=feather.read_dataframe(feather_path)
    # fix the top n issue
    hlds['top_n']=hlds.groupby(['ticker_fund','asof'])['values'].rank(ascending=False)
    return hlds


def TopUp_update_holdings(make_up_mode=True):
    print ('Please make sure no data gap between existing data block and topup data block')
    old=load_fund_holdings(build_from_dump=False,for_top_up=False,for_make_up=False) # load from existing dump, which has been maintained with both topup and makeup
    new=load_fund_holdings(build_from_dump=True,for_top_up=True,for_make_up=False) # rebuid topup from individual dump
    if make_up_mode:
        make_up=load_fund_holdings(build_from_dump=True,for_top_up=False,for_make_up=True) # rebuild makeup from individual dump
        hlds=pd.concat([old,new[old.columns],make_up[old.columns]],axis=0)
    else:
        hlds=pd.concat([old,new[old.columns]],axis=0)
    hlds=hlds.groupby(['ticker_fund','asof','ticker']).last().reset_index()
    # fix the top n issue
    hlds['top_n']=hlds.groupby(['ticker_fund','asof'])['values'].rank(ascending=False)
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % ('hlds'))
    feather.write_dataframe(hlds,feather_path)
    return hlds


def load_nav(build_from_dump=False,for_top_up=False):
    if not for_top_up:
        data_type='nav'
    else:
        data_type='nav_TopUp'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        if data_type=='nav_TopUp':
            files_to_use=[x for x in files_all if data_type in x]
        else:
            files_to_use=[x for x in files_all if (data_type in x and 'TopUp' not in x)]

        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        res['date']=pd.to_datetime(res['date'])
        res['buy_status']=res['buy_status'].map(lambda x: buy_status[x])
        res['sell_status']=res['sell_status'].map(lambda x: sell_status[x])
        feather.write_dataframe(res,feather_path)
    else:
        res=feather.read_dataframe(feather_path)


    return res


def TopUp_update_nav():
    print ('Please make sure no data gap between existing data block and topup data block')
    old=load_nav(build_from_dump=False,for_top_up=False)
    new=load_nav(build_from_dump=True,for_top_up=True)
    nav=pd.concat([old,new],axis=0)
    nav=nav.groupby(['ticker','date']).last().reset_index()
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % ('nav'))
    feather.write_dataframe(nav,feather_path)
    get_current_nav_status()

    return nav




def load_filings(build_from_dump=False):
    data_type='filings'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        files_to_use=[x for x in files_all if data_type in x]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        res['date']=pd.to_datetime(res['date'])
        feather.write_dataframe(res,feather_path)
    else:
        res=feather.read_dataframe(feather_path)

    return res

def load_fund_manager(build_from_dump=False):
    data_type='manager'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        files_to_use=[x for x in files_all if data_type in x]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        res['to']=res['to'].map(lambda x: um.yesterday_date() if x=='至今' else x)
        res['from']=pd.to_datetime(res['from'])
        res['to']=pd.to_datetime(res['to'])
        res=res.groupby(['ticker','from']).last().reset_index()

        collector=[]
        for i,idx in enumerate(res.index):
            entry_i=res.iloc[i]
            manager_list=entry_i['manager'].split(' ')
            manager_name_list=entry_i['manager_name'].split(' ')

            for ii,manager_i in enumerate(manager_list):
                if manager_i!='':
                    entry_i_j=entry_i.copy()
                    entry_i_j['manager']=manager_i
                    entry_i_j['manager_name']=manager_name_list[ii]
                    collector.append(entry_i_j.to_frame().T)
        res_all=pd.concat(collector)
        res_all['manager']=res_all['manager'].map(int)
        feather.write_dataframe(res_all,feather_path)
    else:
        res_all=feather.read_dataframe(feather_path)

    return res_all

def load_split(build_from_dump=False):
    data_type='split'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        files_to_use=[x for x in files_all if data_type in x]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        cols=['institution','individual','internal']
        for col in cols:
            res[col]=res[col].map(lambda x: x.replace('---','0%')).replace('%','').map(lambda x: x.replace('%','')).map(float)
        res['date']=pd.to_datetime(res['date'])
        res['total_shares_out_100mn']=pd.to_numeric(res['total_shares_out_100mn'].map(lambda x: str(x).replace('---','0').replace(',','')))
        res['shares_out_mn']=res['total_shares_out_100mn']*100


        res=res.drop('total_shares_out_100mn',1)
        feather.write_dataframe(res,feather_path)
    else:
        res=feather.read_dataframe(feather_path)

    return res

def load_share_out(build_from_dump=False):
    data_type='shout'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        files_to_use=[x for x in files_all if data_type in x]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        res['date']=pd.to_datetime(res['date'])

        feather.write_dataframe(res,feather_path)
    else:
        res=feather.read_dataframe(feather_path)

    return res


def load_bio(build_from_dump=False):

    data_type='bio'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    if build_from_dump:
        files_all=um.iterate_csv(path.replace('%s.csv',''))
        files_to_use=[x for x in files_all if data_type in x]
        collector=[]
        for file in files_to_use:
            df_i=pd.read_csv(path % (file))
            collector.append(df_i)
        res=pd.concat(collector,axis=0)
        # do some tidy up and parsing
        res=res.rename(columns={'ticker':'manager_idx'}).set_index(['manager_idx'])
        manager_list=get_manager_list()
        res['manager_name']=manager_list['manager']
        #res['house_name']=manager_list['house_name']
        def _get_gender(x):
            if x.find('先生')!=-1 or x.find('男')!=-1:
                return 'M'
            elif x.find('女士')!=-1 or x.find('女')!=-1:
                return 'F'
            else:
                return 'unknown'
        res['gender']=res['bio'].map(lambda x:_get_gender(x))
        def _get_education(x):
            if x.find('博士')!=-1:
                return 'phd'
            elif x.find('MBA')!=-1:
                return 'mba'
            elif x.find('硕士')!=-1 or x.find('研究生')!=-1:
                return 'master'
            elif x.find('学士')!=-1 or x.find('本科')!=-1:
                return 'bachelor'
            else:
                return 'unknown'
        res['education']=res['bio'].map(lambda x: _get_education(x))
        res['is_cfa']=res['bio'].map(lambda x: True if (x.find('CFA')!=-1 or x.find('特许金融分析师')!=-1) else False)
        res['is_frm']=res['bio'].map(lambda x: True if x.find('FRM')!=-1 else False)
        res['is_cpa']=res['bio'].map(lambda x: True if x.find('CPA')!=-1 else False)
        res['is_ccp']=res['bio'].map(lambda x: True if x.find('党')!=-1 else False)
        # iterate through bio to get university
        for j,idx in enumerate(res.index):
            print ('finish university tagging (%s/%s)' % (j+1,len(res)))
            token_i=bltk.tokenize(res['bio'][idx])
            token_i=token_i[token_i['prop'].isin(['nt','nz'])]
            if len(token_i)!=0:
                token_i['is_u']=token_i['item'].map(lambda x: True if x.find('大学')!=-1 else False)
                token_i=token_i[token_i['is_u']]
                if len(token_i)!=0:
                    for i,temp_idx in enumerate(token_i.index):
                        res.at[idx,'university_%s' % (i+1)]=token_i['item'][temp_idx]

        # dump results
        feather.write_dataframe(res.reset_index(),feather_path)
    else:
        res=feather.read_dataframe(feather_path)

    return res


def _helper_clean_ss_ticker(x):
    raw_ticker=str(x['ticker']).replace('.HK','')
    location=x['location']
    if location=='F' or location=='unknown':
        return raw_ticker
    else:
        try:
            ticker_int=int(raw_ticker)
        except ValueError:
            return raw_ticker
        if location=='A':
            return str(ticker_int).zfill(6)+'-CN'
        elif location=='H':
            return str(ticker_int)+'-HK'
        else:
            return 'unknown_ticker'

### functions for topup updte
'''
Some notes:
    static list: updated daily, can help capture new launch and fund taggings. It also serves as the fundation of the fund universe to loop through for other info
    fund basic info: can do top-up refresh daily, can help capture the latest AUM, something about fund benchmark etc
    fund NAV: top up update daily
    fund hlds: manual top-up using launcher, top-up covers current year and last year.
                is only available for fund with equity holdings
                for fund with front/back end version, only one version shows holdings
    fund split & fund manager: just manual refresh for all.
                Split also contains share out but only at semi-annual frequency. But the subscription and redemption can be volatile in short term so the long term share outstanding may not be as useful
'''
def update_basic_info():
    '''
    We assume the same fund will have the same basic info
    '''
    data_type='basics'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    static_list=pd.read_csv(path_static_list)
    old=feather.read_dataframe(feather_path)
    tickers_to_update=static_list[~static_list['ticker'].isin(old['ticker'])]['ticker'].tolist()
    collector=[]
    try:
        for i,ticker in enumerate(tickers_to_update):
            df_i=get_fund_basic_info(str(ticker).zfill(6))
            collector.append(df_i)
            print ('finished %s (%s/%s)' % (ticker,i+1,len(tickers_to_update)))
    except:
        if collector!=[]:
            _update_basic_info_dump_progress(old,collector,feather_path)
        return False
    _update_basic_info_dump_progress(old,collector,feather_path)
    return True

def _update_basic_info_dump_progress(old,collector,feather_path):
    res=pd.concat(collector,axis=0)
    def _get_issuance_date(x):
        try:
            return pd.to_datetime(x,format='%Y年%m月%d日')
        except ValueError:
            return np.nan


    res['issuance_date']=res['发行日期'].map(lambda x: _get_issuance_date(x))
    res=res[res['issuance_date'].fillna('na')!='na'] # we have the chance to add na line if it's updated on eastmoney later
    new=pd.concat([old,res.reset_index()],axis=0)
    new['ticker']=new['ticker'].map(int)
    new=new.sort_values(by='ticker')
    new=new.reset_index().drop('index',1)
    new=new[new['issuance_date'].fillna('na')!='na']
    feather.write_dataframe(new,feather_path)


def get_current_nav_status():
    '''
    we use this func to get the last udpate date of the existing nav dump
    '''
    data_type='nav_record'
    feather_path=path_static_list.replace('static_list.csv','%s.feather' % (data_type))
    old=load_nav()
    to_update=old.reset_index().groupby(['ticker']).max()
    feather.write_dataframe(to_update['date'].reset_index(),feather_path)
    return to_update


def tidy_up_all_data_quick(quick_path=[False,'path']):
    '''
    nav: just load
    hlds: load, merge tag and tidy up
    specs: just load
    static_list: load, and compare with specs, should have similar information
    filings: tidy up
    manager: tidy up
    split: tidy up
    we will dump the tidy-up results
    we no longer merge manager stats with hlds here
    '''
    if not quick_path[0]:
        path=path_static_list.replace('static_list.csv','')
    else:
        path=quick_path[1]
    # load the data
    specs=feather.read_dataframe(path+'basics.feather')
    specs_to_join=specs.rename(columns={'ticker':'ticker_fund'})
    filings=feather.read_dataframe(path+'filings.feather')
    hlds=feather.read_dataframe(path+'hlds.feather')
    managers=feather.read_dataframe(path+'manager.feather')
    nav_full=feather.read_dataframe(path+'nav.feather')
    nav_quick=nav_full.set_index(['date','ticker'])['nav_gross'].unstack().resample('B').last().fillna(method='ffill')
    #nav_ret=nav_quick.pct_change()
    investors=feather.read_dataframe(path+'split.feather')
    static_list=pd.read_csv(path+'static_list.csv')
    bio=feather.read_dataframe(path+'bio.feather')
    print ('raw data loaded')
    # tidy up share out
    shout=feather.read_dataframe(path+'shout.feather')[['ticker','date','share_out_mn']].set_index(['date','ticker'])
    shout_matrix=shout['share_out_mn'].unstack().fillna(method='ffill').fillna(0)
    shout['shout_out_chg_mn']=shout_matrix.diff().stack()
    shout=shout.reset_index().set_index('ticker').join(specs_to_join.set_index('ticker_fund'))
    shout.index.name='ticker'
    shout=shout.reset_index().set_index(['date','ticker'])
    nav_gross_quarterly=nav_full.set_index(['date','ticker'])['nav_gross'].unstack().resample('Q').last().stack()
    nav_net_quarterly=nav_full.set_index(['date','ticker'])['nav_net'].unstack().resample('Q').last().stack()
    shout=shout.join(nav_gross_quarterly.rename('nav_gross')).join(nav_net_quarterly.rename('nav_net'))
    shout['flow_gross_nav_CNYmn']=shout['shout_out_chg_mn']*shout['nav_gross']
    shout['flow_net_nav_CNYmn']=shout['shout_out_chg_mn']*shout['nav_net']
    shout=shout.reset_index()
    print ('share out change tidy up finished')
    # tidy up reporting date
    rpt_dt=filings[filings['type']=='定期报告'].copy()
    rpt_dt['year']=rpt_dt['title'].map(lambda x: x[x.find('年')-4:x.find('年')])
    def _get_periodicity(x):
        if x.find('净值')!=-1:
            return 'unknown'
        else:
            if x.find('季度')!=-1:
                return 'Q%s' % (x[x.find('季度')-1:x.find('季度')]).replace('一','1').replace('二','2').replace('三','3').replace('四','4')
            elif x.find('中期')!=-1 or x.find('半年')!=-1:
                return 'semi'
            elif x.find('季度')==-1 and x.find('中期')==-1 and x.find('半年')==-1 and x.find('年度')!=-1:
                return 'annual'
            else:
                return 'unknown'
    rpt_dt['periodicity']=rpt_dt['title'].map(lambda x: _get_periodicity(x))
    to_keep=['Q1','Q2','Q3','Q4','annual','semi']
    rpt_dt=rpt_dt[(rpt_dt['periodicity'].isin(to_keep)
                   & rpt_dt['year'].isin([str(x) for x in np.arange(2000,um.today_date().year+1)])
                  )]
    rpt_dt['year']=rpt_dt['date'].map(lambda x: x.year)
    rpt_dt['month']=rpt_dt['date'].map(lambda x: x.month)
    rpt_dt['day']=rpt_dt['date'].map(lambda x: x.day)
    rpt_dt=rpt_dt.groupby(['ticker','date']).last().reset_index()
    print ('reporting dates tidy up finished')
    # tidy up manager: we will not include money mkt fund performance as we don't download nav for these types of fund
    managers['to_max']=managers['to'].max()
    managers['to']=managers.apply(lambda x: um.today_date() if x['to_max']==x['to'] else x['to'],axis=1)
    managers=managers.drop(['tenor','return','to_max'],1)
    # add some extra manager stats
    managers=managers.set_index('manager').join(bio.groupby('manager_idx').last()[['gender','education','is_cfa','university_1','university_2','university_3']],how='left')
    managers=managers.reset_index()
    managers=managers.rename(columns={'index':'manager'})
    print ('manager records tidy up finished')

    # tidy up holder splits (not all funds have this data reported)
    investors['date']=investors['date'].map(lambda x: datetime(x.year,x.month,10)+pd.tseries.offsets.MonthEnd())
    investors=investors.groupby(['date','ticker']).last().reset_index()
    investors['month']=investors['date'].map(lambda x: x.month)
    investors=investors[investors['month'].isin([6,12])]
    investors=investors.rename(columns={'date':'asof','ticker':'ticker_fund'})[['asof','ticker_fund','institution','individual','internal']]
    investors['total']=investors[['institution','individual','internal']].sum(axis=1)
    for col in ['institution','individual','internal']:
        investors[col]=investors[col]/investors['total']
    investors['total_new']=investors[['institution','individual','internal']].sum(axis=1)
    investors=investors[['asof','ticker_fund','institution','individual','internal']].set_index(['ticker_fund'])
    to_keep=['aum_mCNY','style','type','type_detail']
    investors_details=investors.join(specs_to_join.set_index('ticker_fund')[to_keep],how='left')
    print ('investor type tidy up finished')
    # tidy up hlds
    to_keep=['style','type','type_detail','fund_parent']
    hlds=(hlds.set_index('ticker_fund').drop('fund_type',1)
        .join(specs_to_join.set_index('ticker_fund')[to_keep],how='left')
        ).reset_index()
    hlds=hlds[hlds['shares'].map(lambda x: np.nan if x==0 else x).fillna(-1)!=-1]
    hlds['aum_mnCNY']=(hlds['values']/hlds['wgt'].map(lambda x: np.nan if x==0 else x)*100)/1000000
    hlds=hlds.set_index(['ticker_fund','asof'])
    hlds['aum_mnCNY']=hlds.groupby(['ticker_fund','asof'])['aum_mnCNY'].median()
    hlds['aum_equity_mnCNY']=hlds.groupby(['ticker_fund','asof'])['values'].sum()/1000000
    hlds=hlds.reset_index()
    hlds['ticker']=hlds['ticker'].map(lambda x: x.replace('.SZ','-CN'))
    hlds=hlds.sort_values(by=['ticker_fund','asof','top_n'])
    tickers_ever_held=hlds.groupby(['ticker_fund','location']).last()['ticker'].unstack().apply(lambda x: '-'.join(x.dropna().sort_index().index.tolist()),axis=1)
    hlds=hlds.set_index('ticker_fund').join(tickers_ever_held.rename('tickers_ever_held')).reset_index()

    hlds=(hlds.set_index(['asof','ticker_fund'])
            #.join(manager_stats_all_to_merge,how='left')
            .join(investors.reset_index().set_index(['asof','ticker_fund']),how='left')
            .reset_index()
        )
    print ('hlds tidy up finished')

    # dump everything
    feather.write_dataframe(static_list,path+'TidyUp_%s.feather' % ('static_list'))
    feather.write_dataframe(specs,path+'TidyUp_%s.feather' % ('specs'))
    feather.write_dataframe(rpt_dt,path+'TidyUp_%s.feather' % ('rpt_dt'))
    feather.write_dataframe(managers,path+'TidyUp_%s.feather' % ('managers'))
    #feather.write_dataframe(manager_stats_all,path+'TidyUp_%s.feather' % ('manager_stats_all'))
    feather.write_dataframe(nav_full,path+'TidyUp_%s.feather' % ('nav_full'))
    feather.write_dataframe(nav_quick.reset_index(),path+'TidyUp_%s.feather' % ('nav_quick'))
    feather.write_dataframe(investors_details.reset_index(),path+'TidyUp_%s.feather' % ('investors_details'))
    feather.write_dataframe(hlds,path+'TidyUp_%s.feather' % ('hlds'))
    feather.write_dataframe(shout,path+'TidyUp_%s.feather' % ('shout'))
    print ('all data dumped')
    return None




if __name__=="__main__":
    print ('ok')
    
    res=get_fund_list(use_static=False)
    
#    res=get_fund_equity_holdings_TopUp(51)
#    must_contain=[False,datetime(2021,6,30),
#              [9158,7115,519667,100053]
#              ]
#
#    res=update_all_fund_info('hlds_TopUp',20,17)
#    get_fund_list(use_static=False)
#
#    path='C:\\Users\\hyin1\\temp_data\\eastmoney_quick\\'
#    tidy_up_all_data_quick(quick_path=[True,path])
#
#    TopUp_update_holdings()
    #check=load_fund_holdings(build_from_dump=True)
    #check=get_fund_equity_holdings(63)
#    load_fund_manager(build_from_dump=True)
#    load_fund_specs(build_from_dump=True)

#    bio=load_bio(build_from_dump=False)
#    get_manager_list()
#    res=load_fund_manager(build_from_dump=True)
#    check=get_fund_manager(1)

#    res=get_manager_bio(30710470)
#    bio=load_bio(build_from_dump=False)
#    from nlp import baidu_nltk as bltk
#    check=bltk.tokenize(bio['bio'][30674126])
#    load_fund_specs(build_from_dump=True)
#    load_nav(build_from_dump=True,for_top_up=False)
#    load_nav(build_from_dump=True,for_top_up=True)
#    TopUp_update_nav()
#    check=get_current_nav_status()
#    TopUp_update_nav()
#    load_nav(build_from_dump=True,for_top_up=True)
#    get_fund_nav(1,TopUpMode=True)
#    update_fund_nav_TopUp()
#    TopUp_update_holdings()

    #split=load_split(build_from_dump=False)
    #filings=load_filings(build_from_dump=False)
    #nav=load_nav(build_from_dump=False)
    #specs=load_fund_specs(build_from_dump=False)
    #manager=load_fund_manager(build_from_dump=False)
#
#    res['to']=res['to'].map(lambda x: um.today_date() if x=='至今' else x)
#    res['from']=pd.to_datetime(res['from'])
#    res['to']=pd.to_datetime(res['to'])
#    hlds=load_fund_holdings(build_from_dump=True, for_top_up=True)

    #check=get_fund_basic_info(160119)
    #check=get_fund_nav(7962,start=datetime(2020,10,31))
    #check=get_fund_manager(110011)
    #res=get_fund_equity_holdings(2380)
#    for i in np.arange(0,30,1):
#        update_all_fund_info('hlds',batch_total=30,batch=i,email_notice=False)
#    check=get_basics_from_dump()
#    # we use 1 run to get the list of data. Slices is hard coded here
#
#    data_type='basics'
#    batch_total=30
#    batch=14
#
#    update_all_fund_info('basics',batch_total=batch_total,batch=batch,)
##
#
#    #double check if all tickers are there
#    all_files=um.iterate_csv(path.replace('%s.csv',''))
#    collector=[]
#    for file in all_files:
#        df=pd.read_csv(path % (file))
#        collector.append(df)
#    res=pd.concat(collector,0)
#
#    res=res.set_index('ticker').sort_index()
#    res.index=res.index.map(lambda x:str(x).zfill(6))
#
#    res_comp=get_fund_list().set_index('ticker')
#
#    res.index.isin(res_comp.index)
#
#    df=get_basics_from_dump()







