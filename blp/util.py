# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:47:19 2019

@author: hyin1
"""

import utilities.constants as uc
import pandas as pd
import numpy as np
from utilities.misc import iterate_csv,yesterday_date,today_date
from blp.bdx import bdp,bdh,bds
from datetime import datetime


def tidy_up_buyback_dump(path,ah_adjustment=False):
    '''
    use this function to hold together the CACX buyback dump
    This one is used to deal with dollar amount type of announcement, mixed with shares num annoucement, and pct!
    Output unit would be million, (shares or value in local FX)
    For AH, sometimes the ticker is in H but the program is for A, indicated in a column later. We need to tidy this up
    '''
    def _parse_amount(x):
        if x!='empty':
            if 'Shares' in x:
                if 'Mln' in x:
                    try:
                        return float(x[x.find(': ')+2:x.find(' Mln Shares')])
                    except ValueError:
                        return np.nan
                else:
                    try:
                        return float(x[x.find(': ')+2:x.find(' Shares')])/1000000
                    except ValueError:
                        return np.nan
            elif ' % shares' in x:
                try:
                    return float(x[x.find(': ')+2:x.find(' % shares')])/100
                except ValueError:
                    return np.nan
            else:
                if 'Mln' in x:
                    try:
                        return float(x[x.find(': ')+2:x.find(' Mln')])
                    except ValueError:
                        return np.nan
                elif 'Bln' in x:
                    try:
                        return float(x[x.find(': ')+2:x.find(' Bln')])*1000
                    except ValueError:
                        return np.nan
                else:
                    try:
                        return float(x[x.find(': ')+2:x.find(' Shares')])/1000000
                    except ValueError:
                        return np.nan
        else:
            return np.nan
    nice_col_name={'Security ID':'Ticker',
           'Announce/Declared Date':'Declared Date',
           'Effective Date':'Effective Date',
           'Summary':'Name',
           'Summary.1':'Buyback Type',
           'Summary.2':'bb_shares_m',
           'Summary.3':'Security Type',
           'Summary.4':'share_out_m',
           }
    files=iterate_csv(path)
    data=pd.DataFrame()
    collector=[]
    for file in files:
        current_data=pd.read_csv(path+file+'.csv',header=3).iloc[1:]
        current_data=current_data.rename(columns=nice_col_name)
        #pdb.set_trace()
        current_data=current_data[['bb_shares_m','Declared Date','Effective Date','share_out_m','Ticker','Security Type','Buyback Type']]
        collector.append(current_data)
    data=pd.concat(collector,0)

    data['declared_date']=pd.to_datetime(data['Declared Date'])
    data['effective_date']=pd.to_datetime(data['Effective Date'])
    data=data.dropna() #dropping na as for A-share some odd lot bb in earlier years has entry record mismatch. Dropping them shouldn't cause too much problem
    data['bb_amount_mn']=data['bb_shares_m'].map(lambda x: _parse_amount(x))
    data['share_out_m']=data['share_out_m'].fillna('na').map(lambda x: _parse_amount(x))
    #data['type']=data['Security Type'].map(lambda x: 'p' if 'Common' not in x else 'c')
    #data['pct_bb']=data['bb_shares_m']/data['share_out_m']
    # mark the entry to distinguish between dollar and share announcement
    def _bb_unit(x):
        if ' Shares' in x:
            return 'shares'
        elif '% shares' in x:
            return 'pct'
        else:
            return 'value'
    data['bb_amount_unit']=data['bb_shares_m'].map(lambda x: _bb_unit(x))
    # for AH
    if ah_adjustment:
        from ah_arb.ah import AH
        ah=AH()
        data['is_A_share']=data[['Security Type','Ticker']].apply(lambda x: True if ('Class A' in x['Security Type'] and ' HK' in x['Ticker']) else False,axis=1)
        ah_map_raw=ah.get_ah_map(direction='h_to_a')
        ah_map={}
        for k,v in ah_map_raw.items():
            ah_map[k.replace('-HK',' HK Equity')]=v.replace('-CN',' CH Equity')
        data['Ticker']=data[['Ticker','is_A_share']].apply(lambda x: ah_map[x['Ticker']] if x['is_A_share'] else x['Ticker'],axis=1)
#    data=data.sort_values(by=['Ticker','declared_date','effective_date'])
#    data_sum=data.reset_index().groupby(['effective_date','Ticker']).sum().swaplevel(1,0,0).sort_index()
#    data_last=data.reset_index().groupby(['effective_date','Ticker']).last().swaplevel(1,0,0).sort_index()[['bb_amount_unit','Buyback Type']]
#    data_first=data.reset_index().groupby(['effective_date','Ticker']).first().swaplevel(1,0,0).sort_index()['declared_date']
#
#    data_res=pd.concat([data_sum,data_last,data_first],axis=1).drop('is_A_share',axis=1)
    res=data.copy()
    res=res.reset_index().groupby(['Ticker','declared_date']).last()
    return res
#    # Here we tag the buyback program announcement
#    # This is really market dependent. For Thailand this seems working.
#    # For Japan, it's really a dirty treatment. USUALLY for announcement event the Announcement date and effective date are the same
#    # US is always about announcement. It's declared and effective date are always the same
#    tickers=res.index.levels[0]
#    collector=[]
#    for ticker in tickers:
#        bb_i=res.loc[ticker].reset_index().set_index('declared_date')
#        counter=0
#        for dt_d in bb_i.index:
#            dt_e=bb_i.loc[dt_d]['effective_date']
#            try:
#                if dt_d!=dt_e:
#                    counter=counter+1
#            except:
#                import pdb
#                pdb.set_trace()
#            bb_i.at[dt_d,'program']=counter
#        bb_i=bb_i.reset_index()
#        bb_i['ticker']=ticker
#        collector.append(bb_i)
#
#    res_announcement=pd.concat(collector,axis=0)
#    res_announcement['program']=res_announcement['program'].map(lambda x: 1 if x==0 else x)
#
#    return res_announcement


def get_bbg_usual_col(ticker_list, short_sector=True, add_short_industry=False,dropna=True):

    # not that BBG calculate marcap with all share classes included.
    # quick fix is to use float / float pct
    # but we still need direct marcap for ADR

    ss_df=pd.DataFrame(index=ticker_list,columns=['Name','MarketCap (US$ bn)','3m ADV (US$ mn)','Sector'])
    ss_df['Name']=bdp(ss_df.index.tolist(),['SHORT_NAME'])['SHORT_NAME']

    check=(bdp(ss_df.index.tolist(),['EQY_FLOAT','EQY_FREE_FLOAT_PCT','CRNCY_ADJ_PX_LAST'],
                 overrides={'EQY_FUND_CRNCY':'USD'}))
    ss_df['MarketCap (US$ bn)']=check['EQY_FLOAT']/check['EQY_FREE_FLOAT_PCT']*check['CRNCY_ADJ_PX_LAST']*100/1000
    to_replace=(bdp(ss_df.index.tolist(),['CRNCY_ADJ_MKT_CAP'],overrides={'EQY_FUND_CRNCY':'USD'})/1000)['CRNCY_ADJ_MKT_CAP']
    ss_df['MarketCap (US$ bn)']=ss_df['MarketCap (US$ bn)'].fillna(to_replace)
    ss_df['3m ADV (US$ mn)']=(bdp(ss_df.index.tolist(),['INTERVAL_AVG'],overrides={'MARKET_DATA_OVERRIDE':'turnover','CRNCY':'USD','CALC_INTERVAL':'63D',
                                                                                           'END_DATE_OVERRIDE':yesterday_date().strftime('%Y%m%d')})/1000000)['INTERVAL_AVG']
    ss_df['Sector']=bdp(ss_df.index.tolist(),['GICS_SECTOR_NAME'])['GICS_SECTOR_NAME']
    if dropna:
        ss_df=ss_df.dropna()
    if short_sector:
        ss_df['Sector']=ss_df['Sector'].map(lambda x: uc.short_sector_name[x])
    if add_short_industry:
        ss_df['Industry']=bdp(ss_df.index.tolist(),['GICS_INDUSTRY_GROUP_NAME'])['GICS_INDUSTRY_GROUP_NAME']
        ss_df['Industry']=ss_df['Industry'].map(lambda x: uc.short_industry_name[x])
    return ss_df


def get_bbg_nice_compo_hist(ticker,asof,load_local=False):

    '''
    This function returns actual shares but is limited to only 700 tickers
    use the _no_limit version of this function if no need for actual shares
    input basket or index ticker and snapshot day, output nicely formatted compo
    modify tickers so that:
    equity
    JP
    CH
    '''
    local_path=uc.index_compo_path+ticker+'\\%s.csv' % (asof.strftime('%Y%m%d'))
    if not load_local:
        compo=bds([ticker],'INDX_MWEIGHT_PX',overrides={'END_DATE_OVERRIDE':asof.strftime('%Y%m%d')})
    else:
        compo=pd.read_csv(local_path)
    compo=compo.rename(columns={'Actual Weight':'shares','Current Price':'px','Index Member':'ticker','Percent Weight':'wgt'})
    compo['wgt']=compo['wgt']/100
    compo['ticker']=compo['ticker'].map(lambda x: x.replace(' CS',' CH').replace(' CG',' CH').replace(' JT',' JP')
                              .replace(' C1',' CH').replace(' C2',' CH')
                             .replace(' UN',' US').replace(' UQ',' US').replace(' UW',' US')
                                            +' Equity')
    compo=compo.set_index('ticker')
    compo['asof']=asof

    return compo

def get_bbg_nice_compo_hist_no_limit(ticker,asof,exclude_a_shares=False):
    '''
    This function returns tickers and weight with limited to upto 5000 tickers
    We disable load function for this one
    '''
    compo=bds([ticker],'INDX_MWEIGHT_HIST',overrides={'END_DATE_OVERRIDE':asof.strftime('%Y%m%d')})
    compo=compo.rename(columns={'Index Member':'ticker','Percent Weight':'wgt'})
    compo['wgt']=compo['wgt']/100
    compo['ticker']=compo['ticker'].map(lambda x: x.replace(' CS',' CH').replace(' CG',' CH').replace(' JT',' JP')
                              .replace(' C1',' CH').replace(' C2',' CH')
                             .replace(' UN',' US').replace(' UQ',' US').replace(' UW',' US')
                                            +' Equity')
    compo.index.name='index'
    if exclude_a_shares:
        compo=compo[compo['ticker'].map(lambda x: False if x.find(' CH Equity')!=-1 else True)]
    compo=compo.set_index('ticker')
    compo['asof']=asof
    return compo



def get_bbg_calendarized_level_and_revision(ticker,field,step_bday,
                                  output_fy=[0,1,2],start_year=2010):

    end_year=today_date().year+3
    years=np.arange(start_year,end_year+1)
    announcement_dts=pd.DataFrame()
    collector=[]
    for year in years:
        res=bdp([ticker],['ANNOUNCEMENT_DT'],overrides={'EQY_FUND_YEAR':year,'FUND_PER':'Y'})
        res['fy']=year
        collector.append(res)
    announcement_dts=pd.concat(collector,0)
    delta=announcement_dts.dropna().diff().iloc[-1].values[0]
    announcement_dts=announcement_dts.set_index('fy')
    announcement_dts=announcement_dts.fillna(0)

    for i,fy in enumerate(announcement_dts.index):
        if announcement_dts.loc[fy]['ANNOUNCEMENT_DT']==0:
            try:
                announcement_dts.at[fy,'ANNOUNCEMENT_DT']=announcement_dts.loc[fy-1]['ANNOUNCEMENT_DT']+delta
            except KeyError:
                continue

    announcement_dts=announcement_dts[announcement_dts['ANNOUNCEMENT_DT']!=0]

    announcement_dts['ANNOUNCEMENT_DT']=pd.to_datetime(announcement_dts['ANNOUNCEMENT_DT'])
    announcement_dts=announcement_dts.reset_index().set_index('ANNOUNCEMENT_DT')
    announcement_dts_d=announcement_dts.resample('D').last().fillna(method='bfill') # need bfill!
    #get the fiscal year 20xx EPS estimate
    all_ts=pd.DataFrame()
    collector=[]
    for year in years:
        try:
            fperiod='%sY' % (year-2000)
            ts=bdh([ticker],[field],datetime(2000,1,1),today_date(),overrides={'BEST_FPERIOD_OVERRIDE':fperiod}).loc[ticker]
            ts['fy']=year
            ts['next_fy_asof_now']=announcement_dts_d['fy'] # this ensures when the date moves to next fy the fy column changes accordingly
            ts=ts.resample('B').last().fillna(method='ffill')
            ts['%s_%s_revision' % (field,step_bday)]=ts[field].pct_change(step_bday)
            collector.append(ts)
        except:
            print ('No data for %s on FY %s' % (ticker,year))
    all_ts=pd.concat(collector,0)
    output=all_ts.dropna().copy()
    output['fy_rel']=output['fy']-output['next_fy_asof_now']
    output=output.reset_index().set_index(['date','fy_rel'])
    output_level=output.reset_index().groupby(['date','fy_rel']).mean().sort_index()[field].unstack()[output_fy]
    output_revision=output.reset_index().groupby(['date','fy_rel']).mean().sort_index()['%s_%s_revision' % (field,step_bday)].unstack()[output_fy]
    return output_level,output_revision


def get_bbg_period_return(tickers,start,end):
    from blp.bdx import bdp
    overrides={'CUST_TRR_START_DT':start.strftime('%Y%m%d'),
               'CUST_TRR_END_DT':end.strftime('%Y%m%d')}
    res=bdp(tickers,['CUST_TRR_RETURN_HOLDING_PER'],overrides=overrides)['CUST_TRR_RETURN_HOLDING_PER']
    return res/100


def group_marcap(x,return_number=True):
    #unit should be in bUSD
    if x<=0.5:
        return 'Micro Cap' if not return_number else 1
    elif x>0.5 and x<=2:
        return 'Small Cap' if not return_number else 2
    elif x>2 and x<=10:
        return 'Mid Cap' if not return_number else 3
    elif x>10 and x<=50:
        return 'Large Cap' if not return_number else 4
    elif x>50:
        return 'Mega Cap' if not return_number else 5
def group_to(x,return_number=False):
    #unit should be in bUSD
    if x<=5:
        return 'Less than 5mn' if not return_number else 1
    elif x>5 and x<=10:
        return '5-10mn' if not return_number else 2
    elif x>10 and x<=20:
        return '10-20mn' if not return_number else 3
    elif x>20 and x<=50:
        return '20-50mn' if not return_number else 4
    elif x>50:
        return 'Above 50mn' if not return_number else 5


def get_ric_for_cml(x):
    x=x.replace(' Equity','')
    if ' HK' in x:
        return str((int(x.replace(' HK','')))).zfill(4)+'.HK'
    if ' CH' in x and x[0]!='6':
        return str(((x.replace(' CH',''))))+'.SZ'
    if ' CH' in x and x[0]=='6':
        return str(((x.replace(' CH',''))))+'.SS'
    if ' JP' in x or 'JT' in x:
        return x.replace(' JP','.T').replace(' JT','.T')
    else:
        print ('unknown BBG ticker')
        return np.nan




def correlation_finder(x,y_compo,
                       start,end,
                       rolling=252,
                       x_method=['pct',1],
                       y_method=['pct',1],
                       min_points=30
                       ):
    '''
    mimic CFND correlation calculation
    y_compo needs to have memb (or somthing similar)
    '''
    all_compo=get_bbg_nice_compo_hist_no_limit(y_compo,yesterday_date())
    tickers=all_compo.index.tolist()
    if x not in tickers:
        tickers.append(x)
    px=bdh(tickers,['px_last'],start,end,currency='USD')['px_last'].unstack().T
    px_to_calc=px.resample('B').last().fillna(method='ffill') # rolling corr cannot handle missing value
    px_calc_x=px_to_calc[x]
    px_calc_y=px_to_calc.drop(x,1)
    if x_method[0]=='pct':
        px_calc_x=px_calc_x.pct_change(x_method[1])
    elif x_method[0]=='diff':
        px_calc_x=px_calc_x.diff(x_method[1])
    elif x_method[0]=='abs':
        pass
    if y_method[0]=='pct':
        px_calc_y=px_calc_y.pct_change(y_method[1])
    elif y_method[0]=='diff':
        px_calc_y=px_calc_y.diff(y_method[1])
    elif y_method[0]=='abs':
        pass
    corr_hist=px_calc_y.rolling(rolling,min_periods=min_points).corr(other=px_calc_x)
    corr_last=corr_hist.iloc[-1].sort_values()
    return corr_hist,corr_last

def get_region(x):
    '''
    For China only
    '''
    x=x.replace(' Equity','')
    try:
        if x[-2:]=='HK':
            return 'HK'
        elif x[-2:]=='CH':
            if int(x[0])==6 or int(x[0])==9: # 9 is for B share in SH
                return 'SH'
            else: # 200 is B share in SZ
                return 'SZ'
        elif x[-2:]=='US':
            return 'ADR'
        else:
            return 'others'
    except:
        return 'others'

def get_board(x):
    '''
    For China only, needs to use apply with region tagged already
    '''
    x=x.replace(' Equity','')
    if x['region']=='SH':
        if int(x['ticker'][0])==6:
            if int(x['ticker'][:3])==688 or int(x['ticker'][:3])==689:
                return 'StarBoard'
            else:
                return 'MainBoard (SH)'
        else:
            return 'B-share'
    elif x['region']=='SZ':
        if int(x['ticker'][0])==0:
            if x['ticker'][:3]=='002':
                return 'SME'
            else:
                return 'MainBoard (SZ)'
        elif int(x['ticker'][0])==3:
            return 'ChiNext'
        else:
            return 'B-share'
    elif x['region']=='HK':
        if len(x['ticker'])==7 and int(x['ticker'][0])==8:
            return 'GEM'
        else:
            return 'MainBoard'
    elif x['region']=='ADR':
        return 'MainBoard'
    elif x['region']=='others':
        return 'others'


def get_ashare_exchange(x):
    if x[0]=='6':
        if x[:3]=='688':
            return 'Shanghai STAR'
        else:
            return 'Shanghai Mainboard'
    elif x[:3]=='300':
        return 'Shenzhen ChiNext'
    elif x[:3]=='002' or x[:3]=='003':
        return 'Shenzhen SME'
    elif x[:3]=='000' or x[:3]=='001':
        return 'Shenzhen Mainboard'
    elif x[:3]=='200':
        return 'Shenzhen B-share'
    elif x[:3]=='900':
        return 'Shanghai B-share'
    else:
        return 'unknown'


def load_compo(index_name):
    import feather
    compo_path=uc.compo_path
    return feather.read_dataframe(compo_path % (index_name)).rename(columns={'asof':'date'}).set_index(['date','ticker'])['wgt']




if __name__=='__main__':
    print ('ok')
    
    
    
#    path="Z:\\dave\\data\\buyback\\china_hk\\"
#    bb=tidy_up_buyback_dump(path,ah_adjustment=True)
#    ticker_list=['728 HK Equity','1398 HK Equity','601398 CH Equity']
#    short_sector=True
#    add_short_industry=True
#
#
#
#    ss_df=pd.DataFrame(index=ticker_list,columns=['Name','MarketCap (US$ bn)','3m ADV (US$ mn)','Sector'])
#    ss_df['Name']=bdp(ss_df.index.tolist(),['SHORT_NAME'])['SHORT_NAME']
#
#    check=(bdp(ss_df.index.tolist(),['EQY_FLOAT','EQY_FREE_FLOAT_PCT','CRNCY_ADJ_PX_LAST'],
#                 overrides={'EQY_FUND_CRNCY':'USD'}))
#    ss_df['MarketCap (US$ bn)']=check['EQY_FLOAT']/check['EQY_FREE_FLOAT_PCT']*check['CRNCY_ADJ_PX_LAST']*100/1000
#
#    ss_df['3m ADV (US$ mn)']=(bdp(ss_df.index.tolist(),['INTERVAL_AVG'],overrides={'MARKET_DATA_OVERRIDE':'turnover','CRNCY':'USD','CALC_INTERVAL':'63D',
#                                                                                           'END_DATE_OVERRIDE':yesterday_date().strftime('%Y%m%d')})/1000000)['INTERVAL_AVG']
#    ss_df['Sector']=bdp(ss_df.index.tolist(),['GICS_SECTOR_NAME'])['GICS_SECTOR_NAME']
#    ss_df=ss_df.dropna()
#    if short_sector:
#        ss_df['Sector']=ss_df['Sector'].map(lambda x: uc.short_sector_name[x])
#
#    if add_short_industry:
#        ss_df['Industry']=bdp(ss_df.index.tolist(),['GICS_INDUSTRY_GROUP_NAME'])['GICS_INDUSTRY_GROUP_NAME']
#        ss_df['Industry']=ss_df['Industry'].map(lambda x: uc.short_industry_name[x])
#
#

#    path="Z:\\dave\\data\\buyback\\thailand\\large_elliott\\"

#    res=tidy_up_buyback_dump(path)
#    bb_path="Z:\\dave\\data\\buyback\\us\\"
#    bb_data=tidy_up_buyback_dump(bb_path+'raw\\')
#
#    x='JNK US Equity'
#    y_compo='MXAP Index'
#    start=datetime(2018,3,9)
#    end=datetime(2020,3,9)
#    rolling=260 # 65 calendar day is about 3m
#    x_method=['pct',1]
#    y_method=['pct',1]
#    min_points=rolling
#
#    corr_hist,corr_last=correlation_finder(x,y_compo,start,end,rolling=rolling,
#                            x_method=x_method,y_method=y_method)
#

