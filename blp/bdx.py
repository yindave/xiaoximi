# SimpleHistoryExample.py

import blpapi
import pandas as pd
import pdb
import numpy as np
from datetime import datetime
import datetime as datetime_

'''
Use both the Excel example and Python example from WAPI to create new function if needed
'''

GMT_adj=datetime_.timedelta(hours=8)

def bdh(tickers,fields,start,end,overrides={},**kwargs):

    '''
    Do not add any ffill option, do this outside
    use kwargs for fields like:
    adjustmentFollowDPDF='No'
    currency='USD'
    periodicitySelection='MONTHLY','QUARTERLY'
    use overrides for overridable fields
    '''


    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost("localhost")
    sessionOptions.setServerPort(8194)

    #print "Connecting to %s:%s" % (options.host, options.port)
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        #print "Failed to start session."
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            #print "Failed to open //blp/refdata"
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")
        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)

        request.set("periodicityAdjustment", "ACTUAL")
        request.set("startDate", start.strftime('%Y%m%d'))
        request.set("endDate", end.strftime('%Y%m%d'))
        #other kwargs
        if len(kwargs)!=0:
            for key, value in kwargs.items():
                request.set(key, value)
        #handle the override
        if len(overrides)!=0:
            ovr=request.getElement("overrides")
            for k,v in overrides.items():
                ovrt=ovr.appendElement()
                ovrt.setElement("fieldId", k)
                ovrt.setElement("value", v)

        # Send the request
        session.sendRequest(request)
        res=pd.DataFrame()
        # Process received events
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            event = session.nextEvent()
            if event.eventType()  in [5,6]:
                '''eventtype and eventType=6 or 5 are the key to differentiate, need to extract data from event now'''

                output=pd.DataFrame()
                for msg in event:
                    fieldDataArray = msg.getElement('securityData').getElement('fieldData')
                    size = fieldDataArray.numValues()
                    fieldDataList = [fieldDataArray.getValueAsElement(i) for i in range(0,size)]
                    outDates = [x.getElementAsDatetime('date') for x in fieldDataList]
                    output = pd.DataFrame(index=outDates,columns=fields)
                    for strD in fields:
                        outData = []

                        for x in fieldDataList:
                            try:
                                outData.append(x.getElementAsFloat(strD))
                            except:
                                outData.append(np.nan)
                        output[strD] = outData
                    output.replace('#N/A History',np.nan,inplace=True)
                    output.index = pd.to_datetime(output.index)
                    output['ticker']=msg.getElement('securityData').getElement('security').getValue()
                    if len(res)==0:
                        res=output.copy()
                    else:
                        res=pd.concat([res,output],axis=0)
            if event.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                break

    finally:
        # Stop the session
        session.stop()
    res.index.name='date'
    res=res.reset_index().set_index(['ticker','date'])
    return res


def bdp(tickers,fields,overrides={}):

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost("localhost")
    sessionOptions.setServerPort(8194)

    #print "Connecting to %s:%s" % (options.host, options.port)
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        #print "Failed to start session."
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            #print "Failed to open //blp/refdata"
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("ReferenceDataRequest")
        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)

        #handle the override
        if len(overrides)!=0:
            ovr=request.getElement("overrides")
            for k,v in overrides.items():
                ovrt=ovr.appendElement()
                ovrt.setElement("fieldId", k)
                ovrt.setElement("value", v)

        # Send the request
        session.sendRequest(request)
        res=pd.DataFrame(index=tickers,columns=fields)
        # Process received events
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            event = session.nextEvent()
            if event.eventType()  in [5,6]:
                '''eventtype and eventType=6 or 5 are the key to differentiate, need to extract data from event now'''
                for msg in event:
                    for i in np.arange(0,len(tickers)):
                        try:
                            ticker=msg.getElement('securityData').getValue(int(i)).getElement('security').getValue()
                            for field in fields:
                                try:
                                    v=msg.getElement('securityData').getValue(int(i)).getElement('fieldData').getElement(field).getValue()
                                except:
                                    v=np.nan
                                res.at[ticker,field]=v
                        except:
                            continue
                            #print ('unknown error potentially caused by bad ticker')
            if event.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                break

    finally:
        # Stop the session
        session.stop()
    return res


def bds(tickers,field,overrides={}):
    '''
    we download one field per time because each bds field is different
    '''
    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost("localhost")
    sessionOptions.setServerPort(8194)

    #print "Connecting to %s:%s" % (options.host, options.port)
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        #print "Failed to start session."
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            #print "Failed to open //blp/refdata"
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("ReferenceDataRequest")
        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)

        request.getElement("fields").appendValue(field)

        #handle the override
        if len(overrides)!=0:
            ovr=request.getElement("overrides")
            ovrt=ovr.appendElement()
            for k,v in overrides.items():
                ovrt.setElement("fieldId", k)
                ovrt.setElement("value", v)

        # Send the request
        session.sendRequest(request)
        res=pd.DataFrame()
        # Process received events
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            event = session.nextEvent()
            if event.eventType()  in [5,6]:
                '''eventtype and eventType=6 or 5 are the key to differentiate, need to extract data from event now'''
                for msg in event:
                    for q in np.arange(0,len(tickers)):
                        ticker=msg.getElement('securityData').getValue(int(q)).getElement('security').getValue()

                        blk=msg.getElement('securityData').getValue(int(q)).getElement('fieldData').getElement(field)
                        output=pd.DataFrame(index=np.arange(0,blk.numValues()),columns=['dummy'])
                        for i in np.arange(0,blk.numValues()):
                            for j in np.arange(0,blk.getValue(0).numElements()):
                                output.at[i,str(blk.getValue(int(i)).getElement(int(j)).name())]=blk.getValue(int(i)).getElement(int(j)).getValue(0)
                        output['ticker']=ticker
                        if len(res)==0:
                            res=output.copy()
                        else:
                            res=pd.concat([res,output],axis=0)

            if event.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                break

    finally:
        # Stop the session
        session.stop()
    res=res.drop('dummy',axis=1)
    res=res.set_index('ticker')
    return res


def bdh_intraday_bar(ticker,start,end,interval,evt_type='TRADE'):
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost("localhost")
        sessionOptions.setServerPort(8194)
    
        #print "Connecting to %s:%s" % (options.host, options.port)
        # Create a Session
        session = blpapi.Session(sessionOptions)
        
        # Start a Session
        if not session.start():
            #print "Failed to start session."
            return
        
        try:
            session.openService("//blp/refdata")
            refDataService = session.getService("//blp/refdata")
            request = refDataService.createRequest("IntradayBarRequest")
            
        
            request.set("security",ticker)
            request.set("startDateTime", start-GMT_adj) # convert to GMT as BBG only accept this
            request.set("endDateTime", end-GMT_adj)
            request.set("interval", interval)
            request.set("eventType", evt_type)
            session.sendRequest(request)
            
            session.sendRequest(request)
            
            res=pd.DataFrame()
            # Process received events
            while(True):
                # We provide timeout to give the chance for Ctrl+C handling:
                event = session.nextEvent()
                if event.eventType()  in [5,6]:
                    '''eventtype and eventType=6 or 5 are the key to differentiate, need to extract data from event now'''
        
                    output=pd.DataFrame()
                    for msg in event:
                        fieldDataArray = msg.getElement('barData').getElement('barTickData')
                        # use this to extract intraday data
                        fieldDataArray.getValue(1).getElement(7).getValue()
                        # to b finished
                        size = fieldDataArray.numValues()
                        fieldDataList = [fieldDataArray.getValueAsElement(i) for i in range(0,size)]
                        outDates = [x.getElement(0).getValue() for x in fieldDataList]
                        output = pd.DataFrame(index=outDates,columns=['open','high','low','close','volume'])
                        for i,strD in enumerate(output.columns):
                            outData = []
                            for x in fieldDataList:
                                try:
                                    outData.append(x.getElement(i+1).getValue())
                                except:
                                    outData.append(np.nan)
                            output[strD] = outData
                        
                        output.replace('#N/A History',np.nan,inplace=True)
                        output.index = pd.to_datetime(output.index)
                        output.index=output.index.map(lambda x: x+GMT_adj)
                        output['ticker']=ticker
                        output['evt_type']=evt_type
                        output['interval']=interval
                        if len(res)==0:
                            res=output.copy()
                        else:
                            res=pd.concat([res,output],axis=0)
                if event.eventType() == blpapi.Event.RESPONSE:
                    # Response completly received, so we could exit
                    break
        finally:
            # Stop the session
            session.stop()    
            
        res.index.name='datetime'
        return res



if __name__ == "__main__":
    print ('ok')
    
    
    
    
    
        

    
    # bfs=['1BF']#,'2BF']
    # check=pd.DataFrame()
    # for bf in bfs:
    #     overrides={'BEST_FPERIOD_OVERRIDE':bf}
    #     res=bdh(['700 HK Equity'],['BEST_EPS'],datetime(2005,1,1),datetime(2018,9,2),overrides=overrides).loc['700 HK Equity']

    # #---- test bdp
    # res=bdp(['700 HK Equity','5 HK Equity','7203 JP Equity'],['SHORT_NAME','GICS_SECTOR_NAME'])


    # overrides={'MARKET_DATA_OVERRIDE':'TURNOVER',
    #           'CALC_INTERVAL':'63D',
    #           'CRNCY':'USD',}
    # res=bdp(['700 HK Equity','5 HK Equity','7203 JP Equity'],['INTERVAL_AVG','INTERVAL_HIGH','INTERVAL_MEDIAN'],
    #         overrides=overrides
    #         )

    # #---- test bds
    # overrides={'END_DATE_OVERRIDE':datetime(2018,5,31).strftime('%Y%m%d')}
    # res=bds(['XIN9I Index','HSCEI Index'],'INDX_MWEIGHT_HIST',overrides=overrides)























