# -*- coding: utf-8 -*-

import utilities.constants as uc
import pandas as pd
import os
import smtplib,ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import time
import mimetypes
from email import encoders
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
import numpy as np
import pdb
import win32com.client as wc
from datetime import datetime
import feather


def merge_pdf(path,input_list=['name_1','name_2','name_3'],output_name='output'):
    '''
    use this function to merge all pdf in one path and output the merged on in the same location
    '''
    from PyPDF2 import PdfFileMerger
    pdfs =[path+'%s.pdf' % (x) for x in input_list]
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))
    with open(path+'%s.pdf' % (output_name), 'wb') as fout:
        merger.write(fout)
    return None


def now_time():
    return datetime.now()

def today_date():
    return datetime.strptime(today_string(),'%Y-%m-%d')
def yesterday_date():
    return datetime.strptime(yesterday_string(),'%Y-%m-%d')
def yesterday_string(strf='%Y-%m-%d'):
    return (datetime.today()-pd.offsets.BDay()).strftime(strf)
def today_string(strf='%Y-%m-%d'):
    return datetime.today().strftime(strf)


def dump_data(frame,name,folder='misc',
              dump_format='csv'#can do feather or other format
              ):
    if frame.empty:
        print('Frame is empty')
        return
    filename=uc.python_dump%(folder,name,today_string('%Y%m%d'))
    if os.path.exists(filename):
        i = 0
        while True:
            filename =  uc.python_dump%(folder,name+"_"+str(i),today_string('%Y%m%d'))
            if not os.path.exists(filename):
                break
            i = i+1
        print('%s already exists - printing in %s' % (
        uc.python_dump%(folder,name,today_string('%Y%m%d')),
        (name+"_"+str(i),today_string('%Y%m%d'))))
    if dump_format=='csv':
        frame.to_csv(filename)
    elif  dump_format=='feather':
        feather.write_dataframe(frame,filename.replace('.csv','.feather'))
        
def dump_png(fig,name,folder='Misc',dpi=200):

    filename=uc.python_dump_png%(folder,name,today_string('%Y%m%d'))
    if os.path.exists(filename):
        i = 0
        while True:
            filename =  uc.python_dump_png%(folder,name+"_"+str(i),today_string('%Y%m%d'))
            if not os.path.exists(filename):
                break
            i = i+1
        print('%s already exists - printing in %s' % (
        uc.python_dump%(folder,name,today_string('%Y%m%d')),
        (name+"_"+str(i),today_string('%Y%m%d'))))
    fig.savefig(filename,bbox_inches='tight',dpi=dpi)



def send_mail_comobj(To, Subject, Body, Bcc='', attachment=[], auto=False,pictures={},subject_pre_fix='(CS QSS Auto-email)'):
	outlook=wc.Dispatch("Outlook.Application")
	mail=outlook.CreateItem(0)
	mail.To=To
	if Bcc!='':
		mail.Bcc=Bcc
	mail.Subject=subject_pre_fix+' '+Subject
	mail.HTMLBody=Body
	
	if len(attachment)!=0:
		for file in attachment:
			mail.Attachments.Add(Source=file)

	if len(pictures)!=0:
		for k,v in pictures.items():
			attachment_pic = mail.Attachments.Add(v)
			attachment_pic.PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F", k)
	if auto:
		mail.Send()
	else:
		mail.Display(True)
		
	return None

def send_mail(subject,body,sendto,copyto=[],blindcopyto=[],
			  pictures=None,attachments=None,
			  #distinguish_users=False,
			  #using_gmail_host=False
			  ):

	'''
    pictures are dict with id and path. Should be used in conjuction with HTML builder in utilities.display
    attachements are dict with attachement name and path. Can be used independently
	'''

	sender="quant-systematic.autoemail@credit-suisse.com"

	msg_root = MIMEMultipart('related')
	msg_root['From'] = sender
	msg_root['To'] = ','.join(sendto)
	msg_root['Cc'] = ','.join(copyto)
# msg_root['BCc'] = ','.join(blindcopyto)
	msg_root['Subject'] = subject

	msg_alt = MIMEMultipart('alternative')
	msg_root.attach(msg_alt)

	msg_text = MIMEText(body, 'html')
	msg_alt.attach(msg_text)

	if pictures is not None:
		for pic_name, pic_path in list(pictures.items()):
			fp = open(pic_path, 'rb')
			msg_image = MIMEImage(fp.read())
			fp.close()
			msg_image.add_header('Content-ID', '<%s>' % pic_name)
			msg_root.attach(msg_image)

	if attachments is not None:
		for attachment_name,attachment in attachments.items():
			ctype, encoding = mimetypes.guess_type(attachment)
			if ctype is None or encoding is not None:
				ctype = "application/octet-stream"
				
			maintype, subtype = ctype.split("/", 1)

			if maintype == "text":
				fp = open(attachment)
				# Note: we should handle calculating the charset
				attachment = MIMEText(fp.read(), _subtype=subtype)
				fp.close()
			elif maintype == "image":
				fp = open(attachment, "rb")
				attachment = MIMEImage(fp.read(), _subtype=subtype)
				fp.close()
			elif maintype == "audio":
				fp = open(attachment, "rb")
				attachment = MIMEAudio(fp.read(), _subtype=subtype)
				fp.close()
			else:
				fp = open(attachment, "rb")
				attachment = MIMEBase(maintype, subtype)
				attachment.set_payload(fp.read())
				fp.close()
				encoders.encode_base64(attachment)

			attachment.add_header("Content-Disposition", "attachment", filename='%s' % (attachment_name))
			msg_root.attach(attachment)

	text = msg_root.as_string()

	s = smtplib.SMTP(uc.smtp)
	s.sendmail(sender,sendto+copyto+blindcopyto,text)
	s.quit()

def quick_auto_notice(msg=''):
	#send_mail(msg,'',uc.dl['self'])
	send_mail_comobj(';'.join(uc.dl['self']), msg, '',subject_pre_fix='(Quick Notificaiton)',auto=True)


def iterate_csv(path,timeliness=[False,pd.Timedelta('30 minutes'),False]
                                                    ,override_email_content='',override_email_subject='',
                                                    iterate_others=[False,'.png or all for everything']):
    email_sent_to=uc.dl['self']
    output=[]
    for filename in os.listdir(path):
        if not iterate_others[0]:
            if '.csv' in filename:
                if not timeliness[0]:
                    current_file_name=str(filename[:filename.find('.csv')])
                    output.append(current_file_name)
                else:
                    check=pd.to_datetime(time.ctime(os.path.getmtime(path+filename)))
                    now=datetime.now()
                    diff=now-check
                    if diff>timeliness[1]:
                        print ('%s result has not been updated for at least %s. Program terminated' % (filename,diff))
                        if timeliness[2]:
                            #send email notification
                            if override_email_content=='':
                                send_mail(subject='spot and fx refresh failed due to timeliness issue',body='',sendto=email_sent_to)
                            else:
                                send_mail(subject=override_email_subject,body=override_email_content,sendto=email_sent_to)
                        return False
                    else:
                        current_file_name=str(filename[:filename.find('.csv')])
                        output.append(current_file_name)
        else:
            if iterate_others[1]!='all':
                if iterate_others[1] in filename:
                    current_file_name=str(filename[:filename.find(iterate_others[1])])
                    output.append(current_file_name)
            else:
                current_file_name=str(filename)
                output.append(current_file_name)
    return output



def iterate_file_time(path,file_type='.csv'):
    file_time_dict={}
    for filename in os.listdir(path):
        if file_type in filename:
            file_time_dict[filename]=pd.to_datetime(time.ctime(os.path.getmtime(path+filename)))
    return pd.Series(file_time_dict)

def mask_symmetric_df(data):
    mask=pd.DataFrame(index=data.columns,columns=data.columns).fillna(0)
    mask=mask.where(np.triu(np.ones(mask.shape)).astype(np.bool)).applymap(lambda x: np.nan if not np.isnan(x) else True)
    pair_index=mask.stack()
    return data.stack().reindex(pair_index.index).unstack()


def display_df_all_rows_columns(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # more options can be specified
        print(df)


def trigger_internet():
    #This is for CS only. Run this before any webscraping and see if it helps
    url="https://www.google.com/"
    ie = wc.Dispatch("InternetExplorer.Application")
    ie.visible=1
    ie.Navigate(url)
    print ('internet triggered')
    return None



def retry_func(func, retries=5,
               error_msg_print='Unknown error try again.',
               error_msg_email='Some func error, try again',
               to_trigger_internet=False
               ):
    '''
    func without ()
    '''
    for retry in np.arange(0,retries):
        try:
            func()
            return True
            #break
        except:
            print (error_msg_print)
            quick_auto_notice('%s. (retry: %s)' % (error_msg_email,retry))
            if to_trigger_internet:
                trigger_internet()
        #failure after many retires
        if retry+1==retries:
             raise TypeError("Error raised")
    return False


def is_fresh(file_name,level_h=8):
    if not os.path.isfile(file_name):
        return False
    else:
        last_update_time=pd.to_datetime(time.ctime(os.path.getmtime(file_name)))
        now=datetime.now()
        diff=(now-last_update_time).total_seconds()/3600
        return False if diff>level_h else True
    
def drop_zero_row(df):
    index_to_use=df.count(1)[df.count(1)!=0].index
    return df.loc[index_to_use]



if __name__ == "__main__":
    
    # ---- main
    print ('ok')
    # from utilities.display import HTML_Builder
    # html=HTML_Builder()
    # html.insert_title('Title')
    # html.insert_body('Content')
    # html.insert_body( 'Content (not bold)',bold=False)
    # picture="Z:\\elita\\Crowding\\weekly\\aspc.png"
    # html.insert_picture(picture)
    # To='dave.yin@credit-suisse.com'
    # Subject='test com auto email'
    # Body=html.body#'test body'
    # #Bcc='daveyin2@bloomberg.net;elai95@bloomberg.net'
    # Bcc=''
    # attachment=[
    #             "Z:\\elita\\Crowding\\weekly\\20200417_CSQSS_StockCrowding.pdf",
    #             "Z:\\elita\\Crowding\\weekly\\20200331_CSQSS_stockCrowding_longs.csv",
    #             "Z:\\elita\\Crowding\\weekly\\20200331_CSQSS_stockCrowding_shorts.csv",
    #             ]
    # send_mail_comobj(To, Subject, Body, Bcc=Bcc,attachment=attachment, auto=True)
    #check=get_bbg_nice_compo_hist('HSI Index',datetime(2019,7,11),load_local=True)
    #check_usual=get_bbg_usual_col(check.index.tolist())
    #best_eps,best_eps_chg=get_bbg_calendarized_level_and_revision('700 HK Equity','BEST_SALES',63)
    #('test','body',['davehanzhang@gmail.com'],using_gmail_host=True)

