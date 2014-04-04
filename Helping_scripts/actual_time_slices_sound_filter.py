import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.dates as md
from pandas import *
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as plt
import time

"""
---------------------------------------
Main Program
---------------------------------------
"""

if __name__ == '__main__':

    
    #actual start_time and end_time
    start_time=['20:59:26-05/12/2013','17:49:31-05/12/2013','04:06:48-05/12/2013','17:00:18-05/12/2013','17:05:04-05/12/2013','17:28:31-05/12/2013','18:14:13-05/12/2013','18:20:10-05/12/2013','18:57:24-05/12/2013','20:01:17-05/12/2013',
                '20:51:30-05/12/2013','21:00:07-05/12/2013','22:12:13-05/12/2013','16:51:29-05/12/2013',
                '17:08:40-05/12/2013','17:49:31-05/12/2013','20:50:59-05/12/2013','16:06:34-05/12/2013',
                '17:50:59-05/12/2013','19:53:05-05/12/2013','20:12:46-05/12/2013','20:12:31-05/12/2013','20:51:40-05/12/2013',
                '17:28:35-05/12/2013','20:52:32-05/12/2013','19:24:29-05/12/2013','20:50:00-05/12/2013']
    end_time=['21:45:49-05/12/2013','18:40:29-05/12/2013','16:53:25-05/12/2013','17:04:05-05/12/2013','17:10:54-05/12/2013','17:59:32-05/12/2013','18:16:07-05/12/2013','18:54:49-05/12/2013','19:51:12-05/12/2013','20:29:27-05/12/2013',
                '20:52:23-05/12/2013','21:44:00-05/12/2013','23:10:28-05/12/2013','23:08:44-05/12/2013',
                '17:33:56-05/12/2013','20:26:38-05/12/2013','23:08:46-05/12/2013','17:46:43-05/12/2013',
                '19:19:32-05/12/2013','20:09:39-05/12/2013','20:22:33-05/12/2013','20:22:33-05/12/2013','22:46:53-05/12/2013',
                '17:34:13-05/12/2013','21:15:32-05/12/2013','20:36:18-05/12/2013','21:32:00-05/12/2013']
    

    s_tstamp=[]
    e_tstamp=[]
    s_tstamp = [long(time.mktime(time.strptime(start_time[x],"%H:%M:%S-%d/%m/%Y"))*1000) for x in range(len(start_time))]
    e_tstamp = [long(time.mktime(time.strptime(end_time[x],"%H:%M:%S-%d/%m/%Y"))*1000) for x in range(len(end_time))]
        
    print "timestamp corresponding to start time:\n"+str(s_tstamp)
    print "timestamp corresponding to end time:\n"+str(e_tstamp)
    #Reads Sound csv Files to filter sound events
    db_paths='/media/New Volume/IPSN/sense/5_6Nov/dbs/'
    dbs=['c003_room_db.csv','c004_room_db.csv','c005_room_db.csv','c006_room_db.csv']
   

    # # Step1: Edge Detection
    # edge_list_df = edge_detection(df_l, df_p, l_phase, p_phase)
    # #sys.exit()

    # # Step2: Edge Matching
    # time_slices = edge_matching(df_l, df_p, edge_list_df)
    df_b=DataFrame()
    df_a=DataFrame()
    print 'start time to next 1 minutes:'
    for t in range(len(s_tstamp)):
        from_t=s_tstamp[t]
        end_t=e_tstamp[t]
        
        i=0
        fig, axes = plt.subplots(nrows=2, ncols=4)
        for db in dbs:
            
            
            print 'opening db file'+db_paths+db
            list_db=pd.read_csv(db_paths+db)
            
            #print 'Filtering sound samples before event'
            db_list=list_db.index[(list_db['time']>= from_t-100000) & (list_db['time']<= from_t)]
            db_list_before=list_db.ix[db_list]
            # print "before data" 
            # print db_list_before

            #print 'Filtering sound samples between events'
            db_list=list_db.index[(list_db['time']>=from_t) & (list_db['time'] <= end_t)]
            db_list_between=list_db.ix[db_list]
           

            #print 'Filtering sound events after event'
            db_list=list_db.index[(list_db['time']>=from_t) & (list_db['time'] <= from_t+100000)]
            db_list_after=list_db.ix[db_list]
            # print 'after data'
            # print db_list_after


            df_before=DataFrame(db_list_before['db'],columns=['Before_'+db+'_'+str(s_tstamp[t])])
            df_after=DataFrame(db_list_after['db'],columns=['After_'+db+'_'+str(e_tstamp[t])])
            df_between=DataFrame(db_list_between['db'],columns=['Between_'+db+'_'+str(s_tstamp[t])+'_'+str(e_tstamp[t])])
            
            mean_bf_db=float(df_before.mean())
            sd_bf_db=float(df_before.std())

            mean_bt_db=float(df_between.mean())
            sd_bt_db=float(df_between.std())

            mean_af_db=float(df_after.mean())
            sd_af_db=float(df_after.std())
            print "start_time:"+str(start_time[t])
            print "end time:"+str(end_time[t])
            print 'Before_'+'Between_'+'After'
            
            print mean_bf_db,mean_bt_db,mean_af_db
            print sd_bf_db,sd_bt_db,sd_af_db


            #t_before = np.array([dt.datetime.fromtimestamp(x/1000) for (x) in db_list_before['time']])
            #t_after = np.array([dt.datetime.fromtimestamp(x/1000) for (x)in db_list_after['time']])
            state=''

            if df_before.empty == False:
                
                state+='T'
                #df_b=pd.concat([df_b,df_before],ignore_index=True)
            else:
                
                state+='F'
            if df_after.empty == False:
               
                state+='T'
                #df_b=pd.concat([df_b,df_after],ignore_index=True)
            else:
                
                state+='F'

            #print state

            fig = plt.gcf()

            if state=='TT':
                # row1
                ax1 = plt.subplot(3, 1, 1)
                 # ax1.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
                plt.plot( df_before)
                plt.title("Sound Before_ " +str(s_tstamp[t])+'_'+db )
                plt.ylabel("Sound in db")

                ax2 = plt.subplot(3, 1, 2)
                # ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
                plt.plot( df_between)
                plt.title("Sound Between_ " + str(s_tstamp[t])+'_'+str(e_tstamp[t])+'_'+db)
                plt.ylabel("Sound in db")

                ax2 = plt.subplot(3, 1, 3)
                # ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
                plt.plot( df_after)
                plt.title("Sound After_ " + str(e_tstamp[t])+'_'+db)
                plt.ylabel("Sound in db")


                

            elif (state == 'TF') | (state == 'FT'):
                if state=='TF':
                    plt.plot( df_before)
                    plt.title("Sound Before_ " +str(s_tstamp[t])+'_'+db )
                    plt.ylabel("Sound in db") 
                   

                if state=='FT':
                    plt.plot(df_after)
                    plt.title("Sound After_ " +str(e_tstamp[t])+'_'+db )
                    plt.ylabel("Sound in db")
            
            pp = PdfPages('/media/New Volume/IPSN/sense/5_6Nov/dbs/plots/actual_sound_events/'+db+'_'+str(s_tstamp[t])+'.pdf')        
            plt.savefig(pp, format='pdf')
            pp.savefig()
            pp.close()
            plt.show()
