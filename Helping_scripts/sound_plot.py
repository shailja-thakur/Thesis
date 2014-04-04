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

    
    
    #Reads Sound csv Files to filter sound events
    db_paths='/media/New Volume/IPSN/sense/5_6Nov/dbs/c006_room_db.csv'
    #dbs=['c003_room_db.csv','c004_room_db.csv','c005_room_db.csv','c006_room_db.csv']
    dbs=['c006_room_db.csv']

    # # Step1: Edge Detection
    # edge_list_df = edge_detection(df_l, df_p, l_phase, p_phase)
    # #sys.exit()

    # # Step2: Edge Matching
    # time_slices = edge_matching(df_l, df_p, edge_list_df)
    df_b=DataFrame()
    df_a=DataFrame()
    
    list_db=pd.read_csv(db_paths)
    print list_db['time']
    s_tstamp=[]
    
    try:
        s_tstamp = [dt.datetime.fromtimestamp(x) for x in list_db['time']]
    except Exception,e:
        print e
        
            
            
    
            
    print s_tstamp
    #fig = plt.gcf()

            
    # # row1
    # ax1 = plt.subplot(3, 1, 1)
    # # ax1.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
    # plt.plot(s_tstamp, list_db['db'])
    # plt.title("Sound "  )
    # plt.ylabel("Sound in db")

                

           
            
    #pp = PdfPages('/media/New Volume/IPSN/sense/5_6Nov/dbs/plots/actual_sound_events/'+db+'_'+str(s_tstamp[t])+'.pdf')        
    #plt.savefig(pp, format='pdf')
    #pp.savefig()
    #pp.close()
    plt.show()
