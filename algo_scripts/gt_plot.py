"""
Script to plot the output of the algorithm:
    1. Time Slices
    2. Location Prediction
    3. Activity Prediction
"""

import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pytz import timezone
import datetime
import matplotlib.dates as md
from sklearn.metrics import *
from CONFIGURATION import TIMEZONE
import localize as cl
from path import *
from scipy import *
from evaluate_timeslices import *
#folder = 'CompleteDataSets/Apartment/Evaluation/exp10/'


day = sys.argv[1]
ch = sys.argv[2]
#room = sys.argv[2]

dates = {'first' : '10_11', 'second' : '11_12','third' : '12_13','fourth' : '13_14', 'fifth':'14_15'}
folder = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/MeterData/CompleteDataSets/Hostel/' + dates[day] + '/'  
csv_p = folder + 'Power.csv'
phase = ['lightphase1','lightphase2','lightphase3','powerphase2','powerphase3']

df_p = pd.read_csv(csv_p)
# #df_p = df_p.ix[df_p['time'] >= 1392035400]
t_p = pd.DatetimeIndex(pd.to_datetime(df_p['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
start = t_p[0]
end = t_p[-1]

#output = OUTPUT + day + '/' + room + '.csv'

# p_df = pd.read_csv(csv_p)
# t_p = pd.DatetimeIndex(pd.to_datetime(df_p['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
# start = t_p[0]

# end = t_p[-1]


# Plotting parameters
colors = ['0.5', '#8B2500', 'm', 'r', '#FFA500', 'g', '#388E8E']
fontsz = 15
font = {'family': 'Arial',
        # 'weight': 'bold',
        'size': fontsz}

mpl.rc('font', **font)


#print df



#Plotting time slices over meter data

def plot_output(output_df):

    

    

    number_of_subplots = len(phase)
    fig, ax1 = plt.subplots()
    ax1 = plt.gcf()
    v = 0
    for p in phase:
        print p
        v = v + 1
        ax1 = plt.subplot(number_of_subplots,1,v)
    
        ax1 = plt.gca()
        ax1.xaxis.set_major_formatter(md.DateFormatter('%H:%M', timezone(TIMEZONE)))
    
        if 'light' in p:
            p_df = pd.read_csv(folder + 'Light.csv')
            
            
            if 'lightphase1' in p:
                #print len(p_df)
                ids = p_df.index[p_df[p] == 0.0]
                #print ids
                if len(ids) > 0:
                    #print p_df.ix[ids][p]
                    p_df = p_df.ix[p_df.index - ids] 
                    
                #print p_df[p].min()
                p_df[p] = p_df[p] - p_df[p].min()
                #print p_df[p].min()
            t_p = pd.DatetimeIndex(pd.to_datetime(p_df['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
            


        else:
            p_df = pd.read_csv(folder + 'Power.csv')
            t_p = pd.DatetimeIndex(pd.to_datetime(p_df['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
            if 'powerphase2' in p:
               
                #print len(p_df)
                ids = p_df.index[p_df[p] == 0.0]
                #print ids
                if len(ids) > 0:
                    #print p_df.ix[ids][p]
                    p_df = p_df.ix[p_df.index - ids] 
                    
                #print p_df[p].min()
                p_df[p] = p_df[p] - p_df[p].min()
                #print p_df[p].min()
                t_p = pd.DatetimeIndex(pd.to_datetime(p_df['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
            #     ids = p_df['powerphase2'].idxmin()
            #     p_df = p_df.ix[p_df.index - [ids]] 
            #     if p_df[p].min() != 0:
            #         p_df[p] = p_df[p] - p_df[p].min()

            #     idx_min = p_df.index[p_df.powerphase2 <= 350]
            #     idx_max = p_df.index[p_df.powerphase2 >= 300]
            #     p_df.loc[idx_max, 'powerphase2'] = p_df.ix[idx_min]['powerphase2'].mean()
            #     t_p = pd.DatetimeIndex(pd.to_datetime(p_df['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
            #     # ids = p_df.index[p_df.powerphase2 > 500]
            #     # p_df.ix[ids][p] = p_df[p].mean()
            # else:
            #     t_p = pd.DatetimeIndex(pd.to_datetime(p_df['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
        # print len(p_df), len(t_p)
        # print 'min plotting',p_df[p].min()
        ax1.plot(t_p, p_df[p], 'b-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(p, color='b')

        fig.autofmt_xdate()
        #ax1.grid(True, which="both")

        op_df = output_df.ix[output_df.phase == p]
    
        op_df.index = arange(0, len(op_df))
        #ax1.set_ylim([0, max(p_df[p])])
        ax1.set_xlim([start, end])
    
        print op_df
        for i, idx in enumerate(op_df.index):

            start_time = op_df.ix[idx]['start_time']
            end_time = op_df.ix[idx]['end_time']
            magnitude = op_df.ix[idx]['magnitude']
            location = op_df.ix[idx]['room']
            appliance = op_df.ix[idx]['appliance']
       
            time = [datetime.datetime.fromtimestamp(x, timezone(TIMEZONE))
                for x in range(start_time, end_time + 1)]
            annotation_xpos = datetime.datetime.fromtimestamp(
                long(start_time), timezone(TIMEZONE))
            print annotation_xpos, start_time
            #print magnitude, len(time)
            points = [magnitude] * len(time)
            points[0] = points[-1] = 0
            plt.plot(time, points, color=colors[int(op_df.ix[idx]['room'])],linestyle = '-', linewidth=2)

    fig.autofmt_xdate()
    plt.legend(loc='lower right', fancybox=True, prop={'size': 18})
    plt.savefig(OUTPUT + 'images/output_time_slice.png', dpi=100, bbox='tight')
    #plt.tight_layout()
    plt.show()




# PLOTTING GROUND TRUTH

def plot_gt(df):

    
    number_of_subplots = len(phase)
    fig, ax = plt.subplots()
    fig = plt.gcf()

    v = 0
    for p in phase:
        print p
        v = v + 1
        ax = plt.subplot(number_of_subplots,1,v)
    


        ax = plt.gca()
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M', timezone(TIMEZONE)))
        x_xcord = [600, 600]
        op_df = df.ix[df.phase == p]
    
        op_df.index = arange(0, len(op_df))
        ax.set_ylim([0, 100])
        ax.set_xlim([start, end])
    
        print op_df
        for i, idx in enumerate(op_df.index):
            start_time = op_df.ix[idx]['start_time']
            end_time = op_df.ix[idx]['end_time']
            magnitude = op_df.ix[idx]['magnitude']
            location = op_df.ix[idx]['room']
            appliance = op_df.ix[idx]['appliance']
            annotation_str = '{' + str(location) + ',' + str(appliance) + '}'
            #print start_time, end_time, magnitude, location, appliance
            annotation_xpos = datetime.datetime.fromtimestamp(
            long(start_time), timezone(TIMEZONE))

            #print annotation_xpos, start_time, end_time
        

            time = [datetime.datetime.fromtimestamp(x, timezone(TIMEZONE))
                for x in range(start_time, end_time)]
            #print magnitude, len(time)
            points = [magnitude] * len(time)
            points[0] = points[-1] = 0
        
            plt.plot(time, points, color=colors[int(op_df.ix[idx]['room'])], linewidth=3)
            ax.set_xlabel('Time', color='k')
            ax.set_ylabel(p, color='k')
    # plt.annotate(annotation_str, xy=(annotation_xpos, magnitude + 500),
    # xytext=(annotation_xpos, magnitude + 500), rotation=90)
    #     plt.xlabel("Time", fontsize=fontsz)
    #     plt.ylabel("Power (in Watts)", fontsize=fontsz)
            ax.grid(True, which="both")

    fig.autofmt_xdate()
    plt.savefig(OUTPUT + 'images/annotated.png', dpi=100, bbox='tight')
    #plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    filename_op = OUTPUT  + 'predicted_timeslices_' + day + '.csv'
    time_slices = pd.read_csv(filename_op)

    gt_path = GROUNDTRUTH_PATH + 'gt_timeslices_' + day + '.csv'
    gt = pd.read_csv(gt_path)


    if str(ch) == 'gt':
        plot_gt(gt)


    elif str(ch) == 'output':

        for ph in phase:
            time_slice = time_slices[time_slices.phase == ph]
            GT = gt[gt.phase == ph]
            calc_ts_accuracy(time_slice, GT)
        plot_output(time_slices)

        
    else:
        print 'Enter (gt/output) as second argument to plot the ground truth data or output data '

    
