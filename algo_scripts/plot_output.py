"""
Script to plot the output of the algorithm:
    1. Time Slices
    2. Location Prediction
    3. Activity Prediction
"""

import sys
import numpy as np
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
#folder = 'CompleteDataSets/Apartment/Evaluation/exp10/'

day = sys.argv[1]
room = sys.argv[2]
folder = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/MeterData/10_11/'  
csv_p = folder + 'Power.csv'
csv_l = folder + 'Light.csv'
#s_csv = folder + 'Sound3.csv'
#w_csv = folder + 'Wifi3.csv'

df_p = pd.read_csv(csv_p)
df_l = pd.read_csv(csv_l)

#df_s = pd.read_csv(s_csv)
# df_w = pd.read_csv(w_csv)

output = OUTPUT + day + '/' + room + '.csv'
op_df = pd.read_csv(output)

# Plotting parameters
colors = ['0.5', 'b', 'm', 'r', 'c', 'g', '0.6', 'k']
fontsz = 15
font = {'family': 'Arial',
        # 'weight': 'bold',
        'size': fontsz}

mpl.rc('font', **font)
df_l = df_l.ix[df_l['time'] >= 1392035400]


# Plotting output data
t_p = pd.DatetimeIndex(pd.to_datetime(df_p['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
df_p.set_index(t_p, inplace=True)
t_l = pd.DatetimeIndex(pd.to_datetime(df_l['time'], unit='s'), tz='UTC').tz_convert(TIMEZONE)
df_l.set_index(t_l, inplace=True)

# print df_p.head()
# print df_l.head()
print op_df


# Plotting meter data
# fig, ax1 = plt.subplots()
# ax1.xaxis.set_major_formatter(md.DateFormatter('%H:%M', timezone(TIMEZONE)))
# ax1.plot(t_p, df_p['power'], 'b-', linewidth=2)
# ax1.set_xlabel('Time')
# Make the y-axis label and tick labels match the line color.
# ax1.set_ylabel('Power (in Watts)', color='b')
# for tl in ax1.get_yticklabels():
#     tl.set_color('b')
# ax2 = ax1.twinx()
# ax2.plot(t_l, df_l['lightpower'], 'g-', linewidth=2)
# ax2.set_ylabel('Power (in Watts)', color='g')
# for tl in ax2.get_yticklabels():
#     tl.set_color('g')
# fig.autofmt_xdate()
# plt.savefig('../E-energy2014/images/meter.png', dpi=100, bbox='tight')
# plt.tight_layout()
# plt.show()

# sys.exit(1)

# Plotting time slices over meter data
fig, ax1 = plt.subplots()
ax1.xaxis.set_major_formatter(md.DateFormatter('%H:%M', timezone(TIMEZONE)))
ax1.set_ylim([260, 500])
# ax1.plot(t_p, df_p['powerphase2'], 'b-', linewidth=1)
# ax1.set_xlabel('Time')
# # Make the y-axis label and tick labels match the line color.
# ax1.set_ylabel('Power (in Watts)', color='k')
# for tl in ax1.get_yticklabels():
#     tl.set_color('k')
# ax1.grid(True, which="both")

#ax2 = ax1.twinx()
ax1.plot(t_l, df_l['lightphase1'], 'g-', linewidth=2)
ax1.set_ylabel('Power (in Watts)', color='k')
for tl in ax1.get_yticklabels():
    tl.set_color('k')
fig.autofmt_xdate()
ax1.grid(True, which="both")

for i, idx in enumerate(op_df.index):

    start_time = op_df.ix[idx]['start_time']
    end_time = op_df.ix[idx]['end_time']
    magnitude = op_df.ix[idx]['magnitude']
    print start_time, end_time, magnitude
    time = [datetime.datetime.fromtimestamp(x, timezone(TIMEZONE))
            for x in range(start_time, end_time + 1)]
    points = [magnitude] * len(time)
    points[0] = points[-1] = 0
    plt.plot(time, points, color=colors[i], linestyle='--', linewidth=3)

fig.autofmt_xdate()
leg = plt.legend(loc='lower right', fancybox=True, prop={'size': 18})
plt.savefig(OUTPUT + 'images/output_time_slice.png', dpi=100, bbox='tight')
plt.tight_layout()
plt.show()

sys.exit(1)

# Annotating time slices
fig = plt.gcf()
ax = plt.gca()
ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M', timezone(TIMEZONE)))
x_xcord = [600, 600]
for i, idx in enumerate(op_df.index):
    start_time = op_df.ix[idx]['start_time']
    end_time = op_df.ix[idx]['end_time']
    magnitude = op_df.ix[idx]['magnitude']
    location = op_df.ix[idx]['room']
    appliance = op_df.ix[idx]['appliance']
    annotation_str = '{' + location + ',' + appliance + '}'

    annotation_xpos = datetime.datetime.fromtimestamp(
        long(start_time), timezone(TIMEZONE))
    print annotation_xpos

    time = [datetime.datetime.fromtimestamp(x, timezone(TIMEZONE))
            for x in range(start_time, end_time + 1)]
    points = [magnitude] * len(time)
    points[0] = points[-1] = 0
    plt.plot(time, points, color=colors[i], linewidth=3)
    ax.set_xlabel('Time', color='k')
    ax.set_ylabel('Power (in Watts)', color='k')
# plt.annotate(annotation_str, xy=(annotation_xpos, magnitude + 500),
# xytext=(annotation_xpos, magnitude + 500), rotation=90)
#     plt.xlabel("Time", fontsize=fontsz)
#     plt.ylabel("Power (in Watts)", fontsize=fontsz)
ax.grid(True, which="both")

fig.autofmt_xdate()
plt.savefig(OUTPUT + 'images/annotated.png', dpi=100, bbox='tight')
plt.tight_layout()
plt.show()


# ------------------------------------------
# fig, axes = plt.subplots(nrows=1, ncols=2)
# df_l['lightpower'].plot(ax=axes[0, 0])
# axes[0, 0].set_title('light')
# df_p['power'].plot(ax=axes[0, 1])
# axes[0, 1].set_title('power')

# df_p.plot(x='date_time', y='powep')


# For plotting time slices
# time_range = range(op_df.ix[0].['start_time'], op_df.ix[1]['end_time'] + 1)
# for idx in op_df.index:
#     s_time = op_df.ix[idx]['start_time']
#     e_time = op_df.ix[idx]['end_time']
#     mag = op_df.ix[idx]['magnitude']
