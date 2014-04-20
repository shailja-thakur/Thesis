import pandas as pd
from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
from path import *
import sys
import datetime
import os
from af_hostel import *
import glob
import re
rm = sys.argv[1]
day = sys.argv[2]
# Enter day as 10_11/11_12/12_13/13_14.....
path = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/MeterData/' 
meters = ['Light', 'Power']
metadata_rooms = {
    "rooms" : ['1', '2', '3', '4', '5', '6'],
    "power" : [['powerphase1','lightphase1','powerphase2'], ['powerphase1','powerphase2','lightphase1'],  ['powerphase3','powerphase2','lightphase2'], ['powerphase3','powerphase2','lightphase2'] 
    		, ['powerphase3','powerphase2','lightphase3'],  ['powerphase3','powerphase2','lightphase3'] ],
}
df = pd.DataFrame(metadata_rooms)

room = df.index[df['rooms'] == rm]
phase_power =  df.loc[room, 'power']
phases = str(phase_power[int(rm) - 1]).strip(']').strip('[').split(',')
#standby_battery_df = pd.read_csv(PATH + '/performance_standby/battery/Battery.csv')
line_style = ['r', 'g', 'b']
number_of_subplots = 3

fig, ax = plt.subplots()
v = 0
i = 0
for phase,style in zip(phases, line_style):
	v = v + 1
	ax = plt.subplot(number_of_subplots,1,v)
	hfmt = dates.DateFormatter('%d/%m/%Y %H:%M:%S')
	ax.xaxis.set_major_formatter(hfmt)
	
	if 'light' in str(phase.split()[0]):
		fname = path + day + '/' + str(meters[0]) + '.csv'
	else :
		fname = path + day + '/' + str(meters[1]) + '.csv'
	if os.path.isfile(fname):
		df_meter = pd.read_csv(fname)
		# if 'power' in str(phase.split()[0]):
		# 	df_meter = smoothening(df_meter, WINDOW)
		if day == '10_11':
			df_meter = df_meter.ix[df_meter['time'] >= 1392042600]
		df_meter = df_meter.sort_index(by = 'time')
		df_time = [datetime.datetime.fromtimestamp(x) for x in df_meter['time']]
		df_values = df_meter[phase.split()[0][1:-1]]
		#print df_time, df_values
		ax.plot( df_time,df_values, style, label = str(room) + '/' + str(phase.split()[0][1:-1])  ,linewidth = 1)
		ax.set_ylabel( str(phase.split()[0][1:-1]))
fig.autofmt_xdate()
plt.show()