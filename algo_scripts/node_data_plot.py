import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from path import *
import sys
from matplotlib import dates
import datetime
import os
day = sys.argv[1]
rooms = ['C001', 'C002','C003','C004', 'C005', 'C006']
sensors = [ 'PIRSensor', 'LightSensor']
#standby_battery_df = pd.read_csv(PATH + '/performance_standby/battery/Battery.csv')
line_style = ['r--', 'g', 'b:']
number_of_subplots = len(rooms)
fig, ax = plt.subplots()
v = 0
for i in rooms:
	print i
	v = v + 1
	ax = plt.subplot(number_of_subplots,1,v)
	
	for sensor, style in zip(sensors, line_style):
		fname = NODE_CSV_PATH + day + '/'  + i + '/' + 'manaswis_NodeHDN' + str(i[3]) + '_' + str(sensor) + '_1' + '.csv'
		if os.path.isfile(fname):
			df_sensor = pd.read_csv(NODE_CSV_PATH + day + '/'  + i + '/' + 'manaswis_NodeHDN' + str(i[3]) + '_' + str(sensor) + '_1' + '.csv')
		
			if sensor == 'PIRSensor':
				idx = df_sensor.index[df_sensor['value'] > 1]
				if len(idx) > 1:
					df_sensor.loc[idx, 'value'] = 1

			if sensor == 'LightSensor':
				idx = df_sensor.index[df_sensor['value'] >= 10]
				if len(idx) >= 1:
					df_sensor.loc[idx, 'value'] = 5

			df_time = [datetime.datetime.fromtimestamp(x) for x in df_sensor['timestamp']]
			df_values = df_sensor['value']
			ax.plot( df_time,df_values, style, label = 'room' + str(i) + str(sensor) ,linewidth = 3)
			ax.set_ylabel('room' + str(i))
	
hfmt = dates.DateFormatter('%d/%m/%Y %H:%M:%S')
ax.xaxis.set_major_formatter(hfmt)
fig.autofmt_xdate()

plt.show()

