import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from path import *
import numpy
import sys
import datetime
from matplotlib import dates
import os
day = sys.argv[1]
window_size = 50
path = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/WifiData/' + day + '/Location_Data/Location_Formatted/'
rooms = ['C001', 'C002', 'C003','C004', 'C005','C006']
sensors = ['TemperatureSensor', 'PIRSensor', 'LightSensor']
#standby_battery_df = pd.read_csv(PATH + '/performance_standby/battery/Battery.csv')
line_style = ['r--', 'g', 'b:']
number_of_subplots = len(rooms)
fig, ax = plt.subplots()
v = 0
for i in rooms:
	print i
	v = v + 1
	ax = plt.subplot(number_of_subplots,1,v)
	hfmt = dates.DateFormatter('%d/%m/%Y %H:%M:%S')
	ax.xaxis.set_major_formatter(hfmt)
	#for sensor, style in zip(sensors, line_style):
	fname = path + str(i) + '_location_fomatted.csv'
	if os.path.isfile(fname):
		print fname
		df_sensor = pd.read_csv(path + str(i) + '_location_fomatted.csv' )

		# window = numpy.ones(int(window_size))/float(window_size)
		# df_sensor[i] = numpy.convolve(df_sensor[i], window, 'same')

		df_sensor = df_sensor.sort_index(by = 'timestamp')
		df_time = [datetime.datetime.fromtimestamp(x) for x in df_sensor['timestamp']]
		
		df_values = df_sensor[i]
	
		df_values[0] = 3
		ax.plot( df_time,df_values, label = str(i)  ,linewidth = 1)
		ax.set_ylabel( str(i))
	
fig.autofmt_xdate()
#plt.tight_layout()
plt.show()

