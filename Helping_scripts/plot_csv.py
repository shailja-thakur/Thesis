import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime,pytz
import time, sys
import csv

#Deriving name of the csv from the name of the script name
# sys.argv[0].replace('.py', '.csv')
val = []
timestamp = []
csvfile =  sys.argv[1] # 'manaswis_RPi-1_TemperatureSensor_1.csv'
TIMEZONE = 'Asia/Kolkata'

fileip = open(csvfile,'rt')
reader = csv.reader(fileip)
header = reader.next()
yaxislabel = header[1]
try:
	for data1 in reader:
		datenums = datetime.datetime.fromtimestamp(long(data1	[0])/1000,pytz.timezone(TIMEZONE));
		timestamp.append(datenums)
		val.append(float(data1[1]))
except Exception,e:
	print e
ax=plt.gca()
figure=plt.gcf()
plt.axes().relim()
plt.title( yaxislabel + " v/s Time")
plt.xlabel("Time")
plt.ylabel(yaxislabel)
plt.axes().autoscale_view(True,True,True)
ax.fmt_xdata=mdates.DateFormatter('%H-%M-%S',TIMEZONE)
ax.grid(True)
figure.autofmt_xdate()
plt.plot(timestamp,val)
plt.show()
