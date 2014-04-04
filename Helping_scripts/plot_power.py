import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime,pytz
import time, sys
import csv

#Deriving name of the csv from the name of the script name
# sys.argv[0].replace('.py', '.csv')
x = []
y=[]
z=[]
p=[]



timestamp = []

csvfile =  sys.argv[1] # 'manaswis_RPi-1_TemperatureSensor_1.csv'

TIMEZONE = 'Asia/Kolkata'

light = open(csvfile,'rt')
reader = csv.reader(light)
header = reader.next()




yaxislabel1 = header[1]
yaxislabel2 = header[2]
yaxislabel3 = header[3]
yaxislabel4 = header[4]



for data1 in reader:
	datenums = datetime.datetime.fromtimestamp(float(data1[0]),pytz.timezone(TIMEZONE));
	timestamp.append(datenums)
	x.append(float(data1[1]))
	y.append(float(data1[2]))
	z.append(float(data1[3]))
	p.append(float(data1[4]))



ax=plt.gca()
figure=plt.gcf()

plt.subplot(411)
#plt.axes().relim()
ax.fmt_xdata=mdates.DateFormatter('%H-%M-%S',TIMEZONE)
ax.grid(True)
figure.autofmt_xdate()
plt.title( yaxislabel1 + " v/s Time")
plt.xlabel("Time")
plt.ylabel(yaxislabel1)
plt.plot(timestamp,x,'g')

plt.subplot(412)

ax.fmt_xdata=mdates.DateFormatter('%H-%M-%S',TIMEZONE)
ax.grid(True)
figure.autofmt_xdate()
plt.title( yaxislabel2 + " v/s Time")
plt.xlabel("Time")
plt.ylabel(yaxislabel2)
plt.plot(timestamp,y,'r')

plt.subplot(413)

ax.fmt_xdata=mdates.DateFormatter('%H-%M-%S',TIMEZONE)
ax.grid(True)
figure.autofmt_xdate()
plt.title( yaxislabel3 + " v/s Time")
plt.xlabel("Time")
plt.ylabel(yaxislabel3)
plt.plot(timestamp,z,'b')


plt.subplot(414)
ax.fmt_xdata=mdates.DateFormatter('%H-%M-%S',TIMEZONE)
ax.grid(True)
figure.autofmt_xdate()
plt.title( yaxislabel4 + " v/s Time")
plt.xlabel("Time")
plt.ylabel(yaxislabel4)
plt.plot(timestamp,p,'y')




plt.show()
