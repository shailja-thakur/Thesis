import matplotlib.pyplot as plt
import matplotlib.dates as md
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
	#datenums = datetime.datetime.fromtimestamp(float(data1[0]),pytz.timezone(TIMEZONE));
	timestamp.append(data1[0])
	x.append(float(data1[1]))
	y.append(float(data1[2]))
	z.append(float(data1[3]))
	p.append(float(data1[4]))

dates2=[datetime.datetime.fromtimestamp(float(ts)) for ts in timestamp]

ax=plt.gca()
figure=plt.gcf()

plt.subplot(411)
#plt.axes().relim()
#ax.fmt_xdata=mdates.DateFormatter('%Y-%m-%d %H:%M:%S',TIMEZONE)

xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
ax.grid(True)
figure.autofmt_xdate()
#plt.title( yaxislabel1 + " v/s Time")
#plt.xlabel("Time")
plt.ylabel(yaxislabel1)



plt.plot(dates2,x,'g')

plt.subplot(412)

#ax.fmt_xdata=mdates.DateFormatter('%Y-%m-%d %H:%M:%S',TIMEZONE)
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
ax.grid(True)
figure.autofmt_xdate()
#plt.title( yaxislabel1 + " v/s Time")
#plt.xlabel("Time")
plt.ylabel(yaxislabel2)
plt.plot(dates2,y,'r')

plt.subplot(413)
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
#ax.fmt_xdata=mdates.DateFormatter('%Y-%m-%d %H:%M:%S',TIMEZONE)
ax.grid(True)
figure.autofmt_xdate()
#plt.title( yaxislabel3 + " v/s Time")
#plt.xlabel("Time")
plt.ylabel(yaxislabel3)
plt.plot(dates2,z,'r')


plt.subplot(414)
#ax.fmt_xdata=mdates.DateFormatter('%Y-%m-%d %H:%M:%S',TIMEZONE)
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
ax.grid(True)
figure.autofmt_xdate()
#plt.title( yaxislabel4 + " v/s Time")
#plt.xlabel("Time")
plt.ylabel(yaxislabel4)
plt.plot(dates2,p,'r')




plt.show()
