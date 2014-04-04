import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import glob
lis=[]
lights=[]
light=pd.DataFrame()
lis1=[]
powers=[]
power=pd.DataFrame()
path='/home/shailja/DataCollection-23_10-25_10/'
files_in_dir = os.listdir(path)

for file_in_dir in files_in_dir:
	files=os.listdir(path+file_in_dir)

	for f in files:
		if f=='Light.csv':
			light=pd.read_csv(path+file_in_dir+'/'+f)
			lis.append(light)
			lights=pd.concat(lis)

			
		if f=='Power.csv':
			power=pd.read_csv(path+file_in_dir+'/'+f)
			lis1.append(power)
			powers=pd.concat(lis1)



powers.sort(columns=['time'],inplace=True)
powertime=powers['time']
series1=np.asarray(powertime)
dates=[datetime.datetime.fromtimestamp(float(ts)) for ts in series1]


lights.sort(columns=['time'],inplace=True)
time=lights['time']
series=np.asarray(time)
dates1=[datetime.datetime.fromtimestamp(float(ts)) for ts in series]

figure=plt.gcf()
ax=plt.gca()

plt.subplot(421)

xfmt = mdates.DateFormatter(' %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("Phase R")
plt.xlabel("first")
ax.grid(True)
plt.plot(dates1,lights['lightphase1'],'r')


plt.subplot(422)
xfmt = mdates.DateFormatter(' %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("Phase R")
plt.xlabel("Fifth")
ax.grid(True)
plt.plot(dates,powers['powerphase1'],'r')


plt.subplot(423)
xfmt = mdates.DateFormatter(' %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("Phase Y")
plt.xlabel("second")
ax.grid(True)
plt.plot(dates1,lights['lightphase2'],'y')

plt.subplot(424)
xfmt = mdates.DateFormatter(' %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("Phase Y")
plt.xlabel("Sixth")
ax.grid(True)
plt.plot(dates,powers['powerphase2'],'y')


plt.subplot(425)
xfmt = mdates.DateFormatter(' %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("phase B")
plt.xlabel("Third")
ax.grid(True)
plt.plot(dates1,lights['lightphase3'],'b')



plt.subplot(426)
xfmt = mdates.DateFormatter(' %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("Phase B")
plt.xlabel("Seventh")
ax.grid(True)
plt.plot(dates,powers['powerphase3'],'b')


plt.subplot(427)
ax=plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.xlabel("Light Meter(EL)")
plt.ylabel("Total power")
ax.grid(True)
plt.plot(dates1,lights['lightpower'],'g')


plt.subplot(428)
ax=plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
plt.ylabel("Total Power")
plt.xlabel("Power Meter(LP)")
ax.grid(True)
plt.plot(dates,powers['power'],'g')


plt.show()
		


