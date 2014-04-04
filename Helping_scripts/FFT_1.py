from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
from scipy import *
from pylab import *
from numpy import *
import MySQLdb
import datetime as dt
import sys
import csv
import numpy as np
Fs = 8000;  # sampling rate
nFFT=1024

def plotSpectru(y,Fs):
 n = len(y) # lungime semnal
 k = arange(n)
 
 T = n/Fs
 frq = k/T # two sides frequency range
 frq = frq[range(n/2)] # one side frequency range

 Y = fft(y)/n # fft computing and normalization
 Y = Y[range(n/2)]
 
 plot(frq,abs(Y),'r') # plotting the spectrum
 xlabel('Freq (Hz)')
 ylabel('|Y(freq)|')

def show_Specgram(Kettle):
    	spectrogram = specgram(Kettle,nFFT,Fs)
	return spectrogram




path=sys.argv[1]
path1=sys.argv[2]
path2=sys.argv[3]
f=open(path,'rb')

reader=csv.reader(f,delimiter=',')
header=reader.next()
time=[]
Fan=[]
AC=[]
TV=[]
Microwave=[]
Kettle=[]


f1=open(path1,'rb')

reader1=csv.reader(f1,delimiter=',')
header1=reader1.next()
time1=[]
Fan1=[]
AC1=[]
TV1=[]
Microwave1=[]
Kettle1=[]


f2=open(path2,'rb')

reader2=csv.reader(f2,delimiter=',')
header2=reader2.next()
time2=[]
Fan2=[]
AC2=[]
TV2=[]
Microwave2=[]
Kettle2=[]



for row in reader :

		if row[2]=='Fan':
       			time.append(row[0])
			Fan.append(long(row[1]))
		if row[2]=='AC':
       	
			AC.append(long(row[1]))
	
		if row[2]=='TV':
       
			TV.append(long(row[1]))
		if row[2]=='Microwave':
       
			Microwave.append(long(row[1]))
	
		if row[2]=='Kettle':
       
			Kettle.append(long(row[1]))


for row in reader1 :
	if row[2]=='Fan':
       
		Fan1.append(long(row[1]))
	if row[2]=='AC':
       
		AC1.append(long(row[1]))
	if row[2]=='TV':
       
		TV1.append(long(row[1]))
	if row[2]=='Microwave':
       
		Microwave1.append(long(row[1]))
	if row[2]=='Kettle':
       
		Kettle1.append(long(row[1]))

for row in reader2 :
	if row[2]=='Fan':
       
		Fan2.append(long(row[1]))
	if row[2]=='AC':
       
		AC2.append(long(row[1]))
	if row[2]=='TV':
       
		TV2.append(long(row[1]))
	if row[2]=='Microwave':
       
		Microwave2.append(long(row[1]))
	if row[2]=='Kettle':
       
		Kettle2.append(long(row[1]))





ac=[AC,AC1,AC2]
fan=[Fan,Fan1,Fan2]
tv=[TV,TV1,TV2]
kettle=[Kettle,Kettle1,Kettle2]
microwave=[Microwave,Microwave1,Microwave2]

lungime=len(ac[0])
timp=len(ac[0])/8000.
#t=linspace(0,timp,len(ac[0]))
ax=subplot(3,5,1)
fig=gcf()
setp(ax.get_xticklabels(),visible=False)
setp(ax.get_yticklabels(),visible=False)
ylabel("Galaxy chat 1")
title("A/C")
#plot(dates1,ac)
show_Specgram(ac[0])	
	

lungime=len(ac[1])
timp=len(ac[1])/8000.
#t=linspace(0,timp,len(ac[1]))

ax=subplot(3,5,6)
ax.set_ylabel("Galaxy chat 2")
setp(ax.get_xticklabels(),visible=False)
setp(ax.get_yticklabels(),visible=False)


#plot(dates1,ac)
show_Specgram(ac[1])


lungime=len(ac[2])
timp=len(ac[2])/8000.
#t=linspace(0,timp,len(ac[2]))

ax=subplot(3,5,11)
setp(ax.get_xticklabels(),visible=False)
setp(ax.get_yticklabels(),visible=False)
ylabel("Nexus S")
#plot(dates1,ac)
show_Specgram(ac[2])


lungime=len(tv[0])
timp=len(tv[0])/8000.
#t=linspace(0,timp,len(tv[0]))

subplot(3,5,2)
axis('off')
title("TV")
#plot(dates1,ac)
show_Specgram(tv[0])


lungime=len(tv[1])
timp=len(tv[1])/8000.
#t=linspace(0,timp,len(tv[1]))

subplot(3,5,7)

axis('off')
#plot(dates1,ac)
show_Specgram(tv[1])

lungime=len(tv[2])
timp=len(tv[2])/8000.
#t=linspace(0,timp,len(tv[2]))

subplot(3,5,12)
axis('off')
#plot(dates1,ac)
show_Specgram(tv[2])


lungime=len(kettle[0])
timp=len(kettle[0])/8000.
#t=linspace(0,timp,len(kettle[0]))

subplot(3,5,3)
axis('off')
title("Kettle")
#plot(dates1,ac)
show_Specgram(kettle[0])

lungime=len(kettle[1])
timp=len(kettle[1])/8000.
#t=linspace(0,timp,len(kettle[1]))

subplot(3,5,8)
axis('off')
#plot(dates1,ac)
show_Specgram(kettle[1])

lungime=len(kettle[2])
timp=len(kettle[2])/8000.
#t=linspace(0,timp,len(kettle[2]))

subplot(3,5,13)
axis('off')
#plot(dates1,ac)
show_Specgram(kettle[2])



lungime=len(fan[0])
timp=len(fan[0])/8000.
#t=linspace(0,timp,len(fan[0]))

subplot(3,5,4)
axis('off')
title("Fan")
#plot(dates1,ac)
show_Specgram(fan[0])

lungime=len(fan[1])
timp=len(fan[1])/8000.
#t=linspace(0,timp,len(fan[1]))

subplot(3,5,9)
axis('off')
#plot(dates1,ac)
show_Specgram(fan[1])

lungime=len(fan[2])
timp=len(fan[2])/8000.
#t=linspace(0,timp,len(fan[2]))

subplot(3,5,14)
axis('off')
#plot(dates1,ac)
show_Specgram(fan[2])

lungime=len(microwave[0])
timp=len(microwave[0])/8000.
#t=linspace(0,timp,len(microwave[0]))

subplot(3,5,5)
axis('off')
title("Microwave")
#plot(dates1,ac)
show_Specgram(microwave[0])

lungime=len(microwave[1])
timp=len(microwave[1])/8000.
#t=linspace(0,timp,len(microwave[1]))

subplot(3,5,10)

#plot(dates1,ac)
show_Specgram(microwave[1])
axis('off')
lungime=len(microwave[2])
timp=len(microwave[2])/8000.
#t=linspace(0,timp,len(microwave[2]))

subplot(3,5,15)
axis('off')
#plot(dates1,ac)
show_Specgram(microwave[2])
plt.savefig('/home/shailja/sense/test.png', bbox_inches='tight')

show()
















