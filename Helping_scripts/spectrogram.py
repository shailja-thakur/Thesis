import sys
from scipy import *
from pylab import *
from numpy import *
import csv
import struct
import MySQLdb
import sys

def show_Specgram(Kettle):
    	

	# Extracts feature info from sound file with scipy module
	spectrogram = specgram(Kettle,512,8000,1024)

	
	#Generates soectrogram with matplotlib specgram
    	#title('Spectrogram of %s'%sys.argv[1])
	ylabel("Phone"+path1)
	show()
	#sound.close()
	return spectrogram




path=sys.argv[1]
path1=sys.argv[2]
f=open(path,'rb')

reader=csv.reader(f,delimiter=',')
header=reader.next()
time=[]
Fan=[]
AC=[]
TV=[]
Microwave=[]
Kettle=[]



for row in reader :
	
	if row[2]=='Fan':
       
		Fan.append(long(row[1]))
	
	if row[2]=='AC':
       
		AC.append(long(row[1]))
	
	if row[2]=='TV':
       
		TV.append(long(row[1]))
	if row[2]=='Microwave':
       
		Microwave.append(long(row[1]))
	
	if row[2]=='Kettle':
       
		Kettle.append(long(row[1]))





show_Specgram(Kettle)

