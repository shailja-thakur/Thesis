import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime,pytz
import time, sys
import csv
import pandas as pd
import glob
TIMEZONE = 'Asia/Kolkata'
pdlist = []
try:
	sensor = 'Sound'
	for name in glob.glob('/home/shailja/Dropbox/Apps/MobiStatSense/mobistatsense-c002/'+ sensor +'*'):
		print name
		pdlist.append(pd.read_csv(name))
	newpd = pd.concat(pdlist)
	print newpd
	newpd.describe()
	newpd = newpd.sort(['time'])
	newpd.to_csv('c002/'+ sensor +'.csv', index=False)
	
except Exception,e:
	print e
