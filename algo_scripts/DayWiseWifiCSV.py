import glob
import pandas as pd
from pandas import *
from path import *
import datetime
import os
import csv
import shutil
import sys
import time

user = sys.argv[1]
#users = [ str(user) ]
stime = ['10/02/2014 20:00:00','10/02/2014 21:59:00', '11/02/2014 21:59:00','12/02/2014 21:59:00','13/02/2014 21:59:00','14/02/2014 21:59:00', '15/02/2014 21:59:00', '16/02/2014 21:59:00']
etime = ['11/02/2014 03:00:00','12/02/2014 03:00:00','13/02/2014 03:00:00','14/02/2014 03:00:00','15/02/2014 03:00:00','16/02/2014 03:00:00','17/02/2014 03:00:00','18/02/2014 03:00:00']
#date = '11/02/2014 00-00-00'

def groupWifiintoDays():
	#for user in users:
		wifipath =  WIFI_CSV_TEST_PATH + user + '/'
		print 'storing wifi csv in path', wifipath
		if not os.path.isdir(wifipath):
   			os.makedirs(wifipath)

		start  = 2000000000000
   		end = 0
   		path = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/PhoneData/mobistatsense-'+ user + '/Wifi_room_' + user 
   		print path
    	# Find minimum and maximum timestamp for each users wifi data
		# for csvs in glob.glob(path + '*.csv'):
		# 	#print csvs
		# 	df = pd.read_csv(csvs)
		# 	df = df.sort_index(by = 'time')
		# 	if int(df.ix[df.index[0]]['time']) < int(start):
		# 		start = int(df.ix[df.index[0]]['time'])

		# 	if int(df.ix[df.index[-1]]['time']) > int(end):
			
		# 		end = int(df.ix[df.index[-1]]['time'])
            
			
		#print datetime.datetime.fromtimestamp(start/1000), datetime.datetime.fromtimestamp(end/1000)
		

		# Read Wifi csvs to create wifi csv day wise 
		df_merged = pd.DataFrame()
		
		for csvs in glob.glob(path + '*.csv'):
			#print csvs
			df = pd.read_csv(csvs)
			df_merged = concat([df_merged, df])
		
		#df_merged = df_merged.sort_index(by = 'time')
		print 'merged wifi'
		print df_merged



		#When wifi csv are created for 00 to 00
		# endtime = time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y %H-%M-%S").timetuple())
		# endtime = endtime * 1000
		#endtime = endtime - 1
		for start, endtime in zip(stime, etime):
			print start, endtime
			endtime = time.mktime(datetime.datetime.strptime(str(endtime), "%d/%m/%Y %H:%M:%S").timetuple())
			endtime = endtime * 1000
			start = time.mktime(datetime.datetime.strptime(str(start), "%d/%m/%Y %H:%M:%S").timetuple())
			start = start * 1000
			print 'starttime', start, 'endtime', endtime

			print 'In range'

			df_day = df_merged.ix[((df_merged['time'] >= start) & (df_merged['time'] <= endtime))]
			#df_day = pd.DataFrame(df_day)
			print df_day
			print 'length of index in range', len(df_day)
			#df_day = df_day.sort_index(by = 'time')
			df_day.to_csv(wifipath  + str(datetime.datetime.fromtimestamp(start/1000)) + '_' + str(datetime.datetime.fromtimestamp(endtime/1000)) + '.csv', index = False, cols = ['time', 'mac', 'ssid', 'rssi', 'label'])
		
			print 'created csv for ' + str(datetime.datetime.fromtimestamp(start/1000)) + ' to ' + str(datetime.datetime.fromtimestamp(endtime/1000))
				



		# while endtime <= end:
		# 	print datetime.datetime.fromtimestamp(endtime/1000)
		# 	if ((endtime >= start) & (endtime <= end)):
		# 	#for time in range (start, endtime):
		# 		print 'In range'

		# 		df_day = df_merged.ix[((df_merged['time'] >= start) & (df_merged['time'] <= endtime))]
		# 		#df_day = pd.DataFrame(df_day)
		# 		print df_day
		# 		print 'length of index in range', len(df_day)
		# 		#df_day = df_day.sort_index(by = 'time')
		# 		df_day.to_csv(wifipath  + str(datetime.datetime.fromtimestamp(start/1000)) + '_' + str(datetime.datetime.fromtimestamp(endtime/1000)) + '.csv', index = False, cols = ['time', 'mac', 'ssid', 'rssi', 'label'])
		
		# 		print 'created csv for ' + str(datetime.datetime.fromtimestamp(start/1000)) + ' to ' + str(datetime.datetime.fromtimestamp(endtime/1000))
				


				 
				
		# 		endtime = time.mktime(datetime.datetime.strptime(etime, "%d/%m/%Y %H-%M-%S").timetuple())
		# 		endtime = endtime * 1000
		# 		start = time.mktime(datetime.datetime.strptime(stime, "%d/%m/%Y %H-%M-%S").timetuple())
		# 		start = start * 1000
		# 		#when wifi files are created for 00 to 00
		# 		# start = endtime + 1
		# 		# endtime = ((start/1000 ) + 86400)* 1000 
		# 		print start, endtime
			
		# 	if endtime > end :
		# 		print 'endtime greater than end'
		# 		endtime = end

		# 		df_day = df_merged.ix[((df_merged['time'] >= start) & (df_merged['time'] <= endtime))]
		# 		print 'length of index in range', len(df_day)
		# 		#df_day = df_day.sort_index(by = 'time')
		# 		df_day.to_csv(wifipath  + str(datetime.datetime.fromtimestamp(start/1000)) + '_' + str(datetime.datetime.fromtimestamp(endtime/1000)) + '.csv', index = False, cols = ['time', 'mac', 'ssid', 'rssi', 'label'])
			
		
		# 		print 'created csv for ' + str(datetime.datetime.fromtimestamp(start/1000)) + ' to ' + str(datetime.datetime.fromtimestamp(endtime/1000))
		# 		start  = 2000000000000
  #  				end = 0

		# 		break
		# 	if start > endtime:
		# 		print 'start greater than end'
		# 		#when wifi files are created for 00 to 00 
		# 		start = endtime + 1
		# 		endtime = ((start/1000 ) + 86400)* 1000
		# 		print 'next start and end time' + str(datetime.datetime.fromtimestamp(start/1000)) + str(datetime.datetime.fromtimestamp(endtime/1000))

			# print 'next start and end time' + str(datetime.datetime.fromtimestamp(start/1000)) + str(datetime.datetime.fromtimestamp(endtime/1000))
			# print 'endtime and end' + str(datetime.datetime.fromtimestamp(endtime/1000)) + str(datetime.datetime.fromtimestamp(end/1000))
	

if __name__ == '__main__':

	#groupWifiintoDays()

	
	# path1 = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/WifiData/WifiRawData/C006/'
	# df_merged = pd.DataFrame()
	# for i in glob.glob(path1 + '2014-02-14*.csv'):
	# 	print i
	# 	df = pd.read_csv(i)
	# 	df_merged = concat([df_merged, df])
	# 	print df_merged

	# df_merged = df_merged.sort_index(by = 'time')
	# df_merged.to_csv(path1 + '2014-02-1400:00:00.003000_2014-02-15.csv', index = False, cols = ['time', 'mac', 'ssid', 'rssi', 'label'])
		




	users = user.split(',')
	wifipath = WIFI_CSV_TEST_PATH 
	days = ['First', 'Second','Third', 'Fourth', 'Fifth', 'Sixth','Seventh','Eighth']
	dates = ['2014-02-10 20:00','2014-02-10', '2014-02-11', '2014-02-12', '2014-02-13', '2014-02-14', '2014-02-15', '2014-02-16']
	
	
	
	for day,date in zip(days, dates):
		daypath = DATA_PATH + day 

		if not os.path.isdir(daypath):
   			os.makedirs(daypath)
   		print 'Created folder ' + daypath 
		
		for user in users:
				wifipath = WIFI_CSV_TEST_PATH + user + '/' 
				for csvs in glob.glob(wifipath + date + '*.csv'):
						print csvs
   						dest_dir = daypath + '/'
   						#print 'File to be copied ' + csvs
   						shutil.move(csvs, dest_dir + user + '.csv')
   						print 'File moved to ' + dest_dir
   				
