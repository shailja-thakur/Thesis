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
#date = ['11/02/2014 00-00-00', '12/02/2014 00-00-00','13/02/2014 00-00-00','14/02/2014 00-00-00','15/02/2014 00-00-00', '16/02/2014 00-00-00', '17/02/2014 00-00-00']
date = '11/02/2014 00-00-00'

def groupAudiointoDays():
	#for user in users:
		audiopath =  AUDIO_FILES_CSV_PATH  + user + '/'
		print 'storing  csv in path', audiopath
		if not os.path.isdir(audiopath):
   			os.makedirs(audiopath)

		start  = 2000000000000
   		end = 0
   		path = AUDIO_FILES_PATH +  'mobistatsense-' + user + '/Sound_room_' + user
   		#path = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/AudioData/AudioFeatureData/c002/Sound_room_' + user
   		print path
   		i = 0
   		cols = ['time','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','label']
    	# Find minimum and maximum timestamp for each users wifi data
		for csvs in glob.glob(path + '*.csv'):
			i =  i + 1
			print i
			print csvs
			df = pd.read_csv(csvs)
			if df.columns.values[0] != cols[0]:
				print 'Column value empty'
				df.columns = cols

			if (df.ix[df.index[-1]][0]) == cols[0]:
				print 'There is duplicate header'
				df = df.ix[:-1]
				sys.exit(1)
			
				
			if int(df.ix[df.index[-1]]['time']) > int(end):
			
					end = int(df.ix[df.index[-1]]['time'])
			if int(df.ix[df.index[0]]['time']) < int(start):
				start = int(df.ix[df.index[0]]['time'])       
		print start, end
		
		
		# Merging all the Sound csv of the user to create a single csv
		df_merged = pd.DataFrame()
		day = 0
		flag = False
	
		for csvs in glob.glob(path + '*.csv'):
			df = pd.read_csv(csvs)
			if df.columns.values[0] != cols[0]:
				print 'Column value empty'
				df.columns = cols
			df_merged = concat([df_merged, df])
	
		print 'merged wifi'
		print df_merged

		endtime = time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y %H-%M-%S").timetuple())
		endtime = endtime * 1000

		endtime = endtime - 1
		while endtime <= end:
			print datetime.datetime.fromtimestamp(endtime/1000)
			if ((endtime >= start) & (endtime <= end)):
			#for time in range (start, endtime):
				print 'In range'

				df_day = df_merged.ix[((df_merged['time'] >= start) & (df_merged['time'] <= endtime))]
				print df_day
				print 'length of index in range', len(df_day)
				df_day.to_csv(audiopath  + str(datetime.datetime.fromtimestamp(start/1000)) + '_' + str(datetime.datetime.fromtimestamp(endtime/1000)) + '.csv', index = False, cols = df.columns.values)
				print 'created csv for ' + str(datetime.datetime.fromtimestamp(start/1000)) + ' to ' + str(datetime.datetime.fromtimestamp(endtime/1000))
				start = endtime + 1
				endtime = ((start/1000 ) + 86400)* 1000 
				print start, endtime
			
			if endtime > end :
				print 'endtime greater than end'
				endtime = end
				df_day = df_merged.ix[((df_merged['time'] >= start) & (df_merged['time'] <= endtime))]
				print 'length of index in range', len(df_day)
				#df_day = df_day.sort_index(by = 'time')
				df_day.to_csv(audiopath  + str(datetime.datetime.fromtimestamp(start/1000)) + '_' + str(datetime.datetime.fromtimestamp(endtime/1000)) + '.csv', index = False, cols = df.columns.values)
				print 'created csv for ' + str(datetime.datetime.fromtimestamp(start/1000)) + ' to ' + str(datetime.datetime.fromtimestamp(endtime/1000))
				start  = 2000000000000
   				end = 0

				break
			if start > endtime:
				print 'start greater than end'
				start = endtime + 1
				endtime = ((start/1000 ) + 86400)* 1000
				print 'next start and end time' + str(datetime.datetime.fromtimestamp(start/1000)) + str(datetime.datetime.fromtimestamp(endtime/1000))

			print 'next start and end time' + str(datetime.datetime.fromtimestamp(start/1000)) + str(datetime.datetime.fromtimestamp(endtime/1000))
			print 'endtime and end' + str(datetime.datetime.fromtimestamp(endtime/1000)) + str(datetime.datetime.fromtimestamp(end/1000))
	

if __name__ == '__main__':

	#groupAudiointoDays()
	
	
	users = user.split(',')
	#audiopath = AUDIO_FILES_PATH
	

	# days = ['First', 'Second','Third', 'Fourth', 'Fifth', 'Sixth','Seventh']
	# dates = ['2014-02-10', '2014-02-11', '2014-02-12', '2014-02-13', '2014-02-14', '2014-02-15', '2014-02-16', '2014-02-17']
	
	days = ['Eighth']
	dates = [ '2014-02-17']
	users = [user]
	for day,date in zip(days, dates):
		daypath = AUDIO_FILES_CSV_PATH + day 

		if not os.path.isdir(daypath):
   			os.makedirs(daypath)
   		print 'Created folder ' + daypath 
		
		for user in users:
				audiopath = AUDIO_FILES_CSV_PATH  + user + '/' 
				for csvs in glob.glob(audiopath + date + '*.csv'):
					
   						dest_dir = daypath + '/'
   						#print 'File to be copied ' + csvs
   						shutil.move(csvs, dest_dir + user + '.csv')
   						print 'File moved to ' + dest_dir
   				
