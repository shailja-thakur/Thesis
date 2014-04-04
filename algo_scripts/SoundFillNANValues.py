import os
import pandas as pd
import sys
import glob
import numpy as np
from path import *
import shutil
user = sys.argv[1]
#path = AUDIO_FILES_CSV_PATH + user + '/' 
#path1 = AUDIO_FILES_CSV_PATH + 'c002' + '/'
path = AUDIO_FILES_PATH +  'mobistatsense-' + user + '/Sound_room_' + user
cols = ['time','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','label']


####

# Searches the large file (> 1MB) moves them to SoundBig folder splits them with samples 500 and add extension .csv to the files 

##########

temppath = AUDIO_FILES_PATH +  'mobistatsense-' + user  + '/SoundBig/'
if not os.path.isdir(temppath):
   			os.makedirs(temppath)

print path
for csvs in glob.glob(path + '*.csv'):
	s = os.path.getsize(csvs) /(1024* 1024)
	if float(s) > 1:
		dest_dir = temppath
   		#print 'File to be copied ' + csvs
   		shutil.move(csvs, dest_dir + str(csvs).split('/')[8].split('.')[0] + '.csv')
   		
os.chdir('/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/PhoneData/mobistatsense-C006/SoundBig/')
print os.getcwd()
for csvs in glob.glob(temppath + '*.csv'):
	#print temppath
	print "split -l 500 " + str(csvs).split('/')[9] + ' '+ AUDIO_FILES_PATH +  'mobistatsense-' + user  + '/' + str(csvs).split('/')[9].split('.')[0] 
	os.system("split -l 500 " + str(csvs).split('/')[9] + ' '+ AUDIO_FILES_PATH +  'mobistatsense-' + user  + '/' + str(csvs).split('/')[9].split('.')[0] )

for filen in glob.glob(path + '*'):
	if '.csv' not in str(filen):
		newname = str(filen) + '.csv'
		print 'newname', newname
		output = os.rename(filen, newname)

		
for csvs in glob.glob(path + '*.csv'):
	print csvs
	df = pd.read_csv(csvs)
	filename = str(csvs).split('/')[8].split('.')[0]

	if df.columns.values[0] != cols[0]:
		print 'Column value empty'
		df.columns = cols
	
	if len(np.where(df['mfcc2'].isnull())[0]) > 0:

		idx_list = df.index[df['mfcc1'] == '-Infinity']

	#print idx_list
		for idx in idx_list:

			idx_prev_list = df.index[df.index < idx]
			idx_next_list = df.index[df.index > idx]

			list1 = [i for i in idx_prev_list if df.ix[i]['mfcc1'] != '-Infinity']
			list2 = [i for i in idx_next_list if df.ix[i]['mfcc1'] != '-Infinity']
		
			if list1 == list():
				prev = min(list2)
			else:
				prev = max(list1)

			#next = min(list2)
		
			for col in df.columns[1:-1]:
				# print 'prev val',df.ix[prev][col]
				
				# print 'next val',df.ix[prev][next]
				#m = np.mean([float(df.ix[prev][col]), float(df.ix[next][col])])
				df.ix[idx, col] = df.ix[prev, col]
		print df
	df.to_csv(AUDIO_FILES_PATH +  'mobistatsense-' + user + '/'  + filename + '.csv', index = False, cols = df.columns.values)
			
			