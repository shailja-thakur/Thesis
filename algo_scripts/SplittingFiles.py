import os
from path import *
import glob
import pandas as pd
# os.chdir("/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/PhoneData/mobistatsense-C002/")
# print 'list of files and directories in path'
# print os.getcwd()
# os.system("split -l 500 Sound_room_C002_2.csv Sound_room_C002_2")
# cols = ['time','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','label']
# audiopath = AUDIO_FILES_PATH + 'C002/'
# path = AUDIO_FILES_PATH + 'mobistatsense-C002/'
os.chdir("/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/WifiData/Second/Location_Data/User_Attribution_Tables/")
# print 'list of files and directories in path'
# print os.getcwd()
os.system("split -l 5000 location_room_set.csv location_room_set")
path = DATA_PATH + 'Second' + USER_ATTRIBUTION_TABLES_PATH
# if not os.path.isdir(audiopath):
#    	os.makedirs(audiopath)
cols = ['start_time', 'end_time', 'room_set']
for filen in glob.glob(path + '*'):
	if '.csv' not in str(filen):
		newname = str(filen) + '.csv'
		print 'newname', newname
		output = os.rename(filen, newname)

for filen in glob.glob(path + '*.csv'):
	print filen
	filename = str(filen).split('/')[10].split('.')[0]
	df = pd.read_csv(filen)
	if df.columns.values[0] != cols[0]:
		print 'Column value empty'
		df.columns = cols
	df.to_csv(path  + filename + '.csv', index = False, cols = df.columns.values)

