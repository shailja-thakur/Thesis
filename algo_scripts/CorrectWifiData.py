import pandas as pd 
import time
import sys
import datetime
loc = '/Location_Data/Location_Formatted/'
path = '/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/WifiData/'


# time format -  '10-02-2014 21:08:00'
day =  sys.argv[1]
user = sys.argv[2]
#from_time = sys.argv[3]
#to_time = sys.argv[4]

def get_time(time):
	
	
	d = time.mktime(datetime.datetime.strptime(frm, "%d-%m-%Y %H:%M:%S").timetuple())
	print d
	return d

def correction(df):
	
	for idx in df.index:
		if ((df.ix[idx]['timestamp'] >= int(1392261360)) & (df.ix[idx]['timestamp'] <= int(1392274560))):
			df.loc[idx, user] = 1
	return df


if __name__ == '__main__':
	df = pd.read_csv(path + day + loc + user + '_location_fomatted' + '.csv' )
	
	#from_time = get_time(to)
	df  = correction(df)

	df.to_csv(path + day + loc + user + '_location_fomatted' + '.csv', cols = df.columns.values, index = False)
