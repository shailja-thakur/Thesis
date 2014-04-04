import csv
import pandas as pd
import sys
from math import *
from pandas import *
import math
from decimal import Decimal
import matplotlib.pyplot as plt

MAX_16_BIT = 32768;
#ARGUMENTS
room=sys.argv[1]
r_type = sys.argv[2]
sound_path=sys.argv[3]
rmno=sys.argv[4]

power_list=[]
time_list=[]
sound_csv = sound_path + '/' + 'mobistatsense' + '/' + 'Sound' + '_' + r_type + '_' + room + '.csv'
df=pd.read_csv(sound_csv)
print 'Parsing'
for idx in df.index[:-1]:
	sm=0
	sqrsm=0
	values=df['values'][idx]
	values=values.split(',')
	samples=[]
	length=len(df['values'][idx])
	for v in values:
		
		""""
		if len(str(v)) > 13:
		
			print "length gt 10 found:"+len(str(v))
			x=str(v)[:-13]+'"'+str(v)[-13:]
			#v.rsplit('"',1)
			samples.append(x)
			print v
			v.to_csv('/media/New Volume/IPSN/sense/5_6Nov/'+rmno+'/mobistatsense/'+rmno+'_'+r_type+'_'+'db.csv')
		
		samples.append(v)
	"""
	for v in values:
		sm+=int(v)*int(v)
	sm/=length
	if sm<0:
		db=np.log10(-sm)*20
	elif sm==0:
		db=0
	else:
		db=np.log10(sm)*20
	
	power_list.append(round(db,2))
	time_list.append(df['time'][idx])
	
print 'Creating dataframe of db values'
data={'time':time_list,'db':power_list}
frame=DataFrame(data)
print 'Writing to csv'
frame.to_csv('/media/New Volume/IPSN/sense/5_6Nov/'+rmno+'/mobistatsense/'+rmno+'_'+r_type+'_'+'db.csv')
plt.plot(frame['time'],frame['db'])
plt.show()
print 'Done!'


