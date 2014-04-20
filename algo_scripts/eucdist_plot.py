import pandas as pd
from path import *
import matplotlib.pyplot as plt
import datetime
import math
from matplotlib import dates
import numpy as np
MFCC_bef = []
MFCC_aft = []
bef_start_time = 1392120540
aft_start_time = bef_end_time = 1392121635
aft_end_time = 1392122700
print '############   AUDIO ##########'
audio_df = pd.read_csv(AUDIO_FILES_CSV_PATH + '/' + 'Second' + '/' + 'C003' + '.csv')

audio_df['time'] = audio_df['time'] / 1000
# bef_idx = audio_df.index[((audio_df.time >= bef_start_time) & (audio_df.time <= bef_end_time))]
# aft_idx = audio_df.index[((audio_df.time >= aft_start_time) & (audio_df.time <= aft_end_time))]
# euc_dist = 0
# euc_distance = []
time = []

# for b,a in zip(bef_idx, aft_idx):
                
#                 MFCC_bef = audio_df.ix[b][audio_df.columns[1:-1]]
#                 MFCC_aft = audio_df.ix[a][audio_df.columns[1:-1]]
#                 time.append(audio_df.ix[b]['time'])
               

#                 # euc distance with MFCC 1-13 feature only
#                 for mi, mj in zip(MFCC_bef, MFCC_aft):
#                         euc_dist = euc_dist + math.pow((float(mi) - float(mj)), 2)

#                 euc_distance.append( math.sqrt(euc_dist) )
#                 print 'euc distance', math.sqrt(euc_dist)
#                 euc_dist = 0



# MFCC_bef = audio_df.ix[bef_idx]['mfcc1']

# MFCC_bef = MFCC_bef.reindex( index=MFCC_bef.index[ ::-1 ] )
# MFCC_aft = audio_df.ix[aft_idx]['mfcc1']
# time  = audio_df.ix[aft_idx]['time']
# print len(MFCC_bef), len(MFCC_aft)

# #euc_dist = euc_dist + math.pow((float(mi) - float(mj)), 2)
# for mi, mj in zip(MFCC_bef, MFCC_aft):
# 	#print mi, mj
# 	euc_dist = math.pow((float(mi) - float(mj)), 2)
# 	#print euc_dist
# 	euc_distance.append( math.sqrt(euc_dist) )




	
	#print 'euc distance', math.sqrt(euc_dist)
#print euc_distance


# plot
audi_idx = audio_df.index[((audio_df.time >= bef_start_time) & (audio_df.time <= aft_end_time))]

MFCC = audio_df.ix[audi_idx]

print MFCC
MFCC  = MFCC.sort_index(by = 'time')
time = MFCC['time']
MFCC_val =  MFCC['mfcc1']	

MFCC_val.replace([np.inf, -np.inf], np.nan)
print MFCC_val
fig, ax = plt.subplots()

hfmt = dates.DateFormatter('%d/%m/%Y %H:%M:%S')
ax.xaxis.set_major_formatter(hfmt)
df_time = [datetime.datetime.fromtimestamp(x) for x in time]
ax.plot( df_time, MFCC  ,linewidth = 1)
ax.set_ylabel('MFCC1')
fig.autofmt_xdate()
plt.show()