# Take ground truth labels for the colleted audio data from user
import csv
import sys
import time
from path import *
import os
import pandas as pd
import math
from scipy import *
import numpy as np
# Variables
label = []
stime_str = []
etime_str = []

# Input arguments
day = sys.argv[1]

filename_gt = GROUNDTRUTH_PATH + 'gt_timeslices_' + day + '.csv'
filename_op = OUTPUT  + 'predicted_timeslices_' + day + '.csv'

# Test Data for Hostel


#################################################################################################
# 10/02/2014 - 11/02/2014
# LightPhase1
# stime = ['1392046686', '1392050535','1392041318','1392036525', '1392050493']
# etime = ['1392046835', '1392053290','1392043319','1392047700', '1392052867']
# mag = ['32', '32', '32', '30', '30']
# phase = ['lightphase1', 'lightphase1','lightphase1','lightphase1','lightphase1']
# label = ['Light', 'Light','Light','Light', 'Light']
# room = ['2', '2','2', '1', '1']

#################################################################################################
# 10/02/2014 - 21:59 - 12/02/2014 - 03:00
# stime = ['10/2/2014-22.11.33','11/2/2014-12.30.09', '11/2/2014-12.51.47',
#         '11/2/2014-20.39.20', '10/2/2014-22.22.15', '11/2/2014-01.01.58',
#         '11/2/2014-02.17.39', '11/2/2014-09.34.00', '11/2/2014-17.47.05',
#         '11/2/2014-18.43.05','11/2/2014-21.12.16','11/2/2014-22.54.25',
#         '11/2/2014-23.06.25','11/2/2014-23.14.14','11/2/2014-23.42.37',
#         '12/2/2014-00.01.48','11/2/2014-00.01.33','11/2/2014-00.37.17',
#         '11/2/2014-14.33.14','11/2/2014-21.17.03','10/2/2014-22.02.01',
#         '11/2/2014-01.08.43','11/2/2014-09.55.25','11/2/2014-09.59.38',
#         '11/2/2014-19.19.05','11/2/2014-22.03.13','11/2/2014-22.05.32',
#         '11/2/2014-22.22.14','11/2/2014-08.59.46','11/2/2014-14.16.32',
#         '11/2/2014-21.05.39','11/2/2014-15.35.32','11/2/2014-10.41.32',
#         '11/2/2014-00.04.21','11/2/2014-01.09.10','11/2/2014-17.44.10',
#         '11/2/2014-14.34.17' ]
# etime = ['10/2/2014-22.51.07','11/2/2014-12.35.02','11/2/2014-13.10.57',
#         '11/2/2014-20.39.49','10/2/2014-22.58.10','11/2/2014-02.16.18',
#         '11/2/2014-03.20.33','11/2/2014-09.59.11','11/2/2014-17.58.26',
#         '11/2/2014-20.34.47','11/2/2014-22.54.03','11/2/2014-22.54.42',
#         '11/2/2014-23.14.11','11/2/2014-23.14.26','12/2/2014-00.08.48',
#         '12/2/2014-01.21.20','11/2/2014-00.36.37','11/2/2014-00.59.41',
#         '11/2/2014-17.57.32','11/2/2014-22.07.39','11/2/2014-00.33.40',
#         '11/2/2014-03.03.51','11/2/2014-09.58.10','11/2/2014-10.05.56',
#         '11/2/2014-20.07.09','11/2/2014-22.03.56','11/2/2014-22.06.32',
#         '12/2/2014-00.46.11','11/2/2014-09.55.08','11/2/2014-20.34.57',
#         '12/2/2014-01.37.52','11/2/2014-22.10.33','11/2/2014-15.13.11',
#         '11/2/2014-00.23.45','11/2/2014-06.14.45','12/2/2014-00.54.03',
#         '11/2/2014-15.45.37']
# mag = ['30','31','34',
#         '30','34','29',
#         '35','27','35',
#         '34','33','35',
#         '31','31','32',
#         '81','60','60',
#         '60','70','30',
#         '31','31','31',
#         '29','30','30',
#         '32','31','36',
#         '32','60','60',
#         '30','30','30',
#         '27']
# phase = ['lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase2','lightphase3','lightphase3',
#         'lightphase3','lightphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase2','powerphase2','powerphase2',
#         'powerphase2']
# label = ['TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'RP','RP','RP',
#         'RP','RP','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','LP','LP',
#         'TL','TL','TL',
#         'TL']
# room = ['1','1','1',
#         '1','2','2',
#         '2','2','2',
#         '2','2','2',
#         '2','2','2',
#         '4','5','5',
#         '6','6','5',
#         '5','5','5',
#         '5','5','5',
#         '5','6','6',
#         '6','6','6',
#         '3','3','3',
#         '4']

##########################################################################################################
# 11/2/2014- 21:59:00 to 13/2/2014 - 03:00:00

# stime = ['12/2/2014-11.00.44','12/2/2014-16.16.49','12/2/2014-18.36.44',
#         '12/2/2014-19.04.09','12/2/2014-22.01.44','13/2/2014-01.28.48',
#         '11/2/2014-22.54.25','11/2/2014-23.06.25','11/2/2014-23.14.14',
#         '11/2/2014-23.42.37','12/2/2014-08.08.25','12/2/2014-23.03.46',
#         '13/2/2014-01.43.57','11/2/2014-23.08.38','12/2/2014-06.57.16',
#         '11/2/2014-22.03.13','11/2/2014-22.05.32','11/2/2014-22.22.14',
#         '12/2/2014-09.28.40','12/2/2014-10.39.17','12/2/2014-12.39.10',
#         '12/2/2014-12.59.13','12/2/2014-17.05.43','12/2/2014-19.43.32',
#         '13/2/2014-01.23.51','12/2/2014-08.29.30','12/2/2014-09.02.35',
#         '12/2/2014-09.25.56','12/2/2014-09.30.34','12/2/2014-16.59.08',
#         '12/2/2014-19.38.36','12/2/2014-21.23.19','12/2/2014-23.41.54',
#         '12/2/2014-01.00.54','12/2/2014-09.43.16','12/2/2014-18.15.25',
#         '12/2/2014-23.18.44','12/2/2014-00.09.48','12/2/2014-00.01.50',
#         '12/2/2014-13.04.51']

# etime = ['12/2/2014-13.35.17','12/2/2014-17.13.27','12/2/2014-18.41.06',
#         '12/2/2014-21.02.05','13/2/2014-01.28.46','13/2/2014-01.28.59',
#         '11/2/2014-22.54.42','11/2/2014-23.14.11','11/2/2014-23.14.26',
#         '12/2/2014-00.08.48','12/2/2014-10.30.34','13/2/2014-00.24.19',
#         '13/2/2014-02.33.01','12/2/2014-02.53.41','12/2/2014-20.43.57',
#         '11/2/2014-22.03.55','11/2/2014-22.06.32','12/2/2014-00.46.11',
#         '12/2/2014-09.29.26','12/2/2014-11.53.12','12/2/2014-12.49.10',
#         '12/2/2014-13.07.55','12/2/2014-17.06.12','12/2/2014-23.57.35',
#         '13/2/2014-01.25.31','12/2/2014-08.33.03','12/2/2014-09.11.57',
#         '12/2/2014-09.26.25','12/2/2014-09.32.17','12/2/2014-17.03.14',
#         '12/2/2014-20.53.35','12/2/2014-23.35.32','13/2/2014-01.25.31',
#         '12/2/2014-01.36.49','12/2/2014-12.50.20','12/2/2014-21.40.06',
#         '12/2/2014-23.51.16','12/2/2014-02.57.24','12/2/2014-01.21.20',
#         '12/2/2014-18.08.33']
# label = ['TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','LP','LP',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'RP','RP','RP',
#         'RP','TL','RP',
#         'RP']
# mag = ['30','32','32',
#         '31','35','35',
#         '32','27','27',
#         '35','32','26',
#         '32','60','65',
#         '29','31','31',
#         '29','33','38',
#         '27','32','38',
#         '33','31','30',
#         '32','31','30',
#         '35','33','28',
#         '66','62','77',
#         '83','25','86',
#         '62']
# phase = ['lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','powerphase2',
#         'powerphase2','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'lightphase3','lightphase3','lightphase3',
#         'lightphase3','powerphase2','lightphase2',
#         'lightphase2']
# room = ['1','1','1',
#         '1','1','1',
#         '2','2','2',
#         '2','2','4',
#         '4','3','3',
#         '5','5','5',
#         '5','5','5',
#         '5','5','5',
#         '5','6','6',
#         '6','6','6',
#         '6','6','6',
#         '6','6','6',
#         '6','3','4',
#         '4']
####################################################################################################

# 12/2/2014-21:59 - 14/2/2014-03:00

# stime = ['12/2/2014-22.01.44', '13/2/2014-01.28.48','13/2/2014-08.04.27',
#         '13/2/2014-11.07.38','13/2/2014-11.23.19','13/2/2014-13.05.06',
#         '13/2/2014-17.53.34','12/2/2014-23.03.46','13/2/2014-01.43.57',
#         '13/2/2014-08.07.51','13/2/2014-13.23.05','13/2/2014-14.23.05',
#         '13/2/2014-01.47.58','13/2/2014-08.19.36','13/2/2014-01.23.49',
#         '13/2/2014-11.24.32','13/2/2014-11.30.04','13/2/2014-17.42.26',
#         '13/2/2014-17.43.47','13/2/2014-17.48.59','12/2/2014-23.41.54',
#         '13/2/2014-07.25.07','13/2/2014-09.10.22','13/2/2014-22.10.21',
#         '13/2/2014-22.11.35','13/2/2014-22.28.18','12/2/2014-23.18.44']

# etime = ['13/2/2014-01.28.46','13/2/2014-01.28.59','13/2/2014-08.34.09',
#         '13/2/2014-11.09.09','13/2/2014-12.22.16','13/2/2014-13.09.20',
#         '13/2/2014-20.47.31','13/2/2014-00.24.19','13/2/2014-02.33.01',
#         '13/2/2014-08.26.26','13/2/2014-13.24.39','13/2/2014-14.33.06',
#         '13/2/2014-15.40.21','13/2/2014-19.48.20','13/2/2014-01.25.31',
#         '13/2/2014-11.26.47','13/2/2014-11.31.27','13/2/2014-17.43.11',
#         '13/2/2014-17.47.29','13/2/2014-17.50.23','13/2/2014-01.34.25',
#         '13/2/2014-08.54.44','13/2/2014-09.22.10','13/2/2014-22.17.15',
#         '13/2/2014-22.28.49','14/2/2014-02.52.51','13/2/2014-00.52.23']

# label = ['TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'RP','LP','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','TL','TL',
#         'TL','LP','LP']

# mag = ['35','35','31',
#         '31','30','34',
#         '30','29','31',
#         '34','31','34',
#         '80','59','29',
#         '28','30','28',
#         '33','32','29',
#         '34','30','35',
#         '30','60','65']
# phase = ['lightphase1','lightphase1','lightphase1',
#         'lightphase1','lightphase1','lightphase1',
#         'lightphase1','powerphase2','powerphase2',
#         'powerphase2','powerphase2','powerphase2',
#         'lightphase2','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','powerphase3',
#         'powerphase3','powerphase3','lightphase3']

# room = ['1','1','1',
#         '1','1','1',
#         '1','4','4',
#         '4','4','4',
#         '4','3','5',
#         '5','5','5',
#         '5','5','6',
#         '6','6','6',
#         '6','3','6']


###################################################################################

# 13/2/2014-21:59 to 15/2/2014-03:00

# stime = ['14/2/2014-18.2.52','15/2/2014-00.11.36','14/2/2014-09.46.43',
# 		'14/2/2014-20.33.58','14/2/2014-00.05.48','13/2/2014-22.28.18',
# 		'14/2/2014-09.05.33','14/2/2014-17.45.50','14/2/2014-18.11.20',
# 		'14/2/2014-21.30.56','14/2/2014-21.56.37','14/2/2014-18.11.20',
# 		'14/2/2014-18.15.20','14/2/2014-18.22.10','14/2/2014-21.57.45',
# 		'14/2/2014-22.14.18','14/2/2014-08.54.01','14/2/2014-10.02.25',
# 		'14/2/2014-13.46.52','14/2/2014-12.56.51','14/2/2014-19.16.07']
# etime = ['14/2/2014-19.38.03','15/2/2014-02.29.23','14/2/2014-11.31.46',
# 		'14/2/2014-23.24.05','14/2/2014-01.06.41','14/2/2014-02.52.51',
# 		'14/2/2014-19.52.44','14/2/2014-17.45.53','14/2/2014-18.14.40',
# 		'14/2/2014-21.31.21','15/2/2014-00.55.55','14/2/2014-18.11.44',
# 		'14/2/2014-18.17.08','14/2/2014-21.30.18','14/2/2014-21.59.10',
# 		'14/2/2014-23.53.09','14/2/2014-09.59.58','14/2/2014-13.22.00',
# 		'14/2/2014-14.57.56','14/2/2014-19.06.15','14/2/2014-22.21.15']
# label = ['TL','TL','RP',
# 		'TL','TL','LP',
# 		'LP','TL','TL',
# 		'TL','TL','TL',
# 		'TL','TL','TL',
# 		'TL','TL','TL',
# 		'TL','RP','RP']
# mag = ['32','33','50',
# 		'29','33','60',
# 		'56','34','35',
# 		'39','39','32',
# 		'27','33','33',
# 		'31','34','29',
# 		'33','60','62']
# phase = ['lightphase1','lightphase1','lightphase1',
# 		'powerphase2','powerphase2','powerphase3',
# 		'powerphase3','powerphase3','powerphase3',
# 		'powerphase3','powerphase3','powerphase3',
# 		'powerphase3','powerphase3','powerphase3',
# 		'powerphase3','powerphase3','powerphase3',
# 		'powerphase3','lightphase3','lightphase3']
# room =['1','1','1',
# 		'4','3','3',
# 		'4','5','5',
# 		'5','5','6',
# 		'6','6','6',
# 		'6','5','5',
# 		'6','6','6']

# ## Test Label

# for row in zip(stime, etime, mag, label, phase, room):
   
#     print "[", row[0], " - ", row[1], " - ", row[2]," - ", row[3]," - ", row[4], " - ", row[5],"]"

# # # Output file
# print "Writing to file - ", filename_gt
# writer = csv.writer(open(filename_gt, "w"))

# # # For test dataset

# writer.writerow(['start_time', 'end_time', 'magnitude', 'appliance','phase' ,'room'])
# no_loop = len(stime)

# # Convert to timestamp and store in a CSV
# for x in range(no_loop):
    
#     s_tstamp = long(
#         time.mktime(time.strptime(stime[x], "%d/%m/%Y-%H.%M.%S")) )
#     e_tstamp = long(
#         time.mktime(time.strptime(etime[x], "%d/%m/%Y-%H.%M.%S")) )

    
# # For test dataset
    
#     writer.writerow([s_tstamp ,e_tstamp, mag[x], label[x],phase[x], room[x]])

#####################################################################################

# Output csv file

df = pd.read_csv(filename_gt)

################################################################################
# 10/02/2014 - 21:59 to 12/02/2014 - 03.00 (corrections)
idx = df.index[df.end_time == 1392053290]
print df.ix[idx]
df.loc[idx,'end_time'] = 1392053167

idx = df.index[df.start_time == 1392142357]
print df.ix[idx]
df.loc[idx,'end_time'] = 1392143824

idx = df.index[df.end_time == 1392136833]
print df.ix[idx]
df.loc[idx,'end_time'] = 1392128418


idx = df.index[df.start_time == 1392059237]
print df.ix[idx]
df = df.ix[df.index - idx]
###################################################################################
# 11/02/2014 - 21:59 to 13/02/2014 - 03.00 (corrections)

# idx = df.index[df.end_time == 1392231259]
# print df.ix[idx]
# df.loc[idx, 'end_time'] = 1392227069

# idx = df.index[df.end_time == 1392153821]
# print df.ix[idx]
# df.loc[idx, 'end_time'] = 1392218037
# idx = df.index[df.start_time == 1392168436]
# print df.ix[idx]
# df = df.ix[df.index - idx]

# idx = df.index[df.start_time == 1392181757]
# print idx
# print df.ix[idx]
# df.loc[idx, 'room'] =  '6'

# idx = df.index[df.end_time == 1392190675]
# print df.ix[idx]
# df.loc[idx, 'end_time'] = 1392190548

#adding an extra row to the dataframe
# row = pd.DataFrame([dict(start_time = '1392143988', end_time = '1392145635', magnitude = '26', appliance='TL', phase = 'powerphase2', room = '3'), ])
# df = df.append(row, ignore_index=True)

# row = pd.DataFrame([dict(start_time = '1392227417', end_time = '1392231259', magnitude = '31', appliance='TL', phase = 'powerphase2', room = '3'), ])
# df = df.append(row, ignore_index=True)
# print row
# df = df[df.start_time != 1392143988]

#######################################################################################

# 12/2/2014-21:59 to 14/2/2014-03:00

# idx = df.index[df.start_time == 1392236278]
# print df.ix[idx]
# df = df.ix[df.index - idx]


# idx = df.index[df.start_time == 1392227324]
# print df.ix[idx]
# df = df.ix[df.index - idx]
###################################################################################

# 13/2/2014-21:59 to 15/2/2014-03:00


# idx = df.index[df.end_time == 1392387764]
# df.loc[idx, 'end_time'] = 1392386984
# print df.ix[idx]

# idx = df.index[df.end_time ==  1392405955]
# df.loc[idx, 'end_time'] = 1392402189

# idx = df.index[df.start_time == 1392396258]
# df = df.ix[df.index - idx]




stime = df['start_time']
etime = df['end_time']
mag = df['magnitude']
label = df['appliance']
phase = df['phase']
room = df['room']
df.index = arange(0, len(df))
print 'Final df'
print df
df.to_csv(filename_op, cols = df.columns.values, index = False)


#######################################################################################################
# for row in zip(stime, etime, mag, label, phase, room):
   
#     print "[", row[0], " - ", row[1], " - ", row[2]," - ", row[3]," - ", row[4], " - ", row[5],"]"

# # # Output file
# print "Writing to file - ", filename_op
# writer = csv.writer(open(filename_op, "w"))

# # # For test dataset

# writer.writerow(['start_time', 'end_time', 'magnitude', 'appliance','phase' ,'room'])
# no_loop = len(stime)

# # Convert to timestamp and store in a CSV
# for x in range(no_loop):
#     # s_tstamp = long(
#     #     time.mktime(time.strptime(stime[x], "%d/%m/%Y-%H.%M.%S")) )
#     # e_tstamp = long(
#     #     time.mktime(time.strptime(etime[x], "%d/%m/%Y-%H.%M.%S")) )

    
# # For test dataset
    
#     #print tstamp, event[x], room[x]
#     print stime[x], etime[x], mag[x], label[x],phase[x], room[x]
#     writer.writerow([stime[x], etime[x], mag[x], label[x],phase[x], room[x]])
