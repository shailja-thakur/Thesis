"""
Script to retrieve meter data from smap db
and store them in csv files

Input: start time and end time
Output: csv file with timestamped power values of all phases
Format: <timestamp, powerphase1, powerphase2, powerphase3, power>

Author: Manaswi Saha
Date: 22nd Sep 2013
"""

import os
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

"""
Time range syntax: %m/%d/%Y, %m/%d/%Y %H:%M, or %Y-%m-%dT%H:%M:%S.
For instance 10/16/1985 and 2/29/2012 20:00 are valid

Example query:
payload="select data in (now -5minutes, now) where
Metadata/Extra/Wing ='BC' and Metadata/Extra/PhysicalParameter='PowerPhase1'
and Metadata/Extra/Type='Power'"

Refer: http://www.cs.berkeley.edu/~stevedh/smap2/archiver.html

"""

# Input
# wing = sys.argv[1]
s_time = sys.argv[1]
e_time = sys.argv[2]
wing = 'BC'
floor = str(0)
# s_time = raw_input("Enter start time in the form MM/DD/YYYY HH:MM :: ")
# e_time = raw_input("Enter end time in the form MM/DD/YYYY HH:MM :: ")

# Processing: Get data for the above time frame for all three phases
power = []
lpower = []
url = "http://192.168.1.40:9101/api/query"

# Store power
payload = ("select data in ('" + s_time + "','" + e_time + "') "
           "limit 200000 "
           "where Metadata/Extra/Wing ='" + wing + "' and "
           "Metadata/Extra/PhysicalParameter='Power'"
           " and Metadata/Extra/Type='Power'"
           " and Metadata/Location/Floor='" + floor + "'")
print payload

r = requests.post(url, data=payload)
readings = np.array(r.json()[0]['Readings'])

df1 = (
    pd.DataFrame(
        np.zeros(
            (len(readings), 5)), columns=['time', 'powerphase1', 'powerphase2',
                                          'powerphase3', 'power']))
df1['time'] = time = readings[:, 0] / 1000
df1['power'] = pwr = readings[:, 1]
tp = np.array([datetime.datetime.fromtimestamp(x) for x in time])

# Phase wise Power
for i in range(1, 4):
    # payload = ("select data in ('9/21/2013 18:30', '9/21/2013 21:30') "
    # 	"where Metadata/Extra/Wing ='"+ wing + "' and "
    # 	"Metadata/Extra/PhysicalParameter='PowerPhase"+ str(i) +
    # "' and Metadata/Extra/Type='Power'")
    payload = ("select data in ('" + s_time + "','" + e_time + "') "
               "limit 200000 "
               "where Metadata/Extra/Wing ='" + wing + "' and "
               "Metadata/Extra/PhysicalParameter='PowerPhase" + str(i) +
               "' and Metadata/Extra/Type='Power' and "
               "Metadata/Location/Floor='" + floor + "'")
    print payload

    r = requests.post(url, data=payload)
    print r
    readings = np.array(r.json()[0]['Readings'])
    df1['powerphase' + str(i)] = p = readings[:, 1]
    power.append(p)

#------------------------------------------------------------------------------

# Store lighting power
payload = ("select data in ('" + s_time + "','" + e_time + "') "
           "limit 200000 "
           "where Metadata/Extra/Wing ='" + wing + "' and "
           "Metadata/Extra/PhysicalParameter='Power'"
           "and Metadata/Extra/Type='Light Backup'"
           " and Metadata/Location/Floor='" + floor + "'")
print payload

r = requests.post(url, data=payload)
readings = np.array(r.json()[0]['Readings'])

df2 = (pd.DataFrame(np.zeros((len(readings), 5)),
       columns=['time', 'lightphase1', 'lightphase2', 'lightphase3', 'lightpower']))
df2['time'] = time = readings[:, 0] / 1000
df2['lightpower'] = lpwr = readings[:, 1]
tl = np.array([datetime.datetime.fromtimestamp(x) for x in time])

for i in range(1, 4):
    # payload = ("select data in ('9/21/2013 18:30', '9/21/2013 21:30') "
    # 	"where Metadata/Extra/Wing ='"+ wing + "' and "
    # 	"Metadata/Extra/PhysicalParameter='PowerPhase"+ str(i) +
    # "' and Metadata/Extra/Type='Power'")
    payload = ("select data in ('" + s_time + "','" + e_time + "') "
               "limit 200000 "
               "where Metadata/Extra/Wing ='" + wing + "' and "
               "Metadata/Extra/PhysicalParameter='PowerPhase" + str(i) +
               "' and Metadata/Extra/Type='Light Backup'"
               " and Metadata/Location/Floor='" + floor + "'")

    print payload

    r = requests.post(url, data=payload)
    print r
    readings = np.array(r.json()[0]['Readings'])
    df2['lightphase' + str(i)] = p = readings[:, 1]
    lpower.append(p)

# Make a single data frame by joining them
# df = pd.merge(df1, df2, on='time')
# time = df['time']
# print df.head()

# Store output in csv
# input time format: 9/21/2013 18:30
# output time format: 21_9_18_30_21_9_21_30
stimestamp = s_time.split(' ')
date1 = (stimestamp[0].split('/'))[1]
month1 = (stimestamp[0].split('/'))[0]
hour1 = (stimestamp[1].split(":"))[0]
minute1 = (stimestamp[1].split(":"))[1]

etimestamp = e_time.split(' ')
date2 = (etimestamp[0].split('/'))[1]
month2 = (etimestamp[0].split('/'))[0]
hour2 = (etimestamp[1].split(":"))[0]
minute2 = (etimestamp[1].split(":"))[1]

seq = (date1, month1, hour1, minute1, date2, month2, hour2, minute2)
directory = ('_').join(seq)
directory = 'CompleteDataSets/Hostel/' + directory
print "Directory:", directory

if not os.path.exists(directory):
    os.makedirs(directory)
new_csv_P = directory + '/Power.csv'
new_csv_L = directory + '/Light.csv'
df1.to_csv(new_csv_P, index=False)
df2.to_csv(new_csv_L, index=False)

# Light and Power Plot
fig = plt.gcf()
# row1
ax1 = plt.subplot(4, 2, 1)
# ax1.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
plt.plot(tp, df1['powerphase1'])
plt.title("Power Plot for " + directory)
plt.ylabel("PowerPhase(R) \n in Watts")

ax2 = plt.subplot(4, 2, 2)
# ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
plt.plot(tl, df2['lightphase1'])
plt.title("Light Plot for " + directory)
plt.ylabel("LP Phase1(R) \n in Watts")

# row2
plt.subplot(4, 2, 3, sharex=ax1)
plt.plot(tp, df1['powerphase2'], 'r')
plt.ylabel("PowerPhase2(Y) \n in Watts")

plt.subplot(4, 2, 4, sharex=ax2)
plt.plot(tl, df2['lightphase2'], 'r')
plt.ylabel("LP Phase2(Y) \n in Watts")

# row3

plt.subplot(4, 2, 5, sharex=ax1)
plt.plot(tp, df1['powerphase3'], 'g')
plt.ylabel("PowerPhase3(B) \n in Watts")

plt.subplot(4, 2, 6, sharex=ax2)
plt.plot(tl, df2['lightphase3'], 'g')
plt.ylabel("LP Phase3(B) \n in Watts")

# row4

plt.subplot(4, 2, 7, sharex=ax1)
plt.plot(tp, df1['power'], 'y')
plt.ylabel("Power")
plt.xlabel("Time")

plt.subplot(4, 2, 8, sharex=ax2)
plt.plot(tl, df2['lightpower'], 'y')
plt.ylabel("LP Power")
plt.xlabel("Time")

fig.autofmt_xdate()

plt.show()
