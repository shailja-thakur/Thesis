"""
Label Sound/Wifi files with ground truth

Input: Unlabeled Collected dataset and ground truth
Output: Labeled dataset

Author: Manaswi Saha
Date: Oct 30, 2013
Updated: Nov 7, 2013
"""

import sys
import glob
import pandas as pd
import datetime as dt

# Common variables
sensor = 'Sound'
# folder = 'CompleteDataSets/Apartment/29_10_18_50_29_10_19_33/phone2/'
# folder = 'CompleteDataSets/Apartment/SoundTrainingSet/testsets/switch/'
# folder = 'CompleteDataSets/Apartment/SoundTrainingSet/Phone2/in-pocket/'

# Input arguments
folder = sys.argv[1]
phno = sys.argv[2]
sensor = sys.argv[3]

# Input files
scsvfile = 'Sound' + phno + '.csv'  # sys.argv[1]
wcsvfile = 'Wifi' + phno + '.csv'  # sys.argv[2]

# Ground truth file
gfile = glob.glob(folder + "ground_truth/*.csv")
gfile.sort()

# Which sensor's training data?
if sensor == "sound":
    k = 0
    # Input files to label
    filenm = folder + scsvfile
    print "Sound File", filenm
    df_sip = pd.read_csv(filenm)

    # Sound ground truth
    gfile = gfile[0]
    # gfile = folder + "ground_truth/train_703_s703_1.csv"
else:
    k = 1
    filenm = folder + wcsvfile
    print "Wifi File", filenm
    df_wip = pd.read_csv(filenm)

    # Wifi ground truth
    gfile = gfile[1]
    # gfile = folder + "ground_truth/train_703_w703_1.csv"


print "Ground Truth File", gfile
# sys.exit(1)
# Getting metadata
# arr = gfile.split('/')
# g_filenm = (arr[-1]).split('_')
# dataset_type = g_filenm[0]
# apt_no = g_filenm[1]
# idx = g_filenm[2].replace('.csv', '')

dataset_type = "train"
apt_no = '102A'
idx = "t9"


# For Sound
# Remove entries where timestamps are incorrect or rows are merged
if k == 0:

    time = (df_sip.time).tolist()
    value = (df_sip.value).tolist()
    label = (df_sip.label).tolist()

    dates = []
    values = []
    labels = []
    for key, val in enumerate(zip(time, value, label)):
        try:
            d = dt.datetime.fromtimestamp(long(val[0]) / 1000)
            if d.year == 2013:
                dates.append(val[0])
                values.append(val[1])
                labels.append(val[2])
            else:
                print "Insufficient data: Error occurred in line no.", key, "Time", val
        except Exception, e:
            print "Error occurred in line no.", key, "Time", val
    df_tmp = pd.DataFrame(
        {'time': dates, 'value': values, 'label': labels}, columns=["time", "value", "label"])
    df_tmp = df_tmp.sort(['time'])
    df_tmp['time'] = (df_tmp.time).astype(long)
    df_tmp['value'] = (df_tmp.value).astype(int)
    # df_tmp.to_csv('tmp1.csv', index=False)
    print "NEW Dataframe:", df_tmp.head()
    print "Total number of records", len(df_tmp.index)

# Temporary generate individual sound/wifi labeled files
df_gt = pd.read_csv(gfile)
label = ''
name = scsvfile

# Make labeled files based on label types
csv_files = []
for i, row in enumerate(df_gt.values):

    if k == 0:
        print "Labeling for Sound data -------------------------------------"
        label = row[2]
    else:
        print "Labeling for Wifi data -------------------------------------"
        df_tmp = df_wip
        name = wcsvfile
        label = row[2]
    # Selecting the desired timestamp range
    df_new = df_tmp[(df_tmp.time >= long(row[0])) & (df_tmp.time <= long(row[1]))]
    # if i == 0:
    #     print "OLD:", df_new_tmp.tail()
    #     print "NEW", df_new.head()

    df_new.label = label
    print "Label", label, row[0], row[1]

    new_loc = ('labeled_tmp/' + str(idx) + "_" + label + "_" +
               apt_no + "_" + str(i) + ".csv")
    df_new.to_csv(new_loc, index=False)

    # Store created files in csv_files
    csv_files.append(new_loc)

print "\n", csv_files

# Concatenate the files into a labeled file
df_list = []
for csvfval in csv_files:
    # print "\nCSVFile", csvfval
    df = pd.read_csv(csvfval)
    df_list.append(df)
# print df_list
df_concat_strain = pd.concat(df_list)
df_concat_strain = df_concat_strain.sort(['label'])
print "HEAD:", df_concat_strain.head()
print "TAIL:", df_concat_strain.tail()
print "Labels:", df_concat_strain.label.unique()

# Store file
new_loc = folder + name
df_concat_strain.to_csv(new_loc, index=False)
print "-" * 20


# ------------------------------------------------------------------
# For Wifi
# Remove entries where timestamps are incorrect or rows are merged
#     time = (df_ip.time).tolist()
#     mac = (df_ip.mac).tolist()
#     ssid = (df_ip.ssid).tolist()
#     rssi = (df_ip.rssi).tolist()
#     label = (df_ip.label).tolist()

#     dates = []
#     macs = []
#     ssids = []
#     rssis = []
#     labels = []
#     for key, val in enumerate(zip(time, value, label)):
#         try:
#             d = dt.datetime.fromtimestamp(long(val[0]) / 1000)
#             if d.year == 2013:
#                 dates.append(val[0])
#                 values.append(val[1])
#                 labels.append(val[2])
#             else:
#                 print "Insufficient data: Error occurred in line no.", key, "Time", val
#         except Exception, e:
#             print "Error occurred in line no.", key, "Time", val
#     time,mac,ssid,rssi,label
#     df_tmp = pd.DataFrame(
#         {'time': dates, 'value': values, 'label': labels}, columns=["time", "value", "label"])
#     df_tmp = df_tmp.sort(['time'])
#     print "NEW Dataframe:", df_tmp.head()
