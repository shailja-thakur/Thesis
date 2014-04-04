# Script to localize a user
# Assumptions:
#   1. Averaged over one second
# Author: Manaswi Saha
# Updated on: Sep 27, 2013


import csv
import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import glob
import re
import math
import numpy as np
from path import *
import os

# import plot_confusion_matrix as pltcm
from CONFIGURATION import *
# import summarize as sm

import warnings

warnings.filterwarnings('ignore')

# Script to convert labeled phone Wifi test data into the desired form
# Format::
# Location Fingerprint: <time, rssi1, rssi2, rssi3,..., rssiN, location>
# Number of RSSI features: number of unique SSID:MACs seen in the data
# Operations done:
#   1. Averaging over a second
#   2. Conversion to proper format
#
# Author: Manaswi Saha
# Date: August 30, 2013


def format_data(wifi_csv, dataset_type, room_no, outcsv):

    filename = room_no + '.csv'
    # Step 1 : Read CSV
    ip_df = pd.read_csv(wifi_csv + filename )

    # Output csv file
    
    outcsv =  outcsv + dataset_type + '_' +  filename 
    writer = csv.writer(open(outcsv, 'w'))

    # Step 2: Find the unique MACID and corresponding RSSI
    mac_id = ip_df.mac.unique()
    ssid_header = mac_id.tolist()
    writer.writerow(['timestamp'] + ssid_header + ['label'])
    #print str(['timestamp'] + ssid_header + ['label'])
    # Step 3: Take average of RSSI value over one second for each unique SSID
    # and store it in the CSV
    sumRssi = {}
    count = {}
    avgRssi = {}
    ploc = ""
    df_size = len(ip_df)
    ptime = 0
    # initialize sumRssi
    for i in mac_id:
        sumRssi[i] = 0
        count[i] = 0

    # taking averages over each SSID
    for i, row in enumerate(ip_df.values):
        
        time_i = (int(row[0])) / 1000
        macid_i = row[1]
        loc_i = row[4]
        # for the first record
        if ptime == 0:
            sumRssi[macid_i] = sumRssi[macid_i] + int(row[3])
            count[macid_i] = count[macid_i] + 1

        # reached last record in the data frame
        elif ptime == time_i and i == (df_size - 1):
            sumRssi[macid_i] = sumRssi[macid_i] + int(row[3])
            count[macid_i] = count[macid_i] + 1
            avg = []
            for i in mac_id:
                if count[i] != 0:
                    rssi = avgRssi[i] = sumRssi[i] / count[i]
                else:
                    rssi = 0
                if rssi < WIFI_THRESHOLD:
                    rssi = 0
                avg.append(rssi)
                sumRssi[i] = 0
                count[i] = 0

            writer.writerow([ptime] + avg + [ploc])
            #print str([ptime] + avg + [ploc])

        elif ptime == time_i:
            sumRssi[macid_i] = sumRssi[macid_i] + int(row[3])
            count[macid_i] = count[macid_i] + 1

        else:
            if count[macid_i] != 0:
                avg = []

                for i in mac_id:
                    if count[i] != 0:
                        rssi = avgRssi[i] = sumRssi[i] / count[i]
                    else:
                        rssi = 0
                    if rssi < WIFI_THRESHOLD:
                        rssi = 0
                    avg.append(rssi)
                    sumRssi[i] = 0
                    count[i] = 0

                writer.writerow([ptime] + avg + [ploc])
                count[macid_i] = 1
                sumRssi[macid_i] = int(row[3])
                #print str([ptime] + avg + [ploc])
        ptime = time_i
        ploc = loc_i
    return outcsv
    # if dataset_type == "train":
    #     sm.cal_window_mean(outcsv, dataset_type, filename, 10)


# Author: Manaswi Saha
# Date: August 29, 2013

def wifi_location(test_csv, room_no, outcsv ):

    # Step1: Get data from radio map test set
    loc_df = pd.read_csv(test_csv)

    # Output csv file
    
    timestamp = []
    loc = []
    #print df
    filename = room_no + '_location_'+ '.csv'
    outcsv = outcsv + filename 
    writer = csv.writer(open(outcsv, 'w'))
    writer.writerow(['timestamp'] + ['location'] )
    # Step2: Testing Hostel access point existence 
    # Retrieve unique mac_ids 
    print 'Original file length', len(loc_df)
    rssi_avg = 0
    avg_mac_0 = []
    mac_id = pd.Series(loc_df.columns[2:-2])
    # MAC address of Access points visible on ground floor
    existing_mac_gnd_flr = mac_id[mac_id.str.contains(GND_AP_MACS, na = False) == True]
    existing_mac_ft_flr = mac_id[(mac_id.str.contains(FIRST_AP_MACS, na = False) == True)]
    
    print "Mac ground floor"
    print existing_mac_gnd_flr
    print "Mac first floor"
    print existing_mac_ft_flr
    done = False
    # Determine in/out(1/0) of the hospital
    for i in loc_df.index:
        # Check for connection with nearest APs
        if (len(existing_mac_gnd_flr) > 0) :
            for mac_id in existing_mac_gnd_flr:
                avg_mac_0.append(loc_df.ix[i][mac_id])

            rssi_avg = np.mean(avg_mac_0)
            avg_mac_0  = []

            if ((rssi_avg < 0) & (rssi_avg >= WIFI_THRESHOLD)):
                location = 1 
                time = loc_df.ix[i]['timestamp']
                loc.append(location)
                timestamp.append(time)
                writer.writerow([time] + [location])
            
            # Check for connection with APs in affinity
            elif (len(existing_mac_ft_flr) > 0) :
                for mac_id in existing_mac_ft_flr:
                    avg_mac_0.append(loc_df.ix[i][mac_id])
                rssi_avg = np.mean(avg_mac_0)
            
                if ((rssi_avg < 0) & (rssi_avg >= WIFI_THRESHOLD)):
                    location = 1
                    time = loc_df.ix[i]['timestamp']
                    loc.append(location)
                    timestamp.append(time)
                    writer.writerow([time] + [location])
                else:
                    
                    location = 0
                    time = loc_df.ix[i]['timestamp']
                    loc.append(location)
                    timestamp.append(time)
                    writer.writerow([time] + [location])
            else:
                location = 0
                time = loc_df.ix[i]['timestamp']
                loc.append(location)
                timestamp.append(time)
                writer.writerow([time] + [location])
                        
        # Check for connection with APs in affinity
        elif (len(existing_mac_ft_flr) > 0) :
            for mac_id in existing_mac_ft_flr:
                avg_mac_0.append(loc_df.ix[i][mac_id])
            rssi_avg = np.mean(avg_mac_0)
            
            if ((rssi_avg < 0) & (rssi_avg > WIFI_THRESHOLD)):
                
                location = 1
                time = loc_df.ix[i]['timestamp']
                loc.append(location)
                timestamp.append(time)
                writer.writerow([time] + [location])
            else:
                print 'first floor macs not satisfying '
                location = 0
                time = loc_df.ix[i]['timestamp']
                loc.append(location)
                timestamp.append(time)
                writer.writerow([time] + [location])
                
        # Outside
        else:
            location = 0
            time = loc_df.ix[i]['timestamp']
            loc.append(location)
            timestamp.append(time)
            writer.writerow([time] + [location])
        done = True

    if done == True:

        df = pd.read_csv(outcsv)
        df = df.sort_index(by = 'timestamp')
        df.to_csv(outcsv, columns = ['timestamp', 'location'], index = False)

    print len(timestamp)
    print len(loc)
    
    #df = pd.DataFrame({'timestamp' : timestamp, 'location' : loc})
    #return df
    
    #df = pd.read_csv(outcsv)
    #print 'Location csv length', len(df)

# def max_min_timestamp(loc_csv ):
#     # Determine all the overlapping and non_overlapping location of the users
#     #    in windows

#     # Step1:Find Minimum Timestamp 
#     min_timestamp = 2000000000000
#     max_timestamp = 0
#     loc_dir = loc_csv + '*.csv'

#     for user_loc in glob.glob(loc_dir):
#         df_loc = pd.read_csv(user_loc)
      
#         for i in df_loc.index:
#             if int(df_loc.ix[df_loc.index[0]]['timestamp']) < int(min_timestamp):
#                 min_timestamp = df_loc.ix[df_loc.index[0]]['timestamp']
#             if int(df_loc.ix[df_loc.index[-1]]['timestamp']) > int(max_timestamp):
#                 max_timestamp = df_loc.ix[df_loc.index[-1]]['timestamp']
        
#     print 'Minimum Timestamp', min_timestamp
#     print 'Maximum Timestamp', max_timestamp

#     return min_timestamp, max_timestamp

    
# def fill_missing_samples(loc_csv):

#     min_timestamp, max_timestamp = max_min_timestamp(USER_LOCATION_PATH)
#     loc_dir = loc_csv + '*.csv'
    
#     for user_loc in glob.glob(loc_dir):
#         print user_loc
#         df_loc = pd.read_csv(user_loc)
    
#         filename = user_loc.split('.')[0].split('/')[9] + 'fomatted'+ '.csv'
#         outcsv = USER_LOCATION_FORMATTED_PATH + filename 
#         writer = csv.writer(open(outcsv, 'w'))
#         writer.writerow(['timestamp'] + [str(user_loc.split('.')[0].split('/')[9].split('_')[0])] )
#         beg = min_timestamp
#         end = max_timestamp
#         for idx in df_loc.index:
            
#             curr_time = df_loc.ix[idx]['timestamp']
#             curr_location = df_loc.ix[idx]['location']
#             if idx == 0:
#                 #print df_loc.ix[idx]['timestamp']
#                 diff = curr_time - beg
#                 #print 'beginning difference', diff
#                 if diff == 0:
                   
#                     location = curr_location
#                     time = beg

#                     #print time, location
#                     writer.writerow([time] + [location])
#                     continue
                
#                 if diff > 0:
#                     count = diff
#                     #print 'count', count
#                     for i in range(0, count+1):
#                         time = (beg + i)
#                         location = 0
#                         #print time, location
#                         writer.writerow([time] + [location])
#                         continue

#             if ((idx > 0) & (idx < df_loc.index[-1])):

#                 prev = df_loc.ix[idx - 1]['timestamp']
#                 prev_location = df_loc.ix[idx - 1]['location']
#                 diff = curr_time - prev
#                 #print 'between missing samples',prev, curr_time
#                 #print 'count', diff
#                 if diff == 1:
#                     location = curr_location
#                     time = prev + 1
#                     #print time, location
#                     writer.writerow([time] + [location])
#                     continue
                        
#                 if ((diff > 0) & (diff <= MIN_STAY_TIME)):
#                     count = diff
#                     for i in range(1, count+1):

#                         time = (prev + i)
#                         location = prev_location
#                         #print time, location
#                         writer.writerow([time] + [location])
#                         continue
#                 if diff > MIN_STAY_TIME:
#                     count = diff
#                     for i in range(1, count+1):

#                         time = prev + i
#                         location = 0
#                         #print time, location
#                         writer.writerow([time] + [location])
#                         continue

#             if idx == df_loc.index[-1]:
#                 print 'index',idx
#                 prev = df_loc.ix[idx - 1]['timestamp']
#                 prev_location = df_loc.ix[idx - 1]['location']
#                 print 'maximum time and last time', end, prev

#                 diff = prev - end

#                 print 'last difference', diff
#                 print 'last missing samples', prev, end
#                 print 'count', diff
#                 if diff == 0:
#                     location = curr_location
#                     time = prev + 1
#                     print time, location
#                     writer.writerow([time] + [location])
#                     continue
#                 if ((int(math.fabs(diff)) > 0) & (int(math.fabs(diff)) <= MIN_STAY_TIME)):
#                     count = int(math.fabs(diff))
#                     #print 'count', count
#                     for i in range(1, count + 1):

#                         time = (prev + i)
#                         location = curr_location
#                         #print time, location
#                         writer.writerow([time] + [location]) 
#                         continue
#                 if int(math.fabs(diff)) > MIN_STAY_TIME:
#                     count = int(math.fabs(diff))
#                     for i in range(1, count + 1):

#                         time = (prev + i)
#                         location = 0  
#                         #print time, location
#                         writer.writerow([time] + [location])
#                         continue
        
        
#         #df = pd.read_csv(outcsv)
#         #print df
        
# def users_location_table():
   
#     min_t, max_t = max_min_timestamp(USER_LOCATION_PATH)
#     df = pd.DataFrame({'timestamp':list(range(min_t, max_t))})
#     filename = 'user_location_table'+ '.csv'
#     outcsv = USER_ATTRIBUTION_TABLES + filename 
#     writer = csv.writer(open(outcsv, 'w'))
#     writer.writerow(['timestamp'] + ['rooms']  )

#     timestamp = []
#     rooms = []
#     for i in glob.glob(USER_LOCATION_FORMATTED_PATH + '*.csv'):
#         frame = pd.read_csv(i)
#         df = pd.merge(df, frame)

#         #print df
#     #print df
#     for idx in df.index:
#         column = ''

#         for col in df.columns[1:]:
            
#             column = column + str(df.ix[idx][col])
#             if col != df.columns[len(df.columns[1:])]:
#                 column = column + ','
#         timestamp.append(df.ix[idx]['timestamp'])
#         rooms.append(column)
#         writer.writerow([df.ix[idx]['timestamp']] +  ['"'+column+'"'])
#     df = pd.DataFrame({'timestamp' : timestamp, 'rooms' : rooms})
#     return df
        
    

# def overlapping_non_overlapping_sets():


#     df = users_location_table()
#     # print 'Finding overlapping and non-overlapping sets'
#     #df = pd.read_csv(USER_ATTRIBUTION_TABLES + 'user_location_table.csv')
   
#     filename = 'location_room_set'+ '.csv'
#     outcsv = USER_ATTRIBUTION_TABLES + filename 
#     writer = csv.writer(open(outcsv, 'w'))
#     writer.writerow(['start_time'] + ['end_time'] + ['room_set'] )
#     prev_room_set = []
#     curr_room_set = []
#     start_time = df.ix[df.index[0]]['timestamp']
#     start = []
#     end = []
#     overlapp_set = []
#     for idx in df.index:
#         if idx == df.index[0]:

#             start_time = df.ix[df.index[0]]['timestamp']
#             end_time = df.ix[df.index[0]]['timestamp']
#             #room_set = df.ix[0]['rooms'].split('"')[1]
#             room_set = df.ix[0]['rooms']           
#             curr_room_set = room_set
                   
#             #print 'index', idx, 'start_time',start_time,'end_time',end_time, 'room_set', room_set

#         if ((idx > df.index[0]) & (idx < df.index[-1])):

#             # curr_room_set = df.ix[idx]['rooms'].split('"')[1]
#             # prev_room_set = df.ix[idx - 1]['rooms'].split('"')[1]
#             curr_room_set = df.ix[idx]['rooms']
#             prev_room_set = df.ix[idx - 1]['rooms']
#             #print 'index > 1','index',idx,'prev_room_set',prev_room_set,'curr_room_set',curr_room_set,'start_time',start_time,'curr_time',df.ix[idx ]['timestamp']
                   
#             if set(curr_room_set) == set(prev_room_set):
#                 continue
#             if set(curr_room_set) != set(prev_room_set):
#                 # print 'room set changed'
#                 # print 'curr_room_set',curr_room_set,'prev_room_set',prev_room_set
#                 #print 'start_time',start_time,'end_time',end_time
#                 end_time = df.ix[idx - 1]['timestamp']
#                 #print 'end_time',end_time
#                 start.append(start_time)
#                 end.append(end_time)
#                 overlapp_set.append(prev_room_set)
#                 writer.writerow([start_time] + [end_time] + [prev_room_set])
#                 print 'index', idx, 'start_time',start_time,'end_time',end_time, 'room_set', prev_room_set
               
                
                
#                 start_time = df.ix[idx]['timestamp']
#                 #print 'start_time',start_time
#         if idx == df.index[-1]:
#             #print 'last index'
#             # curr_room_set = df.ix[idx]['rooms'].split('"')[1]
#             # prev_room_set = df.ix[idx - 1]['rooms'].split('"')[1]
#             curr_room_set = df.ix[idx]['rooms']
#             prev_room_set = df.ix[idx - 1]['rooms']
#             if set(curr_room_set) == set(prev_room_set):
#                 end_time = df.ix[idx ]['timestamp']

#                 start.append(start_time)
#                 end.append(end_time)
#                 overlapp_set.append(prev_room_set)

#                 writer.writerow([start_time] + [end_time] + [prev_room_set])
#                 print 'index', idx, 'start_time',start_time,'end_time',end_time, 'room_set', prev_room_set
#             else:
               
#                 # print 'room set changed'
#                 # print 'curr_room_set',curr_room_set,'prev_room_set',prev_room_set
#                 end_time = df.ix[idx - 1]['timestamp']

#                 start.append(start_time)
#                 end.append(end_time)
#                 overlapp_set.append(prev_room_set)

#                 start.append(df.ix[idx]['timestamp'])
#                 end.append(df.ix[idx]['timestamp'])
#                 overlapp_set.append(curr_room_set)

#                 writer.writerow([start_time] + [end_time] + [prev_room_set])
#                 writer.writerow([df.ix[idx]['timestamp']]+ [df.ix[idx]['timestamp']] + [curr_room_set])
#                 print 'index', idx, 'start_time',start_time,'end_time',end_time, 'room_set', curr_room_set
#     df = pd.DataFrame({'start_time' : start, 'end_time' : end, 'room_set' : overlapp_set})
#     return df

def classify_location(train_csv, test_csv, apt_no, idx):

    # Step1: Get data from radio map (training set) and test set
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df.label != 'Bathroom1']
    train_df = train_df[train_df.label != 'Bedroom3']
    train_df = train_df[train_df.label != 'Bedroom1']
    train_df = train_df[train_df.label != 'Shared Bathroom']
    test_df = pd.read_csv(test_csv)

    # Step2: Check both sets are compatible.
    # If they are not, use the SSIDs which are common to both
    # train_col = (train_df.shape)[1]
    # test_col = (test_df.shape)[1]

    # get rssi columns from each set
    train_col_names = train_df.columns[1:len(train_df.columns) - 1]
    test_col_names = test_df.columns[1:len(test_df.columns) - 2]

    # Find the common rssi columns
    print "-" * 30, "Using multiple access point", "-" * 30
    features = list(set(test_col_names) & set(train_col_names))

    # Testing with the residence access point
    # print "-"*30, "Using single access point", "-"*30
    # features = ['c4:0a:cb:2d:87:a0']
    # print "Training set columns::", train_col_names, "\nTest set::", test_col_names
    print "Features::", features

    # Step3: Localizing the user
    total_rows = (test_df.shape)[0]
    print "Total number of test data points:", total_rows
    if len(features) == 0:
        print "Labeling it with Outside"
        pred_loc = ["Outside" * total_rows]
    else:
        # Run KNN or NNSS algorithm to generate locations for the data points
        n = 5
        print "Using n::", n
        # clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(train_df[features], train_df['label'])
        pred_loc = clf.predict(test_df[features])
    true_loc = test_df['label']

    # Step4: Store results in a csv and plot the csv with labels
    # cols_to_write =  ['timestamp'] + features + ['location']
    train_csv_name = train_csv.split('/')[3].replace('.csv', '')
    new_csv = (test_csv.split('/')[3]).replace(
        '.csv', '_' + train_csv_name + '.csv')
    new_csv = 'Wifi/output/testing/' + apt_no + "_n" + str(n) + "_" + new_csv
    new_df = test_df
    new_df['pred_label'] = pred_loc
    new_df.to_csv(new_csv, index=False)

    # Step5: Check accuracy by comparing it with the ground truth
    # Plot confusion matrix
    classes = sorted(test_df.label.unique().tolist())
    print "Classes: ", classes

    # Data points for the classes in the training set
    print "-" * 10, "Training Set", "-" * 10
    grp_loc_df = [train_df.iloc[np.where(train_df['label'] == i)]
                  for i in classes]
    for j in range(len(grp_loc_df)):
        print "Class", grp_loc_df[j].label.unique(), " : ", len(grp_loc_df[j])

    # Data points for the classes in the test set
    print "-" * 10, "Test Set", "-" * 10
    grp_loc_df = [test_df.iloc[np.where(test_df['label'] == i)]
                  for i in classes]
    for j in range(len(grp_loc_df)):
        print("Class %s : %d" %
              (grp_loc_df[j].label.unique(), len(grp_loc_df[j])))

    cm = confusion_matrix(true_loc.tolist(), pred_loc.tolist())
    print "-" * 20
    print "Confusion Matrix:: "
    print cm

    # Print Classification report
    cls_report = (classification_report(true_loc.tolist(),
                                        pred_loc.tolist(),
                                        labels=classes, target_names=classes))
    print cls_report

    # Overall Accuracy
    true = new_df.label
    pred = new_df.pred_label
    mis_labeled = (true != pred).sum()
    total = len(true)
    accuracy = float(total - mis_labeled) / total * 100
    print("Number of mislabeled points : %d" % (true != pred).sum())
    print("Total number of points: %d" % total)
    print "Overall Accuracy:", accuracy

    # Show confusion matrix in a separate window
    # train_csv_name = train_csv.split('/')[3].replace('.csv', '')
    # imgname = (test_csv.split('/')[3]).replace('.csv', '.png')
    # imgname = imgname.replace(
    #     '.png', '_' + train_csv_name + '_dist_n' + str(n) + '.png')
    # print "Imagename::", imgname, "\n"
    # pltcm.plot_cm(cm, "Wifi", classes, imgname, n)

    return new_df


def classify_location_piecewise(train_csv, test_csv, apt_no, idx):

    # Step1: Get data from radio map (training set) and test set
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Step2: Check both sets are compatible.
    # If they are not, use the SSIDs which are common to both
    # train_col = (train_df.shape)[1]
    # test_col = (test_df.shape)[1]

    # get rssi columns from each set
    train_col_names = train_df.columns[1:len(train_df.columns) - 1]
    test_col_names = test_df.columns[1:len(test_df.columns) - 2]

    # Find the common rssi columns
    # print "-" * 30, "Using multiple access point", "-" * 30
    features = list(set(test_col_names) & set(train_col_names))

    # Testing with the residence access point
    # print "-"*30, "Using single access point", "-"*30
    # features = ['c4:0a:cb:2d:87:a0']
    # print "Training set columns::", train_col_names, "\nTest set::", test_col_names
    print "Features::", features

    # Step3: Localizing the user
    total_rows = (test_df.shape)[0]
    print "Total number of test data points:", total_rows
    if len(features) == 0:
        print "Labeling it with Outside"
        pred_loc = ["Outside" * total_rows]
    else:
        # Run KNN or NNSS algorithm to generate locations for the data points
        n = 5
        print "Using n::", n
        # clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(train_df[features], train_df['label'])
        pred_loc = clf.predict(test_df[features])
    true_loc = test_df['label']

    # Step4: Store results in a csv and plot the csv with labels
    # cols_to_write =  ['timestamp'] + features + ['location']
    train_csv_name = train_csv.split('/')[3].replace('.csv', '')
    new_csv = (test_csv.split('/')[3]).replace(
        '.csv', '_' + train_csv_name + '.csv')
    new_csv = 'Wifi/output/testing/' + apt_no + "_n" + str(n) + "_" + new_csv
    new_df = test_df
    new_df['pred_label'] = pred_loc
    new_df.to_csv(new_csv, index=False)

    # Step5: Check accuracy by comparing it with the ground truth
    # Plot confusion matrix
    train_classes = sorted(train_df.label.unique().tolist())
    test_classes = sorted(test_df.label.unique().tolist())
    pred_test_class = sorted(new_df.pred_label.unique().tolist())
    print "Train Classes: ", train_classes
    print "Test Classes: ", test_classes
    print "Predicted Classes", pred_test_class

    # Data points for the classes in the training set
    print "-" * 10, "Training Set", "-" * 10
    grp_loc_df = [train_df.iloc[np.where(train_df['label'] == i)]
                  for i in test_classes]
    for j in range(len(grp_loc_df)):
        print "Class", grp_loc_df[j].label.unique(), " : ", len(grp_loc_df[j])

    # Data points for the classes in the test set
    print "-" * 10, "Test Set", "-" * 10
    grp_loc_df = [test_df.iloc[np.where(test_df['label'] == i)]
                  for i in test_classes]
    for j in range(len(grp_loc_df)):
        print("Class %s : %d" %
              (grp_loc_df[j].label.unique(), len(grp_loc_df[j])))

    # Print Classification report
    cls_report = (classification_report(true_loc.tolist(),
                                        pred_loc.tolist(),
                                        labels=test_classes, target_names=train_classes))
    print cls_report

    # Overall Accuracy
    true = new_df.label
    pred = new_df.pred_label
    mis_labeled = (true != pred).sum()
    total = len(true)
    accuracy = float(total - mis_labeled) / total * 100
    print("Number of mislabeled points : %d" % (true != pred).sum())
    print("Total number of points: %d" % total)
    print "Overall Accuracy:", accuracy

    # Show confusion matrix in a separate window
    # train_csv_name = train_csv.split('/')[3].replace('.csv', '')
    # imgname = (test_csv.split('/')[3]).replace('.csv', '.png')
    # imgname = imgname.replace(
    #     '.png', '_' + train_csv_name + '_dist_n' + str(n) + '.png')
    # print "Imagename::", imgname, "\n"
    # pltcm.plot_cm(cm, "Wifi", classes, imgname, n)

    # Selecting the label with maximum count
    pred_list = dict((i, list(pred_loc).count(i)) for i in pred_test_class)
    print "\nPredicted list", pred_list
    grpcount_label = pd.DataFrame.from_dict(pred_list, orient="index")
    grpcount_label.columns = ['lcount']
    pred_label = grpcount_label[
        grpcount_label.lcount == grpcount_label.lcount.max()].index[0]
    print "Predicted Location Label:", pred_label
    return pred_label

if __name__ == '__main__':

    # Variables
    # train_csv = sys.argv[1]       # training data set - Wifi/radiomap/apt_no_idx.csv
    # test_csv = sys.argv[2] 
           # test data set - Wifi/test_data/apt_no_idx.csv
    room_no = sys.argv[1]
    day = sys.argv[2]
    # idx = sys.argv[4]
    rooms = room_no.split(',')
    #rooms = [ 'C001','C002', 'C003', 'C004', 'C005', 'C006']

    test_csv_path = DATA_PATH + day + '/'
    location_csv_path = DATA_PATH + day + WIFI_INOUT
    # Classify

    
    for i in rooms:
        formatted_csv_path = (DATA_PATH + day + WIFI_FORMATTED  )
        formatted_csv_path = format_data(test_csv_path, 'train', i, formatted_csv_path)
        print 'Created wifi formatted csv', (formatted_csv_path + i + '.csv')
        wifi_location(formatted_csv_path, i, location_csv_path)
    # #print 'Filling in missing samples'
    #fill_missing_samples(USER_LOCATION_PATH)

    #overlapping_non_overlapping_sets()
        #print 'Created User location csv', (location_path + i + '.csv')
       

    #output_df = classify_location(train_csv, test_csv, apt_no, idx)

    # ------------------------------------------------------------------------
    # TESTING: To check which n gives the highest overall accuracy
    # results = []
    # true_loc = test_df['true_label']

    # for n in range(1, 101, 2):
    #     clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
    #     clf = KNeighborsClassifier(n_neighbors=n)
    #     clf.fit(train_df[features], train_df['location'])
    #     pred_loc = clf.predict(test_df[features])

    #     accuracy = (np.where(pred_loc==test_df['true_label'],
    #                           1, 0).sum() / float(len(test_df.index)))
    #     print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    #     results.append([n, accuracy])

    # results = pd.DataFrame(results, columns=["n", "accuracy"])

    # pl.plot(results.n, results.accuracy)
    # pl.title("Accuracy with Increasing K with weights parameter " + test_csv)
    # pl.show()

    # TESTING: To check which n gives the highest accuracy class wise
    # results = []
    # true_loc = test_df['true_label']
    # true_loc = train_df['location']
    # classes = sorted(test_df.true_label.unique().tolist())
    # print "Classes: ", classes
    # no_of_classes = len(classes)
    # print "Number of classes", no_of_classes
    # acc_col_name = ['accuracy'+str(i) for i in range(no_of_classes)]
    # accuracy = []
    # total_no = test_df.groupby(['true_label']).size()
    # total_no = train_df.groupby(['location']).size()
    # print total_no

    # for n in range(1, 101, 2):
    #   print "------Classifying with n = ", n ,"--------------"
    # clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
    #   clf = KNeighborsClassifier(n_neighbors=n)
    #   clf.fit(train_df[features], train_df['location'])
    # clf.fit(test_df[features], test_df['true_label'])
    #   pred_loc = clf.predict(test_df[features])

    #   cm = confusion_matrix(true_loc, pred_loc)
    #   print cm

    #   for i, row in enumerate(cm):
    #       percent = (float(row[i]) / float(total_no[i])) * 100
    #       accuracy.append(percent)
    #   results.append([n] + accuracy)
    #   accuracy = []

    # cols = ["n"] + acc_col_name
    # results = pd.DataFrame(results, columns=cols)
    # for i in range(3):
    #   pl.plot(results['n'], results['accuracy'+str(i)])
    # pl.title("Accuracy with Increasing K with weights parameter for " + test_csv)
    # pl.title("Accuracy with Increasing K " + test_csv)
    # pl.legend(classes,loc='upper left')
    # pl.show()
