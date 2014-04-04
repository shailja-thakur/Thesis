# Take ground truth labels for the colleted audio data from user
import csv
import sys
import time

# Variables
label = []
stime_str = []
etime_str = []

# Input arguments
folder = sys.argv[1]
set_type = sys.argv[2]  # test or train
#apt_no = sys.argv[3]  # room_no/apartment number
#idx = sys.argv[4]  # index of the new ground truth file

#sensor = idx

# date = sys.argv[3]  # date of experiment DD/MM/YYYY

filename = folder + "ground_truth" + ".csv"

# Time slices with specific sound for the loop count
# no_loop = int(raw_input("Enter number of time slices:: "))

# for x in range(0, no_loop):
# Enter time in the form DD-MM-HH-MM
#     stime_str.append(
#         date + "-" + raw_input("Enter start time in the form HH.MM :: "))
# Enter time in the form DD-MM-HH-MM
#     etime_str.append(
#         date + "-" + raw_input("Enter end time in the form HH.MM :: "))
# Enter location/sound name
#     label.append(raw_input("Enter label:: "))


# Train Data for Hostel
#  t5Nov

stime_str = ['05/12/2013-16.06.48', '05/12/2013-17.00.18', '05/12/2013-17.05.04',
             '05/12/2013-17.28.31', '05/12/2013-18.14.13','05/12/2013-18.20.10',
	     '05/12/2013-18.57.24', '05/12/2013-20.01.17','05/12/2013-20.51.40',
	     '05/12/2013-20.59.26','05/12/2013-22.12.13','05/12/2013-16.51.29',
	     '05/12/2013-17.08.40','05/12/2013-17.47.56','05/12/2013-17.49.31',
	     '05/12/2013-20.50.59','05/12/2013-16.06.34','05/12/2013-17.50.59',
	     '05/12/2013-19.21.05','05/12/2013-20.24.59','05/12/2013-19.53.05',
	     '05/12/2013-20.12.46','05/12/2013-20.51.40','05/12/2013-17.13.20',
	     '05/12/2013-17.28.35','05/12/2013-17.44.18','05/12/2013-20.51.03',
	     '05/12/2013-21.15.32','05/12/2013-19.24.29']

etime_str = ['05/12/2013-16.53.25','05/12/2013-17.04.05','05/12/2013-17.10.54',
	     '05/12/2013-17.59.32','05/12/2013-18.16.07','05/12/2013-18.54.59',
	     '05/12/2013-19.51.12','05/12/2013-20.25.27','05/12/2013-20.52.23',
	     '05/12/2013-21.45.49','05/12/2013-23.10.28','05/12/2013-23.08.44',
	     '05/12/2013-17.33.56','05/12/2013-20.26.38','05/12/2013-18.40.29',
	     '05/12/2013-23.08.46','05/12/2013-17.46.43','05/12/2013-19.19.32',
	     '05/12/2013-19.53.02','05/12/2013-20.51.00','05/12/2013-20.09.39',
	     '05/12/2013-20.22.33','05/12/2013-22.46.53','05/12/2013-17.15.44',
	     '05/12/2013-17.34.11','05/12/2013-20.52.32','05/12/2013-21.15.32',
	     '05/12/2013-02.25.16','05/12/2013-20.36.18']


slabel = ['Light', 'Light', 'Light', 'Light', 'Light', 'Light','Light','Light',
	  'Light','ELight', 'ELight','Adapter','Light', 'Light', 'Light', 
	  'Light','Light','Light','Light','Light','Adapter','Adapter','Adapter',
	  'Light','Light','Light','Light','Light','Adaptor']

wlabel = ['Room6', 'Room6', 'Room6', 'Room6','Room6', 'Room6', 'Room6', 'Room6', 
	  'Room6', 'Room6', 'Room6','Room6','Room5','Room5','Room5','Room5','Room4',
	  'Room4','Room4','Room4','Room4','Room4','Room4','Room3','Room3','Room3','Room3','Room3',
          'Room3']

phase = ['LP-B','LP-B','LP-B','LP-B','LP-B','LP-B','LP-B','LP-B','LP-B','EL-B','EL-B',
         'LP-B','LP-B','LP-B','LP-B','LP-B','LP-Y','LP-Y','LP-Y','LP-Y','EL-Y',
         'EL-Y','EL-Y','LP-Y','LP-Y','LP-Y','LP-Y','LP-Y','EL-Y']


## Test Label
"""
stime_str = ['06/12/2013-17.00.51','06/12/2013-17.21.15','06/12/2013-17.49.02','06/12/2013-19.55.59',
             '06/12/2013-17.49.47','06/12/2013-17.06.11','06/12/2013-15.25.32']


etime_str=['06/12/2013-19.50.18','06/12/2013-17.30.36','06/12/2013-19.53.06','06/12/2013-20.00.50',
           '06/12/2013-19.58.00','06/12/2013-20.53.14','06/12/2013-18.25.18']

slabel=['Light','Light','Light','Light','Adapter','Light','Adapter']
wlabel=['Room5','Room6','Room6','Room6','Room6','Room4','Room4']
phase=['LP-B','LP-B','LP-B','LP-B','EL-B','LP-Y','EL-Y']

"""
for row in zip(stime_str, etime_str):
    print "[", row[0], " - ", row[1], "]"

# Output file
print "Writing to file - ", filename
writer = csv.writer(open(filename, "w"))
"""
if sensor == "sound":
    writer.writerow(['start_time', 'end_time', 'slabel', 'phase'])
    no_loop = len(slabel)
elif sensor == "wifi":
    writer.writerow(['start_time', 'end_time', 'wlabel', 'phase'])
    no_loop = len(wlabel)
"""
# For test dataset

writer.writerow(['start_time', 'end_time', 'slabel', 'wlabel', 'phase'])
no_loop = len(slabel)

# Convert to timestamp and store in a CSV
for x in range(no_loop):
    s_tstamp = long(
        time.mktime(time.strptime(stime_str[x], "%d/%m/%Y-%H.%M.%S")) * 1000)
    e_tstamp = long(
        time.mktime(time.strptime(etime_str[x], "%d/%m/%Y-%H.%M.%S")) * 1000)
# For test dataset
    
    print s_tstamp, e_tstamp, slabel[x], wlabel[x], phase[x]
    writer.writerow([s_tstamp, e_tstamp, slabel[x], wlabel[x], phase[x]])
"""
    # For training dataset
    if sensor == "sound":
        print s_tstamp, e_tstamp, slabel[x]
        writer.writerow([s_tstamp, e_tstamp, slabel[x], phase[x]])
    elif sensor == "wifi":
        print s_tstamp, e_tstamp, wlabel[x]
        writer.writerow([s_tstamp, e_tstamp, wlabel[x], phase[x]])
"""
    
