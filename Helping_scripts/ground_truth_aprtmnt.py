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
apt_no = sys.argv[3]  # room_no/apartment number
idx = sys.argv[4]  # index of the new ground truth file

sensor = idx

# date = sys.argv[3]  # date of experiment DD/MM/YYYY

filename = folder + "ground_truth/" + \
    set_type + "_" + apt_no + "_" + idx + ".csv"

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

# stime_str = ['17/09/2013-11.36', '17/09/2013-11.41',
#              '17/09/2013-11.45', '17/09/2013-11.49']
# etime_str = ['17/09/2013-11.39', '17/09/2013-11.44',
#              '17/09/2013-11.47', '17/09/2013-11.52']
# label = ['Fan','AC','Fan','AC']

# stage 1 - s1
stime_str = ['28/10/2013-16.37', '28/10/2013-16.44', '28/10/2013-16.53',
             '28/10/2013-16.59', '28/10/2013-17.07', '29/10/2013-18.11']
etime_str = ['28/10/2013-16.42', '28/10/2013-16.51', '28/10/2013-16.58',
             '28/10/2013-17.05', '28/10/2013-17.10', '29/10/2013-18.16']

wlabel=['Dining Room','Bedroom','Kitchen','Dining Room','Kitchen',
	'Dining Room']

#stage 2 - s2
"""
stime_str = ['28/10/2013-17.47', '28/10/2013-17.55', '28/10/2013-18.03',
              '28/10/2013-18.10', '28/10/2013-18.18', '28/10/2013-18.23']
etime_str = ['28/10/2013-17.53', '28/10/2013-18.00', '28/10/2013-18.08',
              '28/10/2013-18.15', '28/10/2013-18.20', '28/10/2013-18.29']

wlabel=['Dining Room','Bedroom','Kitchen','Dining Room','Kitchen','Dining Room']

# stage 3 - s3
stime_str = ['29/10/2013-17.15', '29/10/2013-17.21', '29/10/2013-17.28',
              '29/10/2013-17.35', '29/10/2013-17.41', '29/10/2013-17.44']

etime_str = ['29/10/2013-17.20', '29/10/2013-17.26', '29/10/2013-17.33',
              '29/10/2013-17.40', '29/10/2013-17.43', '29/10/2013-17.49']

wlabel=['Dining Room','Bedroom','Kitchen','Dining Room','Kitchen','Dining Room']
"""
# --------------------------------------------------------------------------
# test t1_old
# stime_str = ['29/10/2013-18.50', '29/10/2013-19.01', '29/10/2013-19.14',
#              '29/10/2013-19.18', '29/10/2013-19.23', '29/10/2013-19.30']
# etime_str = ['29/10/2013-19.00', '29/10/2013-19.13', '29/10/2013-19.17',
#              '29/10/2013-19.22', '29/10/2013-19.29', '29/10/2013-19.32']

# --------------------------------------------------------------------------
# Phase 1 Experiments

# test t11 -> t1 - exp1
# stime_str = ['09/11/2013-13.33.48', '09/11/2013-13.39.30', '09/11/2013-13.48.55',
#              '09/11/2013-13.52.36', '09/11/2013-13.58.27', '09/11/2013-14.07.50',
#              '09/11/2013-13.48.43', '09/11/2013-13.58.02', '09/11/2013-14.05.34']
# etime_str = ['09/11/2013-13.39.02', '09/11/2013-13.45.29', '09/11/2013-13.51.55',
#              '09/11/2013-13.57.36', '09/11/2013-14.01.12', '09/11/2013-14.13.56',
#              '09/11/2013-13.52.02', '09/11/2013-14.01.36', '09/11/2013-14.13.55']
# slabel = ['Fan', 'AC', 'Microwave', 'TV', 'Kettle', 'AC', 'Light', 'Light', 'Light']
# wlabel = ['Dining Room', 'Bedroom', 'Kitchen', 'Dining Room',
#           'Kitchen', 'Dining Room', 'Kitchen', 'Kitchen', 'Dining Room']

# test t2 - exp2
# stime_str = [
#     '06/11/2013-17.50.41', '06/11/2013-17.55.06', '06/11/2013-17.58.33', '06/11/2013-18.04.03',
#     '06/11/2013-18.04.15', '06/11/2013-18.06.24', '06/11/2013-18.09.29', '06/11/2013-18.16.26',
#     '06/11/2013-18.16.31']
# etime_str = [
#     '06/11/2013-17.55.02', '06/11/2013-17.55.32', '06/11/2013-18.03.56', '06/11/2013-18.06.17',
#     '06/11/2013-18.06.13', '06/11/2013-18.16.17', '06/11/2013-18.15.27', '06/11/2013-18.20.00',
#     '06/11/2013-18.19.28']

# slabel = ['Fan', 'AC', 'AC', 'Light', 'Microwave', 'Light', 'AC', 'Light', 'Kettle']
# wlabel = ['Dining Room', 'Dining Room', 'Dining Room', 'Kitchen', 'Kitchen', 'Bedroom',
#           'Bedroom', 'Kitchen', 'Kitchen']

# test t12 -> t3 - exp3
# stime_str = [
#     '09/11/2013-12.42.02', '09/11/2013-12.50.23', '09/11/2013-12.53.16', '09/11/2013-12.59.41',
#     '09/11/2013-12.59.54', '09/11/2013-13.03.25', '09/11/2013-13.08.37', '09/11/2013-13.08.42',
#     '09/11/2013-13.12.14']
# etime_str = [
#     '09/11/2013-12.50.04', '09/11/2013-12.59.23', '09/11/2013-12.59.15', '09/11/2013-13.03.06',
#     '09/11/2013-13.02.53', '09/11/2013-13.08.17', '09/11/2013-13.12.00', '09/11/2013-13.11.32',
#     '09/11/2013-13.25.19']

# slabel = ['Fan', 'AC', 'AC', 'Light', 'Microwave', 'TV', 'Light', 'Kettle', 'AC']
# wlabel = ['Dining Room', 'Bedroom', 'Bedroom', 'Kitchen',
#           'Kitchen', 'Dining Room', 'Kitchen', 'Kitchen', 'Dining Room']

# test t4_1 - exp4
# stime_str = ['06/11/2013-17.14.19', '06/11/2013-17.16.30',
#              '06/11/2013-17.22.16', '06/11/2013-17.33.00']
# etime_str = ['06/11/2013-17.16.18', '06/11/2013-17.21.40',
#              '06/11/2013-17.32.16', '06/11/2013-17.38.16']

# slabel = ['Microwave', 'Fan', 'AC', 'TV']
# wlabel = ['Kitchen', 'Dining Room', 'Bedroom', 'Dining Room']

# test t4_2 - exp4
# stime_str = ['06/11/2013-17.14.02', '06/11/2013-17.22.09',
#              '06/11/2013-17.29.29', '06/11/2013-17.32.06']
# etime_str = ['06/11/2013-17.21.43', '06/11/2013-17.29.09',
#              '06/11/2013-17.31.13', '06/11/2013-17.38.16']

# slabel = ['Fan', 'AC', 'Kettle', 'TV']
# wlabel = ['Bedroom', 'Dining Room', 'Kitchen', 'Dining Room']

# test t5_3 - exp5
# stime_str = ['09/11/2013-19.30.32', '09/11/2013-19.40.57',
#              '09/11/2013-19.44.22']
# etime_str = ['09/11/2013-19.40.25', '09/11/2013-19.43.58',
#              '09/11/2013-19.49.27']

# slabel = ['AC', 'Microwave', 'Fan']
# wlabel = ['Bedroom', 'Kitchen', 'Dining Room']

# test t7 - exp7 - showing TV, Fan, Lights, Microwave - realistic setting
# in a controlled environment
# stime_str = ['26/11/2013-20.05.12', '26/11/2013-20.11.57', '26/11/2013-20.20.54',
#              '26/11/2013-20.35.28', '26/11/2013-20.54.01',
#              '26/11/2013-21.04.15', '26/11/2013-21.11.10', '26/11/2013-21.11.47',
#              '26/11/2013-21.13.12', '26/11/2013-21.14.13', '26/11/2013-21.23.11']
# etime_str = ['26/11/2013-21.32.17', '26/11/2013-20.31.33', '26/11/2013-20.24.25',
#              '26/11/2013-20.53.44', '26/11/2013-20.59.25',
#              '26/11/2013-21.22.21', '26/11/2013-21.14.42', '26/11/2013-21.12.48',
#              '26/11/2013-21.19.08', '26/11/2013-21.19.44', '26/11/2013-21.29.17']

# slabel = ['Light', 'TV', 'Light', 'Light', 'Light',
#           'TV', 'Light', 'Microwave', 'Fan', 'Light', 'Fan']
# wlabel = ['Dining Room', 'Dining Room', 'Bedroom', 'Bedroom', 'Kitchen', 'Dining Room',
#           'Kitchen', 'Kitchen', 'Bedroom', 'Bedroom', 'Dining Room']

# test t8 - exp8 - showing TV, Fan, Lights, Microwave - realistic setting
# in a controlled environment - with AudioFeatures calculated on phone
# stime_str = ['28/11/2013-17.02.53', '28/11/2013-17.00.01', '28/11/2013-17.16.52',
#              '28/11/2013-17.16.04', '28/11/2013-17.18.28', '28/11/2013-17.22.37',
#              '28/11/2013-17.22.40', '28/11/2013-17.26.18', '28/11/2013-17.32.48',
#              '28/11/2013-17.34.56', '28/11/2013-17.49.40', '28/11/2013-17.51.44']
# etime_str = ['28/11/2013-17.58.47', '28/11/2013-17.06.28', '28/11/2013-17.17.52',
#              '28/11/2013-17.18.06', '28/11/2013-17.33.09', '28/11/2013-17.32.57',
#              '28/11/2013-17.25.07', '28/11/2013-17.32.48', '28/11/2013-17.34.56',
#              '28/11/2013-17.48.30', '28/11/2013-17.51.44', '28/11/2013-17.58.47']

# slabel = ['None', 'Fan', 'Microwave', 'None', 'None', 'None',
#           'None', 'AC', 'None', 'TV', 'None', 'None']
# wlabel = ['Dining Room', 'Dining Room', 'Kitchen', 'Kitchen', 'Bedroom', 'Bedroom', 'Dining Room',
#           'Bedroom', 'Dining Room', 'Dining Room', 'Kitchen', 'Dining Room']

# test t9 - exp9 - showing TV, Fan, Lights, Microwave, Kettle, AC - realistic setting
# in a controlled environment - with and without AudioFeatures calculated on phone
# stime_str = ['30/11/2013-14.00.07', '30/11/2013-14.03.45', '30/11/2013-14.08.23',
#              '30/11/2013-14.11.52', '30/11/2013-14.12.06', '30/11/2013-14.20.14',
#              '30/11/2013-14.25.48', '30/11/2013-14.32.37']
# etime_str = ['30/11/2013-14.07.33', '30/11/2013-14.04.45', '30/11/2013-14.19.55',
#              '30/11/2013-14.15.08', '30/11/2013-14.17.29', '30/11/2013-14.25.01',
#              '30/11/2013-14.31.59', '30/11/2013-14.38.40']

# slabel = ['Fan', 'Microwave', 'Light', 'Kettle', 'Light',
#           'TV', 'AC', 'AC']
# wlabel = ['Dining Room', 'Kitchen', 'Bedroom', 'Kitchen', 'Kitchen', 'Dining Room',
#           'Dining Room', 'Bedroom']

# test t10 - exp10 - showing TV, Fan, Lights, Microwave, Kettle, AC - realistic setting
# in a controlled environment - with and without AudioFeatures calculated on phone
# stime_str = ['30/11/2013-14.45.02', '30/11/2013-14.48.55', '30/11/2013-14.52.42',
#              '30/11/2013-14.56.14', '30/11/2013-14.57.05', '30/11/2013-15.05.20',
#              '30/11/2013-15.13.47', '30/11/2013-15.24.11']
# etime_str = ['30/11/2013-14.52.20', '30/11/2013-14.49.54', '30/11/2013-15.04.49',
#              '30/11/2013-14.59.08', '30/11/2013-15.01.36', '30/11/2013-15.13.02',
#              '30/11/2013-15.20.47', '30/11/2013-15.28.48']

# slabel = ['Fan', 'Microwave', 'Light', 'Kettle', 'Light',
#           'TV', 'AC', 'AC']
# wlabel = ['Dining Room', 'Kitchen', 'Bedroom', 'Kitchen', 'Kitchen', 'Dining Room',
#           'Dining Room', 'Bedroom']

# --------------------------------------------------------------------------------
# Phase 2 experiments

# training dataset
# s703
# stime_str = ['21/11/2013-19.02.00', '21/11/2013-19.09.00', '21/11/2013-19.20.00',
#              '21/11/2013-19.40.00', '21/11/2013-19.50.00']
# etime_str = ['21/11/2013-19.07.00', '21/11/2013-19.17.00', '21/11/2013-19.24.00',
#              '21/11/2013-19.48.00', '21/11/2013-19.55.00']

# slabel = ['Washing Machine', 'None', 'Microwave', 'Fan', 'TV']

# w703
# stime_str = ['21/11/2013-19.02.00', '21/11/2013-19.09.00', '21/11/2013-19.32.00',
#              '21/11/2013-19.50.00']
# etime_str = ['21/11/2013-19.07.00', '21/11/2013-19.17.00', '21/11/2013-19.38.00',
#              '21/11/2013-20.03.00']
# wlabel = ['Shared Bathroom', 'Master Bedroom', 'Kitchen', 'Dining Room']

# s603
# stime_str = [
#     '23/11/2013-15.40.00', '23/11/2013-15.53.00', '23/11/2013-16.17.00', '23/11/2013-16.25.58',
#     '23/11/2013-16.28.58', '23/11/2013-16.39.59', '23/11/2013-17.20.00', '23/11/2013-16.56.00',
#     '23/11/2013-17.02.00']
# etime_str = [
#     '23/11/2013-15.48.00', '23/11/2013-15.58.00', '23/11/2013-16.18.00', '23/11/2013-16.27.31',
#     '23/11/2013-16.30.31', '23/11/2013-16.41.00', '23/11/2013-17.22.00', '23/11/2013-17.01.15',
#     '23/11/2013-17.05.34']

# slabel = ['None', 'Fan', 'Kettle', 'Microwave', 'Microwave',
#           'Washing Machine', 'Washing Machine', 'TV', 'TV']

# w603
# stime_str = [
#     '23/11/2013-15.40.00', '23/11/2013-15.53.00', '23/11/2013-15.59.00', '23/11/2013-16.05.00',
#     '23/11/2013-16.36.00', '23/11/2013-16.56.00']
# etime_str = [
#     '23/11/2013-15.48.00', '23/11/2013-15.58.00', '23/11/2013-16.04.00', '23/11/2013-16.10.00',
#     '23/11/2013-16.41.00', '23/11/2013-17.17.00']
# wlabel = ['Master Bedroom', 'Bedroom3', 'Bedroom1', 'Kitchen', 'Bathroom1', 'Dining Room']
# -----------------------------------

# Test Data for the Apartments
# 703 - t24-25Nov
"""
stime_str=['24/11/2013-17.23.07','24/11/2013-17.32.08','24/11/2013-19.38.25','24/11/2013-17.09.04','24/11/2013-17.55.14',
	'24/11/2013-15.28.11','24/11/2013-15.24.11','24/11/2013-20.39.21','24/11/2013-20.41.50','25/11/2013-07.59.41',
	'25/11/2013-08.39.34','25/11/2013-17.33.49']
etime_str=['24/11/2013-22.07.40','24/11/2013-22.09.15','24/11/2013-22.09.40','24/11/2013-17.31.09','24/11/2013-22.46.44',
	'24/11/2013-21.58.32','24/11/2013-15.26.44','24/11/2013-20.42.44','24/11/2013-20.44.45','25/11/2013-08.01.42'
	,'25/11/2013-08.41.33',]

slabel=['Light','Light','Light','Light','Light','TV','Microwave','Microwave','Microwave','Microwave','Microwave']
wlabel=['Dining Room','Kitchen','Kitchen','Master Bedroom','Master Bedroom','Dining Room','Kitchen','Kitchen',
	'Kitchen','Kitchen','Kitchen']


# Test Data for the Apartments
# 703 - t25-26Nov

stime_str=['25/11/2013-16.52.18','25/11/2013-16.56.42','25/11/2013-17.32.46','25/11/2013-18.02.49','25/11/2013-21.59.11',
	'25/11/2013-17.33.49','26/11/2013-08.16.18','26/11/2013-08.53.48','26/11/2013-09.03.18']

etime_str=['25/11/2013-21.52.18','25/11/2013-21.57.46','25/11/2013-21.55.45','25/11/2013-22.36.16','26/11/2013-08.02.49',
	'25/11/2013-17.35.49','26/11/2013-08.18.19','26/11/2013-08.55.15','26/11/2013-09.05.19']

slabel=['TV','Light','Light','Light','Fan','Microwave','Microwave','Microwave','Microwave']

wlabel=['Dining Room','Dining Room','Kitchen','Master Sedroom','Master Bedroom','Kitchen','Kitchen','Kitchen'
	,'Kitchen']

# Test Data for the Apartments
# 703 - t26-27Nov

stime_str=['26/11/2013-17.09.15','26/11/2013-17.11.50','26/11/2013-17.11.50','26/11/2013-17.29.46'
	,'26/11/2013-22.29.14','26/11/2013-20.24.02','27/11/2013-10.47.18','26/11/2013-19.27.10'
	,'26/11/2013-21.10.48','26/11/2013-21.13.25','27/11/2013-10.57.16']
etime_str=['26/11/2013-22.26.20','26/11/2013-21.40.17','26/11/2013-21.40.17','26/11/2013-23.14.49'
	,'27/11/2013-08.18.42','26/11/2013-22.23.48','27/11/2013-15.05.49','26/11/2013-19.46.59',
	'26/11/2013-21.12.48','26/11/2013-21.15.23','27/11/2013-10.59.18']

slabel=['Light','Light','Light','Light','Fan','TV','TV','Microwave','Microwave','Microwave','Microwave']
wlabel=['Dining Room','Kitchen','Kitchen','Master Bedroom','Master Bedroom','Dining Room','Dining Room',
	'Kitchen','Kitchen','Kitchen','Kitchen']

# Test Data for the Apartments
# 703 - t27-28Nov

stime_str=['27/11/2013-17.24.37','27/11/2013-17.54.09','27/11/2013-16.07.17','27/11/2013-17.21.00',
	'27/11/2013-18.14.14','27/11/2013-21.58.23','27/11/2013-20.12.08','27/11/2013-20.36.40'
	,'28/11/2013-08.06.16']
etime_str=['27/11/2013-22.00.53','27/11/2013-21.56.53','27/11/2013-17.22.00','27/11/2013-22.40.00',
	'27/11/2013-20.08.42','28/11/2013-07.56.18','27/11/2013-21.53.48','27/11/2013-20.39.40'
	,'28/11/2013-08.08.16']
slabel=['Light','Light','Fan','Light','Fan','Fan','TV','Microwave','Microwave']

wlabel=['Dining Room','Kitchen','Master Bedroom','Master Bedroom','Master Bedroom','Master Bedroom',
	'Dining Room','Kitchen','Kitchen']

# Test Data for the Apartments
# 703 - t28-29Nov

stime_str=['28/11/2013-17.13.43','28/11/2013-17.14.17','28/11/2013-17.14.17','28/11/2013-22.02.18',
	'28/11/2013-17.31.25']
etime_str=['28/11/2013-21.55.17','28/11/2013-21.58.18','28/11/2013-22.59.03','29/11/2013-08.03.57',
	'28/11/2013-17.33.26']

slabel=['Light','Light','Light','Fan','Microwave']
wlabel=['Dining Room','Kitchen','Master Bedroom','Master Bedroom','Kitchen']


"""
"""
stime_str=['24/11/2013-17.23.07','25/11/2013-16.56.42','26/11/2013-17.09.15','27/11/2013-17.24.37',
	'28/11/2013-17.13.43','24/11/2013-17.32.08','24/11/2013-19.38.25','25/11/2013-17.32.46',
	'26/11/2013-17.11.50','26/11/2013-17.11.50','27/11/2013-17.54.09','28/11/2013-17.14.17',
	'24/11/2013-17.09.04','24/11/2013-17.55.14','25/11/2013-18.02.49','25/11/2013-21.59.11',
	'26/11/2013-17.29.46','26/11/2013-22.29.14','27/11/2013-16.07.17','27/11/2013-17.21.00',
	'27/11/2013-18.14.14','27/11/2013-21.58.23','28/11/2013-17.14.17','28/11/2013-22.02.18',
	'24/11/2013-15.28.11','26/11/2013-20.24.02','27/11/2013-10.47.18','27/11/2013-15.11.42',
	'27/11/2013-20.12.08','24/11/2013-15.24.11','24/11/2013-20.39.21','24/11/2013-20.41.50',
	'25/11/2013-07.59.41','25/11/2013-08.39.34','25/11/2013-17.33.49','26/11/2013-08.16.18',
	'26/11/2013-08.53.48','26/11/2013-09.03.18','26/11/2013-19.27.10','26/11/2013-21.10.48',
	'26/11/2013-21.13.25','27/11/2013-10.57.16','27/11/2013-20.36.40','28/11/2013-08.06.16',
	'28/11/2013-17.31.25','28/11/2013-08.43.19','28/11/2013-08.51.20']

etime_str=['24/11/2013-22.07.40','25/11/2013-21.57.46','26/11/2013-22.26.20','27/11/2013-22.00.53',
	'28/11/2013-21.55.17','24/11/2013-22.09.15','24/11/2013-22.09.40','25/11/2013-21.55.45',
	'26/11/2013-21.40.17','26/11/2013-21.40.17','27/11/2013-21.56.53','28/11/2013-21.58.18',
	'24/11/2013-17.31.09','24/11/2013-22.46.44','25/11/2013-22.36.16','26/11/2013-08.02.49',
	'26/11/2013-23.14.49','27/11/2013-08.18.42','27/11/2013-17.22.00','27/11/2013-22.40.00',
	'27/11/2013-20.08.42','28/11/2013-07.56.18','28/11/2013-22.59.03','29/11/2013-08.03.57',
	'24/11/2013-21.58.32','26/11/2013-22.23.48','27/11/2013-15.05.49','27/11/2013-15.55.04',
	'27/11/2013-21.53.48','24/11/2013-15.26.44','24/11/2013-20.42.44','24/11/2013-20.44.45',
	'25/11/2013-08.01.42','25/11/2013-08.41.33','25/11/2013-17.35.49','26/11/2013-08.18.19',
	'26/11/2013-08.55.15','26/11/2013-09.05.19','26/11/2013-19.46.59','26/11/2013-21.12.48',
	'26/11/2013-21.15.23','27/11/2013-10.59.18','27/11/2013-20.39.40','28/11/2013-08.08.16',
	'28/11/2013-17.33.26','28/11/2013-08.45.19','28/11/2013-08.53.20']

slabel=['Light','Light','Light','Light','Light','Light','Light','Light','Light','Light','Light','Light','Light',
	'Light','Light','Light','Fan','Light','Fan','Fan','Light','Fan','Fan','Light','Fan','TV','TV','TV','TV'
	,'TV','Microwave','Microwave','Microwave','Microwave','Microwave','Microwave','Microwave','Microwave'
	,'Microwave','Microwave','Microwave','Microwave','Microwave','Microwave','Microwave','Microwave','Microwave'
	,'Microwave',]

wlabel=['Dining Room','Dining Room','Dining Room','Dining Room','Dining Room','Dining Room','Kitchen','Kitchen'
	,'Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Master Bedroom','Master Bedroom','Master Bedroom'
	,'Master Bedroom','Master Bedroom','Master Bedroom','Master Bedroom','Master Bedroom','Master Bedroom'
	,'Master Bedroom','Master Bedroom','Master Bedroom','Dining Room','Dining Room','Dining Room','Dining Room'
	,'Dining Room','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen'
	,'Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen','Kitchen']

"""

for row in zip(stime_str, etime_str):
    print "[", row[0], " - ", row[1], "]"

# Output file
print "Writing to file - ", filename
writer = csv.writer(open(filename, "w"))

if sensor == "sound":
    writer.writerow(['start_time', 'end_time', 'slabel'])
    no_loop = len(slabel)
elif sensor == "wifi":
    writer.writerow(['start_time', 'end_time', 'wlabel'])
    no_loop = len(wlabel)
# For test dataset
else:
    writer.writerow(['start_time', 'end_time', 'slabel', 'wlabel'])
    no_loop = len(slabel)

# Convert to timestamp and store in a CSV
for x in range(no_loop):
    s_tstamp = long(
        time.mktime(time.strptime(stime_str[x], "%d/%m/%Y-%H.%M")) * 1000)
    e_tstamp = long(
        time.mktime(time.strptime(etime_str[x], "%d/%m/%Y-%H.%M")) * 1000)

    # For training dataset
    if sensor == "sound":
        print s_tstamp, e_tstamp, slabel[x]
        writer.writerow([s_tstamp, e_tstamp, slabel[x]])
    elif sensor == "wifi":
        print s_tstamp, e_tstamp, wlabel[x]
        writer.writerow([s_tstamp, e_tstamp, wlabel[x]])
    # For test dataset
    else:
        print s_tstamp, e_tstamp, slabel[x], wlabel[x]
        writer.writerow([s_tstamp, e_tstamp, slabel[x], wlabel[x]])
