import web
import numpy as np
import time
import os
import pandas as pd
import datetime as dt
import csv
f=open('/home/shailja/Dropbox/DataFiles_Code/1384630317260057.3gp','rb')
reader=csv.reader(f)
for r in f:
        # Extract data
        #data = web.data()cd Drop	
        d = r.split(';')
	print d
	data=d[0].split('\r')
	print data
	
	"""
        last = d[2].split('\r')
        filename_str = last[0].split('=')[1]
        csvdata = last[3][1:]
        upload_time = int(time.time())
        filename = filename_str[1:-1]
        filename = filename.replace('.csv', '_' + str(upload_time) + '.csv')
	
        # Create Directory to save file
        f_parts = filename.split('_')
        sensorname = f_parts[0]
        flat_no = f_parts[2]
        participant = f_parts[3]
       """


