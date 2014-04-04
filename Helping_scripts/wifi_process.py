import csv
import sys
from math import sqrt
import numpy
import scipy
from scipy.cluster.vq import *
import pandas as pd
path=sys.argv[1]
path1=sys.argv[2]
row=""
f=open(path,'rb')
w=open(path1,'wb')
reader=csv.reader(f,delimiter=',')

for r in reader:
	
	if r[0]<'1386288000000':
		row+=r[0]+','+r[1]+','+r[2]+','+r[3]+','+r[4]+'\n'	
#print val

w.write(row)

w.close()

f.close()

