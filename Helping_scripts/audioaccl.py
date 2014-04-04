# import the MySQLdb and sys modules
import MySQLdb
import sys
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as md
import time
import pytz
import math
from matplotlib.backends.backend_pdf import PdfPages
# open a database connection
# be sure to change the host IP address, username, password and database name to match your own
connection = MySQLdb.connect (host = "localhost", user = "root", passwd = "sparrow24", db = "sensors")
TIMEZONE='Asia/Kolkata'
# prepare a cursor object using cursor() method
#cursor = connection.cursor ()
cursor1=connection.cursor()
#cursor2=connection.cursor()


#pp = PdfPages('multipage.pdf')
# execute the SQL query using execute() method.
#cursor.execute ("SELECT timestamp,Lux FROM light WHERE timestamp> 1371195110000")


#cursor2.execute("SELECT timestamp,mean FROM sound_mean WHERE #timestamp>=1374557402000")
cursor1.execute("SELECT timestamp,value FROM soundraw where timestamp>1381938968867")

# fetch all of the rows from the query
#data = cursor.fetchall ()
data1=cursor1.fetchall ()
#data2=cursor2.fetchall ()




t=[]
v=[]

t1=[]
x=[]
y=[]
z=[]

# print the rows
"""
for row in data :
	t.append(row[0])	
	v.append(row[1])
"""
for row in data1 :
        t.append(row[0])
	v.append(row[1])


"""
for row in data2 :
        t1.append(row[0])
	x.append(row[1])
#	y.append(row[2])
#	z.append(row[3])

	
"""





#dates=[dt.datetime.fromtimestamp(float(ts)/1000) for ts in t]
#dates1=[dt.datetime.fromtimestamp(float(ts)/1000) for ts in t]
#dates2=[dt.datetime.fromtimestamp(float(ts)/1000) for ts in t1]


#dates=[dt.datetime.fromtimestamp(float(ts)/1000,pytz.timezone(TIMEZONE)) for ts in t]

fig=plt.figure()
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax=plt.gca()
#xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
#ax.xaxis.set_major_formatter(xfmt)

plt.title('Phone-audio Vs accl')

ax1=fig.add_subplot(111)
ax1.plot(t,v,'g')
ax1.legend(('audio'),
           'upper right', shadow=True, fancybox=True)
									
ax1.set_ylabel('dB unit')
"""
ax11=ax1.twinx()

ax11.plot(dates2,x,'b')
ax11.legend(('x '),
           'lower center', shadow=True, fancybox=True)
ax11.set_ylabel('accl unit')

ax=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)


"""




#plt.setp(ax1.xaxis.get_majorticklabels(),rotation=30)



#plt.savefig(pp, format='pdf')
#pp.savefig()
#pp.close()



plt.show()
cursor1.close()
#cursor2.close()

connection.close ()

# exit the program
sys.exit()
