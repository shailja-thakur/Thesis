import subprocess
import os
import time
import csv
path = '/home/shailja/shailja.thakur90@gmail.com/app_profiling/'
cpucsv =  path + 'cpu.csv'
memcsv = path + 'memory.csv'
cpu_writer = csv.writer(open(cpucsv, 'w'))
mem_writer = csv.writer(open(memcsv, 'w'))
cpu_writer.writerow(['timestamp'] + ['ago_time_ms'] + ['cpu_percentage'])
mem_writer.writerow(['timestamp'] + ['mem_MB'])
os.chdir('/home/shailja/android-sdk-linux/platform-tools')

while True:
	cpu_per = ''
	delay = ''
	cpu_cmd = 'adb shell dumpsys cpuinfo'
	mem_cmd = 'adb shell dumpsys meminfo \'com.iiitd.EnegySense\''
	# cpu_top = 'adb shell top -m 20 -n 1'
	# c_t = subprocess.check_output(cpu_top.split())
	# print c_t
	# c_t = c_t.split('\r\n')
	# #print c_t
	# prev = time.time()
	# for line in c_t:
		
	# 	if 'com.iiitd.EnegySense' in line:

	# 		print line
	# 		cpu_top = line.split()[1].split('%')[0]
	# 		print cpu_top
	# curr = time.time()
	# print 'time elapsed:', curr- prev
	# time.sleep(1)



	# PREVIOUS
	c = subprocess.check_output(cpu_cmd.split())
	m = subprocess.check_output(mem_cmd.split())

	m = m.split('\r\n')
	c = c.split('\r\n')
	for line in c:

		if 'com.iiitd.EnegySense' in line:

			print line
			cpu_per = line.split('%')[0].strip()
			print time.time(), cpu_per
			
			print 'cpu percentage at ',time.time(), 'is', cpu_per
		if 'CPU usage' in line:
			print line
			delay = str(line.split()[5])[:-2]

			print 'delay ', delay

		cpu_writer.writerow([time.time()] + [delay] + [cpu_per] )

	for line in m:
		if 'allocated' in line:
			print line
			mem_MB = float(int(line.split()[4])/1000)
			print time.time(), mem_MB
			mem_writer.writerow([time.time()] + [mem_MB])

	time.sleep(1)


			

