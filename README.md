Thesis
======

Abstract:
		
		
		Increasing concern about energy consumption have motivated energy tracking and monitoring in recent years. Studies have shown that 			having device level energy consumption can cause users to conserve significant amount of energy, but the current electricity meter 			reports only aggregated power consumption data. Previous work [Ref paper] deployed sensors on each appliance in the home to calculate 			individual device power consumption which incurs cost and maintenance overhead.
        
		Thus, we design a system which collects data from smart meters already installed in buildings and  smartphones carried by the 			occupants and developed an  algorithm which disaggregate appliance(light, adapter, fan) consumption in shared settings such as dorm 			and perform user attribution on the disaggregated data. The idea behind the algorithm is to combine data from energy meter and 			smartphones carried by the occupants, the fusion of which detects individual user's energy consumption in shared setting, which is 			hard to detect otherwise  with a simple approach like this and attains high accuracy.
        
		We also have devised mechanism to verify the predicted power consumption with the actual power consumption. We have tested the system 			using real world dataset collected from a set of 6 rooms in a dorm setting in our university campus for a period of 7 days. Our 		approach effectively perform energy attribution to the user which include a tuple of (duration of usage, which appliance, and who) 			information with a precision and recall of  m%.



Algorithm Description in detail

System Components: MetaData, Location Detection,Overlapping and Non-Overlapping Set detection, Sound Detection

I. Meter data processings steps:

	Inputs: 
		light stream L, power stream P
	
	Input description: 
		The hostel settings have two meter in each wing "light" and "power" where Light meter tracks power changes in 
		appliances such as plug points, light and Power meter track power changes in appliances such as light, fan, AC and
		"wing" corresponds to set of rooms. We considered wing with set of rooms equal to 6 for our experiment purpose.
		Each meter goes through three different phases i.e. "R", "Y", & "B". Thus in total there are 6 different phases
		to be considered while doing the analysis meter data.
	Note:
		We have considered meter data phase wise rather then total power for detecting edges []. 

	1. Filter Drops

		Some phase have noise in the collected data samples which are periodic in nature persistent throughout the timeseries data.
		The noise have characteristics of light edges i.e they have magnitude in the range of 25-35 Watts and hence have very high
		probability to be detected during edge detection phase [explained below] thus generating unwanted edges along with 
		true edges hence garbles the detcted edge set. To avoid detecting spurious edges during edge detection phase we have 
		applied edge smoothening on those phases which generate noisy edges. Such edges have certain patterns which can be
		detected and removed. The noisy data occurring in the dataset has certain pattern associated with it i.e they are periodic
		in nature and repeats after every second. Noise has to be differentiated from the actual event so if there is a
		rising edgeof true light event in the data stream then after attaining a rise in magnitude say 30Watt for light event it wont
		drop off by the same amount the next second and will persist untill the user switch off the light and similarly the falling edges
		are differentiated from noisy edges. As can be seen in the fig[1] below figure in the left shows data stream with 
		inherent noise signals and the figure in the right shows the true events[TO BE INCLUDED].

		To filter noisy edges from the data stream we replaced every detected noisy edge with the average of the window 
		before and after the event and thus repeatedly applied this process to the entire data stream thus producing smoothened
		data stream.

		At ith time:
			Detected edge : ei
			pevious window : wpi
			next window : wni

			if ei noisy:

				ei[magnitude] = average(wpi + wni)


	
	2. Edge Detection

		In this step, we generate all the rising and falling edges from the light and power data streams. The algorithm used 
		to detect edges(rising/falling) uses [EnergyLens]. Each detected edge is a tuple ei = (mi, ti)
		of magnitude mi and time ti "when" the event occurred, where mi reflects the power change in electrical event at time ti.
	
II. WIFI stream Processing steps:
	
	Input : WIfi scan data stream for each user 

	1. Raw WIFI formatting and localization

		In this step the wifi data stream from the phone is first summarized into a window of 20sec worth of data by taking mean
		of signal strength samples received from visible access points in the wing. From this summarization we create a table 
		[time, rssi1, rssi2,....rssik] where k is the no of access points visible in the wing.
		The user localization is performed based on the above table by considering only the set of access points visible in the wing
		and whose signal strength is greater than -85dB. The algorithm  determines the location of the user in two categories 
		"in the room(1)" and "not in the room(0)", thus performing a binary location classification of each user corresponding
		to every timestamp time in the above table.

		for every user(u) belonging to set(users):

			At ith timestamp ti:
				APs : {'AP': [ap1, ap2, .. apn], 'RSSI':[rssi1, rssi2,.. rssin]}
				
				if any(APs[AP]) visible:
					if AP[RSSI] > -85 :
						locationi = 1
					else:
						locationi = 0

					
	2. Create Room Sets

		After performing user localization next step is merging the location classification of all the user creating a 
		vector <time, room1,room2,.. roomk> where k is the no of rooms (in our case 6 under experiment). The purpose of
		merging individual users localization result is to  identify all the room sets annotated by its time duration 
		i.e start time and end time. The generated room sets identifies the time duration at which users are present
		simultaneously in their room.
 

	3. Separate Overlapping and non-Overlapping sets

		The room sets are then separated into overlapping and non-Overlapping room sets. 
		The overlapping room sets are those room sets where more than one user is present simultaneously in their room for
		the same time duration annotated with the room set. On the other hand, non-Overlapping room sets are those room sets 
		where one user is present in her room for the time duration.

		Non-Overlapping room set : (tstart, tend, room2, room5, room6)
		[tstart, tend] is the time duration of stay and [room5, room6] are occupied during this time duration.

		Overlapping room set : (tstart, tend, room3)
		Again, [tstart, tend] is the time duration of stay and room3 occupant is the only room occupied during this time duration.
	
III. User Attribution

	Input : Overlapping Set, Non-Overlapping Set, Light Edges, Power Edges

	1. Room Set Edge association

		In this step edges detected in edge detection phase are assigned the room set by comparing the time of event tj
		against the time duration of room sets and associate the edge to the room set if the time tj falls within the
		(si, ei) range. If the edge do not belong to any of the time duration associated with room sets then that edge
		is discarded at this stage which is considered as an event was not done by any of the room occupants. 
		After this step all the edges from Light and Power edges have either been assigned
		room set to which it belongs to or have been discarded because they do not fall in any of the time durations for 
		which sets(overlap/non-overlap) have been created. Edges at this stage gets annotated with their time of occurrence,
		room set, magnitude, phase. Here room set association to edge means that the edge can possibly belong to any of the
		rooms which were occupied during the same time duration.		
		All those edges belonging to Light and Power edge set which have been associated to one of the non-overlapping room
		set need not be further disaggregated because room in non-overlapping set is the only room which was occupied during
		the time duration.
		For all those edges which have been associated to one of the overlapping sets, the edge can belong to any of the rooms
		in the set and with the disaggregated information available till now there is no clue about the room in which the event
		occurred. At this point algorithm can only say about the possible rooms to which event could have happened and not the
		correct room in which event happened since edge is falling in the time duration when many of the rooms were occupied
		and  
		
		We found two possible ways to associate the edge to one of the room in overlapping set, which can be achieved by the
		following two ways:
		First, there are edges which falls in the time duration of overlapping room set, but the event might have occurred on
		the phase which goes to only one of the rooms in the overlap and hence the edge is associated to that room.
		
		Second, the tuple (phase, magnitude, appliance) is such that it is unique to a single room in the overlap, then
		the room gets associated to that room from amogst the overlapping rooms in the time duration.

		When none of the above condition satisfies then the rooms are completely overlapping in terms of all the paramters
		such as phase, appliance, time duration and above condition fails to associate the edge to any of the rooms in the 
		overlap. 


		edge tuple : (ti, ei)		
		room sets time duration : [(tstart1, tend1), ...(tstartn, tendn)]  , where n is the nth set 
		
		For each (ti, ei):
			if ti falls in any (tstartk, tendk):
				if kth set is non-overlapping:
					metadatar <- [(phase, appliance, magnitude)]        , where r is the room
					if ei[phase] matches any metadatar[phase]:
						ei <- associated with room r
					else:
						ei discarded

			
				if kth set is overlapping:
					for each room r in sk
					metadatar <- [(phase, appliance, magnitude)] 

					if phasei matches any one metadatar[phase]:
						ei associated with room r
					if (phasei, magnitudei) matches one of metadatar(phase, magnitude):
						ei<- associated with room r
					else if ei[phase, magnitude] matches many of metadatar:
						ei <- associated with matched rooms in set
					else:
						ei discarded

		Final dataframe consists of ei, ti, phasei, magi, tstart, tend, room set

	2. Audio Detection

		In this step, all the edges from above step which falls in overlapping category and have not been associated to any one 
		room in the overlap
		are considered here for further disaggregation. In this step, we have an additional input MFCC feature(13 feature) of
		audio samples from users smartphone which helps gain more insight about the event occurance location. We consider audio 
		stream for all the user who are in the overlap. Audio stream contains MFCC features from the rooms in the overlap. 
		For each room in the set, we consider a minute's worth of MFCC features just before the event i.e edge occurrence time 
		and after the event and calculate eucledian distance between the two vectors and calculates maximum of the distance 
		vector. Before doing the disaggregation we set a threshold eucledian distance for each room for event like switching on 
		or off the appliance (light, adapter), locking and unlocking the door and found that the eucledian distance for 
		each room was different and the difference was accountable because of the difference in model of the phone being used 
		during the data collection. 
		Finally, max of eucledian distance of the rooms in the overlap are checked against the threshold values rooms which have 
		any matching threshold is retained and rooms for which max of eucledian distance falls below its threshold values is 
		discarded, if only one room satisfies the threshold then edge is associated to that room, if the threshold is satisfied 
		by more than one room in the overlap then edge is associated with all the rooms that satisfies the threshold condition, 
		if none satisfies the condition the edge is not associated to any of the rooms and is discarded at this point.
		After this stage all the edges have been detected which belongs to one of the rooms where it happened with good accuracy.

	3. Edge matching
	
		In this step, we find all the time slices for which the  appliance was used. We consider edges assigned to each rooms 
		separately and apply edge matching to generate the duration of time for which the event occured. Edge matching is done 
		to generate time slices using threshold based matching algorithm [Ref EnergyLens]. The edge matching algorithm matches 
		rising and falling edges of similar magnitude. All the selected rising and falling edges creates a time slice 
		ts = (tr, tf, magt, appl, room, phase) where tr is the start time , tf is the end time, magt is the power consumption 
		of the time slice, appl is the appliance detected from the metadata, room is the associated room with the time slice, 
		phase is the phase on which the event during the time slice occurred. 
		This process of edge matching is repeated  for all the rooms thus generating the time slices of
		events for the room.

	4. Power Consumption

		In this step we caluclate the total power consumption for every individual room using this algorithm by summing the 
		power consumption of each detected time slices.




