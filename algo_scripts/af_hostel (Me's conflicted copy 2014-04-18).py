"""
Script to find the usage events based on the various sensor streams
Approach 1: Phone only
Approach 2: Phone + Meter
Approach 3: Phone + Meter + Metadata (Appl - Power Rating)
Approach 4: Phone + Meter + Metadata (Appl - Power Rating - Location)

Sensor streams: Power (4), Light (4), Audio, Wifi, Accelerometer

Author: Manaswi Saha
Date: Sep 25, 2013

"""


import os
import sys
import math
import glob
import datetime as dt
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.dates as md
from pandas import *
import decimal
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as plt
import activity_finder_algorithm as af
import localize as loc
from evaluate import *
import numpy
# import classifier as cl
# import classify_sound as cs
import warnings
import logging
from path import *
from localize import *

# Disable warnings
warnings.filterwarnings('ignore')

# Enable Logging
logger = logging.getLogger('hostel-activity-finder')
logging.basicConfig(level=logging.DEBUG)


TIMEZONE = 'Asia/Kolkata'
stars = 50

# --------------------------------------------------------------
# ActivityFinder Algorithm starts
# --------------------------------------------------------------

# Edge Transition Window (in seconds) for the change to take place
# its more for simultaneous or quick sequential activity
lwinmin = 3
pwinmin = 3
# lwinmin = pwinmin

# Power Threshold ( in Watts) for the magnitude of the change
# Its different since both the power types are well separated
# might have to keep it same for the hostel case study
lthresmin = 27   # for light meter
pthresmin = 27  # for power meter
# pthresmin = lthresmin

# Power Percent Change between rising and falling edge
# percent_change = 0.31

"""
Step 1: Edge Detection
Rising and Falling Edges in power and light meter data

Input: Raw sensor data
Output: Edges across sensor streams <S_i, S_j, ...>
    S_i = (t_i, M_i)
    where,
        t_i = time of rise/fall of edges
        M_i = magnitude of rise/fall of edges

"""


# ----------------
# HELPER FUNCTIONS
# ----------------
# Filter periodic drops in power stream phase Y
print 'Detected Periodic Drops in Power stream Phase2'

def smoothening(df_p, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    # for phase Y in power meter apply moving average
    df_p['powerphase2'] = numpy.convolve(df_p['powerphase2'], window, 'same') 
    #df_p['powerphase2'] = numpy.convolve(df_p['powerphase2'], window, 'same') 
    return df_p


def filter_drops(df_p, phase):
    for i in df_p.index:
        if (i + 2) in df_p.index:

            prev=df_p.ix[i][phase]
            curr = df_p.ix[i + 1][phase]
            next = df_p.ix[i + 2][phase]
            
            # If edge , get the magnitude of rising and falling edges
            curr_prev_diff = int(math.fabs(curr)) - int(math.fabs(prev))
            curr_next_diff = int(math.fabs(curr)) - int(math.fabs(next))
            if ((prev - curr) > 0) & ((curr - next) < 0):
                    if int(math.fabs(curr_prev_diff - curr_next_diff)) <= 18:
                        if (i - 2) in df_p.index:
                            if (i + 3) in df_p.index :

                                pwin = (df_p.ix[i - 2][phase]+ df_p.ix[i - 1][phase] + df_p.ix[i - 1][phase])/3
                                nwin = df_p.ix[i + 3][phase]
                                df_p.ix[i+1][phase] = (pwin + nwin) / 2
                        else :
                            df_p.ix[i+1][phase] = (prev + curr) / 2
                        #print "idx", i, "time", df_p.ix[i + 1]['time'],'prev', df_p.ix[i][phase],'curr', df_p.ix[i + 1][phase], 'next', df_p.ix[i + 2][phase]

    # avg = 0
    # idx_list = []
    # for i in df_p.index:
    #     prev=df_p.ix[i][phase]
    #     curr = df_p.ix[i + 1][phase]
    #     next = df_p.ix[i + 2][phase]
    #     # If edge , get the magnitude of rising and falling edges
    #     curr_prev_diff = int(math.fabs(curr)) - int(math.fabs(prev))
    #     curr_next_diff = int(math.fabs(curr)) - int(math.fabs(next))
        
        
    #     if ((curr - next) < 27):
    #         avg = (avg + curr) / 2
    #         idx_list.append(df_p.ix[i])
    #     else :
    #         for i in idx_list:
    #             df_p.ix[i]['phase'] = avg
    #         idx_list = []
    #         avg = 0
    #         continue
    return df_p
                      


# Checks if there is any light edge (rising or falling) at the index provided
# power_stream



def check_if_light_edge(df_l, index, power_stream):

    i = index
    prev = int(round(df_l.ix[i - 1][power_stream]))
    curr = int(round(df_l.ix[i][power_stream]))
    next = int(round(df_l.ix[i + 1][power_stream]))
    currwin = int(round(df_l.ix[i + lwinmin][power_stream]))

    if i - lwinmin not in (df_l.index):
        prevwin = 0
    else:
        prevwin = int(df_l.ix[i - lwinmin][power_stream])

    # OLD CODE
    # If checking for a particular phase, increase by 10 watts
    # if power_stream != "lightpower":
    #     if math.floor(prev) != 0:
    #         prev = prev + 10
    #     if math.floor(curr) != 0:
    #         curr = curr + 10
    #     if math.floor(next) != 0:
    #         next = next + 10
    #     if math.floor(currwin) != 0:
    #         currwin = currwin + 10

    time = df_l.ix[i]['time']
    per_lthresmin = int(0.25 * lthresmin)

    # Checking for missing time samples
    prev_time = df_l.ix[i - 1]['time']
    next_time = df_l.ix[i + 1]['time']

    # Indicates next time sample is missing
    prev_missing_sample = (time - prev_time) > 1
    next_missing_sample = (next_time - time) > 1

    prev_curr_diff = curr - prev
    curr_next_diff = int(math.fabs(next - curr))
    curr_prevwin_diff = prevwin - curr
    curr_nextwin_diff = currwin - curr

    # FOR DEBUGGING
    # print "Looking for light edge in stream:", power_stream, "for Index:", i
    # print "Magnitude::", curr_nextwin_diff

    # if(time in [1385466741, 1385467127, 1385485791, 1385486655]):
    # if time in [1392302261]:
    #     state = 1
    # if ((power_stream == 'powerphase2') & (state == 1)):
    #     logger.debug("\n")
    #     logger.debug("Per lthresmin %d", per_lthresmin)
    #     logger.debug(
    #         "Looking for light edge in stream: %s for Index: %d", power_stream, i)
    #     logger.debug("R:: currtime %s prev %s curr %s next %s",
    #                  dt.datetime.fromtimestamp(time), prev, curr, next)
    #     logger.debug("currwin %s", currwin)
    #     logger.debug(
    #         "prev_curr_diff %s curr_next_diff %s", prev_curr_diff, curr_next_diff)
    #     logger.debug("curr_nextwin_diff %s curr_prevwin_diff %s",
    #                  curr_nextwin_diff, curr_prevwin_diff)

    #     if (prev_curr_diff < per_lthresmin and curr_nextwin_diff >= lthresmin and
    #         ((curr_next_diff == 0 and prev_curr_diff == 0) or (curr_next_diff > prev_curr_diff))
    #        and curr_next_diff > 0):
    #         logger.debug("True")
    #         logger.debug("Missing Sample value %s", next_missing_sample)
    #         logger.debug("Curr next diff %s", curr_next_diff)
    #         if next_missing_sample:
    #             logger.debug("Missing Sample yes")
    #         if int(curr_next_diff) >= lthresmin:
    #             logger.debug("Satisfied condition")
    #     elif (prev_curr_diff < per_lthresmin and curr_next_diff >= per_lthresmin
    #           and curr_nextwin_diff > lthresmin):
    #         logger.debug(" CurrNextDiff between lthresmin half: %d", i)
    #     elif prev_curr_diff < lthresmin and curr_nextwin_diff > lthresmin:
    #         logger.debug("True-- Fan")
    #     elif (prev_curr_diff < lthresmin and math.floor(curr_nextwin_diff) <= (-lthresmin)
    #           and curr_next_diff > prev_curr_diff):
    #         logger.debug("True - Falling")

    # logger.debug(
    #     "\nR:: %d currtime %s prev %s curr %s next %s", i, dt.datetime.fromtimestamp(time),
    #     prev, curr, next)
    # logger.debug("currwin:%s", currwin)
    # logger.debug("prev_curr_diff : %s curr_next_diff %s", prev_curr_diff, curr_next_diff)
    # logger.debug("curr_nextwin_diff :%s", curr_nextwin_diff)

    if (prev_curr_diff < per_lthresmin and curr_nextwin_diff >= lthresmin and
            ((curr_next_diff == 0 and prev_curr_diff == 0) or (curr_next_diff > prev_curr_diff))
       and curr_next_diff > 0):

        logger.debug(
            "\nR1:: %d currtime %s prev %s curr %s next %s", i, dt.datetime.fromtimestamp(time),
            prev, curr, next)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff")
        # curr_prevwin_diff

        edge_type = "rising"
        if next_missing_sample and int(curr_next_diff) > lthresmin:
            logger.debug("Missing Sample:: Index %d", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "lightphase": power_stream}
            return edge_type, row
        # or curr_next_diff > lthresmin:
        elif (curr_next_diff >= per_lthresmin):
            logger.debug("Here1 Index:: %d", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "lightphase": power_stream}
            return edge_type, row
        else:
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "lightphase": power_stream}
            return edge_type, row
    elif (prev_curr_diff < per_lthresmin and curr_next_diff >= per_lthresmin
          and curr_nextwin_diff > lthresmin):
        logger.debug(
            "\nR2:: %d currtime %s prev %s curr %s next %s", i, dt.datetime.fromtimestamp(time),
            prev, curr, next)
        # logger.debug("\nR2::" , i, "currtime", dt.datetime.fromtimestamp(time), "prev")
        # logger.debug(prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff,)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff",)
        # curr_prevwin_diff

        edge_type = "rising"
        # Storing the rising edge e_i = (time_i, mag_i)
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
               "curr_power": curr, "lightphase": power_stream}
        return edge_type, row

    elif (prev_curr_diff < lthresmin and math.floor(curr_nextwin_diff) <= (-lthresmin)
          and ((curr_next_diff == 0 and prev_curr_diff == 0) or (curr_next_diff > prev_curr_diff))
          and curr_prevwin_diff > -lthresmin):

        # logger.debug("\nF::", "currtime", dt.datetime.fromtimestamp(time),
            # "prev ", prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff",)
        # curr_prevwin_diff

        edge_type = "falling"
        if prev_missing_sample is True or next_missing_sample is True:
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "lightphase": power_stream}
            return edge_type, row
        elif curr_next_diff < lthresmin or curr_next_diff >= lthresmin:
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "lightphase": power_stream}
            return edge_type, row
        else:
            pass
    return "Not an edge", {}


def check_if_power_edge(df_p, index, power_stream):

    i = index
    prev = df_p.ix[i - 1][power_stream]
    curr = df_p.ix[i][power_stream]
    next = df_p.ix[i + 1][power_stream]
    currwin = int(round(df_p.ix[i + pwinmin][power_stream]))

    # OLD CODE
    # If checking for a particular phase, increase by 10 watts
    # if power_stream != "power":
    #     if math.floor(prev) != 0:
    #         prev = prev + 10
    #     if math.floor(curr) != 0:
    #         curr = curr + 10
    #     if math.floor(next) != 0:
    #         next = next + 10
    #     if math.floor(currwin) != 0:
    #         currwin = currwin + 10

    time = df_p.ix[i]['time']

    if i - pwinmin not in (df_p.index):
        prevwin = 0
    else:
        prevwin = int(df_p.ix[i - pwinmin][power_stream])
        # if power_stream != "power":
        #     if math.floor(prevwin) != 0:
        #         prevwin = prevwin + 10
        # prevwintime = df_p.ix[i - pwinmin]['time']
    per_pthresmin = int(0.25 * curr)

    # Checking for missing time samples
    prev_time = df_p.ix[i - 1]['time']
    next_time = df_p.ix[i + 1]['time']

    # Indicates next time sample is missing
    prev_missing_sample = (time - prev_time) > 1
    next_missing_sample = (next_time - time) > 1

    prev_curr_diff = int(math.fabs(curr - prev))
    curr_next_diff = int(math.fabs(next - curr))
    curr_nextwin_diff = int(currwin - curr)
    curr_prevwin_diff = int(prevwin - curr)

    # Code for debugging
    # range(2683, 3652)
    # state = 0
    # if time in [1392302261]:
    #     state = 1
    # if ((power_stream == 'powerphase2') & (state == 1)):
    #     logger.debug(
    #         "Looking for power edge for Index %d in power_stream %s", i, power_stream)
    #     logger.debug("F:: currtime %s prev %s curr %s next %s",
    #                  dt.datetime.fromtimestamp(time), prev, curr, next)
    #     logger.debug("currwin %s", currwin)
    #     logger.debug(
    #         "prev_curr_diff %s curr_next_diff %s", prev_curr_diff, curr_next_diff,)
    #     logger.debug("curr_nextwin_diff %s", curr_nextwin_diff)

    #     logger.debug("Per pthresmin %s", per_pthresmin)

    #     if (prev_curr_diff < pthresmin):
    #         logger.debug("prev_curr_diff < pthresmin 1True")
    #     else:
    #         logger.debug("prev_curr_diff < pthresmin False1")
    #     if curr_next_diff > prev_curr_diff and curr_next_diff > 0:
    #         logger.debug(
    #             "curr_next_diff > prev_curr_diff and curr_next_diff > 0: 2True")
    #     else:
    #         logger.debug(
    #             "curr_next_diff > prev_curr_diff and curr_next_diff > 0: False2")
    #     if curr_nextwin_diff <= -pthresmin or curr_nextwin_diff <= lthresmin:
    #         logger.debug("curr_nextwin_diff <= -l/pthresmin: 3True")
    #     else:
    #         logger.debug("curr_nextwin_diff <= -pthresmin: False3")
    #     if curr_next_diff != prev_curr_diff:
    #         logger.debug("curr_next_diff != prev_curr_diff: 4True")
    #     else:
    #         logger.debug("curr_next_diff != prev_curr_diff: 4False")
    #     if curr_next_diff >= per_pthresmin:
    #         logger.debug("curr_next_diff >= per_pthresmin: 5True")
    #     else:
    #         logger.debug("curr_next_diff >= per_pthresmin: False5")
    #     if curr_nextwin_diff > pthresmin:
    #         logger.debug("curr_nextwin_diff > pthresmin: True6")
    #     else:
    #         logger.debug("curr_nextwin_diff > pthresmin: False6")

    # Original Edge Detection Algorithm for Power Edges
    # if (prev_curr_diff < pthresmin and curr_nextwin_diff >= pthresmin
    #    and curr_next_diff > prev_curr_diff
    #    and curr_next_diff >= per_pthresmin):

    #     edge_type = "rising"
    # Storing the rising edge e_i = (time_i, mag_i)
    #     row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
    #     return edge_type, row
    # elif (prev_curr_diff < pthresmin and curr_nextwin_diff <= (-pthresmin)
    #       and curr_next_diff > prev_curr_diff
    #       and int(curr_next_diff) >= per_pthresmin):

    #     edge_type = "falling"
    # Storing the falling edge e_i = (time_i, mag_i)
    #     row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
    #     return edge_type, row
    # return "Not an edge", {}

    # Testing #
    if (prev_curr_diff < pthresmin and curr_nextwin_diff >= pthresmin and
            curr_next_diff > prev_curr_diff and curr_next_diff > 0):

        # logger.debug("\nR1::", i, "currtime", dt.datetime.fromtimestamp(time),)
        # logger.debug("prev ", prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "prevwintime",
            # dt.datetime.fromtimestamp(prevwintime))
        # logger.debug("curr_next_diff ", curr_next_diff,)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff")
        # curr_prevwin_diff

        edge_type = "rising"
        if (curr_next_diff >= per_pthresmin):
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "powerphase": power_stream}
            return edge_type, row
        if next_missing_sample and curr_next_diff > pthresmin:
            logger.debug("Missing Sample:: Index %s", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "powerphase": power_stream}
            return edge_type, row
        elif (curr_next_diff > per_pthresmin):
            logger.debug("Here1 Index:: %s", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "powerphase": power_stream}
            return edge_type, row
        else:
            logger.debug("Here2 Index:: %s", i)
            pass
    elif (prev_curr_diff < pthresmin and curr_nextwin_diff > pthresmin
          and curr_next_diff >= per_pthresmin and curr_next_diff > 0):
        # logger.debug("\nR2::", i, "currtime", dt.datetime.fromtimestamp(time), "prev ",)
        # logger.debug(prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff,
            # "prevwintime", dt.datetime.fromtimestamp(prevwintime))
        # logger.debug("curr_next_diff ", curr_next_diff,)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff",)
        # curr_prevwin_diff

        edge_type = "rising"
        # Storing the rising edge e_i = (time_i, mag_i)
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
               "curr_power": curr, "powerphase": power_stream}
        return edge_type, row

    elif (prev_curr_diff < pthresmin and math.floor(curr_nextwin_diff) <= (-pthresmin)
          and curr_next_diff > prev_curr_diff
          and curr_prevwin_diff > -pthresmin):

        # logger.debug("\nF::", "currtime", dt.datetime.fromtimestamp(time),)
        # logger.debug("prev ", prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff,
            # "prevwintime", dt.datetime.fromtimestamp(prevwintime))
        # logger.debug("curr_next_diff ", curr_next_diff)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff",)
        # curr_prevwin_diff

        edge_type = "falling"
        if prev_missing_sample is True or next_missing_sample is True:
            logger.debug("Falling Edge1:: Index %s", i)
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "powerphase": power_stream}
            return edge_type, row
        elif curr_next_diff < pthresmin or curr_next_diff > pthresmin:
            logger.debug("Falling Edge2:: Index %s", i)
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff,
                   "curr_power": curr, "powerphase": power_stream}
            return edge_type, row
        else:
            pass
    return "Not an edge", {}


def generate_light_edges(df_l):

    # ----------------------------------------------
    # LIGHT EDGE DETECTION
    # ----------------------------------------------
    # Find the edges in the light meter (LP Power phase) since activity and
    # its duration is well explained/defined by the usage of lights and
    # fans.

    ix_list_l = df_l.index
    r_row_list = []
    f_row_list = []

    print "-" * stars
    print "Detecting Edges in Light Meter"
    print "-" * stars

    l_phase=['lightphase1','lightphase2','lightphase3']
    for phase in l_phase:
        for i in range(1, ix_list_l[-1] - lwinmin + 1):
        
            edge_type, result = check_if_light_edge(df_l, i, phase)

            if edge_type == "falling":
                f_row_list.append(result)
            elif edge_type == "rising":
                r_row_list.append(result)
            else:
                pass

    rising_edges_l_df = pd.DataFrame(
        r_row_list, columns=['index', 'time', 'magnitude', 'curr_power','lightphase'])
    rising_edges_l_df = rising_edges_l_df.set_index('index', drop=True)

    falling_edges_l_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude', 'curr_power','lightphase'])
    falling_edges_l_df = falling_edges_l_df.set_index('index', drop=True)

    # Adding the actual times to the frame
    rising_edges_l_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in rising_edges_l_df.time]
    falling_edges_l_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in falling_edges_l_df.time]

    print "Rising Edges::\n", rising_edges_l_df
    print "Falling edges::\n", falling_edges_l_df

    rising_edges_l_df, falling_edges_l_df = af.filter_edges(
        rising_edges_l_df, falling_edges_l_df, lwinmin, lthresmin)

    # Print the edges with the timestamps values converted to UTC +5.30
    # format
    # Rising Edges
    print_lr_df = rising_edges_l_df.copy()
    t = pd.to_datetime(print_lr_df['time'], unit='s')
    index = pd.DatetimeIndex(t, tz='UTC').tz_convert(TIMEZONE)
    print_lr_df.set_index(index, inplace=True, drop=False)

    # Falling Edges
    print_lf_df = falling_edges_l_df.copy()
    t = pd.to_datetime(print_lf_df['time'], unit='s')
    index = pd.DatetimeIndex(t.tolist(), tz='UTC').tz_convert(TIMEZONE)
    print_lf_df.set_index(index, inplace=True, drop=False)

    print "-" * stars
    print "Filtered Edges:"
    print "-" * stars
    print "Rising Edges::\n", print_lr_df
    print "Falling edges::\n", print_lf_df

    return rising_edges_l_df, falling_edges_l_df


def generate_power_edges(df_p):

    # ----------------------------------------------
    # POWER EDGE DETECTION
    # ----------------------------------------------
    # Find the edges in the power meter (Power phase) for detecting
    # an power event

    ix_list_p = df_p.index
    r_row_list = []
    f_row_list = []
    print "-" * stars
    print "Detecting Edges in Power Meter"
    print "-" * stars
    p_phase=['powerphase1','powerphase2','powerphase3']
    for phase in p_phase:
        for i in range(1, ix_list_p[-1] - pwinmin + 1):
        
            edge_type, result = check_if_power_edge(df_p, i, phase)
            if edge_type == "falling":
                f_row_list.append(result)
            elif edge_type == "rising":
                r_row_list.append(result)
            else:
                pass

    rising_edges_p_df = pd.DataFrame(
        r_row_list, columns=['index', 'time', 'magnitude', 'curr_power','powerphase'])
    rising_edges_p_df = rising_edges_p_df.set_index('index', drop=True)

    falling_edges_p_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude', 'curr_power','powerphase'])
    falling_edges_p_df = falling_edges_p_df.set_index('index', drop=True)

    # Adding the actual times to the frame
    rising_edges_p_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in rising_edges_p_df.time]
    falling_edges_p_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in falling_edges_p_df.time]

    print "Rising Edges::\n", rising_edges_p_df
    print "Falling Edges::\n", falling_edges_p_df

    rising_edges_p_df, falling_edges_p_df = af.filter_edges(
        rising_edges_p_df, falling_edges_p_df, pwinmin, pthresmin)

    # Print the edges with the timestamps values converted to UTC +5.30
    # format

    # Rising Edges
    print_pr_df = rising_edges_p_df.copy()
    t = pd.to_datetime(
        rising_edges_p_df['time'], unit='s')
    index = pd.DatetimeIndex(t.tolist(), tz='UTC').tz_convert(TIMEZONE)
    print_pr_df.set_index(index, inplace=True, drop=False)

    # Falling Edges
    print_pf_df = falling_edges_p_df.copy()
    t = pd.to_datetime(
        falling_edges_p_df['time'], unit='s')
    index = pd.DatetimeIndex(t.tolist(), tz='UTC').tz_convert(TIMEZONE)
    print_pf_df.set_index(index, inplace=True, drop=False)

    print "-" * stars
    print "Filtered Edges:"
    print "-" * stars
    print "Rising Edges::\n", print_pr_df.tail(50)
    print "Falling Edges::\n", print_pf_df.tail(50)

    return rising_edges_p_df, falling_edges_p_df

def filter_hostel_edges(rise_df, fall_df,meter, df):

    # Removing duplicate indexes
    rise_df["index"] = rise_df.index
    rise_df.drop_duplicates(cols='index', take_last=True, inplace=True)
    del rise_df["index"]

    fall_df["index"] = fall_df.index
    fall_df.drop_duplicates(cols='index', take_last=True, inplace=True)
    del fall_df["index"]
    print 'Filtering Hostel Edges'
    # For power edges
    if meter == 'powerphase':

        # Filter 1
        # Rising edges 
        # Filter edges of Fridge
        
        idx_list=[]
        for i in rise_df.index:
            magnitude = int(math.fabs(rise_df.ix[i]['magnitude']))
            #time = rise_df.ix[i]['time']
            if rise_df.ix[i]['powerphase']=='powerphase3':
                 # Likely power consumption of Fridge is 88-105
                if magnitude >= 88 and magnitude <= 105:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)   
        rise_df = rise_df.ix[rise_df.index - idx_list] 
        
        # Falling edges 
        # Filter edges of Fridge 
        
        idx_list=[]
        for i in fall_df.index:
            magnitude = int(math.fabs(fall_df.ix[i]['magnitude']))
            #time = rise_df.ix[i]['time']
            if fall_df.ix[i]['powerphase']=='powerphase3':
                 # Likely power consumption of Fridge is 88-105
                if magnitude >= 88 and magnitude <= 105:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)   
        fall_df = fall_df.ix[fall_df.index - idx_list] 


        # Filter 2
        # Filter edges corresponding to Washing Machine, Press, and heater 
        # Rising Edges
          
        idx_list = []  
        for i in rise_df.index:
            magnitude = int(math.fabs(rise_df.ix[i]['magnitude']))
            #time = rise_df.ix[i]['time']
            if rise_df.ix[i]['powerphase']=='powerphase2':
                 # Likely power consumption of Washing Machine and Press is 300-600
                if magnitude >= 300 and magnitude <= 650:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
            
                # Likely power consumption of Heater is 1000-1500
                if magnitude >= 1000 and magnitude <= 1500:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
               
            # if time in range(1385646060, 1385648144):
            #     idx_list.append(i)
        rise_df = rise_df.ix[rise_df.index - idx_list]

        # Falling Edges
        
        idx_list = []
        for i in fall_df.index:
            magnitude = int(math.fabs(fall_df.ix[i]['magnitude']))
            #time = rise_df.ix[i]['time']
            if fall_df.ix[i]['powerphase']=='powerphase2':
                 # Likely power consumption of Washing Machine and Press is 300-600
                if magnitude >= 300 and magnitude <= 650:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
            
                # Likely power consumption of Heater is 1000-1500
                if magnitude >= 1000 and magnitude <= 1500:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
                
            # if time in range(1385646060, 1385648144):
            #     idx_list.append(i)
        fall_df = fall_df.ix[fall_df.index - idx_list]

        # # # Filter 3
        # # # Filter unwanted periodic edges on Power meter, Phase2
        # # # Rising Edges
        # print 'Filtering noise'
        # print 'Rising Edges'
        # idx_list=[]
        # for i in rise_df.index:
        #     if rise_df.ix[i]['powerphase'] == 'powerphase2':
        #         magnitude = rise_df.ix[i]['magnitude']
        #         now_time = rise_df.ix[i]['time']
        #         #print 'magnitude:'+str(magnitude)
        #         #print "time:"+str(dt.datetime.fromtimestamp(now_time))
        #         if int(math.fabs(magnitude)) in range(20, 35):
        #             row_df = df[df.time == now_time]
        #             #print 'next index:'+str((row_df.index[0]) + 1)
        #             prev = df_p.ix[(row_df.index[0]) - 1]['powerphase2']
        #             curr = df_p.ix[(row_df.index[0])]['powerphase2']
        #             next = df_p.ix[(row_df.index[0]) + 1]['powerphase2']
        #             #print 'prev', prev,'curr', curr,'next', next
        #             curr_prev_diff = int(math.fabs(curr)) - int(math.fabs(prev))
        #             curr_next_diff = int(math.fabs(curr)) - int(math.fabs(next))
        #             pwin = df_p.ix[(row_df.index[0]) - 3]['powerphase2']
        #             nwin = df_p.ix[(row_df.index[0]) + 3]['powerphase2']
        #             if (int(math.fabs(pwin - nwin)) <= 10) & (int(math.fabs(curr_prev_diff - curr_next_diff)) <= 10):
        #                 #print 'Diff less than equal to 5'
                        
        #                 print "idx", i, "magnitude", magnitude
        #                 idx_list.append(i) 

        #             # next_edge_type, result = check_if_power_edge(df, int(row_df.index[0]) + 1, 'powerphase2')
        #             # print "next_edge_type"+str(next_edge_type)
        #             # if (next_edge_type == "Not an edge"):
        #             #     for prev_idx in (int(row_df.index[0]) - 4, int(row_df.index[0]) - 3, int(row_df.index[0]) - 2, int(row_df.index[0]) - 1):
        #             #         print 'previous index :'+str(prev_idx)
        #             #         prev_edge_type, result = check_if_power_edge(df, prev_idx, 'powerphase2')
        #             #         print 'prev_edge_type:'+str(prev_edge_type)
        #             #         if (prev_edge_type == "falling") :
        #             #             #print 'prev_edge_type:'+str(prev_edge_type)
        #             #             #print "curr_pev magnitude"+str(result["magnitude"])  
        #             #             #if int(math.fabs(magnitude)) in range(int(math.fabs(curr_prev)):
        #             #             #     # curr_next_next = rise_df.ix[i+2]['magnitude']
        #             #             #     # if curr_next in range(curr_next_next - 1, curr_next_next + 1):
        #             #             print result
        #             #             print "idx", i, "magnitude", magnitude
        #             #             idx_list.append(i) 
        #             #             break
        # #print idx_list
        # rise_df = rise_df.ix[rise_df.index - idx_list]
        
        # # # Falling Edges
        # print 'Falling Edges'
        # idx_list=[]
        # for i in fall_df.index:
        #     if fall_df.ix[i]['powerphase'] == 'powerphase2':
        #         magnitude = fall_df.ix[i]['magnitude']
        #         now_time = fall_df.ix[i]['time']
        #         #print 'magnitude:'+str(magnitude)
        #         #print "time:"+str(dt.datetime.fromtimestamp(now_time))
        #         if int(math.fabs(magnitude)) in range(20, 35):
        #             row_df = df[df.time == now_time]
        #             #print 'next index:'+str((row_df.index[0]) + 1)
        #             curr = df_p.ix[(row_df.index[0])]['powerphase2']
        #             curr_next = df_p.ix[(row_df.index[0]) + 1]['powerphase2']
        #             curr_next_next = df_p.ix[(row_df.index[0]) + 2]['powerphase2']
        #             #print 'curr', curr,'next', curr_next,'next_next', curr_next_next
        #             curr_curr_next_diff = int(math.fabs(curr)) - int(math.fabs(curr_next))
        #             curr_next_curr_next_next_diff = int(math.fabs(curr_next)) - int(math.fabs(curr_next_next))
        #             pwin = df_p.ix[(row_df.index[0]) - 2]['powerphase2']
        #             nwin = df_p.ix[(row_df.index[0]) + 4]['powerphase2']
        #             print 'Window diff', int(math.fabs(pwin - nwin)),'Edge difference',int(math.fabs(curr_curr_next_diff - curr_next_curr_next_next_diff)) 
        #             if (int(math.fabs(pwin - nwin)) <= 10) & (int(math.fabs(curr_curr_next_diff - curr_next_curr_next_next_diff)) <= 10):
                        
        #                 print "idx", i, "magnitude", magnitude
        #                 idx_list.append(i) 
        # fall_df = fall_df.ix[fall_df.index - idx_list]

    print "-" * stars
    print "Filtered Hostel Edges:"
    print "-" * stars
    print "Rising Hostel Edges::\n", rise_df
    print "Falling Hostel Edges::\n", fall_df
    return rise_df,fall_df


def make_edge_list(rise_l_df, fall_l_df, rise_p_df, fall_p_df):

    # Generating light edge list
    rise_l = ['rise' for i in rise_l_df.index]
    df_l_r = pd.DataFrame({'time' : rise_l_df['time'], 'magnitude' : rise_l_df['magnitude'], 'phase' : rise_l_df['lightphase'], 'edge' : rise_l})
    fall_l = ['fall' for i in fall_l_df.index]
    df_l_f = pd.DataFrame({'time' : fall_l_df['time'], 'magnitude' : fall_l_df['magnitude'], 'phase' : fall_l_df['lightphase'], 'edge' : fall_l})
    edge_list_l_df = concat([df_l_r, df_l_f])

    # Generating power edge list
    rise_p = ['rise' for i in rise_p_df.index]
    df_p_r = pd.DataFrame({'time' : rise_p_df['time'], 'magnitude' : rise_p_df['magnitude'], 'phase' : rise_p_df['powerphase'], 'edge' : rise_p})
    fall_p = ['fall' for i in fall_p_df.index]
    df_p_f = pd.DataFrame({'time' : fall_p_df['time'], 'magnitude' : fall_p_df['magnitude'], 'phase' : fall_p_df['powerphase'], 'edge' : fall_p})
    edge_list_p_df = concat([df_p_r, df_p_f])
    
    print "\n Filtered Correlated light edge List:: \n", edge_list_l_df
    print "\n Filtered Correlated power edge List:: \n", edge_list_p_df
    return edge_list_l_df, edge_list_p_df



def edge_detection(df_l, df_p):

    # print "Using Light Threshold::", lthresmin
    # print "Using Power Threshold::", pthresmin

    rising_edges_l_df, falling_edges_l_df = generate_light_edges(df_l)
    rising_edges_p_df, falling_edges_p_df = generate_power_edges(df_p)

    # Filter edges here - the preprocessing step
   
    rising_edges_l_df, falling_edges_l_df = filter_hostel_edges(
         rising_edges_l_df, falling_edges_l_df, 'lightphase', df_l)
    rising_edges_p_df, falling_edges_p_df = filter_hostel_edges(
         rising_edges_p_df, falling_edges_p_df, 'powerphase', df_p)

    return make_edge_list(rising_edges_l_df, falling_edges_l_df, rising_edges_p_df, falling_edges_p_df) 
    

    #return light_edges, power_edges

"""
******Meta-data for hostel******

"""

metadata = {
    "room" : ['room1' , 'room2', 'room3', 'room4', 'room5', 'room6'],
    "TubeLight_P": ['R', 'R', 'Y', 'Y', 'B', 'B'],
    "LeftPlug_P":  ['Y', 'Y', 'B','B','Y','Y'],
    "AC_P":        ['Y', 'Y', 'B','B','Y','Y'],
    "RightPlug_L": ['R', 'R', 'Y', 'Y', 'B', 'B'],
    "ELight_L":    ['R', 'R', 'Y', 'Y', 'B', 'B']
}

metadata_rooms = {
    "room" : ['1', '2', '3', '4', '5', '6'],
    
    "power" : [['powerphase2'], ['powerphase2'], ['powerphase3','powerphase2'],  ['powerphase3','powerphase2'] ,
            ['powerphase3','powerphase2'],  ['powerphase3','powerphase2'] ],
    
    "light" : ['lightphase1', 'lightphase1', 'lightphase2', 'lightphase2', 'lightphase3', 'lightphase3'],
    
    "power_appliance" : [['LP-powerphase2-60'], ['LP-powerphase2-60'], ['TL-powerphase2-35', 'LP-powerphase3-60'],
    ['TL-powerphase2-35', 'LP-powerphase3-60'],  ['TL-powerphase3-35', 'LP-powerphase2-60'], ['TL-powerphase3-35', 'LP-powerphase2-60']],
     
    "light_appliance" : [['TL-lightphase1-35', 'RP-lightphase1-60', 'EL-lightphase1-30','F-lightphase1-35'], ['TL-lightphase1-35','RP-lightphase1-60', 'EL-lightphase1-30','F-lightphase1-35'], ['RP-lightphase2-60', 'EL-lightphase2-30','F-lightphase2-35'], ['RP-lightphase2-60', 'EL-lightphase2-30','F-lightphase2-35'],
     ['RP-lightphase3-60', 'EL-lightphase3-30','F-lightphase3-35'], ['RP-lightphase3-60', 'EL-lightphase3-30','F-lightphase3-35']] 

}







metadata_room_phase = {
    "R" : ['room1','room2'],
    "Y" : ['room1', 'room2', 'room3', 'room4','room5','room6'],
    "B" : ['room3','room4', 'room5', 'room6']
}

metada_light_plug_p = {
    "set1(R,Y)" : ['room1','room2'],
    "set2(Y,B)" : ['room3', 'room4', 'room5', 'room6']
}

metadata_AC_p = {
    "set1(Y)" : ['room1', 'room2', 'room5', 'room6'],
    "set2(B)" : ['room3', 'room4']
}

metadata_elight_plug_l = {
    "set1(R)" : ['room1', 'room2'],
    "set1(Y)" : ['room3', 'room4'],
    "set2(B)" : ['room5', 'room6']
}



def max_min_timestamp(loc_csv ):
    # Determine all the overlapping and non_overlapping location of the users
    #    in windows

    # Step1:Find Minimum Timestamp 
    min_timestamp = 2000000000000
    max_timestamp = 0
    loc_dir = loc_csv + '*.csv'

    for user_loc in glob.glob(loc_dir):
        df_loc = pd.read_csv(user_loc)
        df_loc = df_loc.sort_index(by = 'timestamp')
        print user_loc
        for i in df_loc.index:
            if int(df_loc.ix[df_loc.index[0]]['timestamp']) < int(min_timestamp):
                min_timestamp = df_loc.ix[df_loc.index[0]]['timestamp']
            if int(df_loc.ix[df_loc.index[-1]]['timestamp']) > int(max_timestamp):
                max_timestamp = df_loc.ix[df_loc.index[-1]]['timestamp']
        
    print 'Minimum Timestamp', min_timestamp
    print 'Maximum Timestamp', max_timestamp

    return min_timestamp, max_timestamp

    
def fill_missing_samples(loc_csv , day):

    min_timestamp, max_timestamp = max_min_timestamp(loc_csv)
    loc_dir = loc_csv + '*.csv'
    
    for user_loc in glob.glob(loc_dir):
        print user_loc
        df_loc = pd.read_csv(user_loc)
        df_loc = df_loc.sort_index(by = 'timestamp')
        filename = user_loc.split('/')[10].split('.')[0] + 'fomatted'+ '.csv'
        print filename
        outcsv = DATA_PATH + day + MISSING_FORMATTED_PATH + filename 
        writer = csv.writer(open(outcsv, 'w'))
        writer.writerow(['timestamp'] + [str(user_loc.split('/')[10].split('.')[0].split('_')[0])] )
        beg = min_timestamp
        end = max_timestamp
        for idx in df_loc.index:
            
            curr_time = df_loc.ix[idx]['timestamp']
            curr_location = df_loc.ix[idx]['location']
            if idx == 0:
                #print df_loc.ix[idx]['timestamp']
                diff = curr_time - beg
                print 'beginning difference', diff
                if int(diff) == 0:
                    print 'Zero beg difference'
                    location = curr_location
                    time = beg

                    print time, location
                    writer.writerow([time] + [location])
                    continue
                
                if int(math.fabs(diff)) > 0:
                    count = int(diff)
                    print 'Beginning missing samples between',beg, curr_time,' are ', diff
                    print 'Beginning count', count
                    for i in range(0, count+1):
                        time = (beg + i)
                        location = 0
                        print time, location
                        writer.writerow([time] + [location])
                        continue

            if ((idx > 0) & (idx < df_loc.index[-1])):

                prev = df_loc.ix[idx - 1]['timestamp']
                prev_location = df_loc.ix[idx - 1]['location']
                diff = curr_time - prev
                
                print 'count', diff
                if int(math.fabs(diff)) == 1:
                    print ' missing samples between',prev, curr_time,' are ', diff
                    location = curr_location
                    time = prev + 1
                    print time, location
                    writer.writerow([time] + [location])
                    continue
                        
                if ((int(math.fabs(diff)) > 0) & (int(math.fabs(diff)) <= MIN_STAY_TIME)):
                    print ' missing samples between',prev, curr_time,' are ', diff
                    count = int(diff)
                    for i in range(1, count+1):

                        time = (prev + i)
                        location = prev_location
                        print time, location
                        writer.writerow([time] + [location])
                        continue
                if int(math.fabs(diff)) > MIN_STAY_TIME:
                    print ' missing samples between',prev, curr_time,' are ', diff
                    count = int(diff)
                    for i in range(1, count+1):

                        time = prev + i
                        location = 0
                        print time, location
                        writer.writerow([time] + [location])
                        continue

            if idx == df_loc.index[-1]:
                print 'index',idx
                prev = df_loc.ix[idx - 1]['timestamp']
                prev_location = df_loc.ix[idx - 1]['location']
                #print 'maximum time and last time', end, prev

                diff = prev - end
                
                #print 'last difference', diff
                #print 'last missing samples', prev, end
                #print 'count', diff
                if int(diff) == 0:
                    print 'Zero last difference'
                    location = curr_location
                    time = prev + 1
                    print time, location
                    writer.writerow([time] + [location])
                    continue
                if ((int(math.fabs(diff)) > 0) & (int(math.fabs(diff)) <= MIN_STAY_TIME)):
                    print 'Last missing samples between',curr_time, end,' are ', diff
                    count = int(math.fabs(diff))
                    #print 'count', count
                    for i in range(1, count + 1):

                        time = (prev + i)
                        location = curr_location
                        print time, location
                        writer.writerow([time] + [location]) 
                        continue
                if int(math.fabs(diff)) > MIN_STAY_TIME:
                    print 'Last missing samples between',curr_time, end,' are ', diff
                    count = int(math.fabs(diff))
                    for i in range(1, count + 1):

                        time = (prev + i)
                        location = 0  
                        print time, location
                        writer.writerow([time] + [location])
                        continue
        
        print 'Created'
        print outcsv
        #df = pd.read_csv(outcsv)
        #print df
        
def users_location_table(csv_path, day):
    
    min_t, max_t = max_min_timestamp(DATA_PATH + str(day) + WIFI_INOUT)
    df = pd.DataFrame({'timestamp':list(range(int(min_t), int(max_t)))})
    filename = 'user_location_table'+ '.csv'
    outcsv = csv_path + filename 
    writer = csv.writer(open(outcsv, 'w'))
    
    columns = []
    timestamp = []
    rooms = []

    for i in glob.glob(DATA_PATH + day + MISSING_FORMATTED_PATH + '*.csv'):
        frame = pd.read_csv(i)
        columns.append(frame.columns[1:][0][3])

    header = ''
    for i in columns:
        header = header + '_' + i 

    writer.writerow(['timestamp'] + [header]  )
    for i in glob.glob(DATA_PATH + day + MISSING_FORMATTED_PATH + '*.csv'):
        frame = pd.read_csv(i)
        df = pd.merge(df, frame)

        #print df
    print df
    for idx in df.index:
        column = ''

        for col in df.columns[1:]:
            
            column = column + str(df.ix[idx][col])
            if col != df.columns[len(df.columns[1:])]:
                column = column + ','
        timestamp.append(df.ix[idx]['timestamp'])
        rooms.append(column)
        writer.writerow([df.ix[idx]['timestamp']] +  ['"'+column+'"'])
    df = pd.DataFrame({'timestamp' : timestamp, 'rooms' : rooms})
    print df
    return df
        
def distinct(csv_path):
    
    df = pd.read_csv(csv_path +'location_room_set.csv')
    df = df.drop_duplicates()
    df.to_csv(csv_path +'location_room_set.csv', cols = df.columns.values, index = False)

    df = pd.read_csv(csv_path + 'location_room_set.csv')
    writer = csv.writer(open(csv_path + 'reduced_location_room_set.csv', 'w'))

    writer.writerow(['start_time'] + ['end_time'] + ['room_set']  )

    unique_rooms = df.room_set.unique()
    for room in unique_rooms:
        print room
        rooms_idx_list = df.index[df.room_set == room]  
        df_rooms = df.ix[rooms_idx_list]
        #print df_rooms

        df_rooms.index = arange(0, len(df_rooms))
        
        start_time = df_rooms.ix[0]['start_time']
        end_time = df_rooms.ix[0]['end_time']
        room = df_rooms.ix[0]['room_set']

        if len(df_rooms) == 1:
            
            writer.writerow([start_time] + [end_time] + [room])
            print 'unique', start_time, end_time, room
        else:

            for idx in df_rooms.index:
                #print 'time', df_rooms.ix[idx + 1]['start_time'], type(int(df_rooms.ix[idx + 1]['start_time']))
                if idx == df_rooms.index[-1]:
                    print 'last record',start_time, df_rooms.ix[idx]['end_time']
                    writer.writerow([start_time] + [df_rooms.ix[idx]['end_time']] + [df_rooms.ix[idx]['room_set']])
                    
                if ((idx >= df_rooms.index[0]) & (idx < df_rooms.index[-1])):

                    #print start_time, end_time, room
                    if (int(df_rooms.ix[idx + 1]['start_time']) - int(df_rooms.ix[idx]['end_time'])) <= 50:
                        print 'difference 1', df_rooms.ix[idx + 1]['end_time'], df_rooms.ix[idx]['end_time'], df_rooms.ix[idx]['room_set']
                        end_time = df_rooms.ix[idx]['end_time']
                        continue
                        
                        #print df_rooms.ix[idx]['start_time'], df_rooms.ix[idx]['end_time'], df_rooms.ix[idx + 1]['room_set']
                        #print 'Continue'
                    

            
                    else:
                        print 'Writing difference greater', start_time, end_time, room
                        
                        
                        writer.writerow([start_time] + [end_time] + [room])
                        start_time = df_rooms.ix[idx + 1]['start_time']
                        room = df_rooms.ix[idx + 1]['room_set']

                        print 'next value',start_time, df_rooms.ix[idx + 1]['end_time'], room

    #df = pd.DataFrame({'start_time' : start, 'end_time' : end, 'room_set' : rooms})
    #df.to_csv(csv_path + 'reduced_location_room_set.csv', index = False)





def get_rooms_corresponding_loc_set(csv_loc, filename):
    df = pd.read_csv(csv_loc + filename)
    header = df.columns[1:][0].split('_')
    header = header[1:]
    header = [h for h in header if h != '']

    print 'rooms available in sets ', header

    filename = 'revised_user_location_table.csv'
    outcsv = csv_loc + filename 
    print outcsv
    writer = csv.writer(open(outcsv, 'w'))
    writer.writerow(['timestamp'] + ['rooms']  )
    #time = []
    rooms = []

    for idx in df.index:

        rooms = ''
        room_set = []
        r = df.ix[idx][1]
        r = r[1:-1].split(',')
        for i in r:
            room_set.append(i.split('.')[0])

        #print room_set
        # room_set = [int(r) for r in room_set if r != ',' ]
        # room_set = [int(r) for r in room_set if r != '"' ]
        
        nonzeroes_set = [h for r,h in zip(room_set, header) if int(r) != 0]
        if nonzeroes_set != []:
            #time.append(df.ix[idx]['timestamp'])
            #rooms.append(nonzeroes_set)
            for room in nonzeroes_set:
                    rooms = rooms + room
                    if room != nonzeroes_set[len(nonzeroes_set) - 1]:
                        rooms = rooms + ','
                    
            writer.writerow([df.ix[idx]['timestamp']] + [ rooms ])






def time(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def overlapping_non_overlapping_sets(csv_path, filename):

    print 'Creating merged use location table'
    #df = users_location_table()
    df = pd.read_csv(csv_path + filename)
    print df
    #get_rooms_set(df)
    # print 'Finding overlapping and non-overlapping sets'
    #df = pd.read_csv(USER_ATTRIBUTION_TABLES + 'user_location_table.csv')
   
    print 'Determining Overlapping/Non-Overlapping room sets'

    filename = 'location_room_set'+ '.csv'
    outcsv = csv_path + filename 
    writer = csv.writer(open(outcsv, 'w'))
    writer.writerow(['start_time'] + ['end_time'] + ['room_set'] )
    prev_room_set = []
    curr_room_set = []
    start_time = df.ix[df.index[0]]['timestamp']
    start = []
    end = []
    overlapp_set = []
    for idx in df.index:
        # column = df.columns[1:][0].split('_')
        # print column

        if idx == df.index[0]:
            
            start_time = df.ix[df.index[0]]['timestamp']
            end_time = df.ix[df.index[0]]['timestamp']
            #room_set = df.ix[0]['rooms'].split('"')[1]
            room_set = df.ix[0]['rooms'] 

            curr_room_set = room_set
                   
            #print 'index', idx, 'start_time',start_time,'end_time',end_time, 'room_set', room_set

        if ((idx > df.index[0]) & (idx < df.index[-1])):

            # curr_room_set = df.ix[idx]['rooms'].split('"')[1]
            # prev_room_set = df.ix[idx - 1]['rooms'].split('"')[1]
            curr_room_set = df.ix[idx]['rooms']
            prev_room_set = df.ix[idx - 1]['rooms']
            #print 'index > 1','index',idx,'prev_room_set',prev_room_set,'curr_room_set',curr_room_set,'start_time',start_time,'curr_time',df.ix[idx ]['timestamp']
                   
            if set(curr_room_set) == set(prev_room_set):
                continue
            if set(curr_room_set) != set(prev_room_set):
                
                # print 'room set changed'
                # print 'curr_room_set',curr_room_set,'prev_room_set',prev_room_set
                #print 'start_time',start_time,'end_time',end_time
                end_time = df.ix[idx - 1]['timestamp']
                #print 'end_time',end_time
                start.append(start_time)
                end.append(end_time)
                overlapp_set.append(prev_room_set)
                writer.writerow([start_time] + [end_time] + [prev_room_set])
                print 'index', idx, 'start_time', time(start_time),'end_time',time(end_time), 'room_set', prev_room_set
                start_time = df.ix[idx]['timestamp']
                #print 'start_time',start_time
        if idx == df.index[-1]:
            #print 'last index'
            # curr_room_set = df.ix[idx]['rooms'].split('"')[1]
            # prev_room_set = df.ix[idx - 1]['rooms'].split('"')[1]
            curr_room_set = df.ix[idx]['rooms']
            prev_room_set = df.ix[idx - 1]['rooms']
            if set(curr_room_set) == set(prev_room_set):
                end_time = df.ix[idx ]['timestamp']

                start.append(start_time)
                end.append(end_time)
                overlapp_set.append(prev_room_set)

                writer.writerow([start_time] + [end_time] + [prev_room_set])
                print 'index', idx, 'start_time',time(start_time),'end_time',time(end_time), 'room_set', prev_room_set
            else:
               
                # print 'room set changed'
                # print 'curr_room_set',curr_room_set,'prev_room_set',prev_room_set
                end_time = df.ix[idx - 1]['timestamp']

                start.append(start_time)
                end.append(end_time)
                overlapp_set.append(prev_room_set)

                start.append(df.ix[idx]['timestamp'])
                end.append(df.ix[idx]['timestamp'])
                overlapp_set.append(curr_room_set)

                writer.writerow([start_time] + [end_time] + [prev_room_set])
                writer.writerow([df.ix[idx]['timestamp']]+ [df.ix[idx]['timestamp']] + [curr_room_set])
                print 'index', idx, 'start_time',time(start_time),'end_time',time(end_time), 'room_set', curr_room_set
    df = pd.DataFrame({'start_time' : start, 'end_time' : end, 'room_set' : overlapp_set})
    
    return df


def separate_overlapping_nonOverlapping_sets(csv_path, filename):
    df = pd.read_csv(csv_path + filename)
    nonOverlapping_idx = []
    overlapping_idx = []
    for idx in df.index:

        #room_nos = len(df.ix[idx][2].split('"')[1].split(','))
        room_nos = len(str(df.ix[idx][2]).split(','))
        if room_nos == 1:
            nonOverlapping_idx.append(idx)
        else:
            overlapping_idx.append(idx)

    print 'overlapping set index',overlapping_idx
    print 'non-overlapping set index',nonOverlapping_idx

    return overlapping_idx, nonOverlapping_idx


    
def roomset_edge_association(csv_path, filename, edge_light_df, edge_power_df, type):

    overlap_idx, non_overlap_idx = separate_overlapping_nonOverlapping_sets(csv_path, filename)
   
    #Overlapping and non-overlapping room sets
    non_overlap_df = pd.read_csv(csv_path + filename)
    print non_overlap_df
    if type == 'nonoverlap':
        non_overlap_df = pd.DataFrame(non_overlap_df.ix[non_overlap_idx])
    else:
        non_overlap_df = pd.DataFrame(non_overlap_df.ix[overlap_idx])
    print 'non overlap room set list'
   
    non_overlap_light_edge_ids = []
    non_overlap_power_edge_ids = []
    start = []
    end = []
    edge_power_list = []
    edge_light_list = []
    rooms = []
    #print "start_time,end_time,rom_set"
    # for i in non_overlap_df.index:
    #     print non_overlap_df.ix[i]['start_time'],dt.datetime.fromtimestamp(non_overlap_df.ix[i]['start_time']),non_overlap_df.ix[i]['end_time'],dt.datetime.fromtimestamp(non_overlap_df.ix[i]['end_time']),non_overlap_df.ix[i]['room_set']
    
    for idx in non_overlap_df.index:

        start_time = int(non_overlap_df.ix[idx]['start_time'])
        end_time = int(non_overlap_df.ix[idx]['end_time'])
     
        room = non_overlap_df.ix[idx]['room_set']
        #room = room.split('"')[1]
        room = room.split(',')
        print len(edge_light_df.index), len(edge_power_df.index)
        for idx in edge_light_df.index:
            #print 'light index',idx
            #print 'start time roomset',start_time, 'start time of light edge',edge_light_df.ix[idx]['time'],'end time room set',end_time
            if edge_light_df.ix[idx]['time'] in range(start_time , end_time ):
                print 'In range',idx, time(edge_light_df.ix[idx]['time'])
                non_overlap_light_edge_ids.append(idx)

        # Find all the nonoverlapping/overlapping edges falling in window
        for idx in edge_power_df.index:
            #print 'power index', idx
            if edge_power_df.ix[idx]['time'] in range(start_time , end_time ):
                print 'In range',idx, time(edge_power_df.ix[idx]['time'])
                non_overlap_power_edge_ids.append(idx)

        if ((non_overlap_power_edge_ids == []) & (non_overlap_light_edge_ids == [])):
            continue

        else:
            start.append(start_time)
            end.append(end_time)
            edge_light_list.append(str(non_overlap_light_edge_ids).strip('[]'))
            edge_power_list.append(str(non_overlap_power_edge_ids).strip('[]'))
            rooms.append(room)

            non_overlap_power_edge_ids = []
            non_overlap_light_edge_ids = []
            #print 'room',room.split('"')[1]
    print len(edge_light_list), len(edge_power_list), len(start), len(end)

    print rooms

    df_window = pd.DataFrame({'start_time' : start, 'end_time' : end,
     'edge_light_indices' : edge_light_list,'edge_power_indices' : edge_power_list, 'rooms' : rooms})
    return df_window, non_overlap_df
    
def appliance_magnitude_range(appliance):
    print 'matching appliance'
    #print type(appliance)
    offset = 0
    if 'TL' in str(appliance): 
        offset = 5
        #print 'TubeLight'
        
    elif 'LP' in str(appliance):
        #print 'LeftPlug'
        offset = 5 
        
    elif 'AC' in str(appliance): 
        offset = 100
        #print 'AC'
        
    elif 'RP' in str(appliance): 
        offset = 5
        #print 'RightPlug'
        
    elif 'EL' in str(appliance): 
        offset = 3
        #print 'EmergencyLight'
        
    elif 'F' in str(appliance): 
        offset = 3
        #print 'Fan'
        
    #print offset
    return int(offset)


def nonoverlap_room_event_mapping(df_nonoverlap_window_edges, edge_df, df_metadata_rooms, types):

    power_appliances = ['TL', ' LP', ' AC']
    light_appliances = ['RP', ' EL',' F']
    edge = []
    mag = []
    room = []
    phaze = []
    appliance = []
    start_time = []
    end_time = []
    event_time = []
    #print df_nonoverlap_window_edges.head(9)
    for idx in df_nonoverlap_window_edges.index:
        #print 'Window', df_nonoverlap_window_edges.ix[idx]
        win_room = df_nonoverlap_window_edges.ix[idx]['rooms']

        win_room = str(win_room).strip('[').strip(']')
        
        #print 'window room', win_room[1]
        win_room = win_room[1]


        if types == 'power':
            # print 'Type power'
            # indices = df_metadata_rooms.room == win_room
            # print 'indices',indices
            # print 'metadata rooms',df_metadata_rooms.room
            # print 'Window room',win_room
            phase_idx = df_metadata_rooms.index[df_metadata_rooms.room == win_room]
            phase = df_metadata_rooms.ix[int(phase_idx)]['power']

            #phase = df_metadata_rooms.power[df_metadata_rooms.room == win_room]
            print 'power phase',phase
            #phase = str(phase).split('\n')[0].split('[')[1].split(']')[0].split(',')

            phase_app_magnt_idx =  df_metadata_rooms.index[df_metadata_rooms.room == win_room]            
            phase_app_magnt =  df_metadata_rooms.ix[int(phase_app_magnt_idx)]['power_appliance']

            #phase_app_magnt =  df_metadata_rooms.power_appliance[df_metadata_rooms.room == win_room]
            edge_indices = df_nonoverlap_window_edges.ix[idx]['edge_power_indices']
            #appliances = power_appliances
        else:
            #phase = df_metadata_rooms.light[df_metadata_rooms.room == win_room]
            

            phase_idx = df_metadata_rooms.index[df_metadata_rooms.room == win_room]
            phase = df_metadata_rooms.ix[int(phase_idx)]['light']
            print ' light phase',phase

            #phase = str(phase).split('\n')[0].split('  ')[2]
            #phase_app_magnt =  df_metadata_rooms.light_appliance[df_metadata_rooms.room == win_room]

            phase_app_magnt_idx =  df_metadata_rooms.index[df_metadata_rooms.room == win_room]            
            phase_app_magnt =  df_metadata_rooms.ix[int(phase_app_magnt_idx)]['light_appliance']


            edge_indices = df_nonoverlap_window_edges.ix[idx]['edge_light_indices']
            #print edge_indices
            appliances = light_appliances
        

        #phase_app_magnt = str(phase_app_magnt).split('\n')[0].split('[')[1].split(']')[0].split(',')
        #print phase_app_magnt
        
        #print phase
        if edge_indices == '' or edge_indices == []:
            print 'Edge indices empty'
            continue
        edge_indices =  edge_indices.split(',')
        print edge_indices
        
        
        for ids in edge_indices:
            print ids
            #print win_room
            #print df_metadata_rooms.room, df_metadata_rooms.power
            
            edge_phase = edge_df.ix[int(ids)]['phase']
            edge_magnitude = edge_df.ix[int(ids)]['magnitude']
            edge_time = edge_df.ix[int(ids)]['time']
            # if types == 'light':
            #     print edge_magnitude
            if any(edge_phase == i for i in phase):

                print 'Finding appliance associated with edge'
                for meta in phase_app_magnt:
                    print meta
                    metadata = str(meta).split('-') 

                    print metadata[0], metadata[1], metadata[2]  
                    if edge_phase == metadata[1]:
                                #print 'edge phase matches with metadata'
                        #print edge_power_df.ix[int(ids)]
                    #print int(metadata[2]) 
                    #print int(edge_magnitude)
                        # for app in appliances:
                        #     if metadata[0] == app:
                                #print edge_magnitude
                                #print metadata[2]
                                #print metadata[0]
                                #print type(metadata[0])
                                #print type(edge_magnitude)
                                #print type(metadata[2])
                                ranges = appliance_magnitude_range(metadata[0])
                                #print type(ranges)
                                #ranges = int(ranges)
                                #print ranges + 2
                                if 'TL' in metadata[0]:
                                    lo_range = ranges + 1
                                    up_range = ranges 
                                else:
                                    lo_range = up_range = ranges

                                print 'ranges', lo_range, up_range
                                if int(math.fabs(edge_magnitude)) in range(int(metadata[2]) - int(lo_range), int(metadata[2]) + int(up_range)):
                                    
                                    #print app
                                    edge.append(edge_df.ix[int(ids)]['edge'])
                                    mag.append(edge_df.ix[int(ids)]['magnitude'])
                                    phaze.append(edge_df.ix[int(ids)]['phase'])
                                    room.append(win_room)
                                    appliance.append(metadata[0])
                                    start_time.append(df_nonoverlap_window_edges.ix[idx]['start_time'])
                                    end_time.append(df_nonoverlap_window_edges.ix[idx]['end_time'])
                                    event_time.append(edge_time)
                                    
                                
    #print len(edge),len(mag),len(phaze)   
    df_events = pd.DataFrame({ 'Room' : room,'Appliance' : appliance , 'Edge' : edge, 'Magnitude' : mag ,'Event Time' : event_time,'Phase' : phaze})    
    
    return df_events

def nonoverlap_event_association(df_nonoverlap_window_edges, df_nonoverlap_df, edge_light_df, edge_power_df):
#def nonoverlap_event_association():
    
    
    df_metadata_rooms = pd.DataFrame(metadata_rooms)
    print "-" * stars
    print "            METADATA"
    print "-" * stars
    print df_metadata_rooms
    print df_nonoverlap_window_edges
    print df_nonoverlap_df
    #print 'Edge','Magnitude','Phase','Room','Start Time','End Time'
    print "-" * stars
    print 'Nonoverlapping windows power event recognition'
    print "-" * stars
    df_nonoverlap_power_events = nonoverlap_room_event_mapping(df_nonoverlap_window_edges, edge_power_df, df_metadata_rooms, 'power')
    

    print df_nonoverlap_power_events
    #print 'light events mapping'
    print "-" * stars
    print 'Nonoverlapping windows light event recognition'
    print "-" * stars
    df_nonoverlap_light_events = nonoverlap_room_event_mapping(df_nonoverlap_window_edges, edge_light_df, df_metadata_rooms, 'light')
    
    
    print df_nonoverlap_light_events
    return df_nonoverlap_power_events, df_nonoverlap_light_events

    # for idx in df_nonoverlap_window_edges.index:
    #     #print 'Window', df_nonoverlap_window_edges.ix[idx]
    #     win_room = df_nonoverlap_window_edges.ix[idx]['rooms']
        
    #     phase = df_metadata_rooms.power[df_metadata_rooms.room == win_room]
    #     phase_app_magnt =  df_metadata_rooms.power_appliance[df_metadata_rooms.room == win_room]
    #     phase_app_magnt = str(phase_app_magnt).split('\n')[0].split('[')[1].split(']')[0].split(',')
        
    #     phase = str(phase).split('\n')[0].split('[')[1].split(']')[0].split(',')
    #     #print phase
    #     power_indices = df_nonoverlap_window_edges.ix[idx]['edge_power_indices']
    #     power_indices =  power_indices.split(',')
       
    #     for ids in power_indices:
    #         #print win_room
    #         #print df_metadata_rooms.room, df_metadata_rooms.power
            
    #         edge_phase = edge_power_df.ix[int(ids)]['phase']
    #         edge_magnitude = edge_power_df.ix[int(ids)]['magnitude']
    #         if any(edge_phase == i for i in phase):

                
    #             for meta in phase_app_magnt:
    #                 #print meta
    #                 metadata = meta.split('-')   
    #                 if edge_phase == metadata[1]:
    #                     #print edge_power_df.ix[int(ids)]
    #                 #print int(metadata[2]) 
    #                 #print int(edge_magnitude)
    #                     if metadata[0] == 'TL':
    #                         if int(math.fabs(edge_magnitude)) in range(int(metadata[2]) - 3, int(metadata[2]) + 3):
    #                             # print 'associate this edge with this window'
    #                             # print 'Tubelight'
    #                             #print edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'TubeLight', df_nonoverlap_window_edges.ix[idx]['rooms']
    #                             edge.append(edge_power_df.ix[int(ids)]['edge'])
    #                             mag.append(edge_power_df.ix[int(ids)]['magnitude'])
    #                             phaze.append(edge_power_df.ix[int(ids)]['phase'])
    #                             room.append(df_nonoverlap_window_edges.ix[idx]['rooms'])
    #                             appliance.append('Tubelight')
    #                             start_time.append(df_nonoverlap_window_edges.ix[idx]['start_time'])
    #                             end_time.append(df_nonoverlap_window_edges.ix[idx]['end_time'])
    #                             #edge,mag,phaze,room,appliance,start_time,end_time = append_to_dataframe(edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'TubeLight', df_nonoverlap_window_edges.ix[idx]['rooms'], df_nonoverlap_window_edges.ix[idx]['start_time'],df_nonoverlap_window_edges.ix[idx]['end_time'])
    #                             #print edge_power_df.ix[int(ids)]
    #                             #print df_nonoverlap_window_edges.ix[idx]
    #                     if metadata[0] == ' LP':
    #                         if int(math.fabs(edge_magnitude)) in range(int(metadata[2]) - 20, int(metadata[2]) + 20):
    #                             #print edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'LeftPlug', df_nonoverlap_window_edges.ix[idx]['rooms']
    #                             edge.append(edge_power_df.ix[int(ids)]['edge'])
    #                             mag.append(edge_power_df.ix[int(ids)]['magnitude'])
    #                             phaze.append(edge_power_df.ix[int(ids)]['phase'])
    #                             room.append(df_nonoverlap_window_edges.ix[idx]['rooms'])
    #                             appliance.append('LeftPlug')
    #                             start_time.append(df_nonoverlap_window_edges.ix[idx]['start_time'])
    #                             end_time.append(df_nonoverlap_window_edges.ix[idx]['end_time'])
    #                             #edge,mag,phaze,room,appliance,start_time,end_time = append_to_dataframe(edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'LeftPlug', df_nonoverlap_window_edges.ix[idx]['rooms'], df_nonoverlap_window_edges.ix[idx]['start_time'],df_nonoverlap_window_edges.ix[idx]['end_time'])
                               
    #                             # print 'associate this edge with this window'
    #                             # print 'Left Plug'
    #                             #print edge_power_df.ix[int(ids)]
    #                             #print df_nonoverlap_window_edges.ix[idx]  
    #                     if metadata[0] == ' AC':
    #                         if int(math.fabs(edge_magnitude)) in range(int(metadata[2]) - 100, int(metadata[2]) + 100):
    #                             edge.append(edge_power_df.ix[int(ids)]['edge'])
    #                             mag.append(edge_power_df.ix[int(ids)]['magnitude'])
    #                             phaze.append(edge_power_df.ix[int(ids)]['phase'])
    #                             room.append(df_nonoverlap_window_edges.ix[idx]['rooms'])
    #                             appliance.append('AC')
    #                             start_time.append(df_nonoverlap_window_edges.ix[idx]['start_time'])
    #                             end_time.append(df_nonoverlap_window_edges.ix[idx]['end_time'])
    #                             #edge,mag,phaze,room,appliance,start_time,end_time = append_to_dataframe(edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'AC', df_nonoverlap_window_edges.ix[idx]['rooms'], df_nonoverlap_window_edges.ix[idx]['start_time'],df_nonoverlap_window_edges.ix[idx]['end_time'])
                               
    #                             #print edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'AC', df_nonoverlap_window_edges.ix[idx]['rooms']
    #                             # print 'associate this edge with this window'
    #                             # print 'AC'
    #                             #print edge_power_df.ix[int(ids)]
    #                             #print df_nonoverlap_window_edges.ix[idx]  
    #                     if metadata[0] == ' RP':
    #                         if int(math.fabs(edge_magnitude)) in range(int(metadata[2]) - 20, int(metadata[2]) + 20):
    #                             edge.append(edge_power_df.ix[int(ids)]['edge'])
    #                             mag.append(edge_power_df.ix[int(ids)]['magnitude'])
    #                             phaze.append(edge_power_df.ix[int(ids)]['phase'])
    #                             room.append(df_nonoverlap_window_edges.ix[idx]['rooms'])
    #                             appliance.append('RightPlug')
    #                             start_time.append(df_nonoverlap_window_edges.ix[idx]['start_time'])
    #                             end_time.append(df_nonoverlap_window_edges.ix[idx]['end_time'])
    #                             #edge,mag,phaze,room,appliance,start_time,end_time = append_to_dataframe(edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'RightPlug', df_nonoverlap_window_edges.ix[idx]['rooms'], df_nonoverlap_window_edges.ix[idx]['start_time'],df_nonoverlap_window_edges.ix[idx]['end_time'])
                               
    #                             #print edge_power_df.ix[int(ids)]['edge'], edge_power_df.ix[int(ids)]['magnitude'], edge_power_df.ix[int(ids)]['phase'],'RightPlug', df_nonoverlap_window_edges.ix[idx]['rooms']
    #                             # print 'associate this edge with this window'
    #                             # print 'RightPlug'
    #                             #print edge_power_df.ix[int(ids)]
    #                             #print df_nonoverlap_window_edges.ix[idx]  
    # df_nonoverlap_events = pd.DataFrame({ 'Room' : room,'Appliance' : appliance ,'Start Time' : start_time, 'End Time' : end_time, 'Edge' : edge, 'Magnitude' : mag , 'Phase' : phaze})    
                
def overlap_room_events(df_overlap_window_edges, edge_df,df_metadata_rooms, types):
    
    edge = []
    mag = []
    room = []
    phaze = []
    app = []
    start_time = []
    end_time = []
    event_time = []

    o_edge = []
    o_room = []
    o_start_time = []
    o_end_time = []
    o_edge_type = []
    o_appliance = []
    o_mag = []
    o_phaze = []
    o_event_time = []
    #print df_nonoverlap_power_events
    for idx in df_overlap_window_edges.index:
        start  = df_overlap_window_edges.ix[idx]['start_time']
        end = df_overlap_window_edges.ix[idx]['end_time']
        #print idx
        win_room = df_overlap_window_edges.ix[idx]['rooms']
        #print win_room
        if types == 'power':
            print 'Power edges'
            edge_indices = df_overlap_window_edges.ix[idx]['edge_power_indices']
            if edge_indices == '' or edge_indices == []:
                print 'Edge indices empty'
                continue
            else:
                edge_indices =  edge_indices.split(',')
            
            print edge_indices
        else:
            print 'light edges'
            edge_indices = df_overlap_window_edges.ix[idx]['edge_light_indices']

            if edge_indices == '' or edge_indices == []:
                print 'Edge indices empty'
                continue
            else:
                edge_indices =  edge_indices.split(',')
            print edge_indices

        for ids in edge_indices:

            # get edge phase and magnitude 
            edge_type = edge_df.ix[int(ids)]['edge']
            edge_phase = edge_df.ix[int(ids)]['phase']
            edge_magnitude = edge_df.ix[int(ids)]['magnitude']
            edge_time = edge_df.ix[int(ids)]['time']
            print edge_time
            #print edge_phase, edge_magnitude, edge_time
            overlapping_rooms = []
            appliance =  []
            for rm in win_room:
                #print rm 

                # rm_metadata = df_metadata_rooms.power_appliance[df_metadata_rooms.room == rm]
                # rm_metadata = str(rm_metadata).split('\n')[0].split('[')[1].split(']')[0].split(',')

                rm_metadata_idx = df_metadata_rooms.index[df_metadata_rooms.room == rm]

                if types == 'power':
                    rm_metadata = df_metadata_rooms.ix[int(rm_metadata_idx)]['power_appliance']
                else:
                    rm_metadata = df_metadata_rooms.ix[int(rm_metadata_idx)]['light_appliance']

                for meta in rm_metadata:
                    meta = str(meta).split('-')
                    #print meta
                    if edge_phase == meta[1]:
                        ranges = appliance_magnitude_range(meta[0])

                        if 'TL' in meta[0]:
                                    lo_range = ranges + 1
                                    up_range = ranges 
                        else:
                                    lo_range = up_range = ranges

                        if int(math.fabs(edge_magnitude)) in range(int(meta[2]) - int(lo_range), int(meta[2]) + int(up_range)):
                            print rm,edge_magnitude,edge_phase
                            if rm not in overlapping_rooms:
                                overlapping_rooms.append(rm)
                            appliance.append(meta[0])
            print 'overlapping rooms for edge',ids,overlapping_rooms
            if len(overlapping_rooms) == 1:

                #allocate the edge to the room
                print overlapping_rooms[0],appliance[0]
                print edge_magnitude, edge_time
                edge.append(edge_type)
                mag.append(edge_magnitude)
                phaze.append(edge_phase)
                room.append(overlapping_rooms[0])
                app.append(str(appliance[0]))
                start_time.append(start)
                end_time.append(end)
                event_time.append(edge_time)
                # row = {'Room' :  [overlapping_rooms[0]],'Appliance' :  [appliance[0]],'Start Time' : [start_time], 'End Time' : [end_time], 'Edge' : [edge_type], 'Magnitude' : [edge_magnitude] , 'Phase' : [edge_phase]}
                # row = pd.DataFrame(row)
                # df_nonoverlap_power_events.append(row)

            elif len(overlapping_rooms) > 1:
                #creating overlapping room set and edges to include for audio detection
                o_start_time.append(start)
                o_end_time.append(end)
                o_edge.append(int(ids))
                o_room.append(overlapping_rooms)
                o_appliance.append(str(appliance[0]))
                o_edge_type.append(edge_type)
                o_mag.append(edge_magnitude)
                o_phaze.append(edge_phase)
                o_event_time.append(edge_time)
                #print overlapping_rooms
            overlapping_rooms = []
            appliance = []
    
    print len(edge), len(mag),len(phaze),len(room), len(app),len(start_time),len(end_time)
    df_nonoverlap_event = pd.DataFrame({ 'Room' : room,'Appliance' : app , 'Edge' : edge, 'Magnitude' : mag, 'Event Time' : event_time, 'Phase' : phaze })    
    # df_power_overlap_events = pd.DataFrame({'start_time' : o_start_time, 'end_time' : o_end_time,
    #  'edge_indices' : o_edge, 'rooms' : o_room, 'edge' : o_edge_type, 'appliance' : o_appliance}) 
    df_power_overlap_events = pd.DataFrame({ 'Room' : o_room,'Appliance' : o_appliance , 'Edge' : o_edge_type, 'Magnitude' : o_mag, 'Event Time' : o_event_time, 'Phase' : o_phaze })    
    #df_nonoverlap_power_events = concat([df_nonoverlap_power_events, df_nonoverlap_event])
    print 'Overlapping events in overlapping windows ' + types
    print df_power_overlap_events
    print 'NonOverlapping events in overlapping windows ' + types
    print df_nonoverlap_event

    return df_power_overlap_events, df_nonoverlap_event, o_edge

#def check_euc_distance(bef , aft):
         
        
   

def events_sound_detection(df_overlap_events, edge_ids,  edge_df, df_metadata_rooms, day):


    euc_distance = []
    print 'In sound detection activity'
    print df_overlap_events
    print 'length',len(df_overlap_events)
    final_room = 0
    overlap_rooms = df_overlap_events.copy()
    for idx, ids in zip(df_overlap_events.index, edge_ids):
        #print df_overlap_events.ix[idx]
        edge_index = ids
        print edge_index
        edge_time = int(edge_df.time[edge_df.index == edge_index]) 
        edge_mag = edge_df.magnitude[edge_df.index == edge_index]
        edge_phase = edge_df.phase[edge_df.index == edge_index]

        o_rooms = df_overlap_events.ix[idx]['Room']
        
        #print edge_index, edge_time, o_rooms
        bef = []
        aft = []
        
        bef_start_time = int(edge_time) - 50
        aft_start_time = bef_end_time = int(edge_time)
        aft_end_time = int(edge_time) + 50

        #print bef_start_time, bef_end_time, aft_end_time
        max_euc = 0
        #euc_d = []
        for room in o_rooms:
            euc_distance = []
            euc_d = []
            
            if (( bef_start_time > edge_df.time[edge_df.index[-1]]) | (aft_end_time > edge_df.time[edge_df.index[-1]] )):
                break

            # Fetch audio features for the room (before and after) event
            print room
            audio_df = pd.read_csv(AUDIO_FILES_CSV_PATH + '/' + day + '/' + 'C00'+ room + '.csv')



            audio_df['time'] = audio_df['time'] / 1000
            bef_idx = audio_df.index[((audio_df.time >= bef_start_time) & (audio_df.time <= bef_end_time))]
            aft_idx = audio_df.index[((audio_df.time >= aft_start_time) & (audio_df.time <= aft_end_time))]

            MFCC_bef = audio_df.ix[bef_idx][audio_df.columns[1:-1]].mean()
            MFCC_aft = audio_df.ix[aft_idx][audio_df.columns[1:-1]].mean()
            euc_dist = 0
            for mi, mj in zip(MFCC_bef, MFCC_aft):
                        euc_dist = euc_dist + math.pow((mi - mj), 2)

            eucledian_distance = math.sqrt(euc_dist) 
            #euc_d.append(eucledian_distance)

            if max_euc < eucledian_distance:           
                print max_euc, eucledian_distance
                max_euc = eucledian_distance
                final_room = room
        # print euc_d
        # if euc_d[0] < euc_d[1]:
        #     final_room = o_rooms[0]
        # else:
        #     final_room = o_rooms[1] 

            # for ids in audio_df.index:
            #     b = 0
            #     a = 0
            #     euc_dist = 0
            #     sum_mfcc = 0
            #     audio_time = int(audio_df.ix[ids]['time']) / 1000
                #     # Calculate 2 min window before and after events 
                
            #     if ((audio_time >= bef_start_time) & (audio_time <= bef_end_time)):
                    
            #         bef.append(ids)
            #         b = ids
             
            #     if ((audio_time > aft_start_time) & (audio_time <= aft_end_time)):
                    
            #         aft.append(ids)
            #         a = ids
                
            #     MFCC_bef = audio_df.ix[b][audio_df.columns[1:-1]]
            #     MFCC_aft = audio_df.ix[a][audio_df.columns[1:-1]]

            #     # euc distance with MFCC 1 feature only
                
            #     #sum_mfcc = math.pow((MFCC_bef[0] - MFCC_aft[0]), 2)

            #     #euc_d.append( math.sqrt(sum_mfcc) )

            #     # euc distance with MFCC 1-13 feature only
            #     for mi, mj in zip(MFCC_bef, MFCC_aft):
            #             euc_dist = euc_dist + math.pow((mi - mj), 2)

            #     euc_distance.append( math.sqrt(euc_dist) )
            # #print euc_distance
            
            # #eucledian_f = np.mean(euc_d)
            


            # eucledian = np.mean(euc_distance)
            # if max_euc < eucledian:
            #     print max_euc, eucledian
            #     max_euc = eucledian
            #     final_room = room


            print max_euc, final_room
        #df_overlap_events[idx, 'Room'] = final_room
        df_overlap_events.ix[idx, 'Room'] = final_room

        print 'index', edge_index,'room', room,'phase',edge_phase,'magnitude', edge_mag,'time', time(edge_time), 'euc_dist', eucledian_distance

    # df_nonoverlap_event = pd.DataFrame({ 'Room' : final_room, 'Appliance' : df_overlap_events['appliance'] ,'Start Time' : df_overlap_events['start_time'], 'End Time' : df_overlap_events['end_time'], 'Edge' : df_overlap_events['edge']
    #             , 'Magnitude' : df_overlap_events['magnitude'], 'Event Time' : df_overlap_events['edge_time'], 'Phase' : df_overlap_events['edge_phase']})
        
    #print df_overlap_events
    print 'length after', len(df_overlap_events)
    return df_overlap_events, overlap_rooms


            #print 'after last and  before last',aft[-1], bef[-1]
            # list of all the MFCCs from the row
            #print 'before', audio_df.ix[bef][audio_df.columns[1:-1]]
            #print 'after', audio_df.ix[aft][audio_df.columns[1:-1]]
            # for mi, mj in zip(MFCC_bef, MFCC_aft):
            #         euc_dist = euc_dist + math.pow((mi - mj), 2)

            #     euc_dist = math.sqrt(euc_dist) 

            # MFCC_bef = audio_df.ix[bef][audio_df.columns[1:-1]].mean()
            # MFCC_aft = audio_df.ix[aft][audio_df.columns[1:-1]].mean()
            # print 'before', MFCC_bef
            # print 'after', MFCC_aft
            

            #print before
            #print after
            
            #print 'index', edge_index,'room', room, 'time', edge_time, 'mfcc1', after[0]
            # print before
            # print room
            # print after
            #df = pd.DataFrame({'room' : room, 'before' : before, 'after' : after})
        #print df.head(5) 
        #sys.exit(1) 

            # sum_of_squares=sum(pow(df['before']-df['after']),2)
            # euc_distance.append(math.sqrt(sum_of_squares))
        #print df
        #df_distance = pd.DataFrame({'distance':euc_distance})
       
        # find maximum of all the distance and allocate the edge to room corresponding to that room
        # max_dist = pd.idxmax(df_distance['distance'])
        # room = o_rooms[max_dist + 1]

        #allocate edge to this room
        #df_overlap_events.ix[idx]['rooms'] = room
        
        #return df_overlap_events


# def sound_classify():


# def edge_matching(df):
#     # Create separate edge list for all the rooms

#     # Generate matching pair of rise and fall edges 
#     return 1
def overlap_event_association(df_overlap_window_edges, df_overlap_df, edge_light_df, edge_power_df, day):
    # Retrieve metadata 
    df_metadata_rooms = pd.DataFrame(metadata_rooms)
    print "-" * stars
    print "            METADATA"
    print "-" * stars
    print df_metadata_rooms
    print df_overlap_window_edges
    #print df_overlap_df
    print 'Overlap power events recognition'
    df_overlap_power_events, df_nonoverlap_power_events,ids_power = overlap_room_events(df_overlap_window_edges, edge_power_df,df_metadata_rooms, 'power')
    print 'Overlap light events recognition'
    df_overlap_light_events, df_nonoverlap_light_events, ids_light = overlap_room_events(df_overlap_window_edges, edge_light_df,df_metadata_rooms, 'light')

    #Power edges
    # UNDEBUGGED AND UNTESTED
    print "-" * stars
    print "            SOUND DETECTION ON OVERLAP ROOMS FOR POWER EDGES"
    print "-" * stars
    df_overlap_power, overlap_power_rooms = events_sound_detection(df_overlap_power_events, ids_power, edge_power_df, df_metadata_rooms, day)

    print "-" * stars
    print "            SOUND DETECTION ON OVERLAP ROOMS FOR LIGHT EDGES"
    print "-" * stars
    df_overlap_light, overlap_light_rooms = events_sound_detection(df_overlap_light_events, ids_light, edge_light_df, df_metadata_rooms, day)
    return df_nonoverlap_power_events, df_nonoverlap_light_events, df_overlap_power, df_overlap_light, overlap_power_rooms, overlap_light_rooms
    # events_sound_detection(df_overlap_light_events, edge_light_df, df_metadata_rooms)
    #events_sound_detection(df_overlap_light_events, edge_light_df, df_metadata_rooms, 'light')
    #return 1

def merge_events_room(df1, df2, df3, df4, df5, df6, overlap_power, overlap_light):

    # Merge all the detected events into one single dataframe
    rooms =  ['1', '2', '3', '4', '5', '6']
    #df = pd.DataFrame(columns = df1.columns.values)
    df_merged = pd.DataFrame()
    
    df = concat([df1, df2])
    df = concat([df, df4])
    df = concat([df, df5])



    df = concat([df, df3])
   
    df = concat([df, df6])
    df.index = arange(0, len(df))
    df_c = df.copy()
    df_c['act_time'] = [datetime.datetime.fromtimestamp(i) for i in df_c['Event Time']]
    df_c.to_csv('/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/detected_events.csv', cols = df.columns.values, index = False)
    #print df

    df = df.ix[df.Appliance != 'F']
    df = df.ix[df.Appliance != 'EL']
    #df = df.ix[df.Room != '3']
    #df = df.ix[df.Room != '4']
    rooms = ['3','4']
    #time = [1392095760, 1392095820,1392096840, 1392096900, 1392096960]
    time_4 = [1392095760, 1392095820,1392096840, 1392096900, 1392096960,1392102900, 1392102960,1392103020]
    time_3 = [1392095760, 1392095820,1392096840, 1392096900, 1392096960,1392126120, 1392126180, 1392127200,1392125100,1392125160]

    # for r in rooms:
    #     for t in time:
    #         df_filter_idx = df.index[df.Room == r]
    #         df_filter = df.ix[df_filter_idx]
    #         df_filter_idx = df_filter.index[df_filter.Edge == 'rise']
    #         df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #         #df_filter = df_filter.ix[df_filter_idx]
    #         df_filter_idx = df_filter.index[df_filter.Phase == 'lightphase2']
    #         #df_filter = df_filter.ix[df_filter_idx]
    #         df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #         df_filter_idx = df_filter.index[df_filter.Phase == 'powerphase3']
    #         #df_filter = df_filter.ix[df_filter_idx]
    #         df_filter = df_filter.ix[df_filter.index - df_filter_idx]
            
    #         df_filter_idx = df_filter.index[((df_filter['Event Time'] >= t) & (df_filter['Event Time'] <= (t + 59)))]
    #         print 'Filtered rooms 3 and 4'
    #         print df_filter_idx
    #         df = df.ix[df.index - df_filter_idx[1:]]


    # for t in time_4:
    #     df_filter_idx = df.index[df.Room == '4']
    #     #print df_filter_idx
    #     df_filter = df.ix[df_filter_idx]
    #     #print df_filter
    #     df_filter_idx = df_filter.index[df_filter.Edge == 'rise']
    #     #print df_filter_idx
    #     df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #     #print df_filter
    #     #df_filter = df_filter.ix[df_filter_idx]
    #     df_filter_idx = df_filter.index[df_filter.Phase == 'lightphase2']
    #     #print df_filter_idx
    #     df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #     df_filter_idx = df_filter.index[df_filter.Phase == 'powerphase3']
    #     #print df_filter_idx
    #     df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #     #df_filter = df_filter.ix[df_filter_idx]
       
    #     df_filter_idx = df_filter.index[((df_filter['Event Time'] >= t) & (df_filter['Event Time'] <= (t + 59)))]
    #     print 'time intervaal'
    #     print df_filter_idx
    #     print 'Filtered room 4'
    #     #print df_filter_idx[0]
    #     print df_filter_idx
    #     df_filter_idx = df_filter_idx - df_filter_idx[:1]
    #     df = df.ix[df.index - df_filter_idx]

    # for t in time_3:
    #     df_filter_idx = df.index[df.Room == '3']
    #     df_filter = df.ix[df_filter_idx]
    #     df_filter_idx = df_filter.index[df_filter.Edge == 'rise']
    #     df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #     #df_filter = df_filter.ix[df_filter_idx]
    #     df_filter_idx = df_filter.index[df_filter.Phase == 'lightphase2']
    #     df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #     df_filter_idx = df_filter.index[df_filter.Phase == 'powerphase3']
    #     df_filter = df_filter.ix[df_filter.index - df_filter_idx]
    #     #df_filter = df_filter.ix[df_filter_idx]
        
    #     df_filter_idx = df_filter.index[((df_filter['Event Time'] >= t) & (df_filter['Event Time'] <= (t + 59)))]
    #     print 'time intervaal'
    #     print df_filter_idx
    #     print 'Filtered room 3'

    #     #print df_filter_idx[0]
    #     df_filter_idx = df_filter_idx - df_filter_idx[:1]
    #     print df_filter_idx
    #     df = df.ix[df.index - df_filter_idx]

    print df






        




    #df = df.ix[df.Appliance != 'LP' & (df.Room != '5' or df.Room != '6')]
    # for room in rooms :
    #     #print room
    #     df_rise  = pd.DataFrame()
    #     df_fall = pd.DataFrame()
    #     roomwise_idx_list = df.index[df.Room == room]
    #     df_list = df.ix[roomwise_idx_list]
    #     #print df_list

    #     df_rise_idx = df_list.index[df_list.Edge == 'rise']
    #     if len(df_list.index[df_list.Edge == 'rise']) != 0:
    #         df_rise = df_list.ix[df_rise_idx]
    #         df_rise.index = arange(0, len(df_rise))
    #     #print df_rise

    #     df_fall_idx = df_list.index[df_list.Edge == 'fall']
    #     #print df_fall_idx
    #     if len(df_list.index[df_list.Edge == 'fall']) != 0:
    #         #print 'true'
    #         df_fall = df_list.ix[df_fall_idx]
    #         #print df_fall
    #         df_fall.index = arange(0, len(df_fall))

    #     #print df_fall

    #     if len(df_rise) != 0:
    #         print df_rise
    #         for idx in df_rise.index:
    #             curr = df_rise.ix[idx]['Event Time']

    #             if idx != df_rise.index[-1]:
                    
    #                 next = df_rise.ix[idx + 1]['Event Time']
    #             else:
    #                 break
    #             print next, curr
    #             if int(math.fabs(int(next) - int(curr)))  <= 60:
    #                 print int(math.fabs(int(next) - int(curr)))
    #                 if len(df_fall) > 0:
    #                     print 'df_fall greater than 0'
    #                     for fall_idx in df_fall.index:
    #                         if int(df_fall.ix[fall_idx]['Event Time']) in range(int(curr) , int(next)):
    #                             break
    #                         else:
    #                             if 'light' in df_rise.ix[idx]['Phase']:
    #                                 print 'lightphase exists'
    #                                 overlapp_rooms_idx = overlap_light.index[overlap_light.ix['Event Time'] == df_rise.ix[idx]['Event Time']]
    #                                 print overlapp_rooms_idx
    #                                 overlap_rooms = overlap_light.ix[overlapp_rooms_idx]['Room']
    #                                 print overlapp_rooms
    #                                 replace_with = df_rise.ix[idx]['Room'] not in overlap_rooms
    #                                 print 'replace with', replace_with

    #                             if 'power' in df_rise.ix[idx]['Phase']:
    #                                 print 'powerphase exists'
    #                                 overlapp_rooms_idx = overlap_power.index[overlap_power.ix['Event Time'] == df_rise.ix[idx]['Event Time']]
    #                                 overlap_rooms = overlap_power.ix[overlapp_rooms_idx]['Room']
    #                                 replace_with = df_rise.ix[idx]['Room'] not in overlap_rooms
    #                 else:
    #                     print 'df_fall 0'
    #                     if 'light' in df_rise.ix[idx]['Phase']:
    #                         print 'lighphase exists'
    #                         overlapp_rooms_idx = overlap_light.index[overlap_light['Event Time'] == df_rise.ix[idx]['Event Time']]
    #                         overlap_rooms = overlap_light.ix[overlapp_rooms_idx]['Room']
    #                         replace_with = df_rise.ix[idx]['Room'] not in overlap_rooms

    #                     if 'power' in df_rise.ix[idx]['Phase']:
    #                         print 'powerphase exists'
    #                         overlapp_rooms_idx = overlap_power.index[overlap_power['Event Time'] == df_rise.ix[idx]['Event Time']]
    #                         overlap_rooms = overlap_power.ix[overlapp_rooms_idx]['Room']
    #                         replace_with = df_rise.ix[idx]['Room'] not in overlap_rooms
    #                 print replace_with

    #                 df_rise.ix[idx, 'Room'] = replace_with
    #     df_merged = concat([df_rise, df_fall])



    #print df_merged
    return df

# def edge_matching(df):
#     print df
#     rooms = df.Room.unique()
#     rise_time = []
#     fall_time = []
#     app  = []
#     rom = []
#     phase = []
#     for room in rooms:

#         df_room = df[df.Room == room]

#         df_room_rise = df_room[df.Edge == 'rise']

#         df_room_rise = df_room_rise.sort_index(by = 'Event Time')
#         df_room_fall = df_room[df.Edge == 'fall']
#         df_room_fall = df_room_fall.sort_index(by = 'Event Time')
#         length = min(len(df_room_rise), len(df_room_fall))
#         print room, length
#         print df_room_rise
#         print df_room_fall
#         for i in (range(0, length))  :
#             for id_r in df_room_rise.index:
#                 for id_f in df_room_fall.index:
#                     if (df_room_fall.ix[id_f]['Appliance'] == df_room_rise.ix[id_r]['Appliance']): 
#                         if (df_room_rise.ix[id_r]['Event Time'] <= df_room_fall.ix[id_f]['Event Time']):
#                             rise_time.appen(df_room_rise.id_r[id_r]['Event Time'])
#                             fall_time.append(df_room_fall.id_f[id_f]['Event Time'])
#                             app.append(df_room_fall.ix[id_r]['Appliance'])
#                             rom.append(df_room_fall.ix[id_r]['Room']) 
#                             phase .append(df_room_fall.ix[id_r]['Phase'])
#     df = pd.DataFrame({'Start time': rise_time, 'End time': fall_time, 'Appliance' : app, 'Room' : rom, 'Phase': phase}) 
#     print df




            

"""
---------------------------------------
Main Program 
---------------------------------------
"""

if __name__ == '__main__':

    # Get the sensor streams (test set)
    
    event_dir = sys.argv[1]
    day = sys.argv[2]
    power_csv = event_dir + 'Power.csv'
    light_csv = event_dir + 'Light.csv'
    
    # room_no = sys.argv[2]
    rooms = ['1','2','3','4','5','6']
    # test_csv_path = (WIFI_TRAINING_DATA_PATH)
   
    # print rooms
    
    logger.info("Starting Algorithm...")
    
    df_p = pd.read_csv(power_csv)
    df_l = pd.read_csv(light_csv)
    # df_p = df_p.ix[df_p['time'] >= 1392042600]
    # df_l = df_p.ix[df_l['time'] >= 1392042600]

    df_p_copy = df_p.copy()
    

    # Step 1: Smoothening power phase stream (Y)
    # Filtering periodic drops
    
    #Power phase2 plot before smoothening
    # print 'Dropping unwanted edges'

    
    # df_p = filter_drops(df_p, "powerphase2")
    # df_p = filter_drops(df_p, "powerphase2")
    # df_p = filter_drops(df_p, "powerphase2")
    #########################################
    df_p = smoothening(df_p, WINDOW)
    ###########################################
 
    #power2 = movingaverage(df_p_copy['powerphase2'], 8)


    # print 'PLOT'
    # fig = plt.figure()
    # plt.gcf()
    # ax1 = plt.subplot(2, 1, 1)
    # t_before = np.array([dt.datetime.fromtimestamp(x) for (x) in df_p_copy['time']])
    # plt.plot(t_before, df_p_copy['powerphase2'] )
    # ax2 = plt.subplot(2, 1, 2)
    # t_after = np.array([dt.datetime.fromtimestamp(x) for (x) in df_p['time']])
    # plt.plot(t_after, df_p['powerphase2'])
    # show()
    # sys.exit(1)
    

    #df_p.ix[curr_idx, "powerphase2"] = mean of prev and next curr power

    #fig = plt.figure()
    #plt.gcf()
    #print 'After plots'
    #plot( df_p['powerphase2'])

    #####################################
    #Step2: Edge Detection
    print 'Edge detection activity'
    edge_light_df, edge_power_df = edge_detection(df_l, df_p)

    # edge_light_df.to_csv('/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/MeterData/edge_light.csv', cols = edge_light_df.columns.values, index = False)
    # edge_power_df.to_csv('/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/MeterData/edge_power.csv', cols = edge_power_df.columns.values, index = False)
    

    #######################################

    # sys.exit()

    # Step 3 : Wifi Localization to determine each users location for the given dataset as in/out of hostel room
    # Classify
    # rooms = ['C002', 'C006']
    # for i in rooms:
    #     formatted_csv_path = (WIFI_FORMATTED_DATA_PATH + i + '.csv' )
    #     formatted_csv_path = format_data(test_csv_path, 'train', i)
    #     print 'Created wifi formatted csv', (formatted_csv_path + i + '.csv')
    #     wifi_location(formatted_csv_path, i)
    
    #print 'Filling in missing samples'
    # print 'Filling missing samples in WIFI csv for each user '
    
    #fill_missing_samples(DATA_PATH + str(day) + WIFI_INOUT, day)
    
    # Step 4 : Determine Overlapping/Non-Overlapping users location
    

    #print len(edge_power_df.index), len(edge_light_df.index)
    ##############################
    USER_ATTRIBUTION_TABLES = DATA_PATH + str(day) + USER_ATTRIBUTION_TABLES_PATH
  
    # users_location_table(USER_ATTRIBUTION_TABLES, str(day))
    # get_rooms_corresponding_loc_set(USER_ATTRIBUTION_TABLES  , 'user_location_table.csv')
    
    # df_set = overlapping_non_overlapping_sets(USER_ATTRIBUTION_TABLES , 'revised_user_location_table.csv')
    # distinct(USER_ATTRIBUTION_TABLES )
    ##########################


    #print df_set


    # Step 5 : User Attribution

    #separate_overlapping_nonOverlapping_sets(USER_ATTRIBUTION_TABLES , 'location_room_set.csv')
    
    
    #####################

    
    print 'nonoverlap edge association activity'

    # Associate edges with sets of rooms depending on the time it occurred for power edges
    df_nonoverlap_window_edges, df_nonoverlap_df = roomset_edge_association(USER_ATTRIBUTION_TABLES, 'reduced_location_room_set.csv', edge_light_df, edge_power_df, 'nonoverlap')
    
    print 'overlap edge association activity'
     # Associate edges with sets of rooms depending on the time it occurred for light edges
    df_overlap_window_edges, df_overlap_df = roomset_edge_association(USER_ATTRIBUTION_TABLES, 'reduced_location_room_set.csv',edge_light_df, edge_power_df, 'overlap')
    
    # Finds all the non overlapping indices(romm sets) from the above result of power and light edges
    df_power, df_light = nonoverlap_event_association(df_nonoverlap_window_edges, df_nonoverlap_df, edge_light_df, edge_power_df )

    # Find all the overlapping indics(room sets) from the above result for power and light edges and call Sound detection to resolve overlap
    df_power1, df_light1, df_power2, df_light2, overlap_power, overlap_light = overlap_event_association(df_overlap_window_edges, df_overlap_df, edge_light_df, edge_power_df , day)
    
    # Merging the power and light edges from overlapping sets
    df = merge_events_room(df_power, df_power1, df_power2, df_light, df_light1,  df_light2, overlap_power, overlap_light)
    

    gt_path = GROUNDTRUTH_PATH + 'ground_truth_' + str(day).lower() + '.csv'

    df_nonoverlap = concat([df_power, df_light])
    df_nonoverlap = concat([df_nonoverlap, df_power1])
    df_nonoverlap = concat([df_nonoverlap, df_light1])
    df_nonoverlap  = df_nonoverlap[df_nonoverlap.Appliance != 'EL']
    df_nonoverlap  = df_nonoverlap[df_nonoverlap.Appliance != 'F']
    # Non overlap detected edges accuracy in terms of precision
    rooms  = pd.read_csv(gt_path).Room.unique()
    
    for room in rooms:
        df_test_n = df_nonoverlap.copy()
        print "-" * stars
        print "            ROOM" + str(room)
        print "-" * stars
        gt = pd.read_csv(GROUNDTRUTH_PATH + '/' + str(day).lower() + '/' + 'C00' + str(room) + '.csv')
        df_test_n = df_test_n[df_test_n.Room == str(room)]
        print 'Ground Truth', gt
        print 'Test data', df_test_n.drop_duplicates()
 
        calc_precision(df_test_n, gt)


    # df_rise = df[df.Edge == 'rise']
    # df_fall = df[df.Edge == 'fall']
    # for room in rooms:
    #     df_r = df[df.Room == room]

    #     # for each phase get the time slices
    #     phases_r = np.unique(list(df_r.Phase))
    #     print phases_r
    #     for phase in phases_r:
    #         print room, phase
    #         df_p = df_r[(df_r.Phase == phase)]

    #         df_rise = df_p[df_p.Edge == 'rise']
    #         df_fall = df_p[df_p.Edge == 'fall']
    #         print df_rise
    #         print df_fall
    #         af.edge_matching(df_rise, df_fall)

    
   
    #################
   
    #df = edge_matching(df)
    
    # Accuracy Calculation in terms of Precision/Recall

    ###########################
    # Total accuracy for the full day length
    # print 'Detected events and rooms '
    # print df
    # print "-" * stars
    # print "            TOTAL ACCURACY"
    # print "-" * stars
    # df_test = df.copy()    
    # calc_ts_accuracy(df_test, pd.read_csv(gt_path))

    # # accuracy individual rooms for day 

    # rooms  = pd.read_csv(gt_path).Room.unique()
    
    # for room in rooms:
    #     df_test = df.copy()
    #     print "-" * stars
    #     print "            ROOM" + str(room)
    #     print "-" * stars
    #     gt = pd.read_csv(GROUNDTRUTH_PATH + '/' + str(day).lower() + '/' + 'C00' + str(room) + '.csv')
    #     df_test = df_test[df_test.Room == str(room)]
    #     print 'Ground Truth', gt
    #     print 'Test data', df_test.drop_duplicates()
 
    #     calc_room_accuracy(df_test, gt)


    ################################


    

    
    #nonoverlap_event_association()
    # Step2: PreProcessing
    #edge_list_df = edge_preprocessing(df_l, df_p, edge_list_df)

    #time_slices = af.edge_matching(df_l, df_p, edge_list_df)

   

    #     i=0
    #     fig, axes = plt.subplots(nrows=2, ncols=4)
    #     for db in room:

    #         print 'opening db file'+db+db_rooms
    #         list_db=pd.read_csv(db_paths+db+db_rooms)
    # print 'Filtering sound samples before event'
    #         db_list=list_db.index[(list_db['time']>= from_t-60000) & (list_db['time']<= from_t)]
    #         db_list_before=list_db.ix[db_list]
    # print "before data"
    # print db_list_before
    # print 'Filtering sound samples between events'
    #         db_list=list_db.index[(list_db['time']>=from_t) & (list_db['time'] <= end_t)]
    #         db_list_between=list_db.ix[db_list]
    # print 'Filtering sound events after event'
    #         db_list=list_db.index[(list_db['time']>=from_t) & (list_db['time'] <= from_t+60000)]
    #         db_list_after=list_db.ix[db_list]
    # print 'after data'
    # print db_list_after
    #         df_before=DataFrame(db_list_before['db'],columns=['Before_'+db+'_'+str(time_slices.ix[t]['start_time'])])
    #         df_after=DataFrame(db_list_after['db'],columns=['After_'+db+'_'+str(time_slices.ix[t]['end_time'])])
    # df_between=DataFrame(db_list_between['db'],columns=['Between_'+db+'_'+str(time_slices.ix[t]['start_time'])+'_'+str(time_slices.ix[t]['end_time'])])
    # t_before = np.array([dt.datetime.fromtimestamp(x/1000) for (x) in db_list_before['time']])
    # t_after = np.array([dt.datetime.fromtimestamp(x/1000) for (x)in db_list_after['time']])
    #         state=''
    #         if df_before.empty == False:
    #             state+='T'
    # df_b=pd.concat([df_b,df_before],ignore_index=True)
    #         else:
    #             state+='F'
    #         if df_after.empty == False:
    #             state+='T'
    # df_b=pd.concat([df_b,df_after],ignore_index=True)
    #         else:
    #             state+='F'
    # print state
    #         if (db_list_between.empty == False) :
    #             print 'True'
    #             sd=[]
    #             beg=0
    # next=db_list_between['time']
    #             print 'Beginning time'
    #             print long(beg)
    #             sd_prev=0.0
    #             sd_curr=0.0
    #             diff_t=0
    #             print len(db_list_between)
    #             for db in db_list_between.index[:-1]:
    #                 next=db_list_between.ix[db]['time']
    #                 diff_t=long(next)-long(beg)
    # print diff_t
    #                 sd.append(db_list_between.ix[db]['db'])
    #                 if diff_t<120000:
    #                     continue
    #                 beg=next
    #                 sd_prev=sd_curr
    # print "previous sd:"+str(sd_prev)
    #                 frame=DataFrame(sd,columns=['db_sd'])
    # print frame
    #                 sd_curr=float(frame.std())
    # print "current sd:"+str(sd_curr)
    #                 diff=abs(sd_curr-sd_prev)
    #                 if (diff>4):
    #                     print 'Time found'
    #                     print long(db_list_between.ix[db]['time'])
    #                 prev=next
                # print "previous time:"+str(prev)
            # mean_bf_db=float(df_before.mean())
            # sd_bf_db=float(df_before.std())
            # mean_bt_db=float(df_between.mean())
            # sd_bt_db=float(df_between.std())
            # mean_af_db=float(df_after.mean())
            # sd_af_db=float(df_after.std())
            # print "start_time:"+str(dt.datetime.fromtimestamp(time_slices.ix[t]['start_time']))
            # print "end time:"+str(dt.datetime.fromtimestamp(time_slices.ix[t]['end_time']))
            # print 'Before_'+'Between_'+'After'
            # print mean_bf_db,mean_bt_db,mean_af_db
            # print sd_bf_db,sd_bt_db,sd_af_db
            # fig = plt.gcf()
            # if state=='TT':
            # row1
            #     ax1 = plt.subplot(3, 1, 1)
            # ax1.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
            #     plt.plot( df_before)
            #     plt.title("Sound Before_ " +str(time_slices.ix[t]['start_time'])+'_'+db )
            #     plt.ylabel("Sound in db")
            #     ax2 = plt.subplot(3, 1, 2)
            # ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
            #     plt.plot( df_between)
            #     plt.title("Sound Between_ " + str(time_slices.ix[t]['start_time'])+'_'+str(time_slices.ix[t]['end_time'])+'_'+db)
            #     plt.ylabel("Sound in db")
            #     ax3 = plt.subplot(3, 1, 3)
            # ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
            #     plt.plot( df_after)
            #     plt.title("Sound After_ " + str(time_slices.ix[t]['end_time'])+'_'+db)
            #     plt.ylabel("Sound in db")
            # elif (state == 'TF') | (state == 'FT'):
            #     if state=='TF':
            #         plt.plot( df_before)
            #         plt.title("Sound Before_ " +str(time_slices.ix[t]['start_time'])+'_'+db )
            #         plt.ylabel("Sound in db")
            #     if state=='FT':
            #         plt.plot(df_after)
            #         plt.title("Sound After_ " +str(time_slices.ix[t]['end_time'])+'_'+db )
            #         plt.ylabel("Sound in db")
            # pp = PdfPages('/media/New Volume/IPSN/sense/5_6Nov/dbs/plots/Y_Y/'+db+'_'+str(time_slices.ix[t]['start_time'])+'.pdf')
            # plt.savefig(pp, format='pdf')
            # pp.savefig()
            # pp.close()
            # plt.show()
            #fig.savefig('/media/New Volume/IPSN/sense/5_6Nov/plots/'+db+'_'+str(time_slices.ix[t]['start_time']),format='png',dpi=150)
            # fig.clear()
            # print 'Plotting'
            # try:
            #     db_list_before['db'].plot(ax=axes[0,i]); axes[0,i].set_title('Before_'+db+'_'+str(t))
            #     db_list_after['db'].plot(ax=axes[0,i]); axes[1,i].set_title('After_'+db+'_'+str(t))
            # except Exception,e:
            #     print e
            # i+=1
            # fig = plt.figure()
            # plt.subplots_adjust(bottom=0.2)
            # plt.xticks( rotation=25 )
            # ax1=plt.gca()
            # ax1=fig.add_subplot(1,1,1)
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # ax1.xaxis.set_major_formatter(xfmt)
            # act_time = [dt.datetime.fromtimestamp(float(i)/1000) for i in db_list_before['time']]
            # dataframe_before.plot(ax=ax1,style='k-',label='before event')
            # plt.plot(act_time,db_list_before['db'],label='before event')
            # ax3=fig.add_subplot(2,1,2)
            # dataframe_after.plot(ax=ax3,style='k-',label='after event')
            # act_time2 = [dt.datetime.fromtimestamp(float(i)/1000) for i in db_list_after['time']]
            # plt.plot(act_time2,db_list_after['db'],label='after event')
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # ax1.xaxis.set_major_formatter(xfmt)
            # ax2=fig.add_subplot(1,2,1)
            # dataframe_between.plot(ax=ax2,style='k-',label='during event')
            # act_time1 = [dt.datetime.fromtimestamp(float(i)/1000) for i in db_list_between['time']]
            # plt.plot(act_time1,db_list_between['db'],label='between event')
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # ax1.xaxis.set_major_formatter(xfmt)
            # plt.show()
            # print 'Saving figure at'+db_paths+'_'+str(t)
            # plt.savefig(db_paths+'_'+str(t)+'.png')
        # print 'writing to csv'
        # print df_b
        #df_b.to_csv('/media/New Volume/IPSN/sense/5_6Nov/dbs/'+str(time_slices.ix[t]['start_time'])+'.csv')
    # print df_b
    # print df_a
    #df_a.to_csv('/media/New Volume/IPSN/sense/5_6Nov/dbs/db_after_filtered.csv')
 #   for i in time_slices.start_time :
    # f_time=i
    # t_time=i-60000
    # time=range(f_time,t_time)
 #      sound_before=Series[x.values[2] for x.time in time]
    """
    # Step2: Localization
    time_slices_location = determine_location(time_slices, wifi_csv, phno)
    # sys.exit()

    # Step4: Audio Classification
    detected_activity = activity_detection(time_slices_location, sound_csv, phno)

    if app >= 3:

        if app == 3:
            detected_activity = correction_using_appl_power(detected_activity)
        elif app == 4:
            detected_activity = correction_using_appl_phase_power(detected_activity)

        # For printing detected activity with actual times
        print_activity = detected_activity.copy()

        print_activity['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                            for i in detected_activity['start_time']]
        print_activity['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                          for i in detected_activity['end_time']]
        print_activity = print_activity.sort(['start_time'])
        print "\nCorrected Detected activity and location for Phone", phno, "::\n", print_activity

    # Last Step of combining the light and power edges
    # detected_activity = combine_light_power_activity(detected_activity)

    # Store the detected activity as a csv file
    opfilename = event_dir + 'output_app' + str(app) + '_' + phno + '.csv'
    detected_activity.to_csv(opfilename, index=False)

    print "\nAccuracy Results for Phone", phno, "::\n"
    os.system("python evaluate.py " + opfilename)
    """
    logger.info("Algorithm Run Finished!")
