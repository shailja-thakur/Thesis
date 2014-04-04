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
# import classifier as cl
# import classify_sound as cs
import warnings
import logging
# from path import *
from localize import *
# Disable warnings
warnings.filterwarnings('ignore')

# Enable Logging
logger = logging.getLogger('hostel-activity-finder')
logging.basicConfig(level=logging.DEBUG)


TIMEZONE = 'Asia/Kolkata'
stars = 30

# --------------------------------------------------------------
# ActivityFinder Algorithm starts
# --------------------------------------------------------------

# Edge Transition Window (in seconds) for the change to take place
# its more for simultaneous or quick sequential activity
lwinmin = 3
pwinmin = 6
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


def filter_drops(df_p, phase):
    for i in df_p.index:
        if (i + 2) in df_p.index:

            prev = df_p.ix[i][phase]
            curr = df_p.ix[i + 1][phase]
            next = df_p.ix[i + 2][phase]

            # If edge , get the magnitude of rising and falling edges
            curr_prev_diff = int(math.fabs(curr)) - int(math.fabs(prev))
            curr_next_diff = int(math.fabs(curr)) - int(math.fabs(next))
            if ((prev - curr) > 0) & ((curr - next) < 0):
                    if int(math.fabs(curr_prev_diff - curr_next_diff)) <= 18:
                        if (i - 2) in df_p.index:
                            if (i + 3) in df_p.index:

                                pwin = df_p.ix[i - 2][phase]
                                nwin = df_p.ix[i + 3][phase]
                                df_p.ix[i + 1][phase] = (pwin + nwin) / 2
                        else:
                            df_p.ix[i + 1][phase] = (prev + curr) / 2
                        # print "idx", i, "time", df_p.ix[i + 1]['time'],'prev',
                        # df_p.ix[i][phase],'curr', df_p.ix[i + 1][phase], 'next', df_p.ix[i +
                        # 2][phase]

    # avg = 0
    # idx_list = []
    # for i in df_p.index:
    #     prev=df_p.ix[i][phase]
    #     curr = df_p.ix[i + 1][phase]
    #     next = df_p.ix[i + 2][phase]
    # If edge , get the magnitude of rising and falling edges
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

    # If checking for a particular phase, increase by 10 watts
    if power_stream != "lightpower":
        if math.floor(prev) != 0:
            prev = prev + 10
        if math.floor(curr) != 0:
            curr = curr + 10
        if math.floor(next) != 0:
            next = next + 10
        if math.floor(currwin) != 0:
            currwin = currwin + 10

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

    # print "Looking for light edge in stream:", power_stream, "for Index:", i
    # print "Magnitude::", curr_nextwin_diff

    # if(time in [1385466741, 1385467127, 1385485791, 1385486655]):
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
        # Only checking these conditions for cumulative power
        if power_stream == "lightpower":
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
                pass
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

    # If checking for a particular phase, increase by 10 watts
    if power_stream != "power":
        if math.floor(prev) != 0:
            prev = prev + 10
        if math.floor(curr) != 0:
            curr = curr + 10
        if math.floor(next) != 0:
            next = next + 10
        if math.floor(currwin) != 0:
            currwin = currwin + 10

    time = df_p.ix[i]['time']

    if i - pwinmin not in (df_p.index):
        prevwin = 0
    else:
        prevwin = int(df_p.ix[i - pwinmin][power_stream])
        if power_stream != "power":
            if math.floor(prevwin) != 0:
                prevwin = prevwin + 10
        # prevwintime = df_p.ix[i - pwinmin]['time']
    time = df_p.ix[i]['time']
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
    # if time in [1385480507]:
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

    l_phase = ['lightphase1', 'lightphase2', 'lightphase3']
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
        r_row_list, columns=['index', 'time', 'magnitude', 'curr_power', 'lightphase'])
    rising_edges_l_df = rising_edges_l_df.set_index('index', drop=True)

    falling_edges_l_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude', 'curr_power', 'lightphase'])
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
    p_phase = ['powerphase1', 'powerphase2', 'powerphase3']
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
        r_row_list, columns=['index', 'time', 'magnitude', 'curr_power', 'powerphase'])
    rising_edges_p_df = rising_edges_p_df.set_index('index', drop=True)

    falling_edges_p_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude', 'curr_power', 'powerphase'])
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
    print "Rising Edges::\n", print_pr_df
    print "Falling Edges::\n", print_pf_df

    return rising_edges_p_df, falling_edges_p_df


def filter_hostel_edges(rise_df, fall_df, meter, df):

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

        idx_list = []
        for i in rise_df.index:
            magnitude = int(math.fabs(rise_df.ix[i]['magnitude']))
            #time = rise_df.ix[i]['time']
            if rise_df.ix[i]['powerphase'] == 'powerphase3':
                 # Likely power consumption of Fridge is 88-105
                if magnitude >= 88 and magnitude <= 105:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
        rise_df = rise_df.ix[rise_df.index - idx_list]

        # Falling edges
        # Filter edges of Fridge

        idx_list = []
        for i in fall_df.index:
            magnitude = int(math.fabs(fall_df.ix[i]['magnitude']))
            #time = rise_df.ix[i]['time']
            if fall_df.ix[i]['powerphase'] == 'powerphase3':
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
            if rise_df.ix[i]['powerphase'] == 'powerphase2':
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
            if fall_df.ix[i]['powerphase'] == 'powerphase2':
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

        # Filter 3
        # Filter unwanted periodic edges on Power meter, Phase2
        # Rising Edges
        # print 'Filtering noise'
        # print 'Rising Edges'
        # idx_list=[]
        # for i in rise_df.index:
        #     if rise_df.ix[i]['powerphase'] == 'powerphase2':
        #         magnitude = rise_df.ix[i]['magnitude']
        #         now_time = rise_df.ix[i]['time']
        # print 'magnitude:'+str(magnitude)
        # print "time:"+str(dt.datetime.fromtimestamp(now_time))
        #         if int(math.fabs(magnitude)) in range(20, 35):
        #             row_df = df[df.time == now_time]
        # print 'next index:'+str((row_df.index[0]) + 1)
        #             prev = df_p.ix[(row_df.index[0]) - 1]['powerphase2']
        #             curr = df_p.ix[(row_df.index[0])]['powerphase2']
        #             next = df_p.ix[(row_df.index[0]) + 1]['powerphase2']
        # print 'prev', prev,'curr', curr,'next', next
        #             curr_prev_diff = int(math.fabs(curr)) - int(math.fabs(prev))
        #             curr_next_diff = int(math.fabs(curr)) - int(math.fabs(next))
        #             pwin = df_p.ix[(row_df.index[0]) - 3]['powerphase2']
        #             nwin = df_p.ix[(row_df.index[0]) + 3]['powerphase2']
        #             if (int(math.fabs(pwin - nwin)) <= 10) & (int(math.fabs(curr_prev_diff - curr_next_diff)) <= 10):
        # print 'Diff less than equal to 5'

        #                 print "idx", i, "magnitude", magnitude
        #                 idx_list.append(i)

        # next_edge_type, result = check_if_power_edge(df, int(row_df.index[0]) + 1, 'powerphase2')
        # print "next_edge_type"+str(next_edge_type)
        # if (next_edge_type == "Not an edge"):
        # for prev_idx in (int(row_df.index[0]) - 4, int(row_df.index[0]) - 3, int(row_df.index[0]) - 2, int(row_df.index[0]) - 1):
        # print 'previous index :'+str(prev_idx)
        # prev_edge_type, result = check_if_power_edge(df, prev_idx, 'powerphase2')
        # print 'prev_edge_type:'+str(prev_edge_type)
        # if (prev_edge_type == "falling") :
        # print 'prev_edge_type:'+str(prev_edge_type)
        # print "curr_pev magnitude"+str(result["magnitude"])
        # if int(math.fabs(magnitude)) in range(int(math.fabs(curr_prev)):
        # curr_next_next = rise_df.ix[i+2]['magnitude']
        # if curr_next in range(curr_next_next - 1, curr_next_next + 1):
        # print result
        # print "idx", i, "magnitude", magnitude
        # idx_list.append(i)
        # break
        # print idx_list
        # rise_df = rise_df.ix[rise_df.index - idx_list]

        # Falling Edges
        # print 'Falling Edges'
        # idx_list=[]
        # for i in fall_df.index:
        #     if fall_df.ix[i]['powerphase'] == 'powerphase2':
        #         magnitude = fall_df.ix[i]['magnitude']
        #         now_time = fall_df.ix[i]['time']
        # print 'magnitude:'+str(magnitude)
        # print "time:"+str(dt.datetime.fromtimestamp(now_time))
        #         if int(math.fabs(magnitude)) in range(20, 35):
        #             row_df = df[df.time == now_time]
        # print 'next index:'+str((row_df.index[0]) + 1)
        #             curr = df_p.ix[(row_df.index[0])]['powerphase2']
        #             curr_next = df_p.ix[(row_df.index[0]) + 1]['powerphase2']
        #             curr_next_next = df_p.ix[(row_df.index[0]) + 2]['powerphase2']
        # print 'curr', curr,'next', curr_next,'next_next', curr_next_next
        #             curr_curr_next_diff = int(math.fabs(curr)) - int(math.fabs(curr_next))
        #             curr_next_curr_next_next_diff = int(math.fabs(curr_next)) - int(math.fabs(curr_next_next))
        #             pwin = df_p.ix[(row_df.index[0]) - 2]['powerphase2']
        #             nwin = df_p.ix[(row_df.index[0]) + 4]['powerphase2']
        #             print 'Window diff', int(math.fabs(pwin - nwin)),'Edge difference',int(math.fabs(curr_curr_next_diff - curr_next_curr_next_next_diff))
        # if (int(math.fabs(pwin - nwin)) <= 10) &
        # (int(math.fabs(curr_curr_next_diff - curr_next_curr_next_next_diff)) <=
        # 10):

        #                 print "idx", i, "magnitude", magnitude
        #                 idx_list.append(i)
        # fall_df = fall_df.ix[fall_df.index - idx_list]

    print "-" * stars
    print "Filtered Hostel Edges:"
    print "-" * stars
    print "Rising Hostel Edges::\n", rise_df
    print "Falling Hostel Edges::\n", fall_df
    return rise_df, fall_df


def make_edge_list(rise_l_df, fall_l_df, rise_p_df, fall_p_df):

    # Generating light edge list
    rise_l = ['rise' for i in rise_l_df.index]
    df_l_r = pd.DataFrame({'time': rise_l_df['time'], 'magnitude': rise_l_df[
                          'magnitude'], 'phase': rise_l_df['lightphase'], 'edge': rise_l})
    fall_l = ['fall' for i in fall_l_df.index]
    df_l_f = pd.DataFrame({'time': fall_l_df['time'], 'magnitude': fall_l_df[
                          'magnitude'], 'phase': fall_l_df['lightphase'], 'edge': fall_l})
    edge_list_l_df = concat([df_l_r, df_l_f])

    # Generating power edge list
    rise_p = ['rise' for i in rise_p_df.index]
    df_p_r = pd.DataFrame({'time': rise_p_df['time'], 'magnitude': rise_p_df[
                          'magnitude'], 'phase': rise_p_df['powerphase'], 'edge': rise_p})
    fall_p = ['fall' for i in fall_p_df.index]
    df_p_f = pd.DataFrame({'time': fall_p_df['time'], 'magnitude': fall_p_df[
                          'magnitude'], 'phase': fall_p_df['powerphase'], 'edge': fall_p})
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


    # return light_edges, power_edges

"""
******Meta-data for hostel******

"""

metadata = {
    "room": ['room1', 'room2', 'room3', 'room4', 'room5', 'room6'],
    "TubeLight_P": ['R', 'R', 'Y', 'Y', 'B', 'B'],
    "LeftPlug_P": ['Y', 'Y', 'B', 'B', 'Y', 'Y'],
    "AC_P": ['Y', 'Y', 'B', 'B', 'Y', 'Y'],
    "RightPlug_L": ['R', 'R', 'Y', 'Y', 'B', 'B'],
    "ELight_L": ['R', 'R', 'Y', 'Y', 'B', 'B']
}

metadata_room_phase = {
    "R": ['room1', 'room2'],
    "Y": ['room1', 'room2', 'room3', 'room4', 'room5', 'room6'],
    "B": ['room3', 'room4', 'room5', 'room6']
}

metada_light_plug_p = {
    "set1(R,Y)": ['room1', 'room2'],
    "set2(Y,B)": ['room3', 'room4', 'room5', 'room6']
}

metadata_AC_p = {
    "set1(Y)": ['room1', 'room2', 'room5', 'room6'],
    "set2(B)": ['room3', 'room4']
}

metadata_elight_plug_l = {
    "set1(R)": ['room1', 'room2'],
    "set1(Y)": ['room3', 'room4'],
    "set2(B)": ['room5', 'room6']
}


def max_min_timestamp(loc_csv):
    # Determine all the overlapping and non_overlapping location of the users
    #    in windows

    # Step1:Find Minimum Timestamp
    min_timestamp = 2000000000000
    max_timestamp = 0
    loc_dir = loc_csv + '*.csv'

    for user_loc in glob.glob(loc_dir):
        df_loc = pd.read_csv(user_loc)

        for i in df_loc.index:
            if int(df_loc.ix[df_loc.index[0]]['timestamp']) < int(min_timestamp):
                min_timestamp = df_loc.ix[df_loc.index[0]]['timestamp']
            if int(df_loc.ix[df_loc.index[-1]]['timestamp']) > int(max_timestamp):
                max_timestamp = df_loc.ix[df_loc.index[-1]]['timestamp']

    print 'Minimum Timestamp', min_timestamp
    print 'Maximum Timestamp', max_timestamp

    return min_timestamp, max_timestamp


def fill_missing_samples(loc_csv):

    min_timestamp, max_timestamp = max_min_timestamp(USER_LOCATION_PATH)
    loc_dir = loc_csv + '*.csv'

    for user_loc in glob.glob(loc_dir):
        print user_loc
        df_loc = pd.read_csv(user_loc)

        filename = user_loc.split('.')[0].split('/')[10] + 'fomatted' + '.csv'
        print filename
        outcsv = USER_LOCATION_FORMATTED_PATH + filename
        writer = csv.writer(open(outcsv, 'w'))
        writer.writerow(['timestamp'] + [str(user_loc.split('.')[0].split('/')[10].split('_')[0])])
        beg = min_timestamp
        end = max_timestamp
        for idx in df_loc.index:

            curr_time = df_loc.ix[idx]['timestamp']
            curr_location = df_loc.ix[idx]['location']
            if idx == 0:
                # print df_loc.ix[idx]['timestamp']
                diff = curr_time - beg
                # print 'beginning difference', diff
                if diff == 0:

                    location = curr_location
                    time = beg

                    # print time, location
                    writer.writerow([time] + [location])
                    continue

                if diff > 0:
                    count = diff
                    print ' missing samples between', beg, curr_time, ' are ', diff
                    # print 'count', count
                    for i in range(0, count + 1):
                        time = (beg + i)
                        location = 0
                        # print time, location
                        writer.writerow([time] + [location])
                        continue

            if ((idx > 0) & (idx < df_loc.index[-1])):

                prev = df_loc.ix[idx - 1]['timestamp']
                prev_location = df_loc.ix[idx - 1]['location']
                diff = curr_time - prev

                # print 'count', diff
                if diff == 1:
                    location = curr_location
                    time = prev + 1
                    # print time, location
                    writer.writerow([time] + [location])
                    continue

                if ((diff > 0) & (diff <= MIN_STAY_TIME)):
                    print ' missing samples between', prev, curr_time, ' are ', diff
                    count = diff
                    for i in range(1, count + 1):

                        time = (prev + i)
                        location = prev_location
                        # print time, location
                        writer.writerow([time] + [location])
                        continue
                if diff > MIN_STAY_TIME:
                    print ' missing samples between', prev, curr_time, ' are ', diff
                    count = diff
                    for i in range(1, count + 1):

                        time = prev + i
                        location = 0
                        # print time, location
                        writer.writerow([time] + [location])
                        continue

            if idx == df_loc.index[-1]:
                print 'index', idx
                prev = df_loc.ix[idx - 1]['timestamp']
                prev_location = df_loc.ix[idx - 1]['location']
                # print 'maximum time and last time', end, prev

                diff = prev - end

                # print 'last difference', diff
                # print 'last missing samples', prev, end
                # print 'count', diff
                if diff == 0:
                    location = curr_location
                    time = prev + 1
                    print time, location
                    writer.writerow([time] + [location])
                    continue
                if ((int(math.fabs(diff)) > 0) & (int(math.fabs(diff)) <= MIN_STAY_TIME)):
                    print ' missing samples between', curr_time, end, ' are ', diff
                    count = int(math.fabs(diff))
                    # print 'count', count
                    for i in range(1, count + 1):

                        time = (prev + i)
                        location = curr_location
                        # print time, location
                        writer.writerow([time] + [location])
                        continue
                if int(math.fabs(diff)) > MIN_STAY_TIME:
                    print ' missing samples between', curr_time, end, ' are ', diff
                    count = int(math.fabs(diff))
                    for i in range(1, count + 1):

                        time = (prev + i)
                        location = 0
                        # print time, location
                        writer.writerow([time] + [location])
                        continue

        print 'Created'
        print outcsv
        #df = pd.read_csv(outcsv)
        # print df


def users_location_table():

    min_t, max_t = max_min_timestamp(USER_LOCATION_PATH)
    df = pd.DataFrame({'timestamp': list(range(min_t, max_t))})
    filename = 'user_location_table' + '.csv'
    outcsv = USER_ATTRIBUTION_TABLES + filename
    writer = csv.writer(open(outcsv, 'w'))

    columns = []
    timestamp = []
    rooms = []
    for i in glob.glob(USER_LOCATION_FORMATTED_PATH + '*.csv'):
        frame = pd.read_csv(i)
        columns.append(frame.columns[1:][0][3])

    header = ''
    for i in columns:
        header = header + '_' + i

    writer.writerow(['timestamp'] + [header])
    for i in glob.glob(USER_LOCATION_FORMATTED_PATH + '*.csv'):
        frame = pd.read_csv(i)
        df = pd.merge(df, frame)

        # print df
    print df
    for idx in df.index:
        column = ''

        for col in df.columns[1:]:

            column = column + str(df.ix[idx][col])
            if col != df.columns[len(df.columns[1:])]:
                column = column + ','
        timestamp.append(df.ix[idx]['timestamp'])
        rooms.append(column)
        writer.writerow([df.ix[idx]['timestamp']] + ['"' + column + '"'])
    df = pd.DataFrame({'timestamp': timestamp, 'rooms': rooms})
    print df
    return df


def get_rooms_corresponding_loc_set(csv_loc):
    df = pd.read_csv(csv_loc)
    header = df.columns[1:][0].split('_')
    header = [h for h in header if h != '']
    print 'rooms available in sets ', header

    filename = 'revised_user_location_table' + '.csv'
    outcsv = USER_ATTRIBUTION_TABLES + filename
    print outcsv
    writer = csv.writer(open(outcsv, 'w'))
    writer.writerow(['timestamp'] + ['rooms'])
    #time = []
    rooms = []
    for idx in df.index:
        rooms = ''
        room_set = df.ix[idx][1]

        room_set = [r for r in room_set if r != ',']
        room_set = [r for r in room_set if r != '"']
        # print room_set
        nonzeroes_set = [h for r, h in zip(room_set, header) if int(r) != 0]
        if nonzeroes_set != []:
            # time.append(df.ix[idx]['timestamp'])
            # rooms.append(nonzeroes_set)
            for room in nonzeroes_set:
                    rooms = rooms + room
                    if room != nonzeroes_set[len(nonzeroes_set) - 1]:
                        rooms = rooms + ','

            writer.writerow([df.ix[idx]['timestamp']] + ['"' + rooms + '"'])


def overlapping_non_overlapping_sets():

    print 'Creating merged use location table'
    #df = users_location_table()
    df = pd.read_csv(USER_ATTRIBUTION_TABLES + 'revised_user_location_table.csv')
    print df
    # get_rooms_set(df)
    # print 'Finding overlapping and non-overlapping sets'
    #df = pd.read_csv(USER_ATTRIBUTION_TABLES + 'user_location_table.csv')

    print 'Determining Overlapping/Non-Overlapping room sets'

    filename = 'location_room_set' + '.csv'
    outcsv = USER_ATTRIBUTION_TABLES + filename
    writer = csv.writer(open(outcsv, 'w'))
    writer.writerow(['start_time'] + ['end_time'] + ['room_set'])
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

            # print 'index', idx, 'start_time',start_time,'end_time',end_time, 'room_set', room_set

        if ((idx > df.index[0]) & (idx < df.index[-1])):

            # curr_room_set = df.ix[idx]['rooms'].split('"')[1]
            # prev_room_set = df.ix[idx - 1]['rooms'].split('"')[1]
            curr_room_set = df.ix[idx]['rooms']
            prev_room_set = df.ix[idx - 1]['rooms']
            # print 'index >
            # 1','index',idx,'prev_room_set',prev_room_set,'curr_room_set',curr_room_set,'start_time',start_time,'curr_time',df.ix[idx
            # ]['timestamp']

            if set(curr_room_set) == set(prev_room_set):
                continue
            if set(curr_room_set) != set(prev_room_set):

                # print 'room set changed'
                # print 'curr_room_set',curr_room_set,'prev_room_set',prev_room_set
                # print 'start_time',start_time,'end_time',end_time
                end_time = df.ix[idx - 1]['timestamp']
                # print 'end_time',end_time
                start.append(start_time)
                end.append(end_time)
                overlapp_set.append(prev_room_set)
                writer.writerow([start_time] + [end_time] + [prev_room_set])
                print 'index', idx, 'start_time', start_time, 'end_time', end_time, 'room_set', prev_room_set
                start_time = df.ix[idx]['timestamp']
                # print 'start_time',start_time
        if idx == df.index[-1]:
            # print 'last index'
            # curr_room_set = df.ix[idx]['rooms'].split('"')[1]
            # prev_room_set = df.ix[idx - 1]['rooms'].split('"')[1]
            curr_room_set = df.ix[idx]['rooms']
            prev_room_set = df.ix[idx - 1]['rooms']
            if set(curr_room_set) == set(prev_room_set):
                end_time = df.ix[idx]['timestamp']

                start.append(start_time)
                end.append(end_time)
                overlapp_set.append(prev_room_set)

                writer.writerow([start_time] + [end_time] + [prev_room_set])
                print 'index', idx, 'start_time', start_time, 'end_time', end_time, 'room_set', prev_room_set
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
                writer.writerow([df.ix[idx]['timestamp']] + [
                                df.ix[idx]['timestamp']] + [curr_room_set])
                print 'index', idx, 'start_time', start_time, 'end_time', end_time, 'room_set', curr_room_set
    df = pd.DataFrame({'start_time': start, 'end_time': end, 'room_set': overlapp_set})

    return df


def separate_overlapping_nonOverlapping_sets():
    df = pd.read_csv(USER_ATTRIBUTION_TABLES + 'location_room_set.csv')
    nonOverlapping_idx = []
    overlapping_idx = []
    for idx in df.index:
        room_nos = len(df.ix[idx][2].split('"')[1].split(','))
        if room_nos == 1:
            nonOverlapping_idx.append(idx)
        else:
            overlapping_idx.append(idx)

    print 'overlapping set index', overlapping_idx
    print 'non-overlapping set index', nonOverlapping_idx

    return overlapping_idx, nonOverlapping_idx


def nonOverlapping_edge_association(edge_light_df, edge_power_df):

    overlap_idx, non_overlap_idx = separate_overlapping_nonOverlapping_sets()

    # Overlapping and non-overlapping room sets
    set_list_df = pd.read_csv(USER_ATTRIBUTION_TABLES + 'location_room_set.csv')
    non_overlap_df = pd.DataFrame(set_list_df.ix[non_overlap_idx])
    print 'non overlap room set list'

    non_overlap_light_edge_ids = []
    non_overlap_power_edge_ids = []
    start = []
    end = []
    edge_power_list = []
    edge_light_list = []
    room = []
    print "start_time,end_time,rom_set"
    for i in non_overlap_df.index:
        print non_overlap_df.ix[i]['start_time'], dt.datetime.fromtimestamp(non_overlap_df.ix[i]['start_time']), non_overlap_df.ix[i]['end_time'], dt.datetime.fromtimestamp(non_overlap_df.ix[i]['end_time']), non_overlap_df.ix[i]['room_set']
    print 'light edge list'
    print_date = [dt.datetime.fromtimestamp(i) for i in edge_light_df['time']]
    print print_date
    print 'power power edge list'
    print_date = [dt.datetime.fromtimestamp(i)for i in edge_power_df['time']]
    print print_date
    for idx in non_overlap_df.index:

        start_time = non_overlap_df.ix[idx]['start_time']
        end_time = non_overlap_df.ix[idx]['end_time']

        # print 'start_time',start_time,'end_time', end_time
        # Find all the light edges falling in non-overlapping windows

        for idx in edge_light_df.index:
            # print 'start time roomset',start_time, 'start time of light
            # edge',edge_light_df.ix[idx]['time'],'end time room set',end_time
            if edge_light_df.ix[idx]['time'] in range(start_time, end_time):
                print 'In range', edge_light_df.ix[idx]['time']
                non_overlap_light_edge_ids.append(idx)

        # Find all the overlapping edges falling in overlapping windows
        for idx in edge_power_df.index:
            # print 'start time roomset',start_time, 'start time of power edge',edge_power_df.ix[idx]['time'],'end time room set',end_time
            # print 'start time of power edge',edge_power_df.ix[idx]['time']
            if edge_power_df.ix[idx]['time'] in range(start_time, end_time):
                print 'In range', edge_power_df.ix[idx]['time']
                non_overlap_power_edge_ids.append(idx)

        if ((non_overlap_power_edge_ids == []) & (non_overlap_light_edge_ids == [])):
            continue
        else:
            start.append(start_time)
            end.append(end_time)
            # print non_overlap_df.ix[idx]['room_set'].split('"')[1]
            # room.append(non_overlap_df.ix[idx]['room_set'])
            edge_light_list.append(non_overlap_light_edge_ids)
            edge_power_list.append(non_overlap_power_edge_ids)

    df_NonOverlap = pd.DataFrame({'start_time': start, 'end_time': end, 'room': room,
                                  'edge_light_indices': edge_light_list, 'edge_power_indices': edge_power_list})

    #non_overlap_light_edge_list = pd.DataFrame(edge_light_df.ix[non_overlap_light_edge_ids])
    #non_overlap_power_edge_list = pd.DataFrame(edge_power_df.ix[non_overlap_power_edge_ids])
    # print df_NonOverlap
    # print 'non_overlapping power edge list', non_overlap_power_edge_list
    # print 'non_overlapping light edge list', non_overlap_light_edge_list


def location_event_association():
        return 1


"""
---------------------------------------
Main Program 
---------------------------------------
"""

if __name__ == '__main__':

    # Get the sensor streams (test set)

    event_dir = sys.argv[1]

    power_csv = event_dir + 'Power.csv'
    light_csv = event_dir + 'Light.csv'

    room_no = sys.argv[2]
    rooms = room_no.split(',')
    # test_csv_path = (WIFI_TRAINING_DATA_PATH)

    # idx = sys.argv[4]

    print rooms

    #room = sys.argv[2]
    #l_phase = sys.argv[2]
    #p_phase = sys.argv[3]
    # room=sys.argv[1]
    # r_type = sys.argv[2]
   # sound_path=sys.argv[4]
    # phno = sys.argv[2]
    # app = int(sys.argv[3])
    # event_dir = 'CompleteDataSets/Apartment/23Sep - Meter_Wifi_Sound_Accl/'
    # event_dir = 'CompleteDataSets/Apartment/23_9_16_53_23_9_19_11/'

    # sys.argv[2]

    #sound_csv = sound_path + '/' + 'mobistatsense' + '/' + 'AudioSamples' + '_' + r_type + '_' + room + '.csv'
    # accl_csv = event_dir + 'Accl_1BHK_102A.csv'  # sys.argv[3]
    # sound_csv = event_dir + 'Sound_1BHK_102A.csv'  # sys.argv[4]
    # wifi_csv = event_dir + 'Wifi_1BHK_102A.csv'  # sys.argv[5]

    # accl_csv = event_dir + 'Accl_1BHK_102A_' + phno + '.csv'  # sys.argv[3]
    # sound_csv = event_dir + 'Sound' + phno + '.csv'  # sys.argv[4]
    # wifi_csv = event_dir + 'Wifi' + phno + '.csv'  # sys.argv[5]

    logger.info("Starting Algorithm...")
    df_p = pd.read_csv(power_csv)
    df_l = pd.read_csv(light_csv)
    df_p_copy = df_p.copy()

    #db_paths = '/media/New Volume/IPSN/sense/5_6Nov/dbs/'
    #dbs = ['c003_room_db.csv', 'c004_room_db.csv', 'c005_room_db.csv', 'c006_room_db.csv']
    #db_rooms = '_room_db.csv'
    #df_s = pd.read_csv(sound_csv)
    # Step i: Apply moving average to the power data
    # df_l = average_power(df_l)
    # df_p = average_power(df_p)

    # Step 1: Smoothening power phase stream (Y)
    # Filtering periodic drops

    # Power phase2 plot before smoothening
    # df_p = filter_drops(df_p, "powerphase2")
    # df_p = filter_drops(df_p, "powerphase2")
    # df_p = filter_drops(df_p, "powerphase2")

    # print 'PLOT'
    # fig = plt.figure()
    # plt.gcf()
    # ax1 = plt.subplot(2, 1, 1)
    # plt.plot( df_p_copy['powerphase2'] )
    # ax2 = plt.subplot(2, 1, 2)
    # plt.plot( df_p['powerphase2'] )
    # show()
    # sys.exit(1)

    # df_p.ix[curr_idx, "powerphase2"] = mean of prev and next curr power

    #fig = plt.figure()
    # plt.gcf()
    # print 'After plots'
    #plot( df_p['powerphase2'])

    # Step2: Edge Detection
    edge_light_df, edge_power_df = edge_detection(df_l, df_p)

    # sys.exit()

    # Step 3 : Wifi Localization to determine each users location for the given dataset as in/out of hostel room
    # Classify
    """
    for i in rooms:
        formatted_csv_path = (WIFI_FORMATTED_DATA_PATH + i + '.csv' )
        formatted_csv_path = format_data(test_csv_path, 'train', i)
        print 'Created wifi formatted csv', (formatted_csv_path + i + '.csv')
        wifi_location(formatted_csv_path, i)
    """
    # print 'Filling in missing samples'
    # print 'Filling missing samples in WIFI csv for each user '
    """
    fill_missing_samples(USER_LOCATION_PATH)
    """
    # Step 4 : Determine Overlapping/Non-Overlapping users location

    # print df_set
    # users_location_table()
    #get_rooms_corresponding_loc_set(USER_ATTRIBUTION_TABLES + 'user_location_table.csv')
    #df_set = overlapping_non_overlapping_sets()
    # Step 5 : User Attribution
    #separate_overlapping_nonoverlapping_roomsets(USER_ATTRIBUTION_TABLES + 'location_room_set.csv')
    #nonOverlapping_edge_association(edge_light_df, edge_power_df)
    # Step2: PreProcessing
    #edge_list_df = edge_preprocessing(df_l, df_p, edge_list_df)
    #time_slices = af.edge_matching(df_l, df_p, edge_list_df)
    # df_b=DataFrame()
    # df_a=DataFrame()
    # print 'start time to next 1 minutes:'
    # for t in time_slices.index[:-1]:
    #     from_t=int(time_slices.ix[t]['start_time'])*1000
    #     end_t=int(time_slices.ix[t]['end_time'])*1000
    #     room=[]
    # Considering Phase and Magnitude Information
    #     if (time_slices.ix[t]['phase']=='B') & (time_slices.ix[t]['magnitude'] in range(30,36)):
    #         room=['c006','c005']
    #     if (time_slices.ix[t]['phase']=='B') & (time_slices.ix[t]['magnitude'] in range(50,56)):
    #         room=['c003','c004']
    #     if (time_slices.ix[t]['phase']=='Y') & (time_slices.ix[t]['magnitude'] in range(30,36)):
    #         room=['c003','c004']
    #     if (time_slices.ix[t]['phase']=='Y') & (time_slices.ix[t]['magnitude'] in range(50,56)):
    #         room=['c006','c005']
    #     print room
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
