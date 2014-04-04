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
#import activity_finder_algorithm as af
# import classifier as cl
# import classify_sound as cs
import warnings
import logging

# Disable warnings
warnings.filterwarnings('ignore')

# Enable Logging
logger = logging.getLogger('activity-finder')
logging.basicConfig(level=logging.DEBUG)


apt_no = '102A'
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
lthresmin = 30   # for light meter
pthresmin = 30  # for power meter
# pthresmin = lthresmin

# Power Percent Change between rising and falling edge
percent_change = 0.31

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

# Applies a moving average to the power data

def average_power(df):
    pass

# Checks if there is any light edge (rising or falling) at the index provided
# power_stream


def check_if_light_edge(df_l, index, power_stream):

    i = index
    # print "Looking for light edge in stream:", power_stream, "for Index:", i
    prev = int(df_l.ix[i - 1][power_stream])
    curr = int(df_l.ix[i][power_stream])
    next = int(df_l.ix[i + 1][power_stream])
    currwin = int(df_l.ix[i + lwinmin][power_stream])
    if i - lwinmin not in (df_l.index):
        prevwin = 0
    else:
        prevwin = int(df_l.ix[i - lwinmin][power_stream])
    time = df_l.ix[i]['time']
    per_lthresmin = int(0.25 * lthresmin)

    # Checking for missing time samples
    prev_time = df_l.ix[i - 1]['time']
    next_time = df_l.ix[i + 1]['time']

    # Indicates next time sample is missing
    prev_missing_sample = (time - prev_time) > 1
    next_missing_sample = (next_time - time) > 1

    prev_curr_diff = int(math.fabs(curr - prev))
    curr_next_diff = int(math.fabs(next - curr))
    curr_prevwin_diff = prevwin - curr
    curr_nextwin_diff = currwin - curr

    if(time in [1386239808]):
        logger.debug("\n")
        logger.debug("Per lthresmin %d", per_lthresmin)
        logger.debug(
            "Looking for light edge in stream: %s for Index: %d", power_stream, i)
        logger.debug("R:: currtime %s prev %s curr %s next %s",
                     dt.datetime.fromtimestamp(time), prev, curr, next)
        logger.debug("currwin %s", currwin)
        logger.debug(
            "prev_curr_diff %s curr_next_diff %s", prev_curr_diff, curr_next_diff)
        logger.debug("curr_nextwin_diff %s curr_prevwin_diff %s",
                     curr_nextwin_diff, curr_prevwin_diff)

        if (prev_curr_diff < lthresmin and curr_nextwin_diff > lthresmin
           and curr_next_diff > prev_curr_diff):
            logger.debug("True")
            logger.debug("Missing Sample value %s", next_missing_sample)
            logger.debug("Curr next diff %s", curr_next_diff)
            if next_missing_sample:
                logger.debug("Missing Sample yes")
            if int(curr_next_diff) >= lthresmin:
                logger.debug("Satisfied condition")
        elif (curr_next_diff >= per_lthresmin) or curr_next_diff > lthresmin:
            logger.debug(" CurrNextDiff between lthresmin half: %d", i)
        elif prev_curr_diff < lthresmin and curr_nextwin_diff > lthresmin:
            logger.debug("True-- Fan")
        elif (prev_curr_diff < lthresmin and math.floor(curr_nextwin_diff) <= (-lthresmin)
              and curr_next_diff > prev_curr_diff):
            logger.debug("True - Falling")

    # logger.debug("R::", "currtime", time, "prev ", prev, "curr ", curr, "next ", next)
    # logger.debug("currwin", currwin)
    # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff)
    # logger.debug("curr_nextwin_diff", curr_nextwin_diff)

    if (prev_curr_diff < lthresmin and curr_nextwin_diff > lthresmin and
            curr_next_diff > prev_curr_diff and curr_next_diff > 0):

        # logger.debug("\nR1::", i, "currtime", dt.datetime.fromtimestamp(time),
            # "prev ", prev, "curr ", curr, "next ", next)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff")
        # curr_prevwin_diff

        edge_type = "rising"
        # Only checking these conditions for cumulative power
        # if power_stream == "lightpower":
        if next_missing_sample and int(curr_next_diff) > lthresmin:
            logger.debug("Missing Sample:: Index %d", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        # or curr_next_diff > lthresmin:
        elif (curr_next_diff >= per_lthresmin):
            logger.debug("Here1 Index:: %d", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        else:
            print "Not a edge"
            pass

    elif (prev_curr_diff < lthresmin and curr_nextwin_diff > lthresmin
          and curr_next_diff > per_lthresmin):
        # logger.debug("\nR2::", i, "currtime", dt.datetime.fromtimestamp(time), "prev")
        # logger.debug(prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff,)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff",)
        # curr_prevwin_diff

        edge_type = "rising"
        # Storing the rising edge e_i = (time_i, mag_i)
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
        return edge_type, row

    elif (prev_curr_diff < lthresmin and math.floor(curr_nextwin_diff) <= (-lthresmin)
          and curr_next_diff > prev_curr_diff and curr_prevwin_diff > -lthresmin):

        # logger.debug("\nF::", "currtime", dt.datetime.fromtimestamp(time),
            # "prev ", prev, "curr ", curr, "next ", next,)
        # logger.debug("currwin", currwin)
        # logger.debug("prev_curr_diff", prev_curr_diff, "curr_next_diff ", curr_next_diff)
        # logger.debug("curr_nextwin_diff", curr_nextwin_diff, "curr_prevwin_diff",)
        # curr_prevwin_diff

        edge_type = "falling"
        if prev_missing_sample is True or next_missing_sample is True:
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        elif curr_next_diff < lthresmin or curr_next_diff >= lthresmin:
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        else:
            pass
    return "Not an edge", {}


def check_if_power_edge(df_p, index, power_stream):

    i = index
    prev = df_p.ix[i - 1][power_stream]

    curr = df_p.ix[i][power_stream]
    next = df_p.ix[i + 1][power_stream]
    currwin = df_p.ix[i + pwinmin][power_stream]
    
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
        if power_stream != "power":
            if math.floor(prevwin) != 0:
                prevwin = prevwin + 10
        # prevwintime = df_p.ix[i - pwinmin]['time']
    time = df_p.ix[i]['time']
    per_pthresmin = int(0.25 * pthresmin)

    # Checking for missing time samples
    prev_time = df_p.ix[i - 1]['time']
    next_time = df_p.ix[i + 1]['time']
    #print dt.datetime.fromtimestamp(prev_time),dt.datetime.fromtimestamp(time),dt.datetime.fromtimestamp(next_time)

    # Indicates next time sample is missing
    prev_missing_sample = (time - prev_time) > 1
    next_missing_sample = (next_time - time) > 1

    prev_curr_diff = int(math.fabs(curr - prev))
    curr_next_diff = int(math.fabs(next - curr))
    curr_nextwin_diff = int(currwin - curr)
    curr_prevwin_diff = int(prevwin - curr)

    # Code for debugging
    # range(2683, 3652)
    # if i in [1492]:
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
        # Storing the rising edge e_i = (time_i, mag_i)
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
        return edge_type, row
        if next_missing_sample and curr_next_diff > pthresmin:
            logger.debug("Missing Sample:: Index %s", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        elif (curr_next_diff > per_pthresmin) or curr_next_diff > pthresmin:
            logger.debug("Here1 Index:: %s", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        else:
            logger.debug("Here2 Index:: %s", i)
            pass
    elif (prev_curr_diff < pthresmin and curr_nextwin_diff > pthresmin
          and curr_next_diff >= per_pthresmin):
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
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
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
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        elif curr_next_diff < pthresmin or curr_next_diff > pthresmin:
            logger.debug("Falling Edge2:: Index %s", i)
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff}
            return edge_type, row
        else:
            pass
    return "Not an edge", {}


def filter_edges(rise_df, fall_df, winmin):

    # FILTER 1:
    # Handle cases where 2 rising edges corresponding to a single
    # falling edge - combining them to one edge

    for ix_i in rise_df.index:
        sim_edge_set = []
        for ix_j in rise_df.index:
            curr_prev_diff = math.fabs(rise_df.ix[ix_j]['time'] -
                                       rise_df.ix[ix_i]['time'])
            if ix_i == ix_j or ix_i > ix_j:
                continue
            elif curr_prev_diff in range(winmin, winmin + 2 + 1):
                logger.debug("Index i: %d", ix_i)
                logger.debug("Index j: %d", ix_j)
                sim_edge_set.append(ix_j)
        # Add the magnitude of the two edges and convert into a single
        # edge
        if len(sim_edge_set) > 0:
            tmp = rise_df.ix[sim_edge_set]
            sel_idx = tmp[tmp.magnitude == tmp.magnitude.max()].index[0]
            new_mag = (rise_df.ix[ix_i]['magnitude'] +
                       rise_df.ix[sel_idx]['magnitude'])
            rise_df.ix[ix_i]['magnitude'] = new_mag
            logger.debug(
                "Inside Index j: %s New Mag: %s IndexTime: %s", sel_idx, new_mag, ix_i)
            # Remove the second edge
            rise_df.drop(sel_idx)

    # FILTER 2:
    # Filter out spike edges where rising and falling edges
    # are within a small time frame (lwinmin)
    # This is to remove quick successive turn ON and OFFs

    tmp = pd.concat([rise_df, fall_df])
    tmp = tmp.sort('time')
    tmp['ts'] = (tmp.time / 100).astype('int')
    # logger.debug("\nConcatenated TMPDF::\n %s", tmp)
    for i, ix_i in enumerate(tmp.index):
        for j, ix_j in enumerate(tmp.index):
            if(i == j):
                continue
            elif (tmp.ix[ix_i]['ts'] == tmp.ix[ix_j]['ts'] and
                  tmp.ix[ix_i]['magnitude'] > tmp.ix[ix_j]['magnitude'] and
                  tmp.ix[ix_j]['magnitude'] == -tmp.ix[ix_i]['magnitude']):
                logger.debug("Index i:: %s", ix_i)
                logger.debug("Removing %s %s", ix_i, ix_j)
                rise_df.drop(ix_i)
                fall_df.drop(ix_j)

    # for ix_i in rise_df.index:
    #     curr = df_l.ix[ix_i]['lightpower']
    #     currnext = df_l.ix[ix_i + 1]['lightpower']
    #     currnext_next = df_l.ix[ix_i + 2]['lightpower']
    #     curr_next_diff = int(currnext - curr)
    #     succ_fall_mag = math.ceil(int(currnext_next - currnext))
    #     logger.debug("Index:", ix_i, "Succ Fall Mag", \
        # succ_fall_mag, "CurrNextDiff", curr_next_diff)
    # if ix_i ==
    #     if (succ_fall_mag >= lthresmin and
    #        (succ_fall_mag - curr_next_diff) in np.arange(0, 2, 0.1)):
    #         logger.debug("Index:", ix_i, "Succ Fall Mag", succ_fall_mag,)
    #         logger.debug("CurrNextDiff", curr_next_diff)
    #         rise_df.drop(ix_i)
    #         logger.debug("Inside Succ Fall Mag", succ_fall_mag, "Index", i)

    # FILTER 3:
    # Select the edge amongst a group of edges within a small time frame (a minute)
    # and which are close to each other in terms of magnitude
    # with maximum timestamp

    # Rising Edges
    tmp_df = rise_df.copy()
    # tmp_df['ts'] = (rise_df.time / 100).astype('int')
    tmp_df['tmin'] = [str(dt.datetime.fromtimestamp(i).hour) + '-' +
                      str(dt.datetime.fromtimestamp(i).minute) for i in tmp_df.time]
    # print "Grouped Max", tmp_df.groupby('tmin')['magnitude'].max()
    # ts_grouped = tmp_df.groupby('tmin').max()
    # tmp_df = pd.concat([tmp_df[(tmp_df.tmin == i) &
    #                   (tmp_df.magnitude == ts_grouped.ix[i]['magnitude'])]
    #     for i in ts_grouped.index])
    # ts_grouped = tmp_df.groupby('tmin')['time'].max()
    # tmp_df = tmp_df[tmp_df.time.isin(ts_grouped)].sort(['time'])

    # Select the edge with the maximum timestamp lying within a minute
    idx_list = []
    for i, idx in enumerate(tmp_df.index):
        if idx in idx_list:
            continue
        if idx == tmp_df.index[-1]:
            idx_list.append(idx)
            break
        t = tmp_df.ix[idx]['time']
        t_next = tmp_df.ix[tmp_df.index[i + 1]]['time']
        diff = t_next - t
        if diff <= 60:
            t_mag = tmp_df.ix[idx]['magnitude']
            t_next_mag = tmp_df.ix[tmp_df.index[i + 1]]['magnitude']

            if t_mag >= 1000:
                print "in"
                threshold = 0.25
            else:
                threshold = 0.1

            if math.fabs(t_mag - t_next_mag) <= threshold * t_mag:
                idx_list.append(tmp_df.index[i + 1])
            else:
                idx_list.append(idx)
        else:
            idx_list.append(idx)
    print "Rise idx_list", idx_list
    tmp_df = tmp_df.ix[idx_list].sort(['time'])
    rise_df = tmp_df.ix[:, :-1]

    # Falling Edges
    tmp_df = fall_df.copy()
    # tmp_df['ts'] = (fall_df.time / 100).astype('int')
    tmp_df['tmin'] = [str(dt.datetime.fromtimestamp(i).hour) + '-' +
                      str(dt.datetime.fromtimestamp(i).minute) for i in tmp_df.time]
    # print "Grouped Max", tmp_df.groupby('tmin')['magnitude'].max()
    # ts_grouped = tmp_df.groupby('tmin')
    # g = ts_grouped.head()
    # grp_index = g.index.get_level_values(0).unique()

    # idx_list = []
    # for k in grp_index:
    #     tdf = g.ix[k]
    #     grp_idx_list = []
    #     i = 0
    #     new_idx = tdf.index[i]
    #     while new_idx in tdf.index:
    # print new_idx,
    # idx_list.append(new_idx)
    #         if new_idx == tdf.index[-1]:
    #             idx_list.append(new_idx)
    #             break
    #         idx_next = int(tdf.index[i + 1])
    #         mg = math.fabs(tdf.ix[new_idx]['magnitude'])
    #         mg_next = math.fabs(tdf.ix[idx_next]['magnitude'])
    #         if mg <= 60:
    #             threshold = 0.2
    #         elif mg >= 1000:
    #             print "in"
    #             threshold = 0.25
    #         else:
    #             threshold = 0.1
    # print "Using threshold", threshold
    # print "Mag", mg, "Mag next", mg_next
    # Check within 10% of the next value, add in the list
    # and select as new index for next iteration
    #         if math.fabs(mg - mg_next) <= threshold * mg:
    #             grp_idx_list.append(new_idx)
    #             grp_idx_list.append(idx_next)
    #             print "thru"
    #         else:
    #             idx_list.append(new_idx)
    #         new_idx = tdf.index[i + 1]
    #         i = i + 1
    # print "grp_idx_list", grp_idx_list
    #     if len(grp_idx_list) > 0:
    #         grp_time_max = tdf.ix[grp_idx_list]['time'].max()
    #         idx_list = idx_list + [int(tdf.index[np.where(tdf.time == grp_time_max)[0]])]
    # print "idx_list", idx_list
    # print "\n"
    #     idx_list = list(set(idx_list))

    # Select the edge with the maximum timestamp lying within a minute
    idx_list = []
    for i, idx in enumerate(tmp_df.index):
        if idx in idx_list:
            continue
        if idx == tmp_df.index[-1]:
            idx_list.append(idx)
            break
        t = tmp_df.ix[idx]['time']
        t_next = tmp_df.ix[tmp_df.index[i + 1]]['time']
        diff = t_next - t
        if diff <= 60:
            t_mag = math.fabs(tmp_df.ix[idx]['magnitude'])
            t_next_mag = math.fabs(tmp_df.ix[tmp_df.index[i + 1]]['magnitude'])

            if t_mag <= 60:
                threshold = 0.2
            elif t_mag >= 1000:
                print "in"
                threshold = 0.25
            else:
                threshold = 0.1
            if math.fabs(t_mag - t_next_mag) <= threshold * t_mag:
                idx_list.append(tmp_df.index[i + 1])
            else:
                idx_list.append(idx)
        else:
            idx_list.append(idx)
    print "\nFall idx_list", idx_list
    tmp_df = tmp_df.ix[idx_list].sort(['time'])

    # tmp_df = pd.concat([tmp_df[(tmp_df.tmin == i) &
    #                   (tmp_df.magnitude == ts_grouped.ix[i]['magnitude'])]
    #     for i in ts_grouped.index])
    # ts_grouped = tmp_df.groupby('tmin')['time'].max()
    # tmp_df = tmp_df[tmp_df.time.isin(ts_grouped)].sort(['time'])
    fall_df = tmp_df.ix[:, :-1]

    return rise_df, fall_df


def generate_light_edges(df_l, phase):

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
    #print ix_list_l[-1]
    for i in range(1, ix_list_l[-1] - lwinmin + 1):
        edge_type, result = check_if_light_edge(df_l, i, phase)
        if edge_type == "falling":
            f_row_list.append(result)
        elif edge_type == "rising":
            r_row_list.append(result)
        else:
            pass

    rising_edges_l_df = pd.DataFrame(
        r_row_list, columns=['index', 'time', 'magnitude'])
    rising_edges_l_df = rising_edges_l_df.set_index('index', drop=True)

    falling_edges_l_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude'])
    falling_edges_l_df = falling_edges_l_df.set_index('index', drop=True)

    print "Rising Edges::\n", rising_edges_l_df
    print "Falling edges::\n", falling_edges_l_df

    rising_edges_l_df, falling_edges_l_df = filter_edges(
        rising_edges_l_df, falling_edges_l_df, lwinmin)

    # Print the edges with the timestamps values converted to UTC +5.30
    # format
    # Rising Edges
    print_lr_df = rising_edges_l_df.copy()
    t = pd.to_datetime(print_lr_df['time'], unit='s')
    index = pd.DatetimeIndex(t.tolist())
    ts_utc = index.tz_localize('UTC')
    new_tz = ts_utc.tz_convert(TIMEZONE)
    print_lr_df.set_index(new_tz, inplace=True, drop=False)

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


def generate_power_edges(df_p, phase):

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
    for i in range(1, ix_list_p[-1] - pwinmin + 1):
        edge_type, result = check_if_power_edge(df_p, i, phase)
        if edge_type == "falling":
            f_row_list.append(result)
        elif edge_type == "rising":
            r_row_list.append(result)
        else:
            pass

    rising_edges_p_df = pd.DataFrame(
        r_row_list, columns=['index', 'time', 'magnitude'])
    rising_edges_p_df = rising_edges_p_df.set_index('index', drop=True)

    falling_edges_p_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude'])
    falling_edges_p_df = falling_edges_p_df.set_index('index', drop=True)

    print "Rising Edges::\n", rising_edges_p_df
    print "Falling Edges::\n", falling_edges_p_df

    rising_edges_p_df, falling_edges_p_df = filter_edges(
        rising_edges_p_df, falling_edges_p_df, pwinmin)

    # Print the edges with the timestamps values converted to UTC +5.30
    # format

    # Rising Edges
    print_pr_df = rising_edges_p_df.copy()
    t = pd.to_datetime(
        rising_edges_p_df['time'], unit='s')
    index = pd.DatetimeIndex(t.tolist())
    ts_utc = index.tz_localize('UTC')
    new_tz = ts_utc.tz_convert(TIMEZONE)
    print_pr_df.set_index(new_tz, inplace=True, drop=False)

    # Falling Edges
    print_pf_df = falling_edges_p_df.copy()
    t = pd.to_datetime(
        falling_edges_p_df['time'], unit='s')
    index = pd.DatetimeIndex(t.tolist())
    ts_utc = index.tz_localize('UTC')
    new_tz = ts_utc.tz_convert(TIMEZONE)
    print_pf_df.set_index(new_tz, inplace=True, drop=False)

    print "-" * stars
    print "Filtered Edges:"
    print "-" * stars
    print "Rising Edges::\n", print_pr_df
    print "Falling Edges::\n", print_pf_df

    return rising_edges_p_df, falling_edges_p_df


def make_edge_list(rise_l_df, fall_l_df, rise_p_df, fall_p_df):

    # Generating edge list
    edge_list_df = []
    list_df = [rise_l_df, fall_l_df, rise_p_df, fall_p_df]
    for i_no, ip_df in enumerate(list_df):
        logger.debug("Edge List Df No:%s\n" + "-" * stars, i_no + 1)

        if len(ip_df.index) == 0:
            df = pd.DataFrame(columns=['time', 'magnitude',
                                       'act_time'])
            edge_list_df.append(df)
        else:
            ip_df['act_time'] = [dt.datetime.fromtimestamp(int(t)) for t in ip_df.time]
            edge_list_df.append(ip_df)
        # logger.debug("\n OUTDF:: \n", edge_list_df)
    print "\n Filtered Correlated Edge List:: \n", edge_list_df

    return edge_list_df


def edge_detection(df_l, df_p, l_phase, p_phase):

    print "Using Light Threshold::", lthresmin
    print "Using Power Threshold::", pthresmin

    # Light phase
    if l_phase == 'R':
        l_phase = "lightphase1"
    elif l_phase == 'Y':
        l_phase = "lightphase2"
    elif l_phase == 'B':
        l_phase = "lightphase3"

    # Power phases
    if p_phase == 'R':
        p_phase = "powerphase1"
    elif p_phase == 'Y':
        p_phase = "powerphase2"
    elif p_phase == 'B':
        p_phase = "powerphase3"

    
    rising_edges_l_df, falling_edges_l_df = generate_light_edges(df_l, l_phase)
    rising_edges_p_df, falling_edges_p_df = generate_power_edges(df_p, p_phase)

    return make_edge_list(rising_edges_l_df, falling_edges_l_df,
                          rising_edges_p_df, falling_edges_p_df)


"""
Step 3: Edge Matching
Taking time-correlated rising and falling edges, goal is to
match each rising and falling edge to determine the usage time.
Use the Wifi data and meter data, to find the location of
the activity and also act as a filter to distinguish between edges
with similar magnitude (coming from similar fixtures).
The phase of the power event will give the
coarse level location. Combining it with Wifi will give
the user's activity location.

Input: Time-correlated edges with user activity
Output: Time slices of interest <T_i, T_j, ...>
    T_i = (s_i, e_i)
    where,
        s_i = start time of activity
        e_i = end time of activity

"""


def get_phase_information(df_l, df_p, edge_list):
    # Find the phase information for all filtered rising and falling edges
    phases = {1: "R", 2: "Y", 3: "B"}
    
    for i, df in enumerate(edge_list):

        df['phase'] = pd.Series(np.zeros(len(df)))
        # df['location'] = pd.Series(np.zeros(len(df)))
        # Checking for light edges
        if i in [0, 1]:
            logger.debug("Finding phases for light edges......")
            logger.debug("-" * stars)
            # Get indexes
            idx_list = df.index
            phase_list = []
            # loc_list = []
            for idx in idx_list:
                # Checking for each phase
                for j in range(1, 4):
                    edge_type, result = check_if_light_edge(
                        df_l, idx, "lightphase" + str(j))
                    if edge_type == "Not an edge":
                        logger.debug(
                            "Not a light edge in %s for index %s", phases[j], idx)
                    else:
                        logger.debug(
                            "Found light edge in phase %s for index %s", phases[j], idx)
                        phase_list.append(phases[j])
                        # loc_list.append(phase_loc[phases[j]])
                        break
                    if j == 3:
                        # Find edge in the previous entry
                        flag = False
                        for j in range(1, 4):
                            edge_type, result = check_if_light_edge(
                                df_l, idx - 1, "lightphase" + str(j))
                            if edge_type == "Not an edge":
                                logger.debug(
                                    "Not a light edge in %s for index %s", phases[j], idx)
                                flag = True
                            else:
                                logger.debug(
                                    "Found light edge in phase %s for index %s", phases[j], idx)
                                phase_list.append(phases[j])
                                flag = False
                                break
                                # loc_list.append(phase_loc[phases[j]])
                        if flag:
                            # Find in the next entry
                            flag = False
                            for j in range(1, 4):
                                edge_type, result = check_if_light_edge(
                                    df_l, idx + 1, "lightphase" + str(j))
                                if edge_type == "Not an edge":
                                    logger.debug(
                                        "Not a light edge in %s for index %s", phases[j], idx)
                                    flag = True
                                else:
                                    logger.debug(
                                        "Found light edge in phase %s for index %s", phases[j], idx)
                                    phase_list.append(phases[j])
                                    flag = False
                                    break
                                    # loc_list.append(phase_loc[phases[j]])
                            if flag:
                                phase_list.append("Not found")
                                # loc_list.append("Not found")
            # logger.debug("Length of phase list:: %s", len(phase_list))
            # logger.debug("Length of data frame:: %s", len(df))
            # logger.debug("Length of loc list:: %s", len(loc_list))

            df['phase'] = phase_list
            # df['location'] = loc_list

        # Checking for power edges
        else:
            logger.debug("Finding phases for power edges......")
            logger.debug("-" * stars)
            # Get indexes
            idx_list = df.index
            # loc_list = []
            phase_list = []
            for idx in idx_list:
                # Checking  for each phase
                for j in range(1, 4):
                    edge_type, result = check_if_power_edge(
                        df_p, idx, "powerphase" + str(j))
                    if edge_type == "Not an edge":
                        logger.debug(
                            "%d Not a power edge in %s", idx, phases[j])
                    else:
                        logger.debug("Found power edge in phase %s", phases[j])
                        phase_list.append(phases[j])
                        # loc_list.append(phase_loc[phases[j]])
                        break
                    if j == 3:
                        # Find edge in the previous entry
                        flag = False
                        for j in range(1, 4):
                            edge_type, result = check_if_power_edge(
                                df_p, idx - 1, "powerphase" + str(j))
                            if edge_type == "Not an edge":
                                logger.debug(
                                    "%d Not a power edge in %s", idx, phases[j])
                                flag = True
                            else:
                                logger.debug("Found power edge in phase %s", phases[j])
                                phase_list.append(phases[j])
                                flag = False
                                break
                                # loc_list.append(phase_loc[phases[j]])
                        # If edge not found in the previous entry, find
                        # in the next entry
                        if flag:
                            flag = False
                            for j in range(1, 4):
                                edge_type, result = check_if_power_edge(
                                    df_p, idx + 1, "powerphase" + str(j))
                                if edge_type == "Not an edge":
                                    logger.debug(
                                        "%d Not a power edge in %s", idx, phases[j])
                                    flag = True
                                else:
                                    logger.debug("Found power edge in phase %s", phases[j])
                                    phase_list.append(phases[j])
                                    flag = False
                                    break
                                    # loc_list.append(phase_loc[phases[j]])

                            # Edge not found in any of the entry
                            if flag:
                                phase_list.append("Not found")
            # logger.debug("Length of phase list:: %s", len(phase_list))
            # logger.debug("Length of data frame:: %s", len(df))
            df['phase'] = phase_list
            # df['location'] = loc_list
        logger.debug("Index: %s Frame: \n %s", i, df)

    return edge_list


def make_pairs(rise_fall_dict, rise_df, fall_df, edge_type):

    row_df_list = []
    new_rel_idx = rise_fall_dict

    for fall_idx, rise_idx_list in new_rel_idx.items():
        # Processing for falling edges where rising edges exist
        if len(rise_idx_list) > 0:

            logger.debug(
                "Fall_index: %s Rise index: %s", fall_idx, rise_idx_list)
            logger.debug("-" * stars)

            # Falling edge
            f_time = int(fall_df.ix[fall_idx]['time'])
            f_mag = (-1) * fall_df.ix[fall_idx]['magnitude']
            logger.debug("Fall_time %s", dt.datetime.fromtimestamp(f_time))

            min_diff = (rise_idx_list[0], math.fabs(
                f_mag - rise_df.ix[rise_idx_list[0]]['magnitude']))
            print "First min", min_diff
            for rid in rise_idx_list:
                if rid != 0:
                    r_mag = int(rise_df.ix[rid]['magnitude'])
                    diff = math.fabs(f_mag - r_mag)
                    if min_diff[1] > diff:
                        min_diff = (rid, diff)
                        print "New min", min_diff
            # Taking the rising edge which is the closest to the fall magnitude
            r_index = min_diff[0]
            r_time = int(rise_df.ix[r_index]['time'])
            logger.debug("Rise Index:: %s", r_index)
            logger.debug("Rise Time: %s", dt.datetime.fromtimestamp(r_time))

            # Creating edge pair entry
            row_df = pd.DataFrame(
                {"start_time": rise_df.ix[r_index]['time'],
                 "end_time": f_time,
                 "magnitude": f_mag,
                 "type": edge_type, "phase": fall_df.ix[fall_idx]['phase']},
                index=[0])
            logger.debug("Testdf: \n%s", row_df)
            row_df_list.append(row_df)

            # Filter 4: Removing this rising edge which has been associated with
            # a falling edge
            for fid, rise_idx_list in new_rel_idx.items():

                if r_index in rise_idx_list and fid != fall_idx:
                    # logger.debug("Fall_index: %d Rise index:%s",
                                 # fall_index, rise_indx_list)
                    rise_idx_list.remove(r_index)
            logger.debug("New Rising Edge List::\n%s", new_rel_idx)
            logger.debug("-" * stars)

    return row_df_list


def edge_matching(df_l, df_p, edge_list):

    # Output
    time_slices = pd.DataFrame(
        columns=['start_time', 'end_time', 'magnitude', 'type', 'phase'])

    # Find the phase information for all filtered rising and falling edges
    # if app == 4:
    edge_list = get_phase_information(df_l, df_p, edge_list)

    # TODO: Filter out voltage fluctuations seen across multiple phases

    # Find the time slices (start and end time of activity) by matching
    # based on magnitude of the edges

        # for rising and its corresponding falling edge
        # l_power = 5
        # p_power = 100

    rise_l_df = edge_list[0].reset_index(drop=True)
    fall_l_df = edge_list[1].reset_index(drop=True)
    rise_p_df = edge_list[2].reset_index(drop=True)
    fall_p_df = edge_list[3].reset_index(drop=True)

    # if app == 4:
    rise_l_df = rise_l_df[rise_l_df.phase != "Not found"].reset_index(drop=True)
    fall_l_df = fall_l_df[fall_l_df.phase != "Not found"].reset_index(drop=True)
    rise_p_df = rise_p_df[rise_p_df.phase != "Not found"].reset_index(drop=True)
    fall_p_df = fall_p_df[fall_p_df.phase != "Not found"].reset_index(drop=True)

    print "-" * stars
    print "Edge Matching Process"
    print "-" * stars

    # For every falling edge, find the matching rising edge
    full_df_list = []
    for k in [0, 1]:
        if k == 0:
            rise_df = rise_l_df
            fall_df = fall_l_df
            # power = l_power
            edge_type = "light"
            logger.debug("Matching for light edges......")
            logger.debug("-" * stars)
        else:
            rise_df = rise_p_df
            fall_df = fall_p_df
            # power = p_power
            edge_type = "power"
            logger.debug("Matching for power edges......")
            logger.debug("-" * stars)

        # Filter 1: Filter falling edges where it is before any rising edge
        # cols = ['time', 'time_sec', 't_meter','magnitude', 'label', 'pred_label']
        re_l_idx = [np.where(rise_df.time < i)[0] for i in fall_df.time]
        logger.debug("Filter 1 results: %s", re_l_idx)

        # Filter 2: Match falling with rising edges where its magnitude between
        # a power threshold window

        # This dict contains the falling edge index as the key and the corresponding
        # rising edges as list items
        new_rel_idx = defaultdict(list)
        for i, row in enumerate(re_l_idx):
            no_ele = len(row)
            re_l = []
            if no_ele > 0:
                fall_mag = math.fabs(fall_df.ix[i]['magnitude'])
                # Set power threshold for magnitude comparison between
                # rising and falling
                power = fall_mag * percent_change
                logger.debug("Idx %d %s fall_mag=%s %s", i,
                             fall_mag - power, fall_mag, fall_mag + power)
                # For debugging - printing the rising edges corres. to falling
                # edges
                restr = ''
                for idx in row:
                    restr += ' ' + str(rise_df.ix[idx]['magnitude'])
                logger.debug("Rising edges::%s", restr)
                # Generating matching rising edge list for the falling edge i
                re_l = [idx for idx in row if rise_df.ix[idx]['magnitude'] >= fall_mag - power
                        and rise_df.ix[idx]['magnitude'] <= fall_mag + power]
                logger.debug("Matched Rising Edges with fidx %d: %s", i, re_l)
            new_rel_idx[i] = re_l
        logger.debug("Filter 2 results: %s", new_rel_idx.items())

        row_df_list = make_pairs(new_rel_idx, rise_df, fall_df, edge_type)
        full_df_list = full_df_list + row_df_list

    logger.debug("\nMatched Edges: \n%s", full_df_list)
    if len(full_df_list) == 0:
        print "No time slices found!"
        return time_slices
    elif len(full_df_list) == 1:
        time_slices = full_df_list[0]
    else:
        time_slices = pd.concat(full_df_list)

    time_slices = time_slices.reset_index(drop=True)
    # print "Time slices::", time_slices

    # Printing Format
    print_ts_df = time_slices.copy()

    print_ts_df['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                     for i in time_slices['start_time']]
    print_ts_df['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                   for i in time_slices['end_time']]
    print_ts_df = print_ts_df.sort(['start_time'])
    print "\nGenerated Time Slices::\n", print_ts_df
    return print_ts_df


"""
Step 4: Classification of Appliance usage and Location Determination
Use the sound data from that time frame, and classify it

Input: Time slices of activity
Output: Identified Activity + Location <appl_i,...>

"""


def determine_location(time_slices, test_csv, phno):

    print "-" * stars
    print "Location Determination Process"
    print "-" * stars

    # Classify WiFi data and store the csv in df_w
    if phno == str(1):
        train_wifi_csv = 'CompleteDataSets/Apartment/SoundTrainingSet/Phone1/Wifi1.csv'
        # train_wifi_csv = ('CompleteDataSets/Apartment/'
                          # '23Sep - Wifi_TrainingSet/Wifi_1BHK_102A.csv')
    elif phno == str(2):
        # train_wifi_csv = 'CompleteDataSets/Apartment/SoundTrainingSet/Phone1/Wifi1.csv'
        train_wifi_csv = ('CompleteDataSets/Apartment/19Nov-WifiTrainingSet/Wifi2.csv')
    else:
        train_wifi_csv = ('CompleteDataSets/Apartment/23Sep-Wifi_TrainingSet/Wifi3.csv')
    exp_idx = 3

    df_w = cl.classify_location(train_wifi_csv, test_csv, apt_no, exp_idx)

    # Extracting features of the training set
    # train_feat_csv = cll.format_data(train_wifi_csv, "train", apt_no, exp_idx)
    # df_w = pd.read_csv(test_csv)
    # Test CSV to create
    # test_csv = 'Wifi/test_data/tmp/slice.csv'

    time_slices['start_time'] = time_slices['start_time'].astype('int')
    time_slices['end_time'] = time_slices['end_time'].astype('int')
    time_slices['location'] = np.zeros(len(time_slices))
    time_slices['pred_location'] = np.zeros(len(time_slices))

    location = []
    pred_location = []
    logger.debug("For Phone number %s using training file: %s", phno, train_wifi_csv)

    for row_idx in time_slices.index:
        # start_time = time_slices.ix[row_idx]['start_time'] * 1000
        # end_time = time_slices.ix[row_idx]['end_time'] * 1000

        # Create test csv for time slice
        # logger.debug("-" * 20)
        # wifi_df = df_w.ix[(df_w.time >= start_time) & (df_w.time <= end_time)]
        # wifi_df.to_csv(test_csv, index=False)
        # md_st = dt.datetime.fromtimestamp(start_time / 1000)
        # md_et = dt.datetime.fromtimestamp(end_time / 1000)
        # logger.debug("Found %d wifi events for the time", len(wifi_df))
        # logger.debug("slice[%s , %s ]", md_st, md_et)

        # if len(wifi_df.index) == 0:
        #     location.append("Not Found")
        #     pred_location.append("Not Found")
        # time_slices = time_slices[time_slices.index != row_idx]
        # else:
        # Store classified event
        #     sidx = wifi_df.index[0]
        #     location.append(wifi_df.ix[sidx]['label'])
        #     pred_location.append(cl.classify_location_slice(
        #         train_feat_csv, test_csv, apt_no, exp_idx))

        start_time = time_slices.ix[row_idx]['start_time']
        end_time = time_slices.ix[row_idx]['end_time']

        # Extract a time slice from the classified set
        logger.debug("-" * 20)
        df = df_w.ix[(df_w.timestamp >= start_time)
                     & (df_w.timestamp <= end_time)]
        md_st = dt.datetime.fromtimestamp(start_time)
        md_et = dt.datetime.fromtimestamp(end_time)
        logger.debug("Found %d wifi events for the time", len(df))
        logger.debug("slice[%s , %s ]", md_st, md_et)

        if len(df.index) == 0:
            location.append("Not Found")
            pred_location.append("Not Found")
            # time_slices = time_slices[time_slices.index != row_idx]
        else:
            # Set location as the one with maximum number of location label
            label = df.label[df.index[0]]  # Ground Truth label
            grpcount_label = df.groupby('pred_label')['pred_label'].count()
            logger.debug("Predictions:: %s", grpcount_label)
            pred_label = grpcount_label[grpcount_label == grpcount_label.max()].index[0]
            location.append(label)
            pred_location.append(pred_label)
            logger.debug("OLocation:: %s PLocation:: %s", label, pred_label)

    logger.debug("-" * 20)
    time_slices['location'] = location
    time_slices['pred_location'] = pred_location

    # print "Classified Location::\n", time_slices
    # For printing actual times
    print_location = time_slices.copy()

    print_location['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                        for i in time_slices['start_time']]
    print_location['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                      for i in time_slices['end_time']]
    print_location = print_location.sort(['start_time'])
    print "\nDetected activity ::\n", print_location

    return time_slices


def activity_detection(time_slices, test_csv, phno):

    print "-" * stars
    print "Sound Classification Process"
    print "-" * stars

    # For every time slice, classify sound
    train_sound_csv = ('CompleteDataSets/Apartment/'
                       'SoundTrainingSet/Phone' + phno + '_hp.csv')
    exp_idx = 3
    train_feat_csv = cs.extract_features(
        train_sound_csv, "train", apt_no, exp_idx)
    df_s = pd.read_csv(test_csv)

    # Test CSV to create
    test_csv = 'Sound/test_data/tmp/slice.csv'

    time_slices['sound'] = np.zeros(len(time_slices))
    time_slices['pred_sound'] = np.zeros(len(time_slices))
    sound = []
    pred_sound = []
    for row_idx in time_slices.index:
        start_time = time_slices.ix[row_idx]['start_time'] * 1000
        end_time = time_slices.ix[row_idx]['end_time'] * 1000

        # Create test csv for time slice
        logger.debug("-" * 20)
        sound_df = df_s.ix[(df_s.time >= start_time) & (df_s.time <= end_time)]
        sound_df = sound_df.sort(['time'])
        sound_df.to_csv(test_csv, index=False)
        md_st = dt.datetime.fromtimestamp(start_time / 1000)
        md_et = dt.datetime.fromtimestamp(end_time / 1000)
        logger.debug("Found %d sound events for the time", len(sound_df))
        logger.debug("slice[%s , %s ]", md_st, md_et)

        if len(sound_df.index) < 1024:
            # time_slices = time_slices[time_slices.index != row_idx]
            sound.append("Not Found")
            pred_sound.append("Not Found")
        else:
            # Store classified event
            sidx = sound_df.index[0]
            sound.append(sound_df.ix[sidx]['label'])
            pred_sound.append(cl.classify_sound(
                train_feat_csv, test_csv, apt_no, exp_idx))

    logger.debug("-" * 20)
    time_slices['sound'] = sound
    time_slices['pred_sound'] = pred_sound

    logger.debug("Classified Events::\n %s", time_slices)

    # Filter sound slices where the sound predicted is "Others"
    # there is no sound(silence) or or sound not in our list
    # to identify (voice, or some other sound)
    # TODO: Collect "Others" sound samples
    # time_slices = time_slices[time_slices.pred_sound != 'None']

    # detected_activity = combine_light_power_activity(time_slices)

    # For printing actual times
    print_activity = detected_activity.copy()

    print_activity['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                        for i in detected_activity['start_time']]
    print_activity['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                      for i in detected_activity['end_time']]
    print_activity = print_activity.sort(['start_time'])
    print "\nDetected activity ::\n", print_activity

    return detected_activity


def combine_light_power_activity(time_slices):

    # For every light time slice, search for power time slices which lie
    # within the time range of the light time slice and both time slices have the detected
    # same location and sound event. This tells that it is likely that the activity happened
    # in the same room
    # Result expected: "watching TV with lights and fans ON"

    light_ts = time_slices[time_slices.type == 'light']
    power_ts = time_slices[time_slices.type == 'power']
    activity_list = []
    window = 240
    if len(light_ts.index) > 0:
        for idx in light_ts.index:
            light_st = light_ts.ix[idx]['start_time']
            light_et = light_ts.ix[idx]['end_time']
            mod_power_ts = power_ts[
                (power_ts.start_time.isin(
                    range(light_st - window, light_st + window + 1))) &
                (power_ts.end_time.isin(
                    range(light_et - window, light_et + window + 1))) &
                (power_ts.pred_sound == light_ts.ix[idx]['pred_sound']) &
                (power_ts.pred_location == light_ts.ix[idx]['pred_location'])]
            if len(mod_power_ts) == 0:
                activity_list.append(light_ts[light_ts.index == idx])
            else:
                # print mod_power_ts.head()
                activity_list.append(mod_power_ts)
        detected_activity = pd.concat(activity_list)
        # Update with rest of the power time slices
        # i for idx in power_ts.index if idx not in detected_activity.index
        idx_list = []
        for i in power_ts.index:
            if i not in detected_activity.index:
                idx_list.append(i)
        power_list_ts = power_ts.ix[idx_list]
        detected_activity = pd.concat(activity_list + [power_list_ts])
    else:
        detected_activity = time_slices

    return detected_activity
"""
Metadata Used:
1. Appliance - Power Ratings - Location
2. Appliance - Phase - Location + Power Ratings

"""

# Phase - Appliance - Location Mapping
phase_loc = {
    "R": "Bedroom",
    "Y": "Dining Room",
    "B": "Kitchen"
}

# Appliance Power Ratings
metadata = {
    "appliance": ['AC', 'AC', 'Fan', 'Fan', 'TV', 'Microwave', 'Kettle'],
    "rating": [2500, 650, 50, 50, 60, 1100, 830],
    "location": (['Dining Room', 'Bedroom', 'Dining Room', 'Bedroom',
    'Dining Room', 'Kitchen', 'Kitchen']),
    "phase": ['R', 'B', 'Y', 'R', 'Y', 'B', 'B']
}

# Percent change for comparing appliance power with the magnitude of the predicted
# power consumption

per_change = 0.5


def correction_using_appl_phase_power(detected_activity):
    # Take the appliance power location mapping and correct the activity detection

    # md_df = pd.DataFrame(metadata, columns=['appliance', 'location', 'rating', 'phase'])
    md_df = pd.read_csv('Metadata/102A_metadata.csv')
    print "\nMetadata::\n", md_df

    no_ele = len(detected_activity.index)
    if no_ele > 0:
        mappl_l = defaultdict(list)
        for i in detected_activity.index:
            md_l = []
            appl = detected_activity.ix[i]['pred_sound']
            mag = math.fabs(detected_activity.ix[i]['magnitude'])
            loc = detected_activity.ix[i]['pred_location']
            phase = detected_activity.ix[i]['phase']

            # Extract metadata for the current appliance
            mdf = md_df[md_df.appliance == appl]
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                mdf_loc = mdf.ix[md_i]['location']
                mdf_phase = mdf.ix[md_i]['phase']

                max_md_power = md_power + 0.05 * md_power

                # Compare magnitude and metadata power draw
                power = md_power * per_change
                logger.debug("For Appl - %s(%s) :: [%s, md_power=%s]", appl, mag,
                             md_power - power, max_md_power)

                # Matching metadata with appliance power draw and location
                if (mag >= md_power - power and mag <= max_md_power and
                   mdf_loc == loc and mdf_phase == phase):
                    md_l.append(md_i)
            logger.debug(
                "Matched Indexes for Appliance %s with idx %d: %s", appl, i, md_l)
            mappl_l[i] = md_l
    logger.debug("Comparison results: %s\n", mappl_l.items())

    # Empty list entries indicate incorrect classification
    for idx, matched_mdidx_list in mappl_l.items():
        poss_appl = pd.DataFrame()
        if len(matched_mdidx_list) == 0:
            logger.debug("Correction Process starts...")
            appl = detected_activity.ix[idx]['pred_sound']
            mag = math.fabs(detected_activity.ix[idx]['magnitude'])
            loc = detected_activity.ix[idx]['pred_location']
            phase = detected_activity.ix[idx]['phase']

            logger.debug("Appliance %s with index %d and power %d", appl, idx, mag)

            # Extract metadata of appliances with the same location
            mdf = md_df[(md_df.location == loc) & (md_df.phase == phase)]

            # Choose the one with the closest power draw
            # compared to the predicted magnitude
            df_list = []
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                md_appl = mdf.ix[md_i]['appliance']

                max_md_power = md_power + 0.05 * md_power

                # Compare magnitude and metadata power draw
                power = md_power * per_change
                logger.debug("For Appl - %s :: %s md_power=%s", appl,
                             md_power - power, md_power)

                # Matching metadata with prediction
                if mag >= md_power - power and mag <= max_md_power:
                    md_power_diff = md_power - mag
                    df_list.append(
                        pd.DataFrame(
                            {'appl_index': idx, 'md_power_diff': md_power_diff}, index=[md_appl]))

            # Correcting appliance
            if len(df_list) == 1:
                detected_activity.ix[idx, 'pred_sound'] = df_list[0].index[0]
                logger.debug(
                    ".........Corrected Appliance %s with - %s",
                    appl, detected_activity.ix[idx]['pred_sound'])
            elif len(df_list) > 1:
                logger.debug(
                    "TODO: Still need to resolve: Using Feature Extraction in Meter data")
                logger.debug("For now, selecting the closest to the max power draw")
                poss_appl = pd.concat(df_list)
                poss_appl = poss_appl[
                    poss_appl.md_power_diff == poss_appl.md_power_diff.min()]
                detected_activity.ix[idx, 'pred_sound'] = poss_appl.index[0]
                logger.debug(
                    ".........Corrected Appliance %s with - %s",
                    appl, detected_activity.ix[idx]['pred_sound'])
            else:
                logger.debug("ERROR: No match found in the metadata! Recheck Algorithm")

    return detected_activity


def correction_using_appl_power(detected_activity):
    # Take the appliance power location mapping and correct the activity detection

    # md_df = pd.DataFrame(metadata, columns=['appliance', 'location', 'rating', 'phase'])
    md_df = pd.read_csv('Metadata/102A_metadata.csv')
    print "\nMetadata::\n", md_df

    no_ele = len(detected_activity.index)
    if no_ele > 0:
        mappl_l = defaultdict(list)
        for i in detected_activity.index:
            md_l = []
            appl = detected_activity.ix[i]['pred_sound']
            mag = math.fabs(detected_activity.ix[i]['magnitude'])
            loc = detected_activity.ix[i]['pred_location']

            # Extract metadata for the current appliance
            mdf = md_df[md_df.appliance == appl]
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                mdf_loc = mdf.ix[md_i]['location']

                max_md_power = md_power + 0.05 * md_power

                # Compare magnitude and metadata power draw
                power = md_power * per_change
                logger.debug("For Appl - %s(%s) :: [%s, md_power=%s]", appl, mag,
                             md_power - power, max_md_power)

                # Matching metadata with appliance power draw and location
                if mag >= md_power - power and mag <= max_md_power and mdf_loc == loc:
                    md_l.append(md_i)
            logger.debug(
                "Matched Indexes for Appliance %s with idx %d: %s", appl, i, md_l)
            mappl_l[i] = md_l
    logger.debug("Comparison results: %s\n", mappl_l.items())

    # Empty list entries indicate incorrect classification
    for idx, matched_mdidx_list in mappl_l.items():
        poss_appl = pd.DataFrame()
        if len(matched_mdidx_list) == 0:
            logger.debug("Correction Process starts...")
            appl = detected_activity.ix[idx]['pred_sound']
            mag = math.fabs(detected_activity.ix[idx]['magnitude'])
            loc = detected_activity.ix[idx]['pred_location']

            logger.debug("Appliance %s with index %d and power %d", appl, idx, mag)

            # Extract metadata of appliances with the same location
            mdf = md_df[md_df.location == loc]

            # Choose the one with the closest power draw
            # compared to the predicted magnitude
            df_list = []
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                md_appl = mdf.ix[md_i]['appliance']

                max_md_power = md_power + 0.05 * md_power

                # Compare magnitude and metadata power draw
                power = md_power * per_change
                logger.debug("For Appl - %s :: %s md_power=%s", appl,
                             md_power - power, max_md_power)

                # Matching metadata with prediction
                if mag >= md_power - power and mag <= max_md_power:
                    md_power_diff = md_power - mag
                    df_list.append(
                        pd.DataFrame(
                            {'appl_index': idx, 'md_power_diff': md_power_diff}, index=[md_appl]))

            # Correcting appliance
            if len(df_list) == 1:
                detected_activity.ix[idx, 'pred_sound'] = df_list[0].index[0]
                logger.debug(
                    "......Corrected Appliance %s with - %s",
                    appl, detected_activity.ix[idx]['pred_sound'])
            elif len(df_list) > 1:
                logger.debug(
                    "TODO: Still need to resolve: Using Feature Extraction in Meter data")
                logger.debug("For now, selecting the closest to the max power draw")
                poss_appl = pd.concat(df_list)
                poss_appl = poss_appl[
                    poss_appl.md_power_diff == poss_appl.md_power_diff.min()]
                detected_activity.ix[idx, 'pred_sound'] = poss_appl.index[0]
                logger.debug(
                    ".........Corrected Appliance %s with - %s",
                    appl, detected_activity.ix[idx]['pred_sound'])
            else:
                logger.debug("ERROR: No match found in the metadata! Recheck Algorithm")

    return detected_activity


"""
---------------------------------------
Main Program
---------------------------------------
"""

if __name__ == '__main__':

    # Get the sensor streams (test set)

    event_dir = sys.argv[1]
    l_phase = sys.argv[2]
    p_phase = sys.argv[3]
    # room=sys.argv[1]
    # r_type = sys.argv[2]
   # sound_path=sys.argv[4]
    # phno = sys.argv[2]
    # app = int(sys.argv[3])
    # event_dir = 'CompleteDataSets/Apartment/23Sep - Meter_Wifi_Sound_Accl/'
    # event_dir = 'CompleteDataSets/Apartment/23_9_16_53_23_9_19_11/'
    power_csv = event_dir + 'Power.csv'
    # sys.argv[2]
    light_csv = event_dir + 'Light.csv'
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
    print df_l
    db_paths='/media/New Volume/IPSN/sense/5_6Nov/dbs/'
    dbs=['c003_room_db.csv','c004_room_db.csv','c005_room_db.csv','c006_room_db.csv']
    db_rooms='_room_db.csv'
    available_phase_l=['R','Y','B']
    #df_s = pd.read_csv(sound_csv)
    # Step i: Apply moving average to the power data
    # df_l = average_power(df_l)
    # df_p = average_power(df_p)

    # Step1: Edge Detection 


    #edge_list_df = edge_detection(df_l, df_p, l_phase, p_phase)
    #sys.exit()

    # Step2: Edge Matching
    #time_slices = edge_matching(df_l, df_p, edge_list_df)

    # df_b=DataFrame()
    # df_a=DataFrame()
    # print 'start time to next 1 minutes:'
    # for t in time_slices.index[:-1]:
    #     from_t=int(time_slices.ix[t]['start_time'])*1000
    #     end_t=int(time_slices.ix[t]['end_time'])*1000
    #     room=[]
    #     #Considering Phase and Magnitude Information
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
            
    #         #print 'Filtering sound samples before event'
    #         db_list=list_db.index[(list_db['time']>= from_t-60000) & (list_db['time']<= from_t)]
    #         db_list_before=list_db.ix[db_list]
    #         #print "before data" 
    #         #print db_list_before

    #         #print 'Filtering sound samples between events'
    #         db_list=list_db.index[(list_db['time']>=from_t) & (list_db['time'] <= end_t)]
    #         db_list_between=list_db.ix[db_list]
           

    #         #print 'Filtering sound events after event'
    #         db_list=list_db.index[(list_db['time']>=from_t) & (list_db['time'] <= from_t+60000)]
    #         db_list_after=list_db.ix[db_list]
    #         #print 'after data'
    #         #print db_list_after


    #         df_before=DataFrame(db_list_before['db'],columns=['Before_'+db+'_'+str(time_slices.ix[t]['start_time'])])
    #         df_after=DataFrame(db_list_after['db'],columns=['After_'+db+'_'+str(time_slices.ix[t]['end_time'])])
    #         df_between=DataFrame(db_list_between['db'],columns=['Between_'+db+'_'+str(time_slices.ix[t]['start_time'])+'_'+str(time_slices.ix[t]['end_time'])])
            
    #         #t_before = np.array([dt.datetime.fromtimestamp(x/1000) for (x) in db_list_before['time']])
    #         #t_after = np.array([dt.datetime.fromtimestamp(x/1000) for (x)in db_list_after['time']])
    #         state=''

    #         if df_before.empty == False:
                
    #             state+='T'
    #             #df_b=pd.concat([df_b,df_before],ignore_index=True)
    #         else:
                
    #             state+='F'
    #         if df_after.empty == False:
               
    #             state+='T'
    #             #df_b=pd.concat([df_b,df_after],ignore_index=True)
    #         else:
                
    #             state+='F'

    #         #print state
            

    #         if (db_list_between.empty == False) :
    #             print 'True' 
    #             sd=[]
    #             beg=0           
        
    #             #next=db_list_between['time']
    #             print 'Beginning time'
    #             print long(beg)
    #             sd_prev=0.0
    #             sd_curr=0.0
    #             diff_t=0
    #             print len(db_list_between)
    #             for db in db_list_between.index[:-1]:
    #                 next=db_list_between.ix[db]['time']
            
    #                 diff_t=long(next)-long(beg)
    #                 #print diff_t
    #                 sd.append(db_list_between.ix[db]['db'])
    #                 if diff_t<120000:
    #                     continue
    #                 beg=next
                
    #                 sd_prev=sd_curr
    #                 #print "previous sd:"+str(sd_prev)
    #                 frame=DataFrame(sd,columns=['db_sd'])
    #                 #print frame
    #                 sd_curr=float(frame.std())
    #                 #print "current sd:"+str(sd_curr)
    #                 diff=abs(sd_curr-sd_prev)
    #                 if (diff>4):
    #                     print 'Time found'
    #                     print long(db_list_between.ix[db]['time'])

    #                 prev=next
                #print "previous time:"+str(prev)
             
        

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
            #     # row1
            #     ax1 = plt.subplot(3, 1, 1)
            #      # ax1.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
            #     plt.plot( df_before)
            #     plt.title("Sound Before_ " +str(time_slices.ix[t]['start_time'])+'_'+db )
            #     plt.ylabel("Sound in db")


            #     ax2 = plt.subplot(3, 1, 2)
            #     # ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
            #     plt.plot( df_between)
            #     plt.title("Sound Between_ " + str(time_slices.ix[t]['start_time'])+'_'+str(time_slices.ix[t]['end_time'])+'_'+db)
            #     plt.ylabel("Sound in db")

            #     ax3 = plt.subplot(3, 1, 3)
            #     # ax2.xaxis.set_major_formatter( md.DateFormatter('%Y-%m-%d',timezone(TIMEZONE)))
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
            #fig.clear()
            # print 'Plotting'
            # try:
            #     db_list_before['db'].plot(ax=axes[0,i]); axes[0,i].set_title('Before_'+db+'_'+str(t))
            #     db_list_after['db'].plot(ax=axes[0,i]); axes[1,i].set_title('After_'+db+'_'+str(t))
            # except Exception,e:
            #     print e
            #i+=1
            # fig = plt.figure()
            # plt.subplots_adjust(bottom=0.2)
            # plt.xticks( rotation=25 )
            # ax1=plt.gca()
            # ax1=fig.add_subplot(1,1,1)
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # ax1.xaxis.set_major_formatter(xfmt)
            # act_time = [dt.datetime.fromtimestamp(float(i)/1000) for i in db_list_before['time']]
            # #dataframe_before.plot(ax=ax1,style='k-',label='before event')  
            # plt.plot(act_time,db_list_before['db'],label='before event')

            
            
            # ax3=fig.add_subplot(2,1,2)
            # #dataframe_after.plot(ax=ax3,style='k-',label='after event') 
            # act_time2 = [dt.datetime.fromtimestamp(float(i)/1000) for i in db_list_after['time']]
            # plt.plot(act_time2,db_list_after['db'],label='after event') 
            # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # ax1.xaxis.set_major_formatter(xfmt)  

            # # ax2=fig.add_subplot(1,2,1)
            
            # # #dataframe_between.plot(ax=ax2,style='k-',label='during event')  
            # # act_time1 = [dt.datetime.fromtimestamp(float(i)/1000) for i in db_list_between['time']]
            # # plt.plot(act_time1,db_list_between['db'],label='between event')  
            # # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            # # ax1.xaxis.set_major_formatter(xfmt)  

            # plt.show()
            # #print 'Saving figure at'+db_paths+'_'+str(t)
            # #plt.savefig(db_paths+'_'+str(t)+'.png')
        #print 'writing to csv'
        #print df_b
        #df_b.to_csv('/media/New Volume/IPSN/sense/5_6Nov/dbs/'+str(time_slices.ix[t]['start_time'])+'.csv')
        
    #print df_b
    #print df_a 
        
    
    
    #df_a.to_csv('/media/New Volume/IPSN/sense/5_6Nov/dbs/db_after_filtered.csv')
	
		

        
 
            

 #   for i in time_slices.start_time :	
	# f_time=i
	# t_time=i-60000
	# time=range(f_time,t_time)
 #    	sound_before=Series[x.values[2] for x.time in time]
    	
	
    



    """
    # Step3: Localization
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
