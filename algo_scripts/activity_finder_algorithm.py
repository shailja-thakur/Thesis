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
import time
import datetime as dt
from collections import defaultdict
import numpy as np
import pandas as pd
import classifier as cl
import classify_sound as cs
import evaluate as ev
import warnings
import logging
import meter_only as mo

# Disable warnings
warnings.filterwarnings('ignore')

# Enable Logging
logger = logging.getLogger('energy-lens')
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
pthresmin = 60  # for power meter
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
                row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
                return edge_type, row
            # or curr_next_diff > lthresmin:
            elif (curr_next_diff >= per_lthresmin):
                logger.debug("Here1 Index:: %d", i)
                # Storing the rising edge e_i = (time_i, mag_i)
                row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
                return edge_type, row
            else:
                pass
        else:
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
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
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
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
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
            return edge_type, row
        elif curr_next_diff < lthresmin or curr_next_diff >= lthresmin:
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
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
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
            return edge_type, row
        if next_missing_sample and curr_next_diff > pthresmin:
            logger.debug("Missing Sample:: Index %s", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
            return edge_type, row
        elif (curr_next_diff > per_pthresmin):
            logger.debug("Here1 Index:: %s", i)
            # Storing the rising edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
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
        row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
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
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
            return edge_type, row
        elif curr_next_diff < pthresmin or curr_next_diff > pthresmin:
            logger.debug("Falling Edge2:: Index %s", i)
            # Storing the falling edge e_i = (time_i, mag_i)
            row = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
            return edge_type, row
        else:
            pass
    return "Not an edge", {}


def filter_edges(rise_df, fall_df, winmin, thres):

    # Removing duplicate indexes
    rise_df["index"] = rise_df.index
    rise_df.drop_duplicates(cols='index', take_last=True, inplace=True)
    del rise_df["index"]

    fall_df["index"] = fall_df.index
    fall_df.drop_duplicates(cols='index', take_last=True, inplace=True)
    del fall_df["index"]

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
            logger.debug("Inside Index j: %s New Mag: %s ", sel_idx, new_mag)
            logger.debug("Rise time:: %s Rise Time (2): %s",
                         dt.datetime.fromtimestamp(rise_df.ix[ix_i]['time']),
                         dt.datetime.fromtimestamp(rise_df.ix[sel_idx]['time']))
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
            if(ix_i == ix_j) or ix_i > ix_j:
                continue
            elif (tmp.ix[ix_i]['ts'] == tmp.ix[ix_j]['ts'] and
                  tmp.ix[ix_i]['magnitude'] > tmp.ix[ix_j]['magnitude'] and
                  tmp.ix[ix_j]['magnitude'] == -tmp.ix[ix_i]['magnitude']):
                logger.debug("Index i:: %s", ix_i)
                logger.debug("Removing %s %s", ix_i, ix_j)
                if ix_i in rise_df.index:
                    rise_df = rise_df.drop(ix_i)
                fall_df = fall_df.drop(ix_j)

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

        if idx == tmp_df.index[-1]:
            if idx not in idx_list:
                idx_list.append(idx)
            break
        t = tmp_df.ix[idx]['time']
        t_next = tmp_df.ix[tmp_df.index[i + 1]]['time']
        diff = t_next - t
        if diff <= 60:
            t_mag = tmp_df.ix[idx]['magnitude']
            t_next_mag = tmp_df.ix[tmp_df.index[i + 1]]['magnitude']
            t_curr_power = math.fabs(tmp_df.ix[idx]['curr_power'])
            t_next_curr_power = math.fabs(tmp_df.ix[tmp_df.index[i + 1]]['curr_power'])

            curr_diff = math.fabs(t_next_curr_power - t_curr_power)

            if curr_diff < 0.5 * thres:
                if t_mag <= 60:
                    threshold = 0.2
                elif t_mag >= 1000:
                    print "in"
                    threshold = 0.25
                else:
                    threshold = 0.1
                if math.fabs(t_mag - t_next_mag) <= threshold * t_mag:
                    idx_list.append(tmp_df.index[i + 1])
                    if idx in idx_list:
                        idx_list.remove(idx)
                else:
                    if idx not in idx_list:
                        idx_list.append(idx)
            else:
                if idx not in idx_list:
                    idx_list.append(idx)
        else:
            if idx not in idx_list:
                idx_list.append(idx)
    print "Rise idx_list", idx_list
    tmp_df = tmp_df.ix[idx_list].sort(['time'])
    rise_df = tmp_df.ix[:, :-1]

    # Falling Edges
    tmp_df = fall_df.copy()
    # tmp_df['ts'] = (fall_df.time / 100).astype('int')
    tmp_df['tmin'] = [str(dt.datetime.fromtimestamp(i).hour) + '-' +
                      str(dt.datetime.fromtimestamp(i).minute) for i in tmp_df.time]

    # Select the edge with the maximum timestamp lying within a minute
    idx_list = []
    for i, idx in enumerate(tmp_df.index):
        # if idx in idx_list:
        #     continue
        if idx == tmp_df.index[-1]:
            if idx not in idx_list:
                idx_list.append(idx)
            break
        t = tmp_df.ix[idx]['time']
        t_next = tmp_df.ix[tmp_df.index[i + 1]]['time']
        diff = t_next - t
        if diff <= 60:
            t_mag = math.fabs(tmp_df.ix[idx]['magnitude'])
            t_next_mag = math.fabs(tmp_df.ix[tmp_df.index[i + 1]]['magnitude'])
            t_curr_power = math.fabs(tmp_df.ix[idx]['curr_power'])
            t_next_curr_power = math.fabs(tmp_df.ix[tmp_df.index[i + 1]]['curr_power'])

            curr_diff = math.fabs(t_next_curr_power - t_curr_power)

            # print "\n idx ", idx, "now time", dt.datetime.fromtimestamp(t),
            # print "next time", dt.datetime.fromtimestamp(t_next)
            # print "t_mag", t_mag, "t_next_mag", t_next_mag
            # print "t_curr_power", t_curr_power, "t_next_curr_power", t_next_curr_power
            # print "curr_diff", curr_diff
            if curr_diff == 0:
                idx_list.append(tmp_df.index[i + 1])
                if idx in idx_list:
                    idx_list.remove(idx)
                # print "idx in", idx, "next idx", tmp_df.index[i + 1], idx_list
            elif curr_diff < 0.5 * thres:
                if t_mag <= 60:
                    threshold = 0.2
                elif t_mag >= 1000:
                    print "in"
                    threshold = 0.25
                else:
                    threshold = 0.1
                if math.fabs(t_mag - t_next_mag) <= threshold * t_mag:
                    idx_list.append(tmp_df.index[i + 1])
                    if idx in idx_list:
                        idx_list.remove(idx)
                    # print "idx in", idx, "next idx", tmp_df.index[i + 1], idx_list
                else:
                    # print "idx out", idx
                    if idx not in idx_list:
                        idx_list.append(idx)
            else:
                if idx not in idx_list:
                    idx_list.append(idx)

        else:
            if idx not in idx_list:
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


def filter_apt_edges(rise_df, fall_df, apt_no, etype, df_p):

    # Removing duplicate indexes
    rise_df["index"] = rise_df.index
    rise_df.drop_duplicates(cols='index', take_last=True, inplace=True)
    del rise_df["index"]

    fall_df["index"] = fall_df.index
    fall_df.drop_duplicates(cols='index', take_last=True, inplace=True)
    del fall_df["index"]

    # For power edges
    if etype == 'power':

        # Filter 1
        # Filtering out edges with magnitude between 130 to 150
        # Rising Edges
        idx_list = []
        for i in rise_df.index:
            magnitude = rise_df.ix[i]['magnitude']
            time = rise_df.ix[i]['time']
            if apt_no == '603':
                # Likely power consumption of fridge is 110-150
                if magnitude >= 110 and magnitude <= 150:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
            elif apt_no == '703':
                if magnitude >= 130 and magnitude <= 190:
                    idx_list.append(i)
                # 26-27Nov
                # if time in range(1385470967, 1385471880):
                #     idx_list.append(i)
                # 28-29Nov
                if time in range(1385646060, 1385648144):
                    idx_list.append(i)
        rise_df = rise_df.ix[rise_df.index - idx_list]

        # Falling Edges
        idx_list = []
        for i in fall_df.index:
            magnitude = math.fabs(fall_df.ix[i]['magnitude'])
            time = math.fabs(fall_df.ix[i]['time'])
            if apt_no == '603':
                # Likely power consumption of fridge is 110-150
                if magnitude >= 110 and magnitude <= 150:
                    print "idx", i, "magnitude", magnitude
                    idx_list.append(i)
            elif apt_no == '703':
                if magnitude >= 130 and magnitude <= 170:
                    idx_list.append(i)
                # 26-27Nov
                # if time in range(1385470967, 1385471880):
                #     idx_list.append(i)
                # 28-29Nov
                if time in range(1385646060, 1385648144):
                    idx_list.append(i)
        fall_df = fall_df.ix[fall_df.index - idx_list]

        # Filter 2 - removing RO edges
        print "\nFiltering process started.....\n"
        edge_total_list = pd.concat([rise_df, fall_df])
        edge_total_list = edge_total_list.sort(['time'])
        edge_index = edge_total_list.index
        # List containing indexes to remove
        rise_idx_list = []
        fall_idx_list = []
        for i, idx in enumerate(edge_total_list.index):
            now_edge = edge_total_list.ix[idx]['time']
            now_mag = edge_total_list.ix[idx]['magnitude']
            if i + 1 not in range(0, len(edge_index)):
                # Reached the end
                if now_mag < 0 and int(math.fabs(now_mag)) in range(60, 75):
                    # If diff power between curr and prev power last 20 seconds
                    # is similar to the fall mag then select edge
                    curr_power = edge_total_list.ix[idx]['curr_power']
                    for j in range(20, 26):
                        # Find prev power
                        prev_time = now_edge - j
                        row_df = df_p[df_p.time == prev_time]
                        if len(row_df) > 0:
                            prev_sec_power = row_df.ix[row_df.index[0]]['power']
                            diff_power = int(curr_power - prev_sec_power)
                            if now_mag in range(int(diff_power) - 2, int(diff_power) + 3):
                                fall_idx_list.append(idx)
                continue

            next_edge = edge_total_list.ix[edge_index[i + 1]]['time']
            next_mag = edge_total_list.ix[edge_index[i + 1]]['magnitude']
            diff = int(math.fabs(next_edge)) - int(now_edge)

            if ((now_edge < 0 and int(math.fabs(now_mag)) in range(60, 75))
               or now_edge == 1385383788 or next_mag == 1385383788):
                print "\nNow time", dt.datetime.fromtimestamp(now_edge)
                print "Next time", dt.datetime.fromtimestamp(next_edge)
                print "Now mag", now_mag, "next mag", next_mag
                print "Diff", diff

            # Its a falling edge and magnitude is b/w 60 - 70 and previous edge was a rising edge
            if (next_mag < 0 and int(math.fabs(next_mag)) in range(60, 75)
               and now_edge > 0 and diff in range(20, 30)
               and int(now_mag * 0.1) == int(math.fabs(next_mag) * 0.1)):
                # Removing both edges
                rise_idx_list.append(idx)
                fall_idx_list.append(edge_index[i + 1])
            # If rising edge was not detected
            elif now_mag < 0 and int(math.fabs(now_mag)) in range(60, 75):
                print "\nNow time", dt.datetime.fromtimestamp(now_edge)
                print "Next time", dt.datetime.fromtimestamp(next_edge)
                print "Now mag", now_mag, "next mag", next_mag
                print "Diff", diff

                # If diff power between curr and prev power last 20 seconds
                # is similar to the fall mag then select edge
                curr_power = edge_total_list.ix[idx]['curr_power']
                for j in range(20, 30):
                    # Find prev power
                    prev_time = now_edge - j
                    row_df = df_p[df_p.time == prev_time]

                    if len(row_df) > 0:

                        prev_sec_power = row_df.ix[row_df.index[0]]['power']
                        diff_power = int(curr_power - prev_sec_power)
                        if (int(math.fabs(now_mag)) in range(diff_power - 2, diff_power + 3)):
                            print "j=", j
                            print "curr_power", curr_power
                            print "prev_time", prev_time
                            print "row", row_df
                            print "prev_sec_power", prev_sec_power, 'diff_power', diff_power
                            fall_idx_list.append(idx)
        rise_idx_list = list(set(rise_idx_list))
        fall_idx_list = list(set(fall_idx_list))
        # Removing selected edges
        rise_df = rise_df.ix[rise_df.index - rise_idx_list]
        fall_df = fall_df.ix[fall_df.index - fall_idx_list]

    return rise_df, fall_df


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

    for i in range(1, ix_list_l[-1] - lwinmin + 1):
        edge_type, result = check_if_light_edge(df_l, i, "lightpower")
        if edge_type == "falling":
            f_row_list.append(result)
        elif edge_type == "rising":
            r_row_list.append(result)
        else:
            pass

    rising_edges_l_df = pd.DataFrame(
        r_row_list, columns=['index', 'time', 'magnitude', 'curr_power'])
    rising_edges_l_df = rising_edges_l_df.set_index('index', drop=True)

    falling_edges_l_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude', 'curr_power'])
    falling_edges_l_df = falling_edges_l_df.set_index('index', drop=True)

    # Adding the actual times to the frame
    rising_edges_l_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in rising_edges_l_df.time]
    falling_edges_l_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in falling_edges_l_df.time]

    print "Rising Edges::\n", rising_edges_l_df
    print "Falling edges::\n", falling_edges_l_df

    rising_edges_l_df, falling_edges_l_df = filter_edges(
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
    for i in range(1, ix_list_p[-1] - pwinmin + 1):
        edge_type, result = check_if_power_edge(df_p, i, "power")
        if edge_type == "falling":
            f_row_list.append(result)
        elif edge_type == "rising":
            r_row_list.append(result)
        else:
            pass

    rising_edges_p_df = pd.DataFrame(
        r_row_list, columns=['index', 'time', 'magnitude', 'curr_power'])
    rising_edges_p_df = rising_edges_p_df.set_index('index', drop=True)

    falling_edges_p_df = pd.DataFrame(
        f_row_list, columns=['index', 'time', 'magnitude', 'curr_power'])
    falling_edges_p_df = falling_edges_p_df.set_index('index', drop=True)

    # Adding the actual times to the frame
    rising_edges_p_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in rising_edges_p_df.time]
    falling_edges_p_df['act_time'] = [
        dt.datetime.fromtimestamp(int(t)) for t in falling_edges_p_df.time]

    print "Rising Edges::\n", rising_edges_p_df
    print "Falling Edges::\n", falling_edges_p_df

    rising_edges_p_df, falling_edges_p_df = filter_edges(
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


def make_edge_list(rise_l_df, fall_l_df, rise_p_df, fall_p_df):

    # Generating edge list
    edge_list_df = []
    list_df = [rise_l_df, fall_l_df, rise_p_df, fall_p_df]
    for i_no, ip_df in enumerate(list_df):
        logger.debug("Edge List Df No:%s\n" + "-" * stars, i_no + 1)

        if len(ip_df.index) == 0:
            df = pd.DataFrame(columns=['time', 'magnitude', 'curr_power',
                                       'act_time'])
            edge_list_df.append(df)
        else:
            ip_df['act_time'] = [dt.datetime.fromtimestamp(int(t)) for t in ip_df.time]
            edge_list_df.append(ip_df)
        # logger.debug("\n OUTDF:: \n", edge_list_df)
    print "\n Filtered Correlated Edge List:: \n", edge_list_df

    return edge_list_df


def edge_detection(df_l, df_p, apt_no):

    print "Using Light Threshold::", lthresmin
    print "Using Power Threshold::", pthresmin

    rising_edges_l_df, falling_edges_l_df = generate_light_edges(df_l)
    rising_edges_p_df, falling_edges_p_df = generate_power_edges(df_p)

    # Filter edges from the apartment
    if apt_no != '102A':
        rising_edges_l_df, falling_edges_l_df = filter_apt_edges(
            rising_edges_l_df, falling_edges_l_df, apt_no, 'light', df_l)
        rising_edges_p_df, falling_edges_p_df = filter_apt_edges(
            rising_edges_p_df, falling_edges_p_df, apt_no, 'power', df_p)

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

phases = {1: "R", 2: "Y", 3: "B"}


def get_phase(idx, df, meter_type):

    phase = 0
    if meter_type == 'light':
        logger.debug("Looking for light edge for index - %d", idx)
    else:
        logger.debug("Looking for power edge for index - %d", idx)

    # Check for edge phase in every phase stream
    for j in range(1, 4):
        if meter_type == 'light':
            edge_type, result = check_if_light_edge(
                df, idx, "lightphase" + str(j))
        else:
            edge_type, result = check_if_power_edge(
                df, idx, "powerphase" + str(j))
        if edge_type == "Not an edge":
            logger.debug(
                "Not a edge in %s for index %s", phases[j], idx)
        else:
            logger.debug(
                "Found edge in phase %s for index %s", phases[j], idx)
            phase = phases[j]
            # loc_list.append(phase_loc[phases[j]])
            break
        if j == 3:
            # Find edge in the previous entry
            flag = False
            for j in range(1, 4):
                if meter_type == 'light':
                    edge_type, result = check_if_light_edge(
                        df, idx - 1, "lightphase" + str(j))
                else:
                    edge_type, result = check_if_power_edge(
                        df, idx, "powerphase" + str(j))
                if edge_type == "Not an edge":
                    logger.debug(
                        "Not a edge in %s for index %s", phases[j], idx)
                    flag = True
                else:
                    logger.debug(
                        "Found edge in phase %s for index %s", phases[j], idx)
                    phase = phases[j]
                    flag = False
                    break
                    # loc_list.append(phase_loc[phases[j]])
            if flag:
                # Find in the next entry
                flag = False
                for j in range(1, 4):
                    if meter_type == 'light':
                        edge_type, result = check_if_light_edge(
                            df, idx, "lightphase" + str(j))
                    else:
                        edge_type, result = check_if_power_edge(
                            df, idx, "powerphase" + str(j))
                    if edge_type == "Not an edge":
                        logger.debug(
                            "Not a edge in %s for index %s", phases[j], idx)
                        flag = True
                    else:
                        logger.debug(
                            "Found edge in phase %s for index %s", phases[j], idx)
                        phase = phases[j]
                        flag = False
                        break
                        # loc_list.append(phase_loc[phases[j]])
                if flag:
                    phase = "Not Found"
                    # loc_list.append("Not found")
    return phase


def get_phase_information(df_l, df_p, edge_list):
    """
    Find the phase information for all filtered rising and falling edges
    """
    for i, df in enumerate(edge_list):

        df['phase'] = pd.Series(np.zeros(len(df)))

        # Checking for light edges
        if i in [0, 1]:
            logger.debug("Finding phases for light edges......")
            logger.debug("-" * stars)
            # Get indexes
            idx_list = df.index
            phase_list = []
            # loc_list = []
            for idx in idx_list:
                phase_list.append(get_phase(idx, df_l, "light"))

            # logger.debug("Length of phase list:: %s", len(phase_list))
            # logger.debug("Length of data frame:: %s", len(df))
            # logger.debug("Length of loc list:: %s", len(loc_list))

            df['phase'] = phase_list

        # Checking for power edges
        else:
            logger.debug("Finding phases for power edges......")
            logger.debug("-" * stars)
            # Get indexes
            idx_list = df.index
            # loc_list = []
            phase_list = []
            for idx in idx_list:
                phase_list.append(get_phase(idx, df_p, "power"))

            # logger.debug("Length of phase list:: %s", len(phase_list))
            # logger.debug("Length of data frame:: %s", len(df))
            df['phase'] = phase_list
        logger.debug("Index: %s Frame: \n %s", i, df)

    return edge_list


def make_pairs(rise_fall_dict, rise_df, fall_df):

    row_df_list = []
    new_rel_idx = rise_fall_dict

    for fall_idx, rise_idx_list in new_rel_idx.items():
        # Processing for falling edges where rising edges exist
        if len(rise_idx_list) > 0:

            logger.debug(
                "Fall_index: %s Rise index: %s", fall_idx, rise_idx_list)
            logger.debug("-" * stars)

            # Falling edge
            f_time = int(fall_df.ix[fall_idx]['Event_Time'])
            f_mag = (-1) * fall_df.ix[fall_idx]['Magnitude']
            f_phase = fall_df.ix[fall_idx]['Phase']
            logger.debug("Fall_time %s", dt.datetime.fromtimestamp(f_time))

            phase_mag_min_diff = (rise_idx_list[0], math.fabs(
                f_mag - rise_df.ix[rise_idx_list[0]]['Magnitude']),
                rise_df.ix[rise_idx_list[0]]['Phase'])
            print "First min", phase_mag_min_diff
            for rid in rise_idx_list:
                if rid != 0:
                    r_mag = int(rise_df.ix[rid]['Magnitude'])
                    diff = math.fabs(f_mag - r_mag)
                    r_phase = rise_df.ix[rid]['Phase']
                    # if app in [4, 5]:
                    #     if phase_mag_min_diff[1] > diff and f_phase == r_phase:
                    #         phase_mag_min_diff = (rid, diff, r_phase)
                    #         print "New min", phase_mag_min_diff
                    # else:
                    if phase_mag_min_diff[1] > diff:
                        phase_mag_min_diff = (rid, diff, r_phase)
                        print "New min", phase_mag_min_diff
            # Taking the rising edge which is the closest to the fall magnitude
            r_index = phase_mag_min_diff[0]
            r_time = int(rise_df.ix[r_index]['Event_Time'])
            logger.debug("Rise Index:: %s", r_index)
            logger.debug("Rise Time: %s", dt.datetime.fromtimestamp(r_time))

            # Creating edge pair entry
            row_df = pd.DataFrame(
                {"start_time": rise_df.ix[r_index]['Event_Time'],
                 "end_time": f_time,
                 "magnitude": f_mag,
                 "phase": f_phase},
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


def edge_matching( rise_df, fall_df):

    # Output

    time_slices = pd.DataFrame(
        columns=['start_time', 'end_time', 'magnitude', 'phase'])

    # Find the phase information for all filtered rising and falling edges
    # if app == 4:
    
    #edge_list = get_phase_information(df_l, df_p, edge_list)

    # TODO: Filter out voltage fluctuations seen across multiple phases

    # Find the time slices (start and end time of activity) by matching
    # based on magnitude of the edges

        # for rising and its corresponding falling edge
        # l_power = 5
        # p_power = 100

    # rise_l_df = edge_list[0].reset_index(drop=True)
    # fall_l_df = edge_list[1].reset_index(drop=True)
    # rise_p_df = edge_list[2].reset_index(drop=True)
    # fall_p_df = edge_list[3].reset_index(drop=True)

    # if app in [4, 5, 7]:
    #     rise_l_df = rise_l_df[rise_l_df.phase != "Not Found"].reset_index(drop=True)
    #     fall_l_df = fall_l_df[fall_l_df.phase != "Not Found"].reset_index(drop=True)
    #     rise_p_df = rise_p_df[rise_p_df.phase != "Not Found"].reset_index(drop=True)
    #     fall_p_df = fall_p_df[fall_p_df.phase != "Not Found"].reset_index(drop=True)

    print "-" * stars
    print "Edge Matching Process"
    print "-" * stars

    # For every falling edge, find the matching rising edge
    full_df_list = []
    # for k in [0, 1]:
    #     if k == 0:
    #         rise_df = rise_l_df
    #         fall_df = fall_l_df
    #         # power = l_power
    #         edge_type = "light"
    #         logger.debug("Matching for light edges......")
    #         logger.debug("-" * stars)
    #     else:
    #         rise_df = rise_p_df
    #         fall_df = fall_p_df
    #         # power = p_power
    #         edge_type = "power"
    #         logger.debug("Matching for power edges......")
    #         logger.debug("-" * stars)

    # Filter 1: Filter falling edges where it is before any rising edge
    # cols = ['time', 'time_sec', 't_meter','magnitude', 'label', 'pred_label']
    re_l_idx = [np.where(rise_df.Event_Time < i)[0] for i in fall_df.Event_Time]
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
                fall_mag = math.fabs(int(fall_df.ix[i]['Magnitude']))
                print fall_mag
                fall_phase = fall_df.ix[i]['Phase']
                # Set power threshold for magnitude comparison between
                # rising and falling
                power = fall_mag * percent_change
                logger.debug("Idx %d %s fall_mag=%s %s fall phase:%s", i,
                             fall_mag - power, fall_mag, fall_mag + power, fall_phase)
                # For debugging - printing the rising edges corres. to falling
                # edges
                restr = ''
                for idx in row:
                    restr += ' ' + str(rise_df.ix[idx]['Magnitude'])
                logger.debug("Rising edges::%s", restr)
                # Generating matching rising edge list for the falling edge i
                # if app in [4, 5]:
                #     logger.debug("Using Phase for time slice generation")
                #     re_l = [idx for idx in row if rise_df.ix[idx]['Magnitude'] >= fall_mag - power
                #             and rise_df.ix[idx]['Magnitude'] <= fall_mag + power and
                #             rise_df.ix[idx]['Phase'] == fall_phase]
                # else:
                re_l = [idx for idx in row if rise_df.ix[idx]['Magnitude'] >= fall_mag - power
                            and math.fabs(int(rise_df.ix[idx]['Magnitude'])) <= fall_mag + power]
                logger.debug("Matched Rising Edges with fidx %d: %s", i, re_l)
            new_rel_idx[i] = re_l
    logger.debug("Filter 2 results: %s", new_rel_idx.items())

    row_df_list = make_pairs(new_rel_idx, rise_df, fall_df)
    full_df_list = full_df_list + row_df_list

    # logger.debug("\nMatched Edges: \n%s", full_df_list)
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
    return time_slices


def filter_time_slices(time_slices, apt_no, exp_no):
    """
    Remove fridge events and events where the duration is less than 30 seconds
    """
    # Removing the extraneous time slices
    if apt_no == '102A' and exp_no == '3':
        discard_ts = time_slices[
            (time_slices.phase == 'Not Found') & (time_slices.magnitude < 100)]
        time_slices = time_slices.ix[time_slices.index - discard_ts.index]

    elif apt_no == '603':
        print "here"
        # Likely power consumption of fridge is 110-150
        time_slices = time_slices[(time_slices.magnitude < 110) | (time_slices.magnitude > 150) &
                                  (time_slices.type == 'power')]
        # 25-26Nov
        if exp_no == '25-26Nov':
            time_slices = time_slices[time_slices.end_time < 1385404505]
        elif exp_no == '26-27Nov':
            time_slices = time_slices[time_slices.end_time < 1385492334]

    elif apt_no == '703':
        # Likely power consumption of fridge is 130-152
        fridge_ts = time_slices[(time_slices.magnitude >= 130) & (time_slices.magnitude <= 170) &
                                (time_slices.type == 'power')]
        time_slices = time_slices.ix[time_slices.index - fridge_ts.index]

        # Likely power consumption of geyser > 2000 but on light phase > 1000
        geyser_ts = time_slices[(time_slices.magnitude > 1000) & (time_slices.type == 'light')]
        time_slices = time_slices.ix[time_slices.index - geyser_ts.index]

        # 26-27Nov
        if exp_no == '26-27Nov':
            washing_ts = time_slices[
                (time_slices.start_time >= 1385470967) & (time_slices.end_time <= 1385471880)]
            time_slices = time_slices.ix[time_slices.index - washing_ts.index]

        # 28-29Nov
        if exp_no == '28-29Nov':
            time_slices = time_slices[
                (time_slices.start_time < 1385646060) | (time_slices.end_time > 1385648143)]

    # Removing time slices with duration less than 30 seconds
    idx_list = []
    for idx in time_slices.index:
        start_time = time_slices.ix[idx]['start_time']
        end_time = time_slices.ix[idx]['end_time']
        magnitude = time_slices.ix[idx]['magnitude']

        time_diff = end_time - start_time

        if time_diff < 30 and magnitude < 80:
            print "idx", idx, "time_diff", time_diff, "magnitude", magnitude
            # Qualified for filtering it
            idx_list.append(idx)
    time_slices = time_slices.ix[time_slices.index - idx_list]

    return time_slices


"""
Step 4: Classification of Appliance usage and Location Determination
Use the sound data from that time frame, and classify it

Input: Time slices of activity
Output: Identified Activity + Location <appl_i,...>

"""


def determine_location(time_slices, test_csv, phno, apt_no):

    print "-" * stars
    print "Location Determination Process"
    print "-" * stars

    # Classify WiFi data and store the csv in df_w
    if apt_no == '102A':
        if phno == str(1):
            train_wifi_csv = 'CompleteDataSets/Apartment/SoundTrainingSet/Phone1/Wifi1.csv'
            # train_wifi_csv = ('CompleteDataSets/Apartment/'
                              # '23Sep - Wifi_TrainingSet/Wifi_1BHK_102A.csv')
        elif phno == str(2):
            # train_wifi_csv = 'CompleteDataSets/Apartment/SoundTrainingSet/Phone1/Wifi1.csv'
            train_wifi_csv = ('CompleteDataSets/Apartment/19Nov-WifiTrainingSet/Wifi2.csv')
        elif phno in [str(3), str(6)]:
            train_wifi_csv = ('CompleteDataSets/Apartment/23Sep-Wifi_TrainingSet/Wifi3.csv')
        elif phno == str(4):
            train_wifi_csv = ('CompleteDataSets/Apartment/SoundTrainingSet/Phone4/Wifi4.csv')
        else:
            train_wifi_csv = ('CompleteDataSets/Apartment/SoundTrainingSet/Phone5/Wifi5.csv')
    else:
        def_path_apt = 'CompleteDataSets/Apartment/'
        train_wifi_csv = def_path_apt + apt_no + '/train_data/Wifi' + phno + '.csv'

    exp_idx = 3

    df_w = cl.classify_location(train_wifi_csv, test_csv, apt_no, exp_idx)

    # Extracting features of the training set
    # train_feat_csv = cll.format_data(train_wifi_csv, "train", apt_no, exp_idx)
    # df_w = pd.read_csv(test_csv)
    # Test CSV to create
    # test_csv = 'Wifi/test_data/tmp/slice.csv'

    time_slices['start_time'] = time_slices['start_time'].astype('int')
    time_slices['end_time'] = time_slices['end_time'].astype('int')
    # time_slices['true_location'] = np.zeros(len(time_slices))
    time_slices['pred_location'] = np.zeros(len(time_slices))

    # location = []
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
            # location.append("Not Found")
            # pred_location.append("Not Found")
            time_slices = time_slices[time_slices.index != row_idx]
        else:
            # Set location as the one with maximum number of location label
            label = df.label[df.index[0]]  # Ground Truth label
            grpcount_label = df.groupby('pred_label')['pred_label'].count()
            logger.debug("Predictions:: %s", grpcount_label)
            pred_label = grpcount_label[grpcount_label == grpcount_label.max()].index[0]
            # location.append(label)
            pred_location.append(pred_label)
            logger.debug("OLocation:: %s PLocation:: %s", label, pred_label)

    logger.debug("-" * 20)
    # time_slices['true_location'] = location
    time_slices['pred_location'] = pred_location
    # time_slices = time_slices[time_slices.pred_location != 'Not Found']

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


def activity_detection(time_slices, test_csv, phno, apt_no, exp_no):

    print "-" * stars
    print "Sound Classification Process"
    print "-" * stars

    # For every time slice, classify sound
    if apt_no == '102A':
        if phno in [str(4), str(5)]:
            train_sound_csv = ('CompleteDataSets/Apartment/'
                               'SoundTrainingSet/Phone' + phno + '/AudioSamples' + phno + '.csv')
        elif exp_no == '9':
            train_sound_csv = ('CompleteDataSets/Apartment/'
                               'SoundTrainingSet/Phone3_40_10/Sound' + phno + '.csv')
        elif exp_no == '12':
            train_sound_csv = ('CompleteDataSets/Apartment/'
                               'SoundTrainingSet/Phone3_40_10/subsampled/Sound' + phno + '.csv')
            print train_sound_csv
        else:
            train_sound_csv = ('CompleteDataSets/Apartment/'
                               'SoundTrainingSet/Phone' + phno + '_hp.csv')
    else:
        def_path_apt = 'CompleteDataSets/Apartment/'
        # For every time slice, classify sound
        train_sound_csv = def_path_apt + apt_no + '/train_data/Sound' + phno + '.csv'
    exp_idx = 3

    train_feat_csv = cs.extract_features(
        train_sound_csv, "train", apt_no, exp_idx)
    df_s = pd.read_csv(test_csv)

    # Test CSV to create
    test_csv = 'Sound/test_data/tmp/slice.csv'

    # time_slices['true_appliance'] = np.zeros(len(time_slices))
    time_slices['pred_appliance'] = np.zeros(len(time_slices))
    sound = []
    pred_appliance = []
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
            pred_appliance.append("Not Found")
        else:
            # Store classified event
            # sidx = sound_df.index[0]
            # sound.append(sound_df.ix[sidx]['label'])
            psound = cl.classify_sound(
                train_feat_csv, test_csv, apt_no, exp_idx)
            if psound == 'None':
                psound = 'Light'
            pred_appliance.append(psound)

    logger.debug("-" * 20)
    # time_slices['true_appliance'] = sound
    time_slices['pred_appliance'] = pred_appliance

    # Filter sound slices where the sound predicted is "Others"
    # there is no sound(silence) or or sound not in our list
    # to identify (voice, or some other sound)
    # TODO: Collect "Others" sound samples
    # time_slices = time_slices[time_slices.pred_appliance != 'None']

    # detected_activity = combine_light_power_activity(time_slices)
    detected_activity = time_slices.copy()

    # For printing actual times
    print_activity = detected_activity.copy()

    print_activity['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                        for i in detected_activity['start_time']]
    print_activity['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                      for i in detected_activity['end_time']]
    print_activity = print_activity.sort(['start_time'])
    print "\nDetected activity ::\n", print_activity.ix[:,
                                                        print_activity.columns -
                                                       ['start_time', 'end_time']]

    return detected_activity


def activity_detection_ph_features(time_slices, test_feat_csv, phno, exp_no):

    print "-" * stars
    print "Sound Classification Process"
    print "-" * stars

    # For every time slice, classify sound
    if phno in [str(4), str(5)]:
        train_feat_csv = ('CompleteDataSets/Apartment/'
                          'SoundTrainingSet/Phone' + phno + '/AudioFeatures' + phno + '.csv')
    if phno == str(6):
        if exp_no == '14':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/Sound' + phno + '.csv')
            print train_feat_csv
        elif exp_no == '16':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno +
                              '.csv')
        elif exp_no == '17':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_60.csv')
            print train_feat_csv
        elif exp_no == '18':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_80.csv')
            print train_feat_csv
        elif exp_no == '19':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_100.csv')
            print train_feat_csv
        elif exp_no == '20':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_120.csv')
            print train_feat_csv
        elif exp_no == '21':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_140.csv')
            print train_feat_csv
        elif exp_no == '22':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_160.csv')
            print train_feat_csv
        elif exp_no == '23':
            train_feat_csv = ('CompleteDataSets/Apartment/'
                              'SoundTrainingSet/Phone3_MFCC_on_phone/subsampled/Sound' + phno
                              + '_180.csv')
            print train_feat_csv

    else:
        train_feat_csv = ('CompleteDataSets/Apartment/'
                          'SoundTrainingSet/Phone' + phno + '_hp.csv')
    exp_idx = 8

    # Features for test file
    df_s = pd.read_csv(test_feat_csv)
    # df_s = df_s[df_s.mfcc1 != '-Infinity']

    # Test CSV to create
    test_csv = 'Sound/test_data/tmp/slice.csv'

    time_slices['appliance'] = np.zeros(len(time_slices))
    time_slices['pred_appliance'] = np.zeros(len(time_slices))
    sound = []
    pred_appliance = []
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

        if len(sound_df.index) == 0:
            # time_slices = time_slices[time_slices.index != row_idx]
            sound.append("Not Found")
            pred_appliance.append("Not Found")
        else:
            # Store classified event
            sidx = sound_df.index[0]
            sound.append(sound_df.ix[sidx]['label'])
            pred_slabel = cs.classify_sound(
                train_feat_csv, test_csv, apt_no, exp_idx)
            if pred_slabel == 'None':
                pred_slabel = 'Light'
            pred_appliance.append(pred_slabel)

    logger.debug("-" * 20)
    time_slices['sound'] = sound
    time_slices['pred_appliance'] = pred_appliance

    logger.debug("Classified Events::\n %s", time_slices)

    # Filter sound slices where the sound predicted is "Others"
    # there is no sound(silence) or or sound not in our list
    # to identify (voice, or some other sound)
    # TODO: Collect "Others" sound samples
    # time_slices = time_slices[time_slices.pred_appliance != 'None']

    # detected_activity = combine_light_power_activity(time_slices)
    detected_activity = time_slices.copy()

    # For printing actual times
    print_activity = detected_activity.copy()

    print_activity['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                        for i in detected_activity['start_time']]
    print_activity['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                      for i in detected_activity['end_time']]
    print_activity = print_activity.sort(['start_time'])
    print "\nDetected activity ::\n", print_activity.ix[:,
                                                        print_activity.columns -
                                                       ['start_time', 'end_time']]

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
                (power_ts.pred_appliance == light_ts.ix[idx]['pred_appliance']) &
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


def get_light_power_activity(time_slices):

    # For every light time slice, search for power time slices which lie
    # within the time range of the light time slice and both time slices have the detected
    # same location and sound event. This tells that it is likely that the activity happened
    # in the same room
    # Output: For every power event, store the corresponding light event
    # Result expected: "watching TV with lights and fans ON"

    light_ts = time_slices[time_slices.type == 'light']
    power_ts = time_slices[time_slices.type == 'power']
    light_power_activity_list = {}
    window = 240
    if len(light_ts.index) > 0:
        for idx in light_ts.index:
            light_st = light_ts.ix[idx]['start_time']
            light_et = light_ts.ix[idx]['end_time']
            if 'pred_appliance' in light_ts.columns:
                mod_power_ts = power_ts[
                    (power_ts.start_time.isin(
                        range(light_st - window, light_st + window + 1))) &
                    (power_ts.end_time.isin(
                        range(light_et - window, light_et + window + 1))) &
                    (power_ts.pred_appliance == light_ts.ix[idx]['pred_appliance']) &
                    (power_ts.pred_location == light_ts.ix[idx]['pred_location'])]
            else:
                mod_power_ts = power_ts[
                    (power_ts.start_time.isin(
                        range(light_st - window, light_st + window + 1))) &
                    (power_ts.end_time.isin(
                        range(light_et - window, light_et + window + 1))) &
                    (power_ts.pred_location == light_ts.ix[idx]['pred_location'])]
            if len(mod_power_ts) > 0:
                # Not accounting for simultaneous activity in the same location
                i = mod_power_ts.index[0]
                light_power_activity_list[i] = idx
    print "Light Power Activity List", light_power_activity_list

    return light_power_activity_list
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

# How much percent change from the metadata - to define the lower bound of predicted power
lower_mdp_percent_change = 0.15
# How much percent change from the metadata - to define the upper bound of predicted power
upper_mdp_percent_change = 0.2

"""
Stage V: Appliance + Location Correction - Approach 3
"""


def correct_location(md_df, det_activity, light_power_activity_list, mloc_l):

    for idx, matched_mdidx_list in mloc_l.items():
        poss_appl = pd.DataFrame()
        # Empty list entries indicate incorrect classification
        if len(matched_mdidx_list) == 0:
            logger.debug("Location Correction Process starts...")
            mag = math.fabs(det_activity.ix[idx]['magnitude'])
            loc = det_activity.ix[idx]['pred_location']
            phase = det_activity.ix[idx]['phase']

            logger.debug("Location %s with index %d and power %d", loc, idx, mag)

            # Extract metadata of same phase and magnitude
            mdf = md_df[md_df.phase == phase]

            # Choose the one with the closest power draw
            # compared to the predicted magnitude
            df_list = []
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                md_loc = mdf.ix[md_i]['location']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Location - %s :: %s md_power=%s", loc,
                             min_md_power, md_power)

                # Matching metadata with prediction
                if mag >= min_md_power and mag <= max_md_power:
                    md_power_diff = math.fabs(md_power - mag)
                    df_list.append(
                        pd.DataFrame(
                            {'appl_index': idx, 'md_power_diff': md_power_diff}, index=[md_loc]))

            # Correcting location
            if len(df_list) == 1:
                det_activity.ix[idx, 'pred_location'] = df_list[0].index[0]
                if idx in light_power_activity_list.keys():
                    light_idx = light_power_activity_list[idx]
                    det_activity.ix[light_idx, 'pred_location'] = df_list[0].index[0]

                logger.debug(
                    ".........Corrected Location %s with - %s",
                    loc, det_activity.ix[idx]['pred_location'])
            # Locations with similar appliances and on same phase
            elif len(df_list) > 1:
                logger.debug(
                    "TODO: Still need to resolve: Using Feature Extraction in Meter data")
                logger.debug("For now, selecting the closest to the max power draw")
                poss_appl = pd.concat(df_list)
                poss_appl = poss_appl[
                    poss_appl.md_power_diff == poss_appl.md_power_diff.min()]
                print "Possible Appliances::\n", poss_appl
                det_activity.ix[idx, 'pred_location'] = poss_appl.index[0]
                if idx in light_power_activity_list.keys():
                    light_idx = light_power_activity_list[idx]
                    det_activity.ix[light_idx, 'pred_location'] = poss_appl.index[0]

                logger.debug(
                    ".........Corrected Location %s with - %s",
                    loc, det_activity.ix[idx]['pred_location'])
            else:
                logger.debug("ERROR: No match found in the metadata! Recheck Algorithm")
                # logger.debug("Location Classification was incorrect. Do not trust phone")
    return det_activity


def correct_sound(md_df, det_act, light_power_activity_list, mappl_l):

    for idx, matched_mdidx_list in mappl_l.items():
        poss_appl = pd.DataFrame()
        # Empty list entries indicate incorrect classification
        if len(matched_mdidx_list) == 0:
            logger.debug("Correction Process starts...")
            appl = det_act.ix[idx]['pred_appliance']
            mag = math.fabs(det_act.ix[idx]['magnitude'])
            loc = det_act.ix[idx]['pred_location']
            phase = det_act.ix[idx]['phase']

            logger.debug("Appliance %s with index %d and power %d", appl, idx, mag)

            # Extract metadata of appliances with the same location
            mdf = md_df[(md_df.location == loc) & (md_df.phase == phase)]

            # Choose the one with the closest power draw
            # compared to the predicted magnitude
            df_list = []
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                md_appl = mdf.ix[md_i]['appliance']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Appl - %s :: %s md_power=%s", appl,
                             min_md_power, max_md_power)

                # Matching metadata with prediction
                if mag >= min_md_power and mag <= max_md_power:
                    md_power_diff = math.fabs(md_power - mag)
                    df_list.append(
                        pd.DataFrame(
                            {'appl_index': idx, 'md_power_diff': md_power_diff}, index=[md_appl]))

            # Correcting appliance
            if len(df_list) == 1:
                det_act.ix[idx, 'pred_appliance'] = df_list[0].index[0]
                if idx in light_power_activity_list.keys():
                    light_idx = light_power_activity_list[idx]
                    det_act.ix[light_idx, 'pred_appliance'] = df_list[0].index[0]

                logger.debug(
                    ".........Corrected Appliance %s with - %s",
                    appl, det_act.ix[idx]['pred_appliance'])
            elif len(df_list) > 1:
                logger.debug(
                    "TODO: Still need to resolve: Using Feature Extraction in Meter data")
                logger.debug("For now, selecting the closest to the max power draw")
                poss_appl = pd.concat(df_list)
                # print "Possible Appliances::\n", poss_appl
                # print "Miniumum power draw::", poss_appl.md_power_diff.min()
                poss_appl = poss_appl[
                    poss_appl.md_power_diff == poss_appl.md_power_diff.min()]
                det_act.ix[idx, 'pred_appliance'] = poss_appl.index[0]

                if idx in light_power_activity_list.keys():
                    light_idx = light_power_activity_list[idx]
                    det_act.ix[light_idx, 'pred_appliance'] = poss_appl.index[0]

                logger.debug(
                    ".........Corrected Appliance %s with - %s",
                    appl, det_act.ix[idx]['pred_appliance'])
            else:
                logger.debug("ERROR: No match found in the metadata! Recheck Algorithm")
                logger.debug("Location Classification was incorrect. Do not trust phone")
                # corr_act_meter = mo.detect_activity_location_using_meter(
                #     pred_time_slices, app, apt_no)

    return det_act


def detect_incorrect_location(md_df, detected_activity, light_power_activity_list, mclass_l):

    mloc_l = defaultdict(list)
    logger.debug("Detecting Incorrect Location Classification using MD2...")
    for i, matched_mdidx_list in mclass_l.items():
        if len(matched_mdidx_list) == 0:
            if i in light_power_activity_list.values():
                continue
            md_l = []
            mag = math.fabs(detected_activity.ix[i]['magnitude'])
            loc = detected_activity.ix[i]['pred_location']
            phase = detected_activity.ix[i]['phase']

            # Extract metadata for the current appliance
            mdf = md_df[(md_df.location == loc) & (md_df.phase == phase)]
            if len(mdf.index) == 0:
                logger.debug("Incorrect location %s with phase %s with idx %s", loc, phase, i)
            else:
                for md_i in mdf.index:
                    md_power = mdf.ix[md_i]['rating']

                    min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                    max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                    # Compare magnitude and metadata power draw
                    logger.debug("For Location - %s(%s) :: [%s, md_power=%s]", loc, mag,
                                 min_md_power, max_md_power)

                    # Matching metadata with appliance power draw and location
                    if (mag >= min_md_power and mag <= max_md_power):
                        md_l.append(md_i)
            logger.debug(
                "Matched Indexes for Location %s with idx %d: %s", loc, i, md_l)
            mloc_l[i] = md_l
    logger.debug("Comparison results: %s\n", mloc_l.items())
    return mloc_l


def detect_incorrect_sound_md2(md_df, detected_activity, light_power_activity_list, mclass_l):

    mappl_l = defaultdict(list)
    logger.debug("Detecting Incorrect Sound Classification using MD2...")
    for i, matched_mdidx_list in mclass_l.items():
        if len(matched_mdidx_list) == 0:
            if i in light_power_activity_list.values():
                continue
            md_l = []
            appl = detected_activity.ix[i]['pred_appliance']
            mag = math.fabs(detected_activity.ix[i]['magnitude'])
            # loc = detected_activity.ix[i]['pred_location']
            phase = detected_activity.ix[i]['phase']

            # Extract metadata for the current appliance
            mdf = md_df[(md_df.appliance == appl) & (md_df.phase == phase)]
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                # mdf_loc = mdf.ix[md_i]['location']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Appl - %s(%s) :: [%s, md_power=%s]", appl, mag,
                             min_md_power, max_md_power)

                # Matching metadata with appliance power draw and location
                if (mag >= min_md_power and mag <= max_md_power):
                   # and mdf_loc == loc):
                    md_l.append(md_i)

            logger.debug(
                "Matched Indexes for Appliance %s with idx %d: %s", appl, i, md_l)
            mappl_l[i] = md_l
    logger.debug("Comparison results: %s\n", mappl_l.items())
    return mappl_l


def detect_incorrect_classification(md_df, detected_activity, light_power_activity_list, isphase):

    no_ele = len(detected_activity.index)
    if no_ele > 0:
        mappl_l = defaultdict(list)
        logger.debug("Detecting Incorrect Classification...")
        for i in detected_activity.index:
            if i in light_power_activity_list.values():
                continue
            md_l = []
            appl = detected_activity.ix[i]['pred_appliance']
            mag = math.fabs(detected_activity.ix[i]['magnitude'])
            loc = detected_activity.ix[i]['pred_location']
            phase = detected_activity.ix[i]['phase']

            # Extract metadata for the current appliance
            mdf = md_df[md_df.appliance == appl]
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                mdf_loc = mdf.ix[md_i]['location']
                mdf_phase = mdf.ix[md_i]['phase']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Appl - %s(%s) :: [%s, md_power=%s]", appl, mag,
                             min_md_power, max_md_power)

                # Matching metadata with appliance power draw and location
                if (mag >= min_md_power and mag <= max_md_power and
                   mdf_loc == loc):
                    if (isphase and mdf_phase == phase):
                        md_l.append(md_i)
                    elif not isphase:
                        md_l.append(md_i)

            logger.debug(
                "Matched Indexes for Appliance %s with idx %d: %s", appl, i, md_l)
            mappl_l[i] = md_l
    logger.debug("Comparison results: %s\n", mappl_l.items())
    return mappl_l


def correction_using_appl_phase_power(det_act, app, apt_no):
    # Take the appliance power location mapping and correct the activity detection

    # md_df = pd.DataFrame(metadata, columns=['appliance', 'location', 'rating', 'phase'])
    md_df = pd.read_csv('Metadata/' + apt_no + '_metadata.csv')
    print "\nMetadata::\n", md_df

    light_power_activity_list = get_light_power_activity(det_act)

    isphase = True
    # Detect incorrect classification
    mclass_l = detect_incorrect_classification(md_df, det_act, light_power_activity_list, isphase)

    # Correct Location
    if app == 4:
        mloc_l = detect_incorrect_location(md_df, det_act, light_power_activity_list, mclass_l)
        det_act = correct_location(
            md_df, det_act, light_power_activity_list, mloc_l)

    # Correct Appliance
    mappl_l = detect_incorrect_sound_md2(md_df, det_act, light_power_activity_list, mclass_l)
    det_act = correct_sound(
        md_df, det_act, light_power_activity_list, mappl_l)

    return det_act

"""
Stage IV: Appliance Correction - Approach 2
"""


def detect_incorrect_sound_md1(md_df, detected_activity, light_power_activity_list, isphase):

    no_ele = len(detected_activity.index)
    if no_ele > 0:
        mappl_l = defaultdict(list)
        logger.debug("Detecting Incorrect Sound Classification using MD1...")
        for i in detected_activity.index:
            if i in light_power_activity_list.values():
                continue
            md_l = []
            appl = detected_activity.ix[i]['pred_appliance']
            mag = math.fabs(detected_activity.ix[i]['magnitude'])
            loc = detected_activity.ix[i]['pred_location']
            phase = detected_activity.ix[i]['phase']

            # Extract metadata for the current appliance
            mdf = md_df[md_df.appliance == appl]
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                mdf_loc = mdf.ix[md_i]['location']
                mdf_phase = mdf.ix[md_i]['phase']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Appl - %s(%s) :: [%s, md_power=%s]", appl, mag,
                             min_md_power, max_md_power)

                # Matching metadata with appliance power draw and location
                if (mag >= min_md_power and mag <= max_md_power and
                   mdf_loc == loc):
                    if (isphase and mdf_phase == phase):
                        md_l.append(md_i)
                    elif not isphase:
                        md_l.append(md_i)

            logger.debug(
                "Matched Indexes for Appliance %s with idx %d: %s", appl, i, md_l)
            mappl_l[i] = md_l
    logger.debug("Comparison results: %s\n", mappl_l.items())
    return mappl_l


def detect_and_correct_location_md1(md_df, det_act, mappl_l, apt_no):
    # Detecting and Correcting Location using meter data
    logger.debug("Detecting Incorrection Location using MD1...")
    for idx, matched_mdidx_list in mappl_l.items():
        # Empty list entries indicate incorrect classification
        if len(matched_mdidx_list) == 0:
            appl = det_act.ix[idx]['pred_appliance']
            mag = math.fabs(det_act.ix[idx]['magnitude'])
            loc = det_act.ix[idx]['pred_location']

            logger.debug("Appliance %s with index %d and power %d", appl, idx, mag)

            # Extract metadata of appliances with the same location
            mdf = md_df[md_df.location == loc]

            df_list = []
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                md_appl = mdf.ix[md_i]['appliance']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Appl - %s :: %s md_power=%s", appl,
                             min_md_power, max_md_power)

                # Matching metadata with prediction
                if mag >= min_md_power and mag <= max_md_power:
                    md_power_diff = md_power - mag
                    df_list.append(
                        pd.DataFrame(
                            {'appl_index': idx, 'md_power_diff': md_power_diff}, index=[md_appl]))

            # Indicates incorrect Location
            if len(df_list) == 0:
                # Correct Location using meter data only
                print "Correcting Location for:: \n", det_act[det_act.index == idx]
                tsdf_with_loc = mo.detect_activity_location_using_meter(
                    det_act[det_act.index == idx], 6, apt_no)
                if len(tsdf_with_loc) > 0:
                    correct_loc = tsdf_with_loc.ix[tsdf_with_loc.index[0]]['pred_location']
                    if correct_loc != 'Not Found':
                        det_act.ix[idx, 'pred_location'] = correct_loc
                        print "Corrected", loc, "with", correct_loc
    return det_act


def correction_using_appl_power(det_act, apt_no):
    # Take the appliance power location mapping and correct the activity detection

    # md_df = pd.DataFrame(metadata, columns=['appliance', 'location', 'rating', 'phase'])
    md_df = pd.read_csv('Metadata/' + apt_no + '_metadata.csv')
    print "\nMetadata::\n", md_df

    # print "Passed det act\n", det_act
    light_power_activity_list = get_light_power_activity(det_act)
    isphase = False

    # Detecting Incorrect classifications - either sound/location
    mappl_l = detect_incorrect_classification(md_df, det_act, light_power_activity_list, isphase)

    # Detecting and Correcting incorrect locations from the incorrect set
    det_act = detect_and_correct_location_md1(md_df, det_act, mappl_l, apt_no)

    # Detecting Incorrect Appliance Classifications
    mappl_l = detect_incorrect_sound_md1(md_df, det_act, light_power_activity_list, isphase)

    # Correcting Appliance
    for idx, matched_mdidx_list in mappl_l.items():
        poss_appl = pd.DataFrame()
        # Empty list entries indicate incorrect classification
        if len(matched_mdidx_list) == 0:
            logger.debug("Appliance Correction Process starts...")
            appl = det_act.ix[idx]['pred_appliance']
            mag = math.fabs(det_act.ix[idx]['magnitude'])
            loc = det_act.ix[idx]['pred_location']

            logger.debug("Appliance %s with index %d and power %d", appl, idx, mag)

            # Extract metadata of appliances with the same location
            mdf = md_df[md_df.location == loc]

            # Choose the one with the closest power draw
            # compared to the predicted magnitude
            df_list = []
            for md_i in mdf.index:
                md_power = mdf.ix[md_i]['rating']
                md_appl = mdf.ix[md_i]['appliance']

                min_md_power = math.floor(md_power - lower_mdp_percent_change * md_power)
                max_md_power = math.ceil(md_power + upper_mdp_percent_change * md_power)

                # Compare magnitude and metadata power draw
                logger.debug("For Appl - %s :: %s md_power=%s", appl,
                             min_md_power, max_md_power)

                # Matching metadata with prediction
                if mag >= min_md_power and mag <= max_md_power:
                    md_power_diff = math.fabs(md_power - mag)
                    df_list.append(
                        pd.DataFrame(
                            {'appl_index': idx, 'md_power_diff': md_power_diff}, index=[md_appl]))

            # Correcting appliance
            if len(df_list) == 1:
                det_act.ix[idx, 'pred_appliance'] = df_list[0].index[0]
                if idx in light_power_activity_list.keys():
                    light_idx = light_power_activity_list[idx]
                    det_act.ix[light_idx, 'pred_appliance'] = df_list[0].index[0]

                logger.debug(
                    "......Corrected Appliance %s with - %s",
                    appl, det_act.ix[idx]['pred_appliance'])
            elif len(df_list) > 1:
                logger.debug(
                    "TODO: Still need to resolve: Using Feature Extraction in Meter data")
                logger.debug("For now, selecting the closest to the max power draw")
                poss_appl = pd.concat(df_list)
                # print "Minimum power draw::", poss_appl.md_power_diff.min()

                poss_appl = poss_appl[
                    poss_appl.md_power_diff == poss_appl.md_power_diff.min()]
                print "Possible Appliances::\n", poss_appl

                if len(poss_appl.index.unique()) == 1:
                    det_act.ix[idx, 'pred_appliance'] = poss_appl.index[0]
                    if idx in light_power_activity_list.keys():
                        light_idx = light_power_activity_list[idx]
                        det_act.ix[light_idx, 'pred_appliance'] = poss_appl.index[0]

                    logger.debug(
                        ".........Corrected Appliance %s with - %s",
                        appl, det_act.ix[idx]['pred_appliance'])
                else:
                    logger.debug(
                        ".........Could not correct appliance %s: Multiple appliances present",
                        appl)
            else:
                logger.debug("ERROR: No match found in the metadata! Recheck Algorithm")
                logger.debug("Location Classification was incorrect. Do not trust phone")
                # corr_act_meter = mo.detect_activity_location_using_meter(
                #     pred_time_slices, app, apt_no)
    return det_act


def extract_light(detected_activity):

    # Get light + power appliance combination
    light_power_activity_list = get_light_power_activity(detected_activity)

    window = 240

    # Get light + light appliance combination
    light_light_activity_list = {}
    light_ts = detected_activity[detected_activity.type == 'light']
    if len(light_ts.index) > 0:
        for idx in light_ts.index:

            light_st = light_ts.ix[idx]['start_time']
            light_et = light_ts.ix[idx]['end_time']
            light_mag = math.fabs(light_ts.ix[idx]['magnitude'])

            mod_light_ts = light_ts[
                (light_ts.start_time.isin(
                 range(light_st - window, light_st + window + 1))) &
                (light_ts.end_time.isin(
                 range(light_et - window, light_et + window + 1))) &
                (light_ts.pred_appliance == light_ts.ix[idx]['pred_appliance']) &
                (light_ts.pred_location == light_ts.ix[idx]['pred_location'])]

            # Remove the index 'idx' under consideration
            mod_light_ts = mod_light_ts[mod_light_ts.index != idx]
            if len(mod_light_ts) > 0:
                i = mod_light_ts.index[0]
                mod_light_mag = math.fabs(mod_light_ts.ix[i]['magnitude'])
                if int(0.1 * mod_light_mag) == int(0.1 * light_mag):
                    # Indicates there are two Fans running
                    print "Found two fans in the same time slice"
                    pass
                elif mod_light_mag < light_mag:
                    # Indicates its a light (to differentiate between fans and lights)
                    light_light_activity_list[idx] = i
                else:
                    light_light_activity_list[i] = idx
    print "Light light Activity List", light_light_activity_list

    light_activity = dict(light_power_activity_list.items() + light_light_activity_list.items())
    print "Concatenated Light Activity", light_activity, type(light_activity)
    # Set sound for the light entries to "Light"
    for key in light_activity:
        light_idx = light_activity[key]
        etype = detected_activity.ix[light_idx]['type']
        if etype == 'light':
            detected_activity.ix[light_idx, 'pred_appliance'] = "Light"

    return detected_activity


def print_with_actual_times(det_activity):
    # To print the passed data frame with the actual times

    print_activity = det_activity.ix[:, (det_activity.columns -
                                    ['start_time', 'end_time', 'location', 'appliance'])]

    print_activity['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                        for i in det_activity['start_time']]
    print_activity['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                      for i in det_activity['end_time']]
    print_activity = print_activity.sort(['act_start_time'])

    return print_activity

# Bounds for time frame (in seconds) for calculating precision and recall
lower_limit = 3
upper_limit = 5


def convert_to_actual_time(timestamp):
    if len(str(timestamp)) <= 10:
        return dt.datetime.fromtimestamp(timestamp)
    else:
        return dt.datetime.fromtimestamp(timestamp / 1000)


def label_with_ground_truth(time_slices, gfile):
    # Label predicted time slices with ground truth
    print "\nLabeling With Ground Truth....", gfile
    df = time_slices.copy()
    df['true_location'] = ['Not Found'] * len(df)
    df['true_appliance'] = ['Not Found'] * len(df)

    gt = pd.read_csv(gfile)
    gt = gt.sort(['start_time'])
    gt['start_time'] = gt.start_time / 1000
    gt['end_time'] = gt.end_time / 1000

    gt['act_start_time'] = [convert_to_actual_time(i) for i in gt.start_time]
    gt['act_end_time'] = [convert_to_actual_time(i) for i in gt.end_time]

    # print gt
    # print time_slices
    for pidx in df.index:
        pred_st = long(df.ix[pidx]['start_time'])
        pred_et = long(df.ix[pidx]['end_time'])

        for idx in gt.index:
            true_st = long(gt.ix[idx]['start_time'])
            true_et = long(gt.ix[idx]['end_time'])

            if (pred_st in range(true_st - lower_limit, true_st + upper_limit + 1) and
               pred_et in range(true_et - lower_limit, true_et + upper_limit + 1)):
                df.ix[pidx, 'true_location'] = gt.ix[idx]['wlabel']
                df.ix[pidx, 'true_appliance'] = gt.ix[idx]['slabel']
                break
            # else:
            #     print "\nPET", convert_to_actual_time(pred_st), convert_to_actual_time(true_st)
            #     print "PET", convert_to_actual_time(pred_et), convert_to_actual_time(true_et)

    return df


def get_magnitude_metadata(df, idx, power_stream, winmin):

    i = idx
    # print "Looking for edge in stream:", power_stream, "for Index:", i
    prev = int(round(df.ix[i - 1][power_stream]))
    curr = int(round(df.ix[i][power_stream]))
    next = int(round(df.ix[i + 1][power_stream]))
    currwin = int(round(df.ix[i + winmin][power_stream]))

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

    time = df.ix[i]['time']

    curr_nextwin_diff = currwin - curr
    if math.fabs(curr_nextwin_diff) < lthresmin and power_stream == "lightpower":
        # print "Magnitude", curr_nextwin_diff, "is less. Looking for more edges"
        new_winmin = winmin - 1
        while new_winmin != 0:
            # print "New winmin", new_winmin
            currwin = int(round(df.ix[i + new_winmin][power_stream]))
            new_mag = currwin - curr
            if math.fabs(new_mag) < lthresmin:
                # print "INNER:: Magnitude", new_mag, "less. Looking for more edges"
                new_winmin = new_winmin - 1
                continue
            else:
                curr_nextwin_diff = new_mag
                break
    elif (math.fabs(curr_nextwin_diff) < pthresmin and power_stream == "power"
          and curr_nextwin_diff < 0):
        # print "Magnitude", curr_nextwin_diff, "is less. Looking for more edges"
        new_winmin = winmin + 1
        while new_winmin != 0:
            # print "New winmin", new_winmin
            currwin = int(round(df.ix[i + new_winmin][power_stream]))
            new_mag = currwin - curr
            if math.fabs(new_mag) < pthresmin:
                # print "INNER:: Magnitude", new_mag, "is less. Looking for more edges"
                new_winmin = new_winmin + 1
                continue
            else:
                next = int(round(df.ix[i + new_winmin + 1][power_stream]))
                curr_next_diff = math.floor(math.fabs(next - currwin))
                if curr_next_diff < 5:
                    curr_nextwin_diff = new_mag
                break

    edge = {"index": i, "time": time, "magnitude": curr_nextwin_diff, "curr_power": curr}
    return edge


def use_ground_truth_time_slice(gt, df_p, df_l, app):
    """
    Passing ground truth time slices to the next stage
    """

    gt = gt.reset_index(drop=True)
    gt['start_time'] = gt.start_time / 1000
    gt['end_time'] = gt.end_time / 1000
    gt['pred_location'] = np.zeros(len(gt))
    gt['true_location'] = gt.wlabel
    gt['true_appliance'] = gt.slabel

    gt['type'] = np.zeros(len(gt))
    gt['magnitude'] = np.zeros(len(gt))
    gt['phase'] = np.zeros(len(gt))

    print "\n"
    # Label lights and fans as "light" type and magnitude as the fall magnitude of the time slice
    for i in gt.index:
        appl = gt.ix[i]['slabel']
        etype = ''

        if appl in ['Fan', 'Light']:
            gt.ix[i, 'type'] = etype = "light"
        else:
            gt.ix[i, 'type'] = etype = "power"

        # Get index from the main light/power data stream
        fall_time = gt.ix[i]['end_time']
        logger.debug("Fall_time:: %s (%s)", dt.datetime.fromtimestamp(fall_time), fall_time)
        if etype == "light":
            idx = 0
            try:
                idx = np.where(df_l.time == fall_time)[0][0]
            except IndexError, e:
                print "-----Index not found--------", e
                continue

            edge = get_magnitude_metadata(df_l, idx, "lightpower", lwinmin)
            if edge:
                gt.ix[i, 'magnitude'] = math.fabs(edge['magnitude'])
                print i, "Light:: Index", idx, "Magnitude::", math.fabs(edge['magnitude'])
            else:
                logger.debug("--------Edge not found---------")
                continue

            # Get phase of the edge
            gt.ix[i, 'phase'] = get_phase(idx, df_l, "light")
            # print "LIGHT:: Index", idx, "magnitude", magnitude
        else:
            idx = 0
            try:
                idx = np.where(df_p.time == fall_time)[0][0]
            except IndexError, e:
                print "-----Index not found--------", e
                continue

            edge = get_magnitude_metadata(df_p, idx, "power", pwinmin)
            if edge:
                gt.ix[i, 'magnitude'] = math.fabs(edge['magnitude'])
                print i, "Power:: Index", idx, "Magnitude::", math.fabs(edge['magnitude'])
            else:
                logger.debug("--------Edge not found---------")
                continue

            # Get phase of the edge
            gt.ix[i, 'phase'] = get_phase(idx, df_p, "power")

            # print "POWER:: Index", idx, "magnitude", magnitude

    gt.pop('wlabel')
    gt.pop('slabel')

    if app in [4, 7]:
        gt = gt[(gt.phase != 0) & (gt.magnitude > lthresmin)]
    else:
        gt = gt[gt.magnitude > lthresmin]

    gt['act_start_time'] = [dt.datetime.fromtimestamp(i)
                            for i in gt['start_time']]
    gt['act_end_time'] = [dt.datetime.fromtimestamp(i)
                          for i in gt['end_time']]
    gt = gt.sort(['act_start_time'])

    print "\nGround Truth Time slices:\n", gt

    return gt
"""
---------------------------------------
Main Program
---------------------------------------
"""

if __name__ == '__main__':

    # Get the sensor streams (test set)
    exp_no = sys.argv[1]
    apt_no = sys.argv[2]
    phno = sys.argv[3]
    app = int(sys.argv[4])

    def_path = ''
    event_dir = ''
    if apt_no == '102A':
        def_path = 'CompleteDataSets/Apartment/Evaluation/'
        event_dir = def_path + 'exp' + exp_no + '/'
    else:
        def_path = 'CompleteDataSets/Apartment/' + apt_no + '/'
        event_dir = def_path + exp_no + '/'

    # event_dir = 'CompleteDataSets/Apartment/23Sep - Meter_Wifi_Sound_Accl/'
    # event_dir = 'CompleteDataSets/Apartment/23_9_16_53_23_9_19_11/'
    power_csv = event_dir + 'Power.csv'
    # sys.argv[2]
    light_csv = event_dir + 'Light.csv'

    # Read ground truth
    gt_file = def_path + "ground_truth/test" + "_" + apt_no + "_t" + exp_no + ".csv"

    # accl_csv = event_dir + 'Accl_1BHK_102A.csv'  # sys.argv[3]
    # sound_csv = event_dir + 'Sound_1BHK_102A.csv'  # sys.argv[4]
    # wifi_csv = event_dir + 'Wifi_1BHK_102A.csv'  # sys.argv[5]

    accl_csv = event_dir + 'Accl' + phno + '.csv'  # sys.argv[3]
    if phno in [str(4), str(5)]:
        sound_csv = event_dir + 'AudioSamples' + phno + '.csv'
    else:
        sound_csv = event_dir + 'Sound' + phno + '.csv'  # sys.argv[4]
    wifi_csv = event_dir + 'Wifi' + phno + '.csv'  # sys.argv[5]

    st = time.time()
    logger.info("Starting Algorithm...")
    df_p = pd.read_csv(power_csv)
    df_l = pd.read_csv(light_csv)

    # Step i: Apply moving average to the power data
    # df_l = average_power(df_l)
    # df_p = average_power(df_p)

    # Step 1: Edge Detection
    edge_list_df = edge_detection(df_l, df_p, apt_no)
    # sys.exit()

    # Step 2a: Edge Matching
    time_slices = edge_matching(df_l, df_p, edge_list_df, app)
    # sys.exit()

    # Step 2b: Filter extraneous events
    time_slices = filter_time_slices(time_slices, apt_no, exp_no)

    # Step3: Localization
    time_slices_location = determine_location(time_slices, wifi_csv, phno, apt_no)

    # For time-slice accuracy calculation
    # ts_file = event_dir + 'time_slices_exp' + exp_no + '_' + str(app) + '.csv'
    tmp_gt = gt_file
    if apt_no == '603' and exp_no == '27-28Nov':
        gt_file = def_path + "ground_truth/test_603_t27-28Nov_orig.csv"

    print "Using ground_truth file", gt_file
    ts_prec, ts_recall = ev.calc_ts_accuracy(time_slices_location, pd.read_csv(gt_file))
    # sys.exit()

    if apt_no == '603' and exp_no == '27-28Nov':
        gt_file = tmp_gt
        gt = pd.read_csv(gt_file)
        print "Using ground_truth file", gt_file

    # Step 3: Creating Time Slices using ground truth
    # Using ground truth for time slices
    time_slices = use_ground_truth_time_slice(pd.read_csv(gt_file), df_p, df_l, app)

    # sys.exit()

    # Step3: Localization
    time_slices_location = determine_location(time_slices, wifi_csv, phno, apt_no)
    # sys.exit(1)

    # Step4: Audio Classification
    if phno in [str(6)]:
        detected_activity = activity_detection_ph_features(
            time_slices_location, sound_csv, phno, exp_no)
    else:
        detected_activity = activity_detection(
            time_slices_location, sound_csv, phno, apt_no, exp_no)  # , 't' + exp_no + '_' + phno)

    if app >= 3:
        if app == 3:
            detected_activity = correction_using_appl_power(detected_activity, apt_no)
        elif app == 4:
            detected_activity = correction_using_appl_phase_power(detected_activity, app, apt_no)

    # Step5: Differentiate between lights and fans
    detected_activity = extract_light(detected_activity)

    # For ground truth purposes
    t = detected_activity[(detected_activity.pred_appliance != 'Fan')
                          & (detected_activity.pred_appliance != 'Light')
                          & (detected_activity.type == 'light')]
    if len(t) > 0:
        for i in t.index:
            detected_activity.ix[i, 'pred_appliance'] = 'Light'

    # detected_activity

    # Last Step: Label ground truth
    # detected_activity = label_with_ground_truth(detected_activity, gt_file)
    # detected_activity = detected_activity.ix[
    #     :, (detected_activity.columns - ['location', 'appliance'])]
    detected_activity['ts_prec'] = [ts_prec] * len(detected_activity)
    detected_activity['ts_recall'] = [ts_recall] * len(detected_activity)

    # Store the detected activity as a csv file
    opfilename = event_dir + 'output_app' + str(app) + '_' + phno + '.csv'
    print "Making output file", opfilename
    detected_activity.to_csv(opfilename, index=False)

    # For printing detected activity with actual times
    print "\nFinal Detected activity and location for Exp", exp_no, "AppNo.", app, \
        "Phone", phno, "::\n", print_with_actual_times(detected_activity)

    # print "\nAccuracy Results for Phone", phno, "::\n"
    # os.system("python evaluate.py " + str(exp_no) +
    #           " " + str(phno) + " " + str(app) + " " + str(apt_no) + " single")

    logger.info("Algorithm Run Finished!")
    et = time.time()
    print "Algorithm finished in", et - st, "seconds!"
