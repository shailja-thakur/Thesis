"""
Script for running the meter only approach
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import datetime as dt
import activity_finder_algorithm as af
import evaluate as ev
import warnings
import logging
import pprint
# import json

# Disable warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('activity-finder')
logging.basicConfig(level=logging.DEBUG)

lower_mdp_percent_change = 0.15
upper_mdp_percent_change = 0.2


def using_meter_metadata(pred_time_slices, md_df, isphase):

    md_dict = {}
    pp = pprint.PrettyPrinter(indent=4)

    for idx in pred_time_slices.index:

        mag = pred_time_slices.ix[idx]['magnitude']
        phase = pred_time_slices.ix[idx]['phase']

        max_md_power = math.ceil(mag + upper_mdp_percent_change * mag)
        min_md_power = math.floor(mag - lower_mdp_percent_change * mag)

        mdf = pd.DataFrame()
        if isphase:
            mdf = md_df[(md_df.rating >= min_md_power) & (
                        md_df.rating <= max_md_power) & (md_df.phase == phase)]
        else:
            mdf = md_df[(md_df.rating >= min_md_power) & (md_df.rating <= max_md_power)]
        md_dict[idx] = {'md_idx': [], 'md_power_diff': []}

        for md_i in mdf.index:
            md_power = mdf.ix[md_i]['rating']
            md_power_diff = math.fabs(md_power - mag)
            md_dict[idx]['md_idx'].append(md_i)
            md_dict[idx]['md_power_diff'].append(md_power_diff)
    # print "\nMDList for tsidx", idx
    pp.pprint(md_dict)

    return md_dict


def detect_activity_location_using_meter(pred_time_slices, app, apt_no):

    md_df = pd.read_csv('Metadata/' + apt_no + '_metadata.csv')
    print "\nMetadata::\n", md_df

    location = []
    appliance = []
    pred_time_slices['pred_location'] = np.zeros(len(pred_time_slices))
    pred_time_slices['pred_appliance'] = np.zeros(len(pred_time_slices))

    isphase = ''
    if app == 6:
        isphase = False
    else:
        isphase = True
    md_dict = using_meter_metadata(pred_time_slices, md_df, isphase)

    # Detecting location and appliance in use
    for ts_idx in md_dict:
        md_idx = md_dict[ts_idx]['md_idx']
        md_diff = md_dict[ts_idx]['md_power_diff']
        if len(md_idx) > 0:
            print "\nMD", ts_idx, md_idx, md_diff

            min_item = min(md_diff)
            indices = [i for i, x in enumerate(md_diff) if x == min_item]
            print "Minimum Item:", min_item
            print "Minimum Indices", indices
            print "Length indices", len(indices)

            if len(indices) > 1:
                # If all the appliances and corresponding location are same,
                # then assign with it else error
                md_idxes = [md_idx[i] for i in indices]
                appl = [md_df.ix[mdid]['appliance'] for mdid in md_idxes]
                loc = [md_df.ix[mdid]['location'] for mdid in md_idxes]
                if len(set(appl)) == 1 and len(set(loc)) != 1:
                    location.append("Not Found")
                    appliance.append(md_df.ix[mdid]['appliance'])
                elif len(set(appl)) != 1 and len(set(loc)) == 1:
                    location.append(md_df.ix[mdid]['location'])
                    appliance.append("Not Found")
                elif len(set(appl)) == 1 and len(set(loc)) == 1:
                    location.append(md_df.ix[mdid]['location'])
                    appliance.append(md_df.ix[mdid]['appliance'])
                else:
                    location.append("Not Found")
                    appliance.append("Not Found")
            else:
                mdid = md_idx[indices[0]]
                location.append(md_df.ix[mdid]['location'])
                appliance.append(md_df.ix[mdid]['appliance'])
        else:
            print "\nNFMD", ts_idx, md_idx, md_diff
            location.append("Not Found")
            appliance.append("Not Found")
    print "location", location
    print "appliance", appliance

    pred_time_slices = pred_time_slices.sort()
    pred_time_slices['pred_location'] = location
    pred_time_slices['pred_appliance'] = appliance

    print "\nMeter Detected Activity\n"
    print pred_time_slices.ix[:, pred_time_slices.columns -
                             ['start_time', 'end_time']]

    ts_discard = pred_time_slices[(pred_time_slices.pred_location == "Not Found")
                                  & (pred_time_slices.pred_appliance == "Not Found")]
    pred_time_slices = pred_time_slices.ix[pred_time_slices.index - ts_discard.index]

    return pred_time_slices


if __name__ == '__main__':

    # Get the sensor streams (test set)
    exp_no = sys.argv[1]  # experiment number
    app = int(sys.argv[2])  # Can take the value 6 or 7
    apt_no = sys.argv[3]
    algotype = sys.argv[4]
    phno_list = sys.argv[5]

    # algotype = sys.argv[3]  # takes either single or multi
    if apt_no == '102A':
        def_path = 'CompleteDataSets/Apartment/Evaluation/'
        event_dir = def_path + 'exp' + exp_no + '/'
    else:
        def_path = 'CompleteDataSets/Apartment/' + apt_no + '/'
        event_dir = def_path + exp_no + '/'

    power_csv = event_dir + 'Power.csv'
    light_csv = event_dir + 'Light.csv'

    df_p = pd.read_csv(power_csv)
    df_l = pd.read_csv(light_csv)

    # Step1: Edge Detection
    edge_list_df = af.edge_detection(df_l, df_p, apt_no)
    # sys.exit()

    # Step2a: Edge Matching
    time_slices = af.edge_matching(df_l, df_p, edge_list_df, app)

    # Step2b: Filter extraneous events
    time_slices = af.filter_time_slices(time_slices, apt_no, exp_no)

    # Calculate time slice accuracy
    if algotype == 'single':
        gfile = def_path + "ground_truth/test_" + apt_no + "_t" + exp_no + ".csv"
        gt = pd.read_csv(gfile)
    else:
        gt_file = []
        phno_list = phno_list.split(',')
        no_of_occupants = len(phno_list)
        for i in range(no_of_occupants):
            print i
            gt_file.append(def_path + 'ground_truth/test_' +
                           apt_no + '_t' + exp_no + '_' + phno_list[i] + '.csv')
        gt = pd.concat([pd.read_csv(gt_file[0]), pd.read_csv(gt_file[1])])
        # Remove duplicates (shared time slices)
        gt.drop_duplicates(inplace=True)

    ts_prec, ts_recall = ev.calc_ts_accuracy(time_slices, gt)

    # sys.exit(1)

    # Use ground truth time slice for further processing
    time_slices = af.use_ground_truth_time_slice(gt, df_p, df_l, app)

    # Step 3: Determine Activity and Location
    detected_activity = detect_activity_location_using_meter(time_slices, app, apt_no)

    # Step5: Differentiate between lights and fans
    detected_activity = af.extract_light(detected_activity)

    # Storing the time slice accuracy
    detected_activity['ts_prec'] = [ts_prec] * len(detected_activity)
    detected_activity['ts_recall'] = [ts_recall] * len(detected_activity)

    # Store the detected activity as a csv file
    opfilename = event_dir + 'output_app' + str(app) + '_meter.csv'
    print "Making output file", opfilename
    detected_activity.to_csv(opfilename, index=False)

    # For printing actual times
    print "\nFinal Detected activity and location for Exp", exp_no, "AppNo.", app, \
        "::\n", af.print_with_actual_times(detected_activity)

    phno = 'nil'
    # print "Accuracy for approach", app
    # os.system("python evaluate.py " + str(exp_no) +
    #           " " + str(phno) + " " + str(app) + " " + str(apt_no) + " " + algotype)
