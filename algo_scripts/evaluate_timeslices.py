"""
Script to calculate the precision and recall of classified events
"""

import os
import sys
import numpy as np
import pandas as pd
import math
import datetime as dt
import activity_finder_algorithm as af
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import *
from random import randint
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

# ip = sys.argv[1]
def_path = 'CompleteDataSets/Apartment/Evaluation/'

# Bounds for time frame (in seconds) for calculating precision and recall
lower_limit = 3
upper_limit = 5


def calc_precision(time_slices, gt):

    # Calculate precision
    ts_result = time_slices.ix[:, ['start_time', 'end_time', 'room']].sort(['start_time'])
    ts_result = ts_result.sort(['start_time'])
    print ts_result
    gt = gt.reset_index(drop=True)
    gt = gt.sort(['start_time'])
    gt['start_time'] = gt.start_time 
    gt['end_time'] = gt.end_time 
    gt['room'] = gt.room
    print gt
    ts_result['precision'] = np.zeros(len(ts_result))

    for pidx in ts_result.index:
        pred_st = ts_result.ix[pidx]['start_time']
        pred_et = ts_result.ix[pidx]['end_time']
        pred_rm = ts_result.ix[pidx]['room']
        for idx in gt.index:
            true_st = long(gt.ix[idx]['start_time'])
            true_et = long(gt.ix[idx]['end_time'])
            true_rm = gt.ix[idx]['room']
            if pred_st == '':
                pred_st = 0
                pred_et = 0

            if (pred_st in range(true_st - lower_limit, true_st + upper_limit + 1) and
               pred_et in range(true_et - lower_limit, true_et + upper_limit + 1) ):
                ts_result.ix[pidx, 'precision'] = 1
                # print "\n TS ST", dt.datetime.fromtimestamp(pred_st), \
                #     "TS ET", dt.datetime.fromtimestamp(pred_et)
                # print "\n GT ST", dt.datetime.fromtimestamp(true_st), \
                #     "GT ET", dt.datetime.fromtimestamp(true_et)
                # print ".....RESULT:: 1"
                break
            else:
                # print "\n TS ST", dt.datetime.fromtimestamp(pred_st), \
                #     "TS ET", dt.datetime.fromtimestamp(pred_et)
                # print "\n GT ST", dt.datetime.fromtimestamp(true_st), \
                #     "GT ET", dt.datetime.fromtimestamp(true_et)
                # print ".....RESULT:: 0"

                ts_result.ix[pidx, 'precision'] = 0

    # Calculate accuracy
    # ts_acc = total correct/ total points
    tot_pcorrect = len(ts_result[ts_result.precision == 1])
    ts_precision = 0
    try:
        ts_precision = (float(tot_pcorrect) / len(ts_result)) * 100
    except Exception,e:
        print 'no time_slices found'
    # For printing
    ts_result['act_start_time'] = [dt.datetime.fromtimestamp(i)
                                   for i in ts_result['start_time']]
    ts_result['act_end_time'] = [dt.datetime.fromtimestamp(i)
                                 for i in ts_result['end_time']]
    ts_result = ts_result.sort(['act_start_time'])
    print "Precision table\n", ts_result

    return ts_precision


def calc_recall(time_slices, gt):

    # Calculate recall

    ts_result = time_slices.ix[:, ['start_time', 'end_time','room']].sort(['start_time'])
    ts_result = ts_result.sort(['start_time'])

    gt = gt.reset_index(drop=True)
    gt = gt.sort(['start_time'])
    gt['start_time'] = gt.start_time 
    gt['end_time'] = gt.end_time 
    gt['room'] = gt.room
    gt['recall'] = np.zeros(len(gt))
    for pidx in ts_result.index:

        pred_st = ts_result.ix[pidx]['start_time']
        pred_et = ts_result.ix[pidx]['end_time']
        pred_rm = ts_result.ix[pidx]['room']
        for idx in gt.index:

            true_st = long(gt.ix[idx]['start_time'])
            true_et = long(gt.ix[idx]['end_time'])
            true_rm = gt.ix[idx]['room']
            if pred_st == '':
                pred_st = 0
                pred_et = 0

            if (pred_st in range(true_st - lower_limit, true_st + upper_limit + 1) and
               pred_et in range(true_et - lower_limit, true_et + upper_limit + 1) and (str(pred_rm) in str(true_rm))):
                # print "\n TS ST", dt.datetime.fromtimestamp(pred_st), \
                #     "TS ET", dt.datetime.fromtimestamp(pred_et)
                # print "\n GT ST", dt.datetime.fromtimestamp(true_st), \
                #     "GT ET", dt.datetime.fromtimestamp(true_et)
                # print ".....RESULT:: 1"

                gt.ix[idx, 'recall'] = 1
                break

    # Calculate accuracy
    # ts_acc = total correct/ total points
    ts_recall  = 0
    tot_rcorrect = len(gt[gt.recall == 1])
    try:
        ts_recall = (float(tot_rcorrect) / len(gt)) * 100
    except Exception,e:
        print 'no time_slices found '
    # For printing
    gt['act_start_time'] = [dt.datetime.fromtimestamp(i)
                            for i in gt['start_time']]
    gt['act_end_time'] = [dt.datetime.fromtimestamp(i)
                          for i in gt['end_time']]
    gt = gt.sort(['act_start_time'])
    print "Recall table\n", gt

    return ts_recall


def calc_ts_accuracy(time_slices, gt):
    """
    Calculate time slice generation precision/recall
    """
    ts_precision = calc_precision(time_slices, gt)
    ts_recall = calc_recall(time_slices, gt)

    print "Time Slice Precision:", ts_precision
    print "Time Slice Recall:", ts_recall

    return ts_precision, ts_recall


def calc_loc_accuracy(time_slices, gtfile):
    """
    Calculate precision/recall for localization stage
    """
    # Label with ground truth
    df = af.label_with_ground_truth(time_slices, gtfile)
    df = df[df.true_location != 'Not Found']
    print "Location TS With GT\n", df

    j = 'location'
    true = df['true_' + j]
    pred = df['pred_' + j]
    classes = list(set(true.unique().tolist() + pred.unique().tolist()))
    print "Classes", classes
    true = []
    pred = []
    for l in df.index:
        for m in classes:
            if df.ix[l]['true_' + j] == m:
                true.append(classes.index(m))
    for l in df.index:
        for m in classes:
            if df.ix[l]['pred_' + j] == m:
                pred.append(classes.index(m))
    print j, true, pred
    cm = confusion_matrix(true, pred)
    print cm

    precision = math.ceil(precision_score(true, pred) * 100 * 100) / 100
    recall = math.ceil(recall_score(true, pred) * 100 * 100) / 100

    print "Location Precision", precision
    print "Location Recall", recall


def find_precision_recall(true, pred, labels):

    labels = np.array(labels)
    p, r, f1, s = precision_recall_fscore_support(true, pred,
                                                  labels=labels,
                                                  average=None)

    # compute averages
    precision = np.average(p, weights=s)
    recall = np.average(r, weights=s)

    return precision, recall

if __name__ == '__main__':

    # Get all files
    # files = glob.glob("output_app*.csv")

    exp_no = sys.argv[1]
    phno = sys.argv[2]
    app_list = sys.argv[3]
    apt_no = sys.argv[4]
    algo_type = sys.argv[5]
    # try:
    #     a = int(exp_no)
    # except ValueError, e:
    #     apt_no = raw_input("Enter Apt No::")

    app_list = app_list.split(',')
    app_list = app_list + ['6']

    # Calculate precision and recall for all three approaches and
    # store precision and recall values in df
    def_path = ''
    if apt_no not in ['603', '703', '802']:
        def_path = 'CompleteDataSets/Apartment/Evaluation/'
        # if phno == 'nil':
        #     op_file = def_path + 'results/exp' + exp_no + 'meter.csv'
        # else:
        #     op_file = def_path + 'results/exp' + exp_no + '_' + phno + '.csv'
        op_file = def_path + 'results/exp' + exp_no + '_' + phno + '.csv'
    else:
        def_path = 'CompleteDataSets/Apartment/' + apt_no + '/'
        # if phno == 'nil':
        #     op_file = def_path + 'results/' + exp_no + 'meter.csv'
        # else:
        #     op_file = def_path + 'results/' + exp_no + '_' + phno + '.csv'
        op_file = def_path + 'results/' + exp_no + '_' + phno + '.csv'

    op_df = pd.DataFrame(
        columns=['app', 'location_prec', 'location_recall', 'appliance_prec',
                        'appliance_recall', 'both_prec', 'both_recall', 'ts_prec', 'ts_recall'
                        'who_prec', 'who_recall'])

    df_list = []
    appl_classes = pd.Series(['Light', 'Fan', 'Microwave', 'Kettle', 'AC', 'TV'])
    location_classes = pd.Series(['Bedroom', 'Dining Room', 'Kitchen'])
    # For each approach, compile the results of all the approaches together
    for app_no, app_item in enumerate(app_list):
        app = str(app_item)
        who_prec = 'NA'
        who_recall = 'NA'
        if apt_no not in ['603', '703', '802']:
            if app in ['6', '7']:
                if algo_type == 'multi':
                    filenm = def_path + 'exp' + exp_no + \
                        '/output_app' + app + '_' + phno + '_meter.csv'
                else:
                    filenm = def_path + 'exp' + exp_no + '/output_app' + app + '_meter.csv'
            else:
                filenm = def_path + 'exp' + exp_no + '/output_app' + app + '_' + phno + '.csv'
        else:
            if app in ['6', '7']:
                if algo_type == 'multi':
                    filenm = def_path + exp_no + '/output_app' + app + '_' + phno + '_meter.csv'
                else:
                    filenm = def_path + exp_no + '/output_app' + app + '_meter.csv'
            else:
                filenm = def_path + exp_no + '/output_app' + app + '_' + phno + '.csv'

            appl_classes = pd.Series(['Light', 'Fan', 'Microwave', 'Kettle', 'TV'])
            location_classes = pd.Series(['Master Bedroom', 'Dining Room', 'Kitchen'])

        print "\nInput File::",  filenm, "\n"

        if not os.path.isfile(filenm):
            print "No file found", filenm
            continue

        app_df = pd.read_csv(filenm)
        app_df = app_df[app_df.true_location != 'Not Found']

        # print app_df
        ts_prec = app_df['ts_prec'][0]
        ts_recall = app_df['ts_recall'][0]
        if app in ['3', '4', '5'] and (algo_type == 'multi'):
            who_prec = app_df['who_prec'][0]
            who_recall = app_df['who_recall'][0]
        print app_df.ix[:, app_df.columns - ['start_time', 'end_time', 'type']]

        lists = [[0 for _ in range(2)] for _ in range(3)]
        true = []
        pred = []
        catgy = ['location', 'appliance', 'both']

        # Check if "Not Found" label is present in the any of labels - Location/Appliance
        flag = [False] * 3
        for i, j in enumerate(['location', 'appliance']):
            if "Not Found" in list(app_df['pred_' + j]):
                flag[i] = True
                print j, "is", flag[i]

        df = app_df.copy()
        if not flag[0] and not flag[1]:
            df['true_both'] = df.true_appliance + '-' + df.true_location
            df['pred_both'] = df.pred_appliance + '-' + df.pred_location

            # Calculate precision/recall where "Not Found" is not present
            for idx, j in enumerate(catgy):
                true = df['true_' + j]
                pred = df['pred_' + j]
                classes = list(set(true.unique().tolist() + pred.unique().tolist()))
                print "\nClasses", classes

                labels = range(len(classes))
                print "Class Labels", labels

                true = []
                pred = []
                for l in df.index:
                    for m in classes:
                        if df.ix[l]['true_' + j] == m:
                            true.append(classes.index(m))
                for l in df.index:
                    for m in classes:
                        if df.ix[l]['pred_' + j] == m:
                            pred.append(classes.index(m))
                print j, true, pred

                # Calculate precision/recall for the category where "Not Found" entry was found
                prec, recall = find_precision_recall(true, pred, labels)
                lists[idx][0] = math.ceil(prec * 100 * 100) / 100
                lists[idx][1] = math.ceil(recall * 100 * 100) / 100
                print "Precision/Recall for", j, ":\n", lists
        else:

            # Calculate precision/recall for 1000 times and take average
            counter = 1000
            loc_prec_list = []
            loc_recall_list = []
            print "Calculating Precision/Recall..."

            for ctr in range(counter):

                for idx, j in enumerate(catgy):

                    if j != "both":
                        true = df['true_' + j]
                        pred = df['pred_' + j]
                        classes = list(set(true.unique().tolist() + pred.unique().tolist()))

                        # Replace "Not Found" entry with an incorrect label
                        if "Not Found" in classes:
                            flag[i] = True
                            classes.remove("Not Found")

                        labels = range(len(classes))
                        # print "\nClasses", classes
                        # print "Class Labels", labels

                        classes = pd.Series(classes)
                        df_tmp = df[df['pred_' + j] == "Not Found"]
                        for l in df_tmp.index:
                            true_l = df.ix[l]['true_' + j]
                            not_true_lbl = classes[classes != true_l]
                            if len(not_true_lbl) == 0:
                                if j == 'location':
                                    not_true_lbl = location_classes[location_classes != true_l]
                                else:
                                    not_true_lbl = appl_classes[appl_classes != true_l]
                                classes = classes.append(not_true_lbl)
                                classes.reset_index(drop=True, inplace=True)
                            incorrect_label = classes.ix[
                                randint(min(not_true_lbl.index), max(not_true_lbl.index))]
                            df.ix[l, 'pred_' + j] = incorrect_label
                            # print "idx::", l, j, "replaced with", incorrect_label
                        classes = list(classes)

                    else:
                        df['true_both'] = df.true_appliance + '-' + df.true_location
                        df['pred_both'] = df.pred_appliance + '-' + df.pred_location

                        true = df['true_' + j]
                        pred = df['pred_' + j]
                        classes = list(set(true.unique().tolist() + pred.unique().tolist()))
                        labels = range(len(classes))
                        # print "\nClasses", classes
                        # print "Class Labels", labels

                    true = []
                    pred = []
                    for l in df.index:
                        for m in classes:
                            if df.ix[l]['true_' + j] == m:
                                true.append(classes.index(m))
                    for l in df.index:
                        for m in classes:
                            if df.ix[l]['pred_' + j] == m:
                                pred.append(classes.index(m))
                    # print j, true, pred

                    # Calculate precision/recall for the category where "Not Found" entry was found
                    prec, recall = find_precision_recall(true, pred, labels)
                    # print j, "Precision::", prec * 100, "Recall:", recall * 100
                    lists[idx][0] += prec
                    lists[idx][1] += recall
                    # Plotting purposes
                    # if j == 'location':
                    #     print ctr, prec * 100, lists[idx][0] * 100,
                    # lists[idx][0] * 100 / (ctr + 1)
                    #     loc_prec_list.append(lists[idx][0] * 100 / (ctr + 1))
                    #     loc_recall_list.append(lists[idx][1] * 100 / (ctr + 1))

                df = app_df.copy()

            # Take average
            for i in range(3):
                lists[i][0] = math.ceil((lists[i][0] / counter) * 100 * 100) / 100
                lists[i][1] = math.ceil((lists[i][1] / counter) * 100 * 100) / 100
            print "Precision/Recall for", j, ":\n", lists

            # Plot location precision/recall plot
            # plt.plot(loc_prec_list)
            # plt.plot(loc_recall_list)
            # plt.show()

        op_df = pd.DataFrame({'app': app_no + 1, 'location_prec': lists[0][0],
                              'location_recall': lists[0][1], 'appliance_prec': lists[1][0],
                              'appliance_recall': lists[1][1], 'both_prec': lists[2][0],
                              'both_recall': lists[2][1], 'ts_prec': ts_prec,
                              'ts_recall': ts_recall, 'who_prec': who_prec,
                              'who_recall': who_recall}, index=[app_item])
        # print "df_i \n", op_df
        df_list.append(op_df)

        # print "\ndf_list \n", df_list
    # sys.exit(1)

    # print "Len df_list", len(df_list)
    # f_df = op_df
    f_df = pd.concat(df_list)
    print "\nFinal precision/matrix::\n", f_df
    print "Making Results File::", op_file
    f_df.to_csv(op_file, index=False)

    sys.exit(1)
    # ----------------------------- END -------------------------------------

    # Activity
    classes = df.appliance.unique()

    true = df.appliance
    pred = df.pred_appliance
    print "Predicted classes:", pred.unique()

    cm = confusion_matrix(true.tolist(), pred.tolist())
    cls_report = (classification_report(true.tolist(),
                                        pred.tolist(),
                                        labels=classes, target_names=classes))
    print "Confusion Matrix::\n", cm
    print "Classification Report::\n", cls_report

    print precision

    # Location
    classes = df.location.unique()

    true = df.location
    pred = df.pred_location
    print "Predicted classes:", pred.unique()

    cm = confusion_matrix(true.tolist(), pred.tolist())
    cls_report = (classification_report(true.tolist(),
                                        pred.tolist(),
                                        labels=classes, target_names=classes))
    print "Confusion Matrix::\n", cm
    print "Classification Report::\n", cls_report

    # Activity + Location
    classes = df.true_both.unique()

    true = df.true_both
    pred = df.pred_both
    print "Predicted classes:", pred.unique()

    cm = confusion_matrix(true.tolist(), pred.tolist())
    cls_report = (classification_report(true.tolist(),
                                        pred.tolist(),
                                        labels=classes, target_names=classes))
    print "Confusion Matrix::\n", cm
    print "Classification Report::\n", cls_report
