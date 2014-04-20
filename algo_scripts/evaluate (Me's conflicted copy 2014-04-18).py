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
from sklearn.metrics import *
from random import randint
import warnings
import numpy
import csv
from scipy import *
import math
import datetime

# Disable warnings
warnings.filterwarnings('ignore')

# ip = sys.argv[1]
def_path = 'CompleteDataSets/Apartment/Evaluation/'

# Bounds for time frame (in seconds) for calculating precision and recall
lower_limit = 50
upper_limit = 50


def calc_precision(time_slices, gt):

    # Calculate precision
    ts_result = time_slices.sort(['Event Time'])
    ts_result.index = arange(0, len(ts_result))
    ts_result = ts_result.drop_duplicates()
    #ts_result = ts_result.sort(['Event Time'])
    #print ts_result
    gt = gt.reset_index(drop=True)
    gt = gt.sort(['Event Time'])
    gt['Event Time'] = gt['Event Time'] / 1000
    #gt['Event Time'] = gt.end_time / 1000

    ts_result['precision'] = np.zeros(len(ts_result))

    for pidx in ts_result.index:
        pred_st = ts_result.ix[pidx]['Event Time']
        pred_rm = ts_result.ix[pidx]['Room']
        #pred_event = ts_result.ix[pidx]['Event']
        #pred_et = ts_result.ix[pidx]['end_time']

        for idx in gt.index:
            true_st = long(gt.ix[idx]['Event Time'])
            true_rm = gt.ix[idx]['Room']
            #true_event = gt.ix[idx]['Event']
            #true_et = long(gt.ix[idx]['end_time'])
            if pred_st == '':
                pred_st = 0
                #pred_et = 0
            #print type(true_st), type(pred_st), type(pred_rm), type(true_rm)
            if (pred_st in range(int(true_st) - lower_limit, int(true_st) + upper_limit + 1) and (str(pred_rm) in str(true_rm))):
                ts_result.ix[pidx, 'precision'] = 1
                #print 'True'
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
    #print 'precision'
    print tot_pcorrect, len(ts_result)
    if ((tot_pcorrect > 0) & (len(ts_result) > 0)):
        ts_precision = (float(tot_pcorrect) / len(ts_result)) * 100
    else:
        ts_precision = 0
    print 'Precision', ts_precision
    # For printing
    
    # ts_result['act_time'] = [dt.datetime.fromtimestamp(i)
    #                                for i in ts_result['Event Time']]
    

    # ts_result['act_end_time'] = [dt.datetime.fromtimestamp(i)
    #                              for i in ts_result['end_time']]
    

    #ts_result = ts_result.sort(['act_time'])
    print "Precision table\n", ts_result

    writer = csv.writer(open('/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/GroundTruth/precision.csv', "w"))

    # For test dataset

    writer.writerow(ts_result.columns.values)

    for idx in ts_result.index:
        writer.writerow([ts_result.ix[idx]['Edge']] + [datetime.datetime.fromtimestamp(ts_result.ix[idx]['Event Time'])] + [ts_result.ix[idx]['Appliance']] + [ts_result.ix[idx]['Magnitude']] + [ts_result.ix[idx]['Phase']] + [ts_result.ix[idx]['Room']] + [ts_result.ix[idx]['precision']])

    # For test dataset
   
    # For test dataset

   
    #writer.writerow(ts_percision.columns.values)
    #writer.writerow([ts_precision])

    return ts_precision


def calc_recall(time_slices, gt):

    # Calculate recall
    ts_result = time_slices.sort(['Event Time'])
    ts_result.index = arange(0, len(ts_result))
    ts_result = ts_result.drop_duplicates()
    gt = gt.reset_index(drop=True)
    gt = gt.sort(['Event Time'])
    gt['Event Time'] = gt['Event Time'] / 1000
    #gt['end_time'] = gt.end_time / 1000

    gt['recall'] = np.zeros(len(gt))
    for pidx in ts_result.index:

        pred_st = ts_result.ix[pidx]['Event Time']
        pred_rm = ts_result.ix[pidx]['Room']
        #pred_event = ts_result.ix[pidx]['Event']
        #pred_et = ts_result.ix[pidx]['time']

        for idx in gt.index:

            true_st = long(gt.ix[idx]['Event Time'])
            true_rm = gt.ix[idx]['Room']
            #true_event = gt.ix[idx]['Event']
            #true_et = long(gt.ix[idx]['time'])

            if pred_st == '':
                pred_st = 0
                #pred_et = 0
            if (pred_st in range(int(true_st) - lower_limit, int(true_st) + upper_limit + 1) and (str(pred_rm) in str(true_rm))):
                #print 'true'
            #if (pred_st in range(true_st - lower_limit, true_st + upper_limit + 1) and (pred_rm in true_rm)):
                # print "\n TS ST", dt.datetime.fromtimestamp(pred_st), \
                #     "TS ET", dt.datetime.fromtimestamp(pred_et)
                # print "\n GT ST", dt.datetime.fromtimestamp(true_st), \
                #     "GT ET", dt.datetime.fromtimestamp(true_et)
                # print ".....RESULT:: 1"

                gt.ix[idx, 'recall'] = 1
                break

    # Calculate accuracy
    # ts_acc = total correct/ total points
    tot_rcorrect = len(gt[gt.recall == 1])
    #print 'recall'
    print tot_rcorrect, len(gt)
    if ((tot_rcorrect > 0) & (len(gt) > 0)):
        ts_recall = (float(tot_rcorrect) / len(gt)) * 100
    else:
        ts_recall = 0
    #print 'Recall', ts_recall
    # For printing
    # gt['act_time'] = [dt.datetime.fromtimestamp(i)
    #                         for i in gt['Event Time']]
    # gt['act_end_time'] = [dt.datetime.fromtimestamp(i)
    #                       for i in gt['end_time']]
    #gt = gt.sort(['act_time'])
    print "Recall table\n", gt

    writer = csv.writer(open('/home/shailja/shailja.thakur90@gmail.com/Thesis/DataSet/GroundTruth/recall.csv', "w"))

    # For test dataset

    writer.writerow(gt.columns.values)
    for idx in gt.index:
        writer.writerow([datetime.datetime.fromtimestamp(gt.ix[idx]['Event Time'])] + [gt.ix[idx]['Room']] + [gt.ix[idx]['Event']] + [gt.ix[idx]['recall']])

    #writer.writerow([gt])

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

def calc_room_accuracy(time_slices, gt):

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
    print "Labels", labels
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
    # For each approach, compile the results of all the approaches together
    for app_no, i in enumerate(app_list):
        app = str(i)
        who_prec = 'NA'
        who_recall = 'NA'
        if apt_no not in ['603', '703', '802']:
            if app in ['6', '7']:
                filenm = def_path + 'exp' + exp_no + '/output_app' + app + '_meter.csv'
            else:
                filenm = def_path + 'exp' + exp_no + '/output_app' + app + '_' + phno + '.csv'
        else:
            if app in ['6', '7']:
                filenm = def_path + exp_no + '/output_app' + app + '_meter.csv'
            else:
                filenm = def_path + exp_no + '/output_app' + app + '_' + phno + '.csv'

        print "\nInput File::",  filenm, "\n"

        if not os.path.isfile(filenm):
            print "No file found", filenm
            continue

        df = pd.read_csv(filenm)
        df = df[df.true_location != 'Not Found']

        # print df
        ts_prec = df['ts_prec'][0]
        ts_recall = df['ts_recall'][0]
        if app in ['3', '4', '5'] and (algo_type == 'multi'):
            who_prec = df['who_prec'][0]
            who_recall = df['who_recall'][0]
        print df.ix[:, df.columns - ['start_time', 'end_time']]

        lists = [([], []) for _ in range(3)]
        true = []
        pred = []
        catgy = ['location', 'appliance', 'both']

        # Check if "Not Found" is present in the any of labels - Location/Appliance
        flag = [False] * 3
        # if "Not Found"
        df['true_both'] = df.true_appliance + '-' + df.true_location
        df['pred_both'] = df.pred_appliance + '-' + df.pred_location

        for idx, j in enumerate(catgy):
            true = df['true_' + j]
            pred = df['pred_' + j]
            classes = list(set(true.unique().tolist() + pred.unique().tolist()))

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

            for i, j in enumerate(['location', 'appliance']):
                true_lbl = df['true_' + j]
                pred_lbl = df['pred_' + j]
                classes = list(set(true_lbl.unique().tolist() + pred_lbl.unique().tolist()))

                # Replace "Not Found" entry with an incorrect label
                if "Not Found" in classes:
                    flag[i] = True
                    classes.remove("Not Found")
                else:
                    break
                classes = pd.Series(classes)
                df_tmp = df[df.pred_location == "Not Found"]
                for l in df_tmp.index:
                    true_l = df.ix[l]['true_' + j]
                    not_true_lbl = classes[classes != true_l]
                    incorrect_loc_label = classes.ix[
                        randint(min(not_true_lbl.index), max(not_true_lbl.index))]
                    df.ix[l, 'pred_' + j] = incorrect_loc_label
                    print "idx::", l, j, "replaced with", incorrect_loc_label

            # Calculate precision/recall for the category where "Not Found" entry was found
            if flag[idx]:
                sum_prec = 0
                sum_recall = 0
                for i in range(1000):
                    prec, recall = find_precision_recall(true, pred, labels)
                    sum_prec += prec
                    sum_recall += recall
                prec = sum_prec / 1000
                recall = sum_recall / 1000
            else:
                prec, recall = find_precision_recall(true, pred, labels)

            lists[idx][0].append(math.ceil(prec * 100 * 100) / 100)
            lists[idx][1].append(math.ceil(recall * 100 * 100) / 100)
            print "Precision/Recall for", j, ":\n", lists
            # print "\n"
            # print "Classes", classes
            # print "catgy", j
            # print "precision", lists[idx][0]
            # print "recall", lists[idx][1]
            # print "weighted precision", precision_score(true, pred,
                # labels=classes, average='weighted')
            # print "weighted recall", recall_score(true, pred, labels=classes, average='weighted')

        op_df = pd.DataFrame({'app': app_no + 1, 'location_prec': lists[0][0],
                              'location_recall': lists[0][1], 'appliance_prec': lists[1][0],
                              'appliance_recall': lists[1][1], 'both_prec': lists[2][0],
                              'both_recall': lists[2][1], 'ts_prec': ts_prec,
                              'ts_recall': ts_recall, 'who_prec': who_prec,
                              'who_recall': who_recall}, index=[i])
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
