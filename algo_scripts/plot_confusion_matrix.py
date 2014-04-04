# Script to plot confusion matrix
import numpy as np
import matplotlib.pyplot as plt


def plot_cm(conf_arr, sensor_type, classes, imgname, n=None):

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    imgfile = sensor_type + '/confusion_matrix/' + imgname
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix(%s) for n = %d' % (sensor_type,n))
    plt.xticks(range(width), classes[:width])
    plt.yticks(range(height), classes[:height])
    plt.savefig(imgfile, format='png')
