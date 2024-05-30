import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, path, inc=5, colorbar=False):
    classes_num = cm.shape[0]

    plt.figure(dpi=400)
    np.set_printoptions(precision=2)

    plt.imshow(
        cm,
        interpolation='nearest',
        cmap='GnBu',
    )
    if colorbar:
        plt.colorbar()

    plt.axvline(classes_num - inc - 0.5, linestyle='--', color='r')
    plt.axhline(classes_num - inc - 0.5, linestyle='--', color='r')
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    plt.xticks(np.arange(0, classes_num, inc), np.arange(0, classes_num, inc))
    plt.yticks(np.arange(0, classes_num, inc),
               np.arange(0, classes_num, inc),
               rotation='vertical')

    # offset the tick
    tick_marks = np.array(range(classes_num)) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    # plt.grid(True, which='minor', linestyle='-')

    # show confusion matrix
    plt.savefig(path, bbox_inches='tight')