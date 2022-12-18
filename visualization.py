import os

import numpy as np
import seaborn as sns

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import utils

def save_plot(name):
    FIG_PATH = "figures"

    if not os.path.isdir(FIG_PATH):
        os.mkdir(FIG_PATH)

    plt.savefig(os.path.join(FIG_PATH, name + ".pdf"), bbox_inches="tight")

def plot_recording(ax, rec_data, title=None):
    palette = sns.color_palette("Spectral",n_colors=12)
    [ax.plot(x, color=palette[i]) for i, x in enumerate(rec_data)]
    ax.set_xlim(0, 22)
    ax.set_ylim(-1.5, 2)
    if title:
        ax.title.set_text("$\it{" + title + "}$")

def add_axis_labels(fig):
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time Step")
    plt.ylabel("LPCC")

def plot_recordings(X):
    palette = sns.color_palette("Spectral",n_colors=12)
    fig, ax = plt.subplots(nrows=3, ncols=3)
    fig.tight_layout(pad=1.5)


    max_range = 30
    min_range = 0
    count = 0

    for row in ax:
        for col in row:
            random_point = np.random.randint(min_range, max_range)
            min_range += 30
            max_range += 30
            plot_recording(col, X[random_point].T, title="Speaker \;" + str(count))
            count += 1

    utils.seed()
    add_axis_labels(fig)
    fig.suptitle("Random Recordings from each Speaker", y=1.1)
    patches = [mpatches.Patch(color=c, label="c-" + str(i + 1)) for i, c in enumerate(palette)]
    plt.legend(handles=patches, bbox_to_anchor=(1.2, 0.9))
    save_plot("rand_speakers")
    plt.show()

def plot_cnn_training(history):
    plt.plot(history.history['sparse_categorical_accuracy'])
    #plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.0)
    plt.show()

    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_rf_training(cm):
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    plt.xlabel("Predicted Speaker")
    plt.ylabel("True Speaker")
    plt.show()

def plot_tsne(x, y, lim=24):
    palette = dict(zip(np.arange(0, 9), sns.color_palette(n_colors=9)))
    tsne = TSNE(n_components=2, init='random', random_state=utils.RANDOM_STATE, learning_rate='auto')
    X_tsne = tsne.fit_transform(x)
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette=palette)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_rec_len_freq(data_ls, title, xmax=30, xlab="Recording Length"):
    lens = [x.shape[0] for x in data_ls]
    min_, max_ = min(lens), max(lens)
    print("Min length:", min_, "Max length:", max_)
    plt.hist(lens)
    plt.xlim(0, xmax)
    plt.xlabel(xlab)
    plt.ylabel("Frequency")
    plt.title(title)
    save_plot(title + "rec_len")
    plt.show()
