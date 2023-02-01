import os

import numpy as np
import seaborn as sns
import pandas as pd
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
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.show()

    plt.plot(history.history['loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_cross_val(history, name):
    def get_mean(metric):
        sum = []
        sum_val = []
        for i, h in enumerate(history):
            if sum == []:
                sum = h.history[metric]
                sum_val = h.history[f'val_{metric}']
            else:
                sum = np.add(sum, h.history[metric])
                sum_val = np.add(sum_val, h.history[f'val_{metric}'])
        mean = [i/5 for i in sum]
        mean_val = [i/5 for i in sum_val]
        return mean, mean_val
    
    mean_acc, mean_val_acc = get_mean('sparse_categorical_accuracy')
    mean_loss, mean_val_loss = get_mean('loss')
    fig, axs = plt.subplots(1,2, figsize=(14,6))
    axs[0].plot(mean_acc, 'g', label='Training')
    axs[0].plot(mean_val_acc, 'b', label='Validation')
    axs[0].set_ylim(0.6, 1.0)
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_title("Average training- and validation accuracy")
    axs[0].grid()

    axs[1].plot(mean_loss, 'g', label='Training')
    axs[1].plot(mean_val_loss, 'b', label='Validation')
    axs[1].set_ylim(0, 1.0)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Average training- and validation loss")
    axs[1].grid()
    axs[1].legend(loc="upper right")
    fig.suptitle(f"5-fold Cross-Validation on the {name} Training Dataset")
    fig.savefig(f"cross_val_{name}.pdf")
    axs[0].cla()
    axs[1].cla()
    fig.clf()
    plt.clf()

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

def plot_cm(cm, name, hc=None):
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    plt.xlabel("Predicted Speaker")
    plt.ylabel("True Speaker")
    plt.title(f"Confusion Matrix for {name} Training Dataset")
    plt.savefig(f"cm_{name}_{hc}.pdf")
    plt.clf()