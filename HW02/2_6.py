import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from decisionTree import DecisionTreeNode, train_tree, evaluate_tree
import sys
from typing import Callable

# Add matplotlib rc parameters to beautify plots
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Plot scatter plot of D1.txt and D2.txt
def plot_scatter(data, title, save=False):
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('./images/' + title + '_sct.png')
    plt.show()


def draw_decision_boundary(model_function: Callable, grid_abs_bound: float = 1.0, savefile: str = None):
    """`model_function` should be your model's formula for evaluating your decision tree, returning either `0` or `1`.
    \n`grid_abs_bound` represents the generated grids absolute value over the x-axis, default value generates 50 x 50 grid.
    \nUse `grid_abs_bound = 1.0` for question 6 and `grid_abs_bound = 1.5` for question 7.
    \nSet `savefile = 'plot-save-name.png'` to save the resulting plot, adjust colors and scale as needed."""

    colors = ['#91678f', '#afd6d2']  # hex color for [y=0, y=1]

    xval = np.linspace(grid_abs_bound, 0,
                       50).tolist()  # grid generation
    xdata = []
    for i in range(len(xval)):
        for j in range(len(xval)):
            xdata.append([xval[i], xval[j]])

    # creates a dataframe to standardize labels
    df = pd.DataFrame(data=xdata, columns=['x_1', 'x_2'])
    # applies model from model_function arg
    df['y'] = df.apply(model_function, axis=1)

    d_columns = df.columns.to_list()  # grabs column headers
    y_label = d_columns[-1]  # uses last header as label
    d_xfeature = d_columns[0]  # uses first header as x_1 feature
    d_yfeature = d_columns[1]  # uses second header as x_1 feature
    # sorts by label to ensure correct ordering in plotting loop
    df = df.sort_values(by=y_label)

    d_xlabel = f"feature  $\mathit{{{d_xfeature}}}$"  # label for x-axis
    dy_ylabel = f"feature  $\mathit{{{d_yfeature}}}$"  # label for y-axis
    plt.figure(figsize=(5, 5))  # set figure size
    plt.xlabel(d_xlabel, fontsize=10)  # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=10)  # set y-axis label
    legend_labels = []  # create container for legend labels to ensure correct ordering

    # loop through placeholder dataframe
    for i, label in enumerate(df[y_label].unique().tolist()):
        df_set = df[df[y_label] == label]  # sort according to label
        set_x = df_set[d_xfeature]  # grab x_1 feature set
        set_y = df_set[d_yfeature]  # grab x_2 feature set
        # marker='s' for square, s=40 for size of squares large enough
        plt.scatter(set_x, set_y, c=colors[i], marker='s', s=40)
        legend_labels.append(
            f"""{y_label} = {label}""")  # apply labels for legend in the same order as sorted dataframe

    ax = plt.gca()  # grab to set background color of plot
    # set aforementioned background color in hex color
    ax.set_facecolor('#2b2d2e')
    plt.legend(legend_labels)  # create legend with sorted labels

    if savefile is not None:  # save your plot as .png file
        plt.savefig(savefile)
    plt.show()  # show plot with decision bounds


def model_tree1(row):
    return evaluate_tree(tree1, row)


def model_tree2(row):
    return evaluate_tree(tree2, row)


if __name__ == "__main__":
    # Load data
    D1 = np.loadtxt('data/D1.txt')
    D2 = np.loadtxt('data/D2.txt')

    save = sys.argv[3]
    # Plot scatter plot
    plot_scatter(D1, 'D1', save)
    plot_scatter(D2, 'D2', save)

    # Train decision tree
    dataset = sys.argv[1]
    dataset2 = sys.argv[2]

    D1 = np.loadtxt("data/" + dataset + ".txt", delimiter=" ")
    tree1 = train_tree(D1, False)
    D2 = np.loadtxt("data/" + dataset2 + ".txt", delimiter=" ")
    tree2 = train_tree(D2, False)

    draw_decision_boundary(model_tree1, grid_abs_bound=1.0,
                           savefile='./images/' + dataset + '_bound.png')

    draw_decision_boundary(model_tree2, grid_abs_bound=1.0,
                           savefile='./images/' + dataset2 + '_bound.png')
