import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

def main(data1, data2, values):
    '''YOUR CODE GOES HERE'''
    b = 0
    weight_1 = 0
    weight_2 = 0 
    print(values)
    weight_1, weight_2, b, all_weights = perceptron(data1, data2, values, weight_1, weight_2, b)
    print(weight_1, weight_2, b)
    return weight_1, weight_2, b, all_weights

def perceptron(data1, data2, values, weight_1, weight_2, b):

    converge = False
    all_weights = []
    while not converge:
        all_weights.append([weight_1, weight_2, b])
        count = 0

        for i in range(len(values)):
            f_x1 = weight_1*data1[i]
            f_x2 = weight_2*data2[i]
            f_x = f_x1 + f_x2

            if f_x > -b:
                f_x = 1 
            else:
                f_x = -1

            if values[i]*f_x <= 0:
                weight_1 = weight_1 + values[i]*data1[i]
                weight_2 = weight_2 + values[i]*data2[i]
                b = b + values[i]*1
            else:
                count += 1

        print(weight_1, weight_2, b)
        if count == len(values):
            converge = True

    return weight_1, weight_2, b, all_weights

def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1],
                      title=''):
    """
        Scatter plot feat1 vs feat2.
        Assumes +/- binary labels.
        Plots first and second columns by default.
        Args:
          - df: dataframe with feat1, feat2, and labels
          - feat1: column name of first feature
          - feat2: column name of second feature
          - labels: column name of labels
          - weights: [w1, w2, b] 
    """

    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feat1), str(feat2))
    ax.set_title(title)

    plt.show()

if __name__ == "__main__":
    """DO NOT MODIFY"""

    data_file = str(sys.argv[1])
    results_file = str(sys.argv[2])

    data = pd.read_csv(data_file, header=None, usecols=[0, 1, 2], names=['data1', 'data2', 'values'])
    data1 = data['data1'].to_numpy()
    data2 = data['data2'].to_numpy()
    values = data['values'].to_numpy()

    # visualize_scatter(df=data, feat1='data1', feat2='data2', labels='values', title='Data')

    weight_1, weight_2, b, all_weights = main(data1, data2, values)

    print(all_weights)

    results_df = pd.DataFrame(all_weights, columns=['weight1', 'weight2', 'b'])

    results_df.to_csv(results_file, header=False, index=False)

    visualize_scatter(df=data, feat1='data1', feat2='data2', labels='values', weights=[weight_1, weight_2, b], title='Perceptron Result | ' + 'Weight_1=' + str(weight_1) + '; Weight_2=' + str(weight_2) + '; b=' + str(b))







