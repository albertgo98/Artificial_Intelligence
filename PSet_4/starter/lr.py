
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main(age, weight, height):
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    age = normalize_data(age)
    weight = normalize_data(weight)
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.0005]
    b_0 = 0 
    b_age = 0 
    b_weight = 0

    # print('Age: ', age)
    print('Weight: ', weight)

    results = []
    for i in range(len(alphas)):
        limit = 100
        if i == len(alphas)-1:
            limit = 280
        b_0, b_age, b_weight = linear_regression(b_0, b_age, b_weight, age, weight, height, alphas[i], limit)
        results.append([alphas[i], limit, b_0, b_age, b_weight])

        b_0 = 0 
        b_age = 0 
        b_weight = 0

    print('Results: ', results)
    print('-------------------')

    costs = []
    for j in results:
        betas = [j[2], j[3], j[4]]
        R = find_cost(j, age, weight, height)
        costs.append(R)

    print('Costs: ', costs)

    return results, age, weight, height

def linear_regression(b_0, b_age, b_weight, age, weight, height, alpha, limit):
    iterations = 0

    print('Evaluating alpha = ', alpha)
    while iterations < limit:
        b_0, b_age, b_weight = gradient_descent(b_0, b_age, b_weight, age, weight, height, alpha)

        iterations += 1

    return b_0, b_age, b_weight

def gradient_descent(b_0, b_age, b_weight, age, weight, height, alpha):
    sum_errors_b0 = 0
    sum_errors_b_age = 0
    sum_errors_b_weight = 0
    for i in range(len(age)):
        y_pred = b_0 + b_age*age[i] + b_weight*weight[i]
        sum_errors_b0 += (y_pred-height[i])
        sum_errors_b_age += (y_pred-height[i])*age[i]
        sum_errors_b_weight += (y_pred-height[i])*weight[i]

    b_0 = b_0 - (alpha/len(height))*sum_errors_b0
    b_age = b_age - (alpha/len(height))*sum_errors_b_age
    b_weight = b_weight - (alpha/len(height))*sum_errors_b_weight

    return b_0, b_age, b_weight

def normalize_data(data):
    mu = data.mean()
    std_dev = data.std()
    print(mu, std_dev)

    data_scaled = []
    for x in data:
        x_scaled = (x-mu)/std_dev
        data_scaled.append(x_scaled)

    data_scaled = np.array(data_scaled)

    return data_scaled

def find_cost(betas, age, weight, height):
    squared_errors = 0
    n = len(height)
    for i in range(len(height)):
        squared_errors += ((betas[0] + betas[1]*age[i] + betas[2]*weight[i])-height[i])**2
    R = (1/(2*n))*squared_errors

    return R

def visualize_3d(df, lin_reg_weights=[1,1,1], feat1=0, feat2=1, labels=2,
                 xlim=(-2, 2), ylim=(-1.5, 4), zlim=(0, 3),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=''):
    """ 
    3D surface plot. 
    Main args:
      - df: dataframe with feat1, feat2, and labels
      - feat1: int/string column name of first feature
      - feat2: int/string column name of second feature
      - labels: int/string column name of labels
      - lin_reg_weights: [b_0, b_1 , b_2] list of float weights in order
    Optional args:
      - x,y,zlim: axes boundaries. Default to -1 to 1 normalized feature values.
      - alpha: step size of this model, for title only
      - x,y,z labels: for display only
      - title: title of plot
    """

    # Setup 3D figure
    ax = plt.figure().gca(projection='3d')
    # plt.hold(True)

    # Add scatter plot
    ax.scatter(df[feat1], df[feat2], df[labels])
    # ax.scatter(age, weight, height)

    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array( [lin_reg_weights[0] +
                       lin_reg_weights[1]*f1 +
                       lin_reg_weights[2]*f2  # height
                       for f1, f2 in zip(axes1, axes2)] )
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1, alpha=0.75)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    if title == '':
        title = 'LinReg Height with Alpha %f' % alpha
    ax.set_title(title)

    plt.show()

if __name__ == "__main__":

    data_file = str(sys.argv[1])
    results_file = str(sys.argv[2])

    data = pd.read_csv(data_file, header=None, usecols=[0, 1, 2], names=['age', 'weight', 'height'])
    age = data['age'].to_numpy()
    weight = data['weight'].to_numpy()
    height = data['height'].to_numpy()

    # print('Age: ', age)
    # print('Weight: ', weight)
    # print('Height: ', height)

    results, age, weight, height = main(age, weight, height)

    results_df = pd.DataFrame(results, columns=['alphas', 'limit', 'b_0', 'age', 'weight'])
    results_df.to_csv(results_file, header=False, index=False)

    data = {'age': age,
            'weight': weight,
            'height': height}

    for i in results:
        visualize_3d(df=data, lin_reg_weights=[i[2], i[3], i[4]], feat1='age', feat2='weight', labels='height', title='Linear Regression with alpha = ' + str(i[0]))


