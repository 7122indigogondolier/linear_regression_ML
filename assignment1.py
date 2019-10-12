"""
Author: Utkrist P. Thapa '21
CSCI 315: Artificial Intelligence
Assignment 1
This program implements linear regression and uses gradient descent ML algorithm to optimize parameters
for the regression line and plots the intermediates and final graphs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# define constants
COL = 12                              # Choose independant variable by choosing column number
LAST_COL = 13

#define initial values for b0 and b1 
initial_b0 = random.randint(0, 1)    # some random initial value for b0
initial_b1 = random.randint(-1, 1)   # some random initial value for b1


# define hyperparameter L with fine tuned learning rates
learningRate = [0.011, 0.00011, 0.005750, 0.0005, 0.035,
                0.01, 0.00018, 0.00088, 0.00098, 0.0000045,
                0.002, 0.000007, 0.001]

# define column names
COL_NAMES = ["Crime rate by town", "Residential land zoned for lots over 25,000 sq.ft.",
             "Proportion of non retail business acres per town",
             "River variable (= 1 if tract bounds river; 0 otherwise)",
             "Nitric oxides concentration (parts per 10 million)",
             "Average number of rooms per home", "Proportion of owner occupied units built prior to 1940",
             "Weighted distances to five employment centres", "Index of accessibility to radial highways",
             "Full value property tax rate per 10,000s", "Student-teacher ratio by town", "Ethncity Demographic",
             "Percentage lower status of the population", "Median value of owner occupied homes in $1000s"]


def getInputData(filename, col):
    data = pd.read_csv(filename)
    X = data.iloc[:, col]
    Y = data.iloc[:, LAST_COL]
    return [X, Y]

def plotGraphWithLine(x, y, b0, b1):
    plt.figure()
    y_pred = b0 + (b1 * x)
    plt.plot(x, y_pred, color = 'red')
    plt.scatter(x, y)
    plt.xlabel(COL_NAMES[COL])
    plt.ylabel(COL_NAMES[LAST_COL])
    plt.title(COL_NAMES[LAST_COL] + " vs " + COL_NAMES[COL])
    plt.show()
    return None

def plotGraphWithoutLine(x, y):
    plt.scatter(x, y)
    plt.xlabel(COL_NAMES[COL])
    plt.ylabel(COL_NAMES[LAST_COL])
    plt.title(COL_NAMES[LAST_COL] + " vs " + COL_NAMES[COL])
    plt.show()
    return None

def runGradientDescent(X, Y, b0, b1, iterations, L):
    n = float(len(X))
    plotGraphWithLine(X, Y, b0, b1)
    for i in range(iterations):                  # Training the model 
        Y_pred = b0 + b1*X 
        D_b0 = (-2/n) * sum(Y - Y_pred)          # Derivative with respect to b0
        D_b1 = (-2/n) * sum(X * (Y - Y_pred))    # Derivative with respect to b1
        b0 = b0 - L * D_b0                       # updating values of b0 and b1 before next iteration
        b1 = b1 - L * D_b1
        if i % 1000 == 0:
            print("%10s%10s" % ("b0", "b1"))
            print("%10.3f%10.3f" % (b0, b1))
            print("")
            plotGraphWithLine(X, Y, b0, b1)      # Plotting graph every 1000 iterations in order to obtain intermediate graphs
    return [b0, b1]

def calculateError(X, Y, b0, b1):
    Y_pred = b0 + b1 * X
    E = Y - Y_pred
    SE = sum(E ** 2)
    MSE = (1/len(X)) * SE
    return MSE

def main():
    [X, Y] = getInputData("housing.csv", COL) # get data and store in X and Y
    print("%10s%10s" % ("b0", "b1"))
    print("%10.3f%10.3f" % (initial_b0, initial_b1))
    print("")
    plotGraphWithoutLine(X, Y) # plot initial graph
    [b0, b1] = runGradientDescent(X, Y, initial_b0, initial_b1, 10000, learningRate[COL]) # run gradient descent to optimize b0, b1
    error = calculateError(X, Y, b0, b1) # calculate the error from the optimized values
    print("Error: ", error)
    print("%10.3f%10.3f" % (b0, b1))
    print("")
    plotGraphWithLine(X, Y, b0, b1) # plot graph with the new values of b0 and b1
    

main()
    
    
