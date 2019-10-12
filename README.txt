README

Utkrist P. Thapa '21 
CSCI 315
Artificial Intelligence 

There should be folders in this directory named after column number of the independent variables. These folders contain all the intermediate and final graphs for the corresponding column.
Each graph also includes the values of b0 and b1 at that point of iteration. These values are above each of the graphs. The first number is always b0 and second number always b1. 
All functions have been described below.
======================================================
Resources: 
1) Importing matplotlib in python: https://matplotlib.org/tutorials/introductory/pyplot.html
2) Linear Regression using Gradient Descent: https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

Libraries used: 
1) numpy
2) pandas
3) matplotlib
======================================================


---HOW TO RUN LINEAR REGRESSION IN THE PROGRAM ASSIGNMENT1---

1) Open program file assignment1.py
2) Choose value for COL (between 0-12). 0 represents the first column in the housing.csv file and 12 is the last column for the independent variables.
3) Various functions have been defined in order to make the coding easier: 

	- getInputData(filename, col) 
		This function takes in the filename of the input data file as string and the column number of the independent variable we want as int
		It extracts the data from the file and assigns it in array form to two variables X and Y and returns these variables
	
	- plotGraphWithLine(x, y, b0, b1)
		This function takes in variables x and y as array of values obtained from the data file, and b0 and b1 as float (regression parameters)
		This function outputs a scatterplot graph with a regression line according to x, y, b0 and b1
		Returns None
	
	- plotGraphWithoutLine(x, y)
		This function takes in variables x and y as array of values obtained from the data file		
		This function outputs a scatterplot graph according to x and y without a regression line 
		Returns None

	- runGradientDescent(X, Y, b0, b1, iterations, L)
		This function takes in variables X and Y as array of values obtained from the data file, b0 and b1 as float (regression parameters), iterations as int (hyperparameter) and learning rate 		L as float (hyperparameter)
		This function implements the gradient descent formula and optimizes the values of b0 and b1 
		Returns optimized b0 and b1
	
	- calculateError(X, Y, b0, b1)
		This function takes variables X and Y as array of values obtained from the data file, and b0 and b1 as float (regression parameters)
		Calculates the mean squared error 
		Returns the value of the mean squared error as float 


4) Run the program in order to generate the intermediate graphs along with the final optimized best fit line graph. The program will only show one graph at a time so in order to take a look at the next graph, cancel out of the current graph (give it a moment because the iterations take a while to compute). 

Note: The program requires the data file to be in the same location as itself

======================================================


---EXPLAINING THE CODE IN ASSIGNMENT1.PY---

Assignment1.py is the program file for the first assignment in our AI class (CSCI 315). The program implements a linear regression algorithm which uses gradient descent to find the optimal values of regression parameters b0 and b1. 

The linear regression model is: 
	Y = b0 + (b1 * X)
where, 
Y -> Dependent variable or the target variable which stores an array of values (for housing.csv our target variable is the last column of data)
X -> Independent variable which stores an array of values (housing.csv has 13 other columns 0-12 that represent the independent variable)
b0 -> y-intercept (one of the parameters in the regression model that helps predict the relationship between X and Y)
b1 -> slope (another parameter in the regression model that helps predict the relationship between X and Y)

Since the relationship between X and Y is determined by the values of b0 and b1, we need to find values for b0 and b1 such that those values minimize the mean-squared-error (MSE) between the actual values of Y and the predicted values of Y (predicted using the linear regression model). This is called optimization. We use gradient descent for optimization.

To find the MSE, we use the following formula: 
	MSE = (1/n) * sum((Y - Y_pred) ** 2)
where, 
	Y_pred -> Predicted value of Y (predicted by the regression model)
	MSE is also referred to as the cost function or the loss function. 

To run gradient descent, we find partial derivatives of the cost function with respect to b0 and b1 which is denoted by D_b0 and D_b1 in the program. We then update the values of b0 and b1 before iterating again using the following formula: 
	b0 = b0 - L * D_b0
	b0 = b0 - L * D_b0
where, 

	L -> learning rate (hyperparameter)

The learning rate is the variable that determines how large of a step we will take along the cost function during each iteration. Ideally, we want to reach the global minimum where the error/cost is minimum. The learning rate has to be fine tuned for each set of X and Y. The fine tuned learning rates have been stored in an array called learningRate. The index of each of the learning rates inside this array represents the column number of the variable in housing.csv.  Here are the values for the learning rates for each column: 

col = 0, learning rate = 0.011
col = 1, learning rate = 0.00011
col = 2, learning rate = 0.005750	
col = 3, learning rate = 0.0005
col = 4, learning rate = 0.035
col = 5, learning rate = 0.01
col = 6, learning rate = 0.00018
col = 7, learning rate = 0.00088
col = 8, learning rate = 0.00098
col = 9, learning rate = 0.0000045
col = 10, learning rate = 0.002
col = 11, learning rate = 0.000007
col = 12, learning rate = 0.001
  

Once we run the gradient descent algorithm implemented in runGradientDescent() function, we obtain the optimized values for b0 and b1. We can then plot this graph.
======================================================

