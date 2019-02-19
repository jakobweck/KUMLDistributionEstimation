##Exploring 2D GMM's (Multi-dimensional)
##not perfectly implemented - clusters don't seem to separate nicely
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import argparse
import pandas as pd
import random
import sys
import os

class TwoDGaussianMM:
    def __init__(self,X,iterations,clusters,showAll):
        self.iterations = iterations
        self.data = X
        self.initData = X

        self.number_of_sources = clusters
        self.XY = None

        self.means = None
        self.weights = None
        self.covars = None
        self.showAll = showAll
        self.colors = 10*["g","r","c","b","k"]

    
    def runExMax(self):
        # Code sourced/referenced from https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
        # sets reg covariance to initial data
        print(self.initData)
        print(min(self.initData[:,0]))
        print(max(self.initData[:,0]))
        print((self.number_of_sources))
        self.reg_cov = 1e-6 * np.identity(len(self.initData[0]))
        # Creates a grid based on initial data, XY
        x,y = np.meshgrid(np.sort(self.initData[:,0]), np.sort(self.initData[:,1]))
        # Creates a matrix XY out of initial data
        self.XY = np.array([x.flatten(), y.flatten()]).T

        # Define the initial mean, standard deviation, and weights
        self.means = np.random.randint(min(self.initData[:,0]), max(self.initData[:,0])+1, 
            size=(self.number_of_sources, len(self.initData[0])))
        self.covars = np.zeros((self.number_of_sources, len(self.initData[0]), len(self.initData[0])))

        for dim in range(len(self.covars)):
            np.fill_diagonal(self.covars[dim], 5)

        self.weights = np.ones(self.number_of_sources) / self.number_of_sources

        #initial plot based on bounds
        fig = plt.figure(figsize = (10,10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(self.initData[:,0], self.initData[:,1])
        ax0.set_title('Initial state')

        # Initial plot created, now scatter data
        for m,c in zip(self.means, self.covars):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m, cov=c)
            ax0.contour(np.sort(self.initData[:,0]), np.sort(self.initData[:,1]), multi_normal.pdf(self.XY).reshape(len(self.initData), len(self.initData)), colors='black', alpha=0.3)
            ax0.scatter(m[0], m[1], c='grey', zorder=10, s=100)

        for iter in range(self.iterations):
            # EXPECTATION, E Step
            #Create array with datapoints * gaussians /// output probabilities
            r_ic = np.zeros((len(self.initData), len(self.covars)))

            for m,co,p,r in zip(self.means, self.covars, self.weights, range(len(r_ic[0]))):
                co += self.reg_cov
                mn = multivariate_normal(mean=m, cov=co)
                r_ic[:,r] = p*mn.pdf(self.initData)/np.sum([pi_c*multivariate_normal(mean=m, cov=cov_c).pdf(self.data) 
                    for pi_c, mu_c, cov_c in zip(self.weights, self.means, self.covars + self.reg_cov)], axis=0)
            
            # Calculate new mean vector and covariance matrices based on x_i to classes c --> r_ic
            self.means = []
            self.covars = []
            self.weights = []

            for val_c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,val_c], axis = 0)
                mu_c = (1/m_c)*np.sum(self.initData*r_ic[:,val_c].reshape(len(self.initData),1),axis=0)
                self.means.append(mu_c)
                # Calculate the standard deviation / covariance matrix
                self.covars.append(((1/m_c)*np.dot((np.array(r_ic[:,val_c]).reshape(len(self.initData),1)*(self.initData-mu_c)).T,(self.initData-mu_c)))+self.reg_cov)
                # Calculate the new weight
                self.weights.append(m_c/np.sum(r_ic))
                
                """
                This process of E step followed by a M step is now iterated a number of n times. In the second step for instance,
                we use the calculated pi_new, mu_new and cov_new to calculate the new r_ic which are then used in the second M step
                to calculat the mu_new2 and cov_new2 and so on....
                """
            plt.show()
        fig3 = plt.figure(figsize=(10,10))
        ax2 = fig3.add_subplot(111)
        ax2.scatter(self.data[:,0],self.data[:,1])
        for m,c in zip(self.means,self.covars):
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax2.contour(np.sort(self.data[:,0]),np.sort(self.data[:,1]),multi_normal.pdf(self.XY).reshape(len(self.data),len(self.data)),colors='black',alpha=0.3)
            ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
            ax2.set_title('Final state')
        plt.show()
    
def main():
    style.use('fivethirtyeight')
    parser=argparse.ArgumentParser(
        description='''Gaussian mixture model plotter for data from two user-specified columns in CSV format ''')
    parser.add_argument('--file', type=str, help='CSV filename in the working directory')
    parser.add_argument('--colx', type=int, default=1, help='0-based index of the x-axis column')
    parser.add_argument('--coly', type=int, default=2, help='0-based index of the y-axis column')
    parser.add_argument('--rows', type=int, help="Number of data points to load from the specified column")
    parser.add_argument('--iters', type=int, default=10, help="Number of EM iterations to do. <=10 usually prevents weirdness. Default: 10.")
    parser.add_argument('--showallplots', type=bool, default=False, 
    help='If true, show an output plot for each iteration, which must be manually closed to continue iterating. Otherwise, show only the final plot. Default: False.')
    parser.add_argument('--clusters', type=int, help='Number of Gaussian distributions to fit to the data. Max: 10')
    parser.add_argument('--testdata', type=bool, default=False, help='Demonstrate the GMM using 60 nicely clustered, evenly spaced random data points instead of data from a CSV file. Default: False')
    args=parser.parse_args()
    if args.clusters > 10:
        raise ValueError("Clusters must be <=10.")

    dataSet = np.array([[]])
    dataColumns = [args.colx,args.coly]
    if(not args.testdata):
        df = pd.read_csv(args.file, nrows=args.rows, usecols=dataColumns)
        print(df)
        for row in range(0,args.rows):
            rowSet = df.loc[[row]].values
            if dataSet.size == 0:
                dataSet = rowSet
            else:
                dataSet = np.append(dataSet, rowSet, axis=0)
        print("Input Data Collected")
    else:
        initDataX = np.linspace(-5,5,num=20) #get range of 20 evenly spaced numbers from -5 to 5
        data0X = initDataX*np.random.rand(len(initDataX))+15 # Use these to generate 3 random datasets of 20 points - first from 5 to 15
        data1X = initDataX*np.random.rand(len(initDataX))-15 # from -15 to -5
        data2X = initDataX*np.random.rand(len(initDataX)) # from -5 to 5
        dataSetX = np.stack((data0X,data1X,data2X)).flatten() # Combine the clusters to get the random datapoints from above

        initDataY = np.linspace(-5,5,num=20) #do it again for the y column
        data0Y = initDataY*np.random.rand(len(initDataY))+15 
        data1Y = initDataY*np.random.rand(len(initDataY))-15 
        data2Y = initDataY*np.random.rand(len(initDataY))
        dataSetY = np.stack((data0Y,data1Y,data2Y)).flatten() 

        dataSet = np.column_stack((dataSetX, dataSetY)) #concat both 1d arrays into a 2-column 2d arrays
        print("Auto-Generated Data Collected")

    gmm = TwoDGaussianMM(dataSet, args.iters, args.clusters, args.showallplots)
    gmm.runExMax()

def readColumnsFromCSV(filepath, col, numRows):
    if os.name == 'nt':
        csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    else:
        csvFrame = pd.read_csv(filepath)
    csvFrame = csvFrame.iloc[:,[col]]
    return csvFrame.head(min(len(csvFrame.index), numRows))

main()