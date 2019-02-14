import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm
import argparse

def readColumnsFromCSV(filepath, col1, col2, numRows):
    csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    csvFrame = csvFrame.iloc[:,[col1, col2]]
    return csvFrame.head(max(len(csvFrame.index), numRows))

class OneDGaussianMixedModeler:
    def __init__(self, dataSet, iters):
        self.iterations = iters
        self.dataSet = dataSet
        self.mu = None
        self.pi = None
        self.var = None
    def em(self):
        self.mu = [-8, 8, 5]
        self.pi = [1/3,1/3,1/3]
        self.var = [5,3,1]
        #EXPECTATION - Generate the expectation (probability) that random data points belong to arbitrarily chosen gaussian distributions
        #create empty array with one column for each gaussian and one row for each of the 60 datapoints
        for iter in range(self.iterations):
            r = np.zeros((len(self.dataSet),3))  
            print('Dimensionality','=',np.shape(r))
            #create 3 gaussians (normal distributions) with means -5, 8, 1.5 and stdevs 5, 3, 1
            #.pdf can be called on these with a value to get its probability under that distribution
            #for example gauss_1.pdf(-5) returns ~.079
            gauss0 = norm(loc=-5,scale=5) 
            gauss1 = norm(loc=8,scale=3)
            gauss2 = norm(loc=1.5,scale=1)

            m = np.array([1/3,1/3,1/3])
            weights = m/np.sum(m)
            #index gaussians as 0,1,2
            for c,g,w in zip(range(3),[gauss0,gauss1,gauss2], weights):
                r[:,c] = w*g.pdf(self.dataSet) 
                                    # Write the probability of each datapoint under the given gaussian indexed by c in column c of our empty array r
                                    # Weight this probability by the fraction of total datapoints we know to belong to each gaussian (the pi array)
                                    # Get a 60x3 array filled with the probability that each of the 60 x_i belongs to each one of the 3 gaussians

            #normalize probablities - we want each row to sum to 1
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(weights)*np.sum(r,axis=1)[i])
            #MAXIMIZATION
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:,c])
                m_c.append(m) 
            pi_c = []
            for m in m_c:
                pi_c.append(m/np.sum(m_c)) #calculate fraction of points that belong to each cluster
            #new means
            mu_c = np.sum(self.dataSet.reshape(len(self.dataSet),1)*r, axis=0)/m_c
            #new stdevs
            var_c = []
            for c in range(len(r[0])):
                var_c.append((1/m_c[c])*np.dot(((np.array(r[:,c]).reshape(60,1))*(self.dataSet.reshape(len(self.dataSet),1)-mu_c[c])).T,(self.dataSet.reshape(len(self.dataSet),1)-mu_c[c])))
            gauss0 = norm(loc=mu_c[0], scale=var_c[0])
            gauss1 = norm(loc=mu_c[1], scale=var_c[1])
            gauss2 = norm(loc=mu_c[2], scale=var_c[2])
            #update r array
            for c,g,p in zip(range(3),[gauss0,gauss1,gauss2],weights): #should this be weights(original pi) or pi_c?
                r[:,c] = p*g.pdf(self.dataSet) 
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(pi_c)*np.sum(r,axis=1)[i])
                
            print(r)
            print(np.sum(r,axis=1)) # As we can see, as result each row sums up to one, just as we want it.




def main():
    parser=argparse.ArgumentParser(
        description='''Gaussian mixture model plotter for data from two user-specified columns in CSV format ''')
    parser.add_argument('-file', type=str, help='CSV filename in the working directory')
    parser.add_argument('-colx', type=int, help='0-based index of the x-axis column')
    parser.add_argument('-coly', type=int, help='0-based index of the y-axis column')
    parser.add_argument('-rows', type=int, default=100, help='Number of rows to take from head of CSV file. Default: 100')
    args=parser.parse_args()

    df = readColumnsFromCSV(args.file, args.colx, args.coly, args.rows)

    style.use('fivethirtyeight')
    np.random.seed(0)
    initData = np.linspace(-5,5,num=20) #get range of 20 evenly spaced numbers from -5 to 5
    data0 = initData*np.random.rand(len(initData))+10 # Use these to generate 3 random datasets of 20 points - first from 5 to 15
    data1 = initData*np.random.rand(len(initData))-10 # from -15 to -5
    data2 = initData*np.random.rand(len(initData)) # from -5 to 5
    dataSet = np.stack((data0,data1,data2)).flatten() # Combine the clusters to get the random datapoints from above
    gmm = OneDGaussianMixedModeler(dataSet, 10)
    gmm.em()