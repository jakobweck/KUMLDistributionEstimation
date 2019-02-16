##Exploring 2D GMM's
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm
import argparse
import pandas as pd
import random
import sys
import os

class OneDGaussianMixtureModeler:
    def __init__(self,X,iterations,clusters,showAll):
        self.clusters = clusters
        self.iterations = iterations
        self.data = X
        self.initData = X
        self.means = None
        self.weights = None
        self.stdevs = None
        self.showAll = showAll
        self.colors = 10*["g","r","c","b","k"]

    
    def runExMax(self):

        #arbitrarily set initial values
        #ideally we would run a more complex algorithm to decide these
        #1d k-means can apparently be used to set good initial means
        #for nowmeans - evenly spaced ints within the data range
        self.means = np.linspace(min(self.data),max(self.data),num=self.clusters)
        self.weights = [1/self.clusters]*self.clusters
        #pretty much made this up based on some experimentation
        #at least gives us a feasible value where the stdevs aren't bigger than the range of data or anything
        self.stdevs = [(max(self.data)-min(self.data))/(self.clusters*2)]*self.clusters
        
        
        for iter in range(self.iterations):
            #EXPECTATION
            #Create array with dimensions (num. of datapoints)*(num. of gaussians desired)
            #Will hold probabilities that each data point belongs to each gaussian
            r = np.zeros((len(self.initData), self.clusters))  

            gaussians = []
            for c in range (self.clusters):
                gaussians.append(norm(loc=self.means[c],scale=self.stdevs[c]))

            #populate r with these probabilities
            #pair each column index of R with a gaussian generated from the corresponding mean and stdev stored in class members
            #and with a weight representing number of points known to belong to that gaussian
            for c,g,p in zip(range(self.clusters),gaussians,self.weights):
                r[:,c] = p*g.pdf(self.initData) # Write the probability that x belongs to gaussian c in column c. 

            #normalize probabilities so each row of r sums to 1 and weight them
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(self.weights)*np.sum(r,axis=1)[i])
            
            #plotting data pts - plot only if this is the last iteration or user wants to see all intermediate plots
            if(self.showAll or iter==(self.iterations-1)):
                fig = plt.figure(figsize=(10,10))
                ax0 = fig.add_subplot(111)
                for i in range(len(r)):
                    #with arbitary num. of clusters (read:not 3) we can't do the rgb trick that the tutorial did to color points on a gradient
                    #we need to assign each point a color based on which cluster it is closest to
                    highestProbCluster = np.argmax(r[i])
                    ax0.scatter(self.data[i],0,c=self.colors[highestProbCluster],s=100) 
                #plotting normal distribution probability density functions
                #x-axis is a smooth range of points from minimum data pt to max data point
                #y-axis is the result of the pdf for those pts
                for g,c in zip(gaussians, self.colors):
                    ax0.plot(np.linspace(min(self.data),max(self.data),num=len(self.data)),
                    g.pdf(np.linspace(min(self.data),max(self.data),num=len(self.data))),c=c)
                       
            #MAXIMIZATION
    
            #calculate relative membership of each cluster by adding all points' probability to be in that cluster
            clusterMembers = []
            for c in range(len(r[0])):
                m = np.sum(r[:,c])
                clusterMembers.append(m) 
            #recalculate weights based on new membership data
            for k in range(len(clusterMembers)):
                self.weights[k] = (clusterMembers[k]/np.sum(clusterMembers)) # just fraction of that cluster over all clusters in m_c
            #recalculate means
            #for each gaussian, the mean is obtained by summing
            #the probability that each data pt is in the cluster, times the value of that data point (multiplying dataset by r)
            #and dividing the resulting sum by the number of members of the cluster
            self.means = np.sum(self.data.reshape(len(self.data),1)*r,axis=0)/clusterMembers
            #recalculate stdev of each gaussian by taking sqrt of the result of the standard GMM variance equation for that cluster
            variances = []
            for c in range(len(r[0])):
                oneOverMC = 1/clusterMembers[c]
                rC = np.array(r[:,c])
                #get matrix of differences between each data point and the mean of this cluster
                #and the transpose of that matrix
                dataMeanDiff = (self.data.reshape(len(self.data),1)-self.means[c])
                dataMeanDiffT = dataMeanDiff.T
                #matrix-multiply these two to get a summation and divide by cluster's membership
                variance = (oneOverMC * np.dot(rC*dataMeanDiffT,dataMeanDiff))[0][0]
                variances.append(variance)                          
            self.stdevs = (np.sqrt(variances).flatten().tolist())
            if(self.showAll or iter==(self.iterations-1)):
                plt.show()
                print(self.stdevs)
    
def main():
    style.use('fivethirtyeight')
    parser=argparse.ArgumentParser(
        description='''Gaussian mixture model plotter for data from two user-specified columns in CSV format ''')
    parser.add_argument('--file', type=str, help='CSV filename in the working directory')
    parser.add_argument('--colx', type=int, help='0-based index of the x-axis column')
    parser.add_argument('--rows', type=int, help="Number of data points to load from the specified column")
    parser.add_argument('--iters', type=int, default=10, help="Number of EM iterations to do. <=10 usually prevents weirdness. Default: 10.")
    parser.add_argument('--showallplots', type=bool, default=False, 
    help='If true, show an output plot for each iteration, which must be manually closed to continue iterating. Otherwise, show only the final plot. Default: False.')
    parser.add_argument('--clusters', type=int, help='Number of Gaussian distributions to fit to the data. Max: 10')
    parser.add_argument('--testdata', type=bool, default=False, help='Demonstrate the GMM using 60 nicely clustered, evenly spaced random data points instead of data from a CSV file. Default: False')
    args=parser.parse_args()
    if args.clusters > 10:
        raise ValueError("Clusters must be <=10.")

    dataSet = np.array([])
    if(not args.testData):
        df = readColumnsFromCSV(args.file, args.colx, args.rows)
        dataSet = df.to_numpy().flatten()
    else:
        initData = np.linspace(-5,5,num=20) #get range of 20 evenly spaced numbers from -5 to 5
        data0 = initData*np.random.rand(len(initData))+15 # Use these to generate 3 random datasets of 20 points - first from 5 to 15
        data1 = initData*np.random.rand(len(initData))-15 # from -15 to -5
        data2 = initData*np.random.rand(len(initData)) # from -5 to 5
        dataSet = np.stack((data0,data1,data2)).flatten() # Combine the clusters to get the random datapoints from above

    gmm = OneDGaussianMixtureModeler(dataSet, args.iters, args.clusters, args.showallplots)
    gmm.runExMax()

def readColumnsFromCSV(filepath, col, numRows):
    if os.name == 'nt':
        csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    else:
        csvFrame = pd.read_csv(filepath)
    csvFrame = csvFrame.iloc[:,[col]]
    return csvFrame.head(min(len(csvFrame.index), numRows))

main()