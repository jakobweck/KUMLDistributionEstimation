##Exploring 2D GMM's (Multi-dimensional)
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

        # self.clusters = clusters
        self.number_of_sources = clusters
        self.XY = None

        self.means = None
        self.weights = None
        self.stdevs = None
        self.showAll = showAll
        self.colors = 10*["g","r","c","b","k"]

    
    def runExMax(self):
        # Code sourced/referenced from https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
        # sets reg covariance to initial data
        print(type(self.initData))
        print(self.initData)
        print(min(self.initData[:,0]))
        print(max(self.initData[:,0]))
        print((self.number_of_sources))
        self.reg_cov = 1e-6 * np.identity(len(self.initData[0]))
        print(type(self.initData))
        # Creates a grid based on initial data, XY
        x,y = np.meshgrid(np.sort(self.initData[:,0]), np.sort(self.initData[:,1]))
        # Creates a matrix XY out of initial data
        self.XY = np.array([x.flatten(), y.flatten()]).T

        """ 1. Set the iniital mu, cov and pi(weight) values"""
        self.means = np.random.randint(min(self.initData[:,0]), max(self.initData[:,0])+1, 
            size=(self.number_of_sources, len(self.initData[0])))
        self.stdevs = np.zeros((self.number_of_sources, len(self.initData[0]), len(self.initData[0])))

        for dim in range(len(self.stdevs)):
            np.fill_diagonal(self.stdevs[dim], 5)

        self.weights = np.ones(self.number_of_sources) / self.number_of_sources
        log_likelihoods = []

        #initial plot based on bounds
        fig = plt.figure(figsize = (9,9))
        ax0 = fig.add_subplot(111)
        ax0.scatter(self.initData[:,0], self.initData[:,1])
        ax0.set_title('Initial state')

        for m,c in zip(self.means, self.stdevs):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m, cov=c)
            ax0.contour(np.sort(self.initData[:,0]), np.sort(self.initData[:,1]), multi_normal.pdf(self.XY).reshape(len(self.initData), len(self.initData)), colors='blue', alpha=0.3)
            ax0.scatter(m[0], m[1], c='grey', zorder=10, s=100)

        for iter in range(self.iterations):
            #EXPECTATION
            #Create array with dimensions (num. of datapoints)*(num. of gaussians desired)
            #Will hold probabilities that each data point belongs to each gaussian
            r_ic = np.zeros((len(self.initData), len(self.stdevs)))

            for m,co,p,r in zip(self.means, self.stdevs, self.weights, range(len(r_ic[0]))):
                co += self.reg_cov
                mn = multivariate_normal(mean=m, cov=co)
                r_ic[:,r] = p*mn.pdf(self.initData)/np.sum([pi_c*multivariate_normal(mean=m, cov=cov_c).pdf(self.data) 
                    for pi_c, mu_c, cov_c in zip(self.weights, self.means, self.stdevs + self.reg_cov)], axis=0)
            
            #Calculate new mean vector and covariance matrices based on x_i to classes c --> r_ic
            self.means = []
            self.stdevs = []
            self.weights = []

            log_likelihood = []
            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:,c], axis = 0)
                mu_c = (1/m_c)*np.sum(self.initData*r_ic[:,c].reshape(len(self.initData),1),axis=0)
                self.means.append(mu_c)
                # Calculate the covariance matrix per source based on the new mean
                self.stdevs.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.initData),1)*(self.initData-mu_c)).T,(self.initData-mu_c)))+self.reg_cov)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
                self.weights.append(m_c/np.sum(r_ic)) # Here np.sum(r_ic) gives as result the number of instances. This is logical since we know 
                                                # that the columns of each row of r_ic adds up to 1. Since we add up all elements, we sum up all
                                                # columns per row which gives 1 and then all rows which gives then the number of instances (rows) 
                                                # in X --> Since pi_new contains the fractions of datapoints, assigned to the sources c,
                                                # The elements in pi_new must add up to 1
                """Log likelihood"""
                log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.means[i],self.stdevs[j]).pdf(self.data) for k,i,j in zip(self.weights,range(len(self.means)),range(len(self.stdevs)))])))
                
                """
                This process of E step followed by a M step is now iterated a number of n times. In the second step for instance,
                we use the calculated pi_new, mu_new and cov_new to calculate the new r_ic which are then used in the second M step
                to calculat the mu_new2 and cov_new2 and so on....
                """
            # fig2 = plt.figure(figsize=(10,10))
            # ax1 = fig2.add_subplot(111) 
            # ax1.set_title('Log-Likelihood')
            # ax1.plot(range(0,self.iterations,1),log_likelihoods)
            plt.show()
        fig3 = plt.figure(figsize=(10,10))
        ax2 = fig3.add_subplot(111)
        ax2.scatter(self.data[:,0],self.data[:,1])
        for m,c in zip(self.means,self.stdevs):
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax2.contour(np.sort(self.data[:,0]),np.sort(self.data[:,1]),multi_normal.pdf(self.XY).reshape(len(self.data),len(self.data)),colors='black',alpha=0.3)
            ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
            ax2.set_title('Final state')
        plt.show()


        #     ####mmmkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk-old stuff below-mmmmkkkkkk

        #     gaussians = []
        #     for c in range (self.clusters):
        #         gaussians.append(norm(loc=self.means[c],scale=self.stdevs[c]))

        #     #populate r with these probabilities
        #     #pair each column index of R with a gaussian generated from the corresponding mean and stdev stored in class members
        #     #and with a weight representing number of points known to belong to that gaussian
        #     for c,g,p in zip(range(self.clusters),gaussians,self.weights):
        #         r[:,c] = p*g.pdf(self.initData) # Write the probability that x belongs to gaussian c in column c. 

        #     #normalize probabilities so each row of r sums to 1 and weight them
        #     for i in range(len(r)):
        #         r[i] = r[i]/(np.sum(self.weights)*np.sum(r,axis=1)[i])
            
        #     #plotting data pts - plot only if this is the last iteration or user wants to see all intermediate plots
        #     if(self.showAll or iter==(self.iterations-1)):
        #         fig = plt.figure(figsize=(10,10))
        #         ax0 = fig.add_subplot(111)
        #         for i in range(len(r)):
        #             #with arbitary num. of clusters (read:not 3) we can't do the rgb trick that the tutorial did to color points on a gradient
        #             #we need to assign each point a color based on which cluster it is closest to
        #             highestProbCluster = np.argmax(r[i])
        #             ax0.scatter(self.data[i],0,c=self.colors[highestProbCluster],s=100) 
        #         #plotting normal distribution probability density functions
        #         #x-axis is a smooth range of points from minimum data pt to max data point
        #         #y-axis is the result of the pdf for those pts
        #         for g,c in zip(gaussians, self.colors):
        #             ax0.plot(np.linspace(min(self.data),max(self.data),num=len(self.data)),
        #             g.pdf(np.linspace(min(self.data),max(self.data),num=len(self.data))),c=c)
                       
        #     #MAXIMIZATION
    
        #     #calculate relative membership of each cluster by adding all points' probability to be in that cluster
        #     clusterMembers = []
        #     for c in range(len(r[0])):
        #         m = np.sum(r[:,c])
        #         clusterMembers.append(m) 
        #     #recalculate weights based on new membership data
        #     for k in range(len(clusterMembers)):
        #         self.weights[k] = (clusterMembers[k]/np.sum(clusterMembers)) # just fraction of that cluster over all clusters in m_c
        #     #recalculate means
        #     #for each gaussian, the mean is obtained by summing
        #     #the probability that each data pt is in the cluster, times the value of that data point (multiplying dataset by r)
        #     #and dividing the resulting sum by the number of members of the cluster
        #     self.means = np.sum(self.data.reshape(len(self.data),1)*r,axis=0)/clusterMembers
        #     #recalculate stdev of each gaussian by taking sqrt of the result of the standard GMM variance equation for that cluster
        #     variances = []
        #     for c in range(len(r[0])):
        #         oneOverMC = 1/clusterMembers[c]
        #         rC = np.array(r[:,c])
        #         #get matrix of differences between each data point and the mean of this cluster
        #         #and the transpose of that matrix
        #         dataMeanDiff = (self.data.reshape(len(self.data),1)-self.means[c])
        #         dataMeanDiffT = dataMeanDiff.T
        #         #matrix-multiply these two to get a summation and divide by cluster's membership
        #         variance = (oneOverMC * np.dot(rC*dataMeanDiffT,dataMeanDiff))[0][0]
        #         variances.append(variance)                          
        #     self.stdevs = (np.sqrt(variances).flatten().tolist())
        #     if(self.showAll or iter==(self.iterations-1)):
        #         plt.show()
        #         print(self.stdevs)
    
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
        initData = np.linspace(-5,5,num=20) #get range of 20 evenly spaced numbers from -5 to 5
        data0 = initData*np.random.rand(len(initData))+15 # Use these to generate 3 random datasets of 20 points - first from 5 to 15
        data1 = initData*np.random.rand(len(initData))-15 # from -15 to -5
        data2 = initData*np.random.rand(len(initData)) # from -5 to 5
        dataSet = np.stack((data0,data1,data2)).flatten() # Combine the clusters to get the random datapoints from above
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