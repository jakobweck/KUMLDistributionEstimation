import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm
import argparse
import pandas as pd
import sys
import os

class OneDGaussianMixedModeler:
    def __init__(self,X,iterations, showAll):
        self.iterations = iterations
        self.data = X
        self.initData = X
        self.means = None
        self.weights = None
        self.stdevs = None
        self.showAll = showAll
    
    def runExMax(self):

        #arbitrarily set initial values
        #ideally we would run a more complex algorithm to decide these
        #1d k-means can be used to set good initial means
        self.means = [.05,.3,.5]
        self.weights = [1/3,1/3,1/3]
        self.stdevs = [.4,.4,.4]
        
        
        for iter in range(self.iterations):
            #EXPECTATION
            #Create array with dimensions (num. of datapoints)*(num. of gaussians desired)
            #Will hold probabilities that each data point belongs to each gaussian
            r = np.zeros((len(self.initData),3))  
  
            #populate r with these probabilities
            #pair each column index of R with a gaussian generated from the corresponding mean and stdev stored in class members
            #and with a weight representing number of points known to belong to that gaussian
            for c,g,p in zip(range(3),[norm(loc=self.means[0],scale=self.stdevs[0]),
                                       norm(loc=self.means[1],scale=self.stdevs[1]),
                                       norm(loc=self.means[2],scale=self.stdevs[2])],self.weights):
                r[:,c] = p*g.pdf(self.initData) # Write the probability that x belongs to gaussian c in column c. 

            #normalize probabilities so each row of r sums to 1 and weight them
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(self.weights)*np.sum(r,axis=1)[i])
            
            if(self.showAll or iter==(self.iterations-1)):
                #plotting data pts
                fig = plt.figure(figsize=(10,10))
                ax0 = fig.add_subplot(111)
                for i in range(len(r)):
                    twod = []
                    #have to 'trick' this array into being 2D to avoid tons of errors from scatter
                    #each point's rgb color is determined based on which distributions it is closest to
                    #todo scale this to arbitrary cluster count
                    asdf = np.argmax(r[i])
                    twod.append([round(r[i][0], 15), round(r[i][1],15), round(r[i][2],15)])

                    ax0.scatter(self.data[i],0,c=twod,s=100) 
                #plotting normal distribs
                for g,c in zip([norm(loc=self.means[0],scale=self.stdevs[0]).pdf(np.linspace(-20,20,num=60)),
                                norm(loc=self.means[1],scale=self.stdevs[1]).pdf(np.linspace(-20,20,num=60)),
                                norm(loc=self.means[2],scale=self.stdevs[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
                    ax0.plot(np.linspace(min(self.data),max(self.data),num=60),g,c=c)
                       
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
            #for each cluster, the mean is obtained by summing
            #the probability that each data pt is in the cluster, times the value of that data point (multiplying dataset by r)
            #and dividing the resulting sum by the number of members of the cluster
            self.means = np.sum(self.data.reshape(len(self.data),1)*r,axis=0)/clusterMembers
            #recalculate covariances
            #
            variances = []
            for c in range(len(r[0])):
                oneOverMC = 1/clusterMembers[c]
                rC = np.array(r[:,c])
                dataMeanDiff = (self.data.reshape(len(self.data),1)-self.means[c])
                dataMeanDiffT = dataMeanDiff.T
                variance = (oneOverMC * np.dot(rC*dataMeanDiffT,dataMeanDiff))[0][0]
                if(variance==0.0):
                    variances.append(self.stdevs[c]*self.stdevs[c])
                else:
                    variances.append(variance)
                           
            self.stdevs = (np.sqrt(variances).flatten().tolist())
            print(self.stdevs)
            if(self.showAll or iter==(self.iterations-1)):
                print ("")
                plt.show()
    
def main():
    parser=argparse.ArgumentParser(
        description='''Gaussian mixture model plotter for data from two user-specified columns in CSV format ''')
    parser.add_argument('--file', type=str, help='CSV filename in the working directory')
    parser.add_argument('--colx', type=int, help='0-based index of the x-axis column')
    parser.add_argument('--showallplots', type=bool, default=False, 
    help='If true, show output plots for each iteration which must be manually closed to continue iterating. Otherwise, show only the final plot. Default: False.')
    args=parser.parse_args()

    df = readColumnsFromCSV(args.file, args.colx, 200)

    style.use('fivethirtyeight')
    np.random.seed(0)
    # initData = np.linspace(-5,5,num=20) #get range of 20 evenly spaced numbers from -5 to 5
    # data0 = initData*np.random.rand(len(initData))+15 # Use these to generate 3 random datasets of 20 points - first from 5 to 15
    # data1 = initData*np.random.rand(len(initData))-15 # from -15 to -5
    # data2 = initData*np.random.rand(len(initData)) # from -5 to 5
    # dataSet = np.stack((data0,data1,data2)).flatten() # Combine the clusters to get the random datapoints from above
    dataSet = df.to_numpy().flatten()
    gmm = OneDGaussianMixedModeler(dataSet, 100, args.showallplots)
    gmm.runExMax()

def readColumnsFromCSV(filepath, col, numRows):
    if os.name == 'nt':
        csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    else:
        csvFrame = pd.read_csv(filepath)
    csvFrame = csvFrame.iloc[:,[col]]
    return csvFrame.head(min(len(csvFrame.index), numRows))

main()