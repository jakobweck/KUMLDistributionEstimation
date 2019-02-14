import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import style
import argparse

class KMeans:
    def __init__(self, k, tolerance, maxIterations):
        self.k = k #number of centroids
        self.tolerance = tolerance #threshold of centroid movement below which to stop
        self.maxIterations = maxIterations 
    #data is a 2d array (pairs of coords)
    def fit(self, data):
        self.centroids = {}
        #set centroids to first k data points arbitrarily
        #ideally this would be random data points but this is ok
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        for i in range(self.maxIterations):
            self.classifications = {}
            #on each iteration give each centroid an empty classification array
            for i in range(self.k):
                self.classifications[i] = []
            #for each data point calculate a distance from each centroid
            #classify each data point as belonging to the closest centroid
            #by adding it to that centroid's classification array
            for datapoint in data:
                    distances = [np.linalg.norm(datapoint-self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classifications[classification].append(datapoint)
            prevCentroids = dict(self.centroids)
            #move each centroid to the average of the points classified as belonging to it
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True
            #for each centroid, compare the old and new positions
            #if it moved less than the tolerance, keep 'optimized' bool true
            #otherwise, make it false
            for c in self.centroids:
                originalCentroid = prevCentroids[c]
                newCentroid = self.centroids[c]
                #this fails if the originalCentroid is 0 - todo handle this
                centroidMovement = 0
                else:      
                    centroidMovement = np.sum((newCentroid-originalCentroid)/originalCentroid*100.0)
                centroidMoving = centroidMovement > self.tolerance
                if centroidMoving:
                    optimized = False
            #if all centroids are optimized we can stop iterating
            if optimized:
                break
    #return the predicted classification of any given point
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

def readColumnsFromCSV(filepath, col1, col2, numRows):
    csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    csvFrame = csvFrame.iloc[:,[col1, col2]]
    return csvFrame.head(max(len(csvFrame.index), numRows))

def main():
    parser=argparse.ArgumentParser(
        description='''K-means plotter for data from two user-specified columns in CSV format ''')
    parser.add_argument('-file', type=str, help='CSV filename in the working directory')
    parser.add_argument('-colx', type=int, help='0-based index of the x-axis column')
    parser.add_argument('-coly', type=int, help='0-based index of the y-axis column')
    parser.add_argument('-rows', type=int, default=100, help='Number of rows to take from head of CSV file. Default: 100')
    args=parser.parse_args()

    df = readColumnsFromCSV(args.file, args.colx, args.coly, args.rows)

    kMeansClass = KMeans(2, .001, 300)
    kMeansClass.fit(df.values)

    colors = 10*["g","r","c","b","k"]
    #graph the centroids as black circles
    for centroid in kMeansClass.centroids:
        x = kMeansClass.centroids[centroid][0]
        y = kMeansClass.centroids[centroid][1]
        plt.scatter(x, y,marker="o", color="k", s=150, linewidths=5)
    #graph the data points as Xs colored w/r/t their centroid
    for classification in kMeansClass.classifications:
        color = colors[classification]
        for dataPoint in kMeansClass.classifications[classification]:
            plt.scatter(dataPoint[0], dataPoint[1], marker="x", color=color, s=150, linewidths=5)

    print("X axis:" + df.columns[0])
    print("Y axis:" + df.columns[1])
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    #print(kMeansClass.predict([0.6, 9.0]))
    plt.show() 
    while(True):
        xCoord = input("X coordinate for prediction: ")
        yCoord = input("Y coordinate for prediction: ")
        print(kMeansClass.predict([float(xCoord), float(yCoord)]))
main()