import numpy as np
import pandas as pd
import sys
import os


def readColumnsFromCSV(filepath, col1, col2):
    print (os.getcwd()+ "\\" + filepath)
    csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    csvFrame = csvFrame.iloc[:,[10, 11]]
    return csvFrame



class KMeans:
    def __init__(self, k, tolerance, maxIterations):
        self.k = k #number of centroids
        self.tolerance = tolerance #threshold of centroid movement below which to stop
        self.maxIterations = maxIterations 
    #data is a 2d array (pairs of coords)
    def fit(self, data):
        self.centroids = {}
        #set centroids to first k data points arbitrarily
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        for i in range(self.maxIterations):
            self.classifications = {}
            #for each iteration give each centroid an empty classification array
            for i in range(self.k):
                self.classifications[i] = []
            #for each data point calculate a distance from each centroid
            #classify each data point as belonging to the closest centroid
            #by adding it to that centroid's classification array
            for featureset in data:
                    distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classifications[classification].append(featureset)
            prevCentroids = dict(self.centroids)
            #move each centroid to the average of the points classified belonging to it
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True
            #for each centroid, compare the old and new centroids
            #if it moved less than the tolerance, keep 'optimized' bool true
            #otherwise, make it false
            for c in self.centroids:
                originalCentroid = prevCentroids[c]
                newCentroid = self.centroids[c]
                centroidMovement = np.sum((newCentroid-originalCentroid)/originalCentroid*100.0)
                centroidMoving = centroidMovement > self.tolerance
                if centroidMoving:
                    print(centroidMovement)
                    optimized = False
            #if all centroids are optimized we can stop iterating
            if optimized:
                break
    #return the predicted classification of any given point
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


df = readColumnsFromCSV(sys.argv[1], 11, 12)
kMeansClass = KMeans(2, .001, 300)
kMeansClass.fit(df)

for centroid in kMeansClass.centroids:
    print(centroid)
for classification in kMeansClass.classifications:
    for featureset in kMeansClass.classifications[classification]:
        print featureset
 