# KUMLDistributionEstimation
##Distribution Estimation project for EECS738
For our project, we chose to look for clusters in datasets by implementing the k-means and Gaussian mixture model methods.
We based our k-means implementation on a tutorial from pythonprogramming.net and our 1D and 2D GMM implementations on a set of notes from CS274 at UCI and a tutorial from python-course.eu.
We did all our work in Python, first writing a script which runs k-means (with user-specified k) on specified data columns from a CSV file, resulting in a plot of the data, classified into k clusters. We then wrote another script which performs expectation-maximization to plot a one-dimensional Gaussian mixture model (with user-specified number of Gaussians) for a single column in a CSV file. The lessons learned here were applied to a third script for creating two-dimensional GMMs.
We chose as our datasets the Portuguese Red Wine Quality and Glass Classification csv files from the UCIML data on Kaggle.