# KUMLDistributionEstimation
## Distribution Estimation project for EECS738


For our project, we chose to estimate the distributions of datasets and look for clusters by implementing the k-means and Gaussian mixture model methods.
To do this, we set out to write a Gaussian mixture modeler utility capable of algorithmically fitting a normal distribution to an input dataset (or multiple distributions to clusters within a single dataset).
Additionally, we planned to write a k-means utility for the purpose of discovering potentially interesting multivariate clustering between two chosen variables of the input dataset.
We based our k-means implementation on a tutorial from [pythonprogramming.net](https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/) and our 1D and 2D GMM implementations on a [set of notes from CS274](https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf) at UCI and a tutorial from [python-course.eu](https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php).

We did all our work in Python, first writing a script which runs k-means (with user-specified k) on specified data columns from a CSV file, resulting in a plot of the data, classified into k clusters. We then wrote another script which performs expectation-maximization to plot a one-dimensional Gaussian mixture model (with user-specified number of Gaussians) for a single column in a CSV file. The lessons learned here were applied to a third script for creating two-dimensional GMMs.

We chose as our datasets the Portuguese Red Wine Quality and Glass Classification csv files from the UCIML data on Kaggle. 
Although many dimensions in these datasets displayed mostly continuous data without interesting clusters, 
some insights could be obtained. The Gaussian mixture model utility with k=1 is useful for the estimation of a single distribution to represent a set of data without subsets.
The glass dataset classifies its members by glass usage -
whether the glass described is used in a building, a vehicle window, a container, tableware, or for headlamps.
Performing k-means with this type as one axis and the glass' weight-percent content of an element reveals 
that non-window glasses (types 5,6,7) tend to cluster together as opposed to window glasses (types 1,2,3).
For example, when using magnesium content as the x-axis, non-window glasses cluster loosely at the low end of magnesium content,
while window glasses cluster tightly at the high end. When using k=2, the two clusters produced almost always represent
non-window and window glasses, informing us that the two generally display different elemental characteristics.