## Table of Contents
* [K-Means Clustering](#k-means-clustering)
* [Hierarchical Clustering (HC)](#hierarchical-clustering-hc)


## K-Means Clustering
This is used when you have unlabeled data (data that hasn't been categorized). This finds groups in the data, with the number of groups represented by the variable K and works iteratively to assign each data point to one of K groups based on the features that are provided.

![K-Means Clustering](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/k-means-clustering.png)

Step breakdown:
* Step 1 - Choose the number K of clusters
* Step 2 - Select at random K points, the centroids (not necessarily from your dataset)
* Step 3 - Assign each data point to the closest centroid. This forms K clusters
* Step 4 - Compute and place the new centroid of each cluster
* Step 5 - Reassign each data point to the new closest centroid. If any reassignment took place, repeat Step 4, otherwise 
           go to FIN.
* FIN - Your model is ready.

To choose the right number of clusters, we use the 'Within Cluster Sum of Squares' formula.

![Within Cluster Sum of Squares Formula](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/wcss.png)

This is how it works:
* With every point in a cluster group, identify each points distance from the centroid and add those distances together. 
  Then square this number.
* Repeat this process with the other clusters.
* Add the total for each cluster together.

The more clusters you have, the smaller the distance will be.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/3.%20clustering/0.%20k_means.py) for an example of a k-means clustering. To make this use the [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) class from the Scikit-Learn library.

```python
# Using the Elbow Method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

## Hierarchical Clustering (HC)
This involves creating clusters that have a predetermined ordering from top to bottom. There are two types of hierarchical clustering, Divisive and Agglomerative.

Agglomerative clusters go from bottom to top. Divisive clusters go from top to bottom.

Hierarchical Clustering works like this:
* Step 1 - Make each data point a single-point cluster. This forms N clusters.
* Step 2 - Take the two closest data points and make them into one cluster. This forms N - 1 cluster.
* Step 3 - Take the two closest clusters and make them one cluster. That forms N - 2 clusters.
* Step 4 - Repeat Step 3 until there is only one cluster.
* FIN - Your model is ready.

There are a few methods we an take to find the distance between clusters. This could be either:
* Closest points from each cluster.
* Furthest points from each cluster.
* Average distance of each cluster.
* The distance between the centroids.

![Agglomerative HC](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/agglomerative-hc.png)

Each step is stored within a dendrogram. Dendrograms are scaled by distance between each point. Using these dendrograms you can see the hierarchy of each cluster to determine which cluster was created in the relevant order. You can also use these to create thresh holds of where to limit the height of the HC to find the most optimal set of clusters.

Dendrograms are easy to decipher, you can simply calculate how many clusters there are based on the amount of lines the thresh hold crosses through. 

![Dendrograms](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/dendrograms.png)

The longest distance of a horizontal line without it crossing any other lines is the optimal set of clusters to use for your model. Using the image as an example, 2 clusters would be the most optimal. 

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/3.%20clustering/1.%20hierarchical_clustering.py) for an example of a Hierarchical Clustering model. To make this use the [AgglomerativeClustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) class from the Scikit-Learn library.

```python
# Fit HC to dataset
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
```