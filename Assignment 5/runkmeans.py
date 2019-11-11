function[centroids,idx] = runKmeans(x,initialCentroids,iter):
	m = size(x,1)
	k = size(initialCentroids,1)
	centroids= initialCentroids
	idx = 	zeros(m,1)
	for i in range(1,itr):
		fprintf('K-means' +i+ ''+iter)
		idx = assignCentroids(x,centroids)
		centroids =  computeCentroids(x,idx,k)
		end
	end