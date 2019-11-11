def centroids = initCentroids(x,k):
	centroids = zeros(k,size(k,2))
	randIndex = randperm(size(x,1))
	centroids = x(randIndex(1:K),:)
	end