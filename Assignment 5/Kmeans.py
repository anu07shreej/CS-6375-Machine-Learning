function centroids = computeCentroids(x,idx,k):
	n = size(x,2);
	centroids = zeros(k,n);
	for i in range(1,k):
		centroids(1,:) = mean(x(findx(idx==i),:));
	end
	end
