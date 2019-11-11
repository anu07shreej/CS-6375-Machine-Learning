def centroids(x,idx,k):
	n = size(x,2)
	centroids = zeros(k,n)
	for i in range(1,k):
		centroids(i,:) = mean(x(find(idx==i), :))
	end
	end