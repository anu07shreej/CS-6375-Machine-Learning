function idx = assignCentroids(x,Centroids):

	k = size(centroids,1);
	idx = zeroes(size(x,1),1);
	for i = 1:size(x,1):
		distances = zeroes(1,k);
		for j = 1:k:
			distances(1,j) = sumsq(X(i, :) - centroids(j, :));
		end
		[one, idx(i)] = min(distances);
	end
end
		