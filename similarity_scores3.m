function scores = similarity_scores3(sp, K, opts)
% Returns similarity scores for all uperpixel pairs in K

scores = zeros(1,size(K,1));

for k = 1:size(K,1)
    g = sort([K(k,1), K(k,2)]);
    scores(k) = similarity3(sp{K(k,1)}, sp{K(k,2)}, opts);
end
