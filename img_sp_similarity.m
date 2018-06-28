function scores_AB = img_sp_similarity(spA,spB,opts)

nA = length(spA);
nB = length(spB);
 ores_AB = zeros(nA,nB);
for i = 1:nA
    for j = 1:nB
        scores_AB(i,j) = similarity(spA{i},spB{j},opts);
    end
end

end

