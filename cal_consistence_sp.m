function scores = cal_consistence_sp(feaA,spA_center,spB_center,final_sp_match,spA)
% A consistency criterion to calculate a coherence error for a group of
% neighbour superpixels.

scores = zeros(size(feaA,1),1);
coor_match = spB_center(final_sp_match,:);
T = coor_match-spA_center;  %平移向量
Tx = T(:,1); Ty = T(:,2);
x1 = spA_center(:,1);
y1 = spA_center(:,2);
for i = 1:length(spA)
    neighbors = spA{i}.neighbors;
    dTx = Tx(neighbors) - Tx(i);
    dTy = Ty(neighbors) - Ty(i);
    dT = sqrt(dTx.^2+dTy.^2); %平移向量的差值
%     dx1 = x1(neighbors) - x1(i);
%     dy1 = y1(neighbors) - y1(i);
%     d1 = sqrt(dx1.^2+dy1.^2); %相邻超像素的中心位置差
%     C = dT./d1;
    C = dT;
    % 根据颜色差确定 权值
    fea_i = feaA(i,:);
    fea_i = repmat(fea_i,length(neighbors),1);
    fea_neigh = feaA(neighbors,:);
    d_fea = sqrt(sum((fea_neigh-fea_i).^2,2));
    w = exp(-0.02*d_fea);
    w = w./sum(w);
    scores(i) = w'*C;
end
end