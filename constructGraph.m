function [gph] = constructGraph(sp,K)
% Input
%   sp  -  the superpixels of image, n_sp x 1 (cell)
%   K   -  the adjacency relationship between superpixels, n_edge x 2 
%              
%            
% Output     
%   gph      -  graph
%     Pt     -  graph node, d x n
%     Eg     -  graph edge, 2 x 2m
%     vis    -  binary indicator of nodes that have been kept, 1 x n | []
%     G      -  node-edge adjacency, n x m
%     H      -  augumented node-edge adjacency, n x (m + n)
%     PtD    -  edge feature, 2 x 2m
%     dsts   -  distance, 1 x 2m
%     angs   -  angle, 1 x 2m
%     angAs  -  angle, 1 x 2m
n = length(sp);    
% center cordinate of each superpixel
d_c = length(sp{1}.hist{1});
sp_center = zeros(2,n); colors = zeros(d_c,n);
for i = 1:n
    sp_center(:,i)= mean(sp{i}.pixels,1);
    colors(:,i) = sp{i}.hist{1};
end
%data point
Pt = [sp_center;colors];
%edge
K2(:,1) = K(:,2); K2(:,2) = K(:,1);
Eg = [K;K2]; 
ind = sub2ind([n,n],Eg(:,1),Eg(:,2));
Eg = Eg';
m2 = size(Eg,2);
% val = ones(m2,1);

vis = zeros(n,n);
vis(ind)=1;

% incidence matrix
[G, H] = gphEg2IncU(Eg, n);

% second-order feature
[PtD, dsts, angs, angAs] = gphEg2Feat(Pt, Eg);

% store
gph.Pt = Pt;
gph.Eg = Eg;
gph.vis = vis;
gph.G = G;
gph.H = H;
gph.PtD = PtD;
gph.dsts = dsts;
gph.angs = angs;
gph.angAs = angAs;

end