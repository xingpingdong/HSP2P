function [gphs] = constructGraphs(sp_s,K_s)
% Input
%   sp_s  -  the superpixels of images, 1 x nI (cell)
%   K_s   -  the adjacency relationship between superpixels, 1 x nI (cell)
%              
%            
% Output     
%   gphs      -  graphs
%     Pt     -  graph node, d x n
%     Eg     -  graph edge, 2 x 2m
%     vis    -  binary indicator of nodes that have been kept, 1 x n | []
%     G      -  node-edge adjacency, n x m
%     H      -  augumented node-edge adjacency, n x (m + n)
%     PtD    -  edge feature, 2 x 2m
%     dsts   -  distance, 1 x 2m
%     angs   -  angle, 1 x 2m
%     angAs  -  angle, 1 x 2m
for i = 1:length(sp_s)
    [gphs{i}] = constructGraph(sp_s{i},K_s{i});    
end
end