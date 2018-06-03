function newA = handle_sat_change(A,newA)
% handle saturation changes for rgb color space
% we project pixel colors from both images along the gray line
% (eliminating luminance variation) and optimize for the scale factor
% s that best fits the corresponding chrominances alone
% a uniform scale about the gray line
w_rgb = [-1,1,1;...
         1,-1,1;...
         1,1,-1]./3; % gray model
E = diag(ones(3,1));

diff_A = newA - A;


[h,w,d] = size(diff_A);

V = reshape(diff_A,[h*w,d]);
Vw = V*w_rgb;
up = sum(sum(V.*Vw,2));
down = sum(sum(V.*V,2))+eps;
s = -mean(up./down);
w_u = s*E + w_rgb;
Ar = reshape(newA,[h*w,d]);
newA = Ar*(w_u'/w_rgb);
newA = reshape(newA,[h,w,d]);

end