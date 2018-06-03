function label = getLabel(I_rgb,sp)
[h,w,d] = size(I_rgb);
label = zeros(h*w,1);
for i = 1:length(sp)
    sub = double(sp{i}.pixels);
    ind = sub2ind([h,w],sub(:,1),sub(:,2));
    label(ind) = i;
end

end