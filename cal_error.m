function error = cal_error(A,B)
 [hA,wA,dA] = size(A);
 error = double(A*255)-double(B*255);
 error = reshape(error,hA*wA,dA);
 error = sqrt(sum((sum(error.^2,2))))/(hA*wA);
end