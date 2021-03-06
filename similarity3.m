function m = similarity3(sp1, sp2, opts)
% Returns the similarity score of two superpixels using vari s methods
% As seen with Matlab profiler, this function is responsible for majority
% of calculation for the whole algorithm.
sigma = 0.02;
EPSILON = 1e-5;
N = length(sp1.hist);
d = zeros(1, N);
% a = 1;

for fn = 1:N
    d(fn) = sqrt(sum((sp1.hist{fn} - sp2.hist{fn}).^2));     
end
m = exp(-sigma*mean(d))+ EPSILON;



