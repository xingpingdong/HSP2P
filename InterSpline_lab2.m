function [ paraMat ] = InterSpline_lab2( A, B, candinate1,candinate2, channel )

% Cubic spline interpolation in the single channel of the image
% Input :
% A, B                   :  nA x 3, nB x 3. The average color of superpiels 
% candinate1,candinate2  :  nMatch x 1. The indexes of the matches. 
% channel          :  r, g, b
% Output :
% paraMat         : The coefficient of the spline

sC_A = A(:, channel); % single channel of image
sC_B = B(:, channel);
Nmatches = size(candinate1, 1);

vec_sCA = zeros(1, Nmatches); 
vec_sCB = zeros(1, Nmatches); % the value of the matched pixels

for i = 1 : Nmatches    
    
    tempValA = sC_A(candinate1(i)); tempValB = sC_B(candinate2(i));
    vec_sCA(1, i) = tempValA;
    vec_sCB(1, i) = tempValB;
end

% Sort and delete the repetitive elements
two_sC = [vec_sCA; vec_sCB];
st_sC = sortrows(two_sC', 1)';
diff_sC = diff(st_sC(1,:));
ind_sC = (diff_sC == 0);
st_sC(:, ind_sC) = [];

% Choose five Break Points
Nst_sC = size(st_sC, 2);
NbreakP = 7;
indInter = round(Nst_sC/NbreakP) : round(Nst_sC/NbreakP) : Nst_sC;
indInter = indInter(1 : (NbreakP-1));

% Add the head and tail constraint
% BreakPs = st_sC(:, indInter)/255;
if channel == 1
    BreakPs = st_sC(:, indInter)/100;
    BreakPs = [[-0.1; -0.1] BreakPs [1.1; 1.1]];
else
    BreakPs = (st_sC(:, indInter)+128)/255;
    BreakPs = [[-0.1; -0.1] BreakPs [1.1; 1.1]];
end
% BreakPs = [[-0.1; -0.1] BreakPs [1.1; 1.1]];
paraMat = csape(BreakPs(1, :), BreakPs(2, :), 'variational');

%Test Display
xx = 0 : 0.05 : 1; yy = ppval(paraMat, xx);
figure,plot(BreakPs(1, :), BreakPs(2, :), 'o', xx, yy);

end

