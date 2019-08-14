%% Input Argument
% X - Target Domain Matrix
% XS - Source Domain Matrix
% J - Rank of the matrix
% maxiter - Maximum number of Iterations

% You may need Matlab Tensor Toolbox library

%% Random Initialization of Factor matrices 
N = ndims(XS);
Uinit = cell(N,1);    
for n = 1:N 
    UinitS{n} = normalize_factor(rand(size(XS,n),J),2);        
end
W_S_NMF = UinitS{1};
H_S_NMF = UinitS{2};

Uinit = cell(N,1);    
for n = 1:N 
    Uinit{n} = normalize_factor(rand(size(X,n),J),2);        
end

W_O = Uinit{1};
H_O = Uinit{2};

%% Learn features of source domain
fprintf('\n NMF on Source Domain\n')
tic
opt = statset('MaxIter',maxiter,'Display','final');
[W_S,H_S,~] = nnmf(X_S,J,'w0',W_S_NMF,'h0',H_S_NMF','options',opt,'algorithm','mu');
toc

%% NMFTLFS
% W_S - Factor matrix from source domain
fprintf('\n Transfer Learning via Feature Selection based NMF (FSNMF) \n')
tic
[FSNMF_W, FSNMF_H] = FSNMF(X, W_S, W_O,H_O, maxiter);
toc

