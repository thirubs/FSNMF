function [W,H] = FSNMF(V,W_S,W_T,H0,maxiter)
W_S = normalize_factor(W_S,2);
H = H0; W = W_S;
epsilon = 0.0000001;
for iter=1:maxiter,
    rows_and_cols = size(W_S);
    rows_S = rows_and_cols(1);
    cols_S = rows_and_cols(2);
    cols_T = size(W,2);
    Hmat = (H' *H)+epsilon;
    
    %% Learn W_S 
    %if iter == 1
    U_in{1,1} = W_S;
    U_in{2,1} = H;
    tmpmat=mttkrp(V,U_in,1);
    grad_S= -(tmpmat-(W_S *Hmat));
    [W_S, S_diff] = goWiter(grad_S,W_S,Hmat,rows_S, cols_S);
    %[W_S, S_diff] = goWiter2(grad_S,W_S,Hmat,rows_S, cols_S); % using this gWiter2 for W_S and W_T is improving the performance drastically
    %Check nonnegativity 
    W_S(W_S<=epsilon)=epsilon;
    %Normalization (if you need)
    W_S=normalize_factor(W_S,2);
    
    %% Learn W_T
    %end

    U_in{1,1} = W_T;
    U_in{2,1} = H;
    tmpmat = mttkrp(V,U_in,1);
    grad_T= -(tmpmat-(W_T *Hmat));
    [W_Tn, T_diff] = goWiter(grad_T,W_T,Hmat,rows_S, cols_S);
    %[W_Tn, T_diff] = goWiter2(grad_T,W_T,Hmat,rows_S, cols_S);
    if sum(sum(W_Tn)) == 0
        fprintf('True')
       [W_Tn, T_diff] = goWiter2(grad_T,W_T,Hmat,rows_S, cols_S);
    end
    W_T = W_Tn;
    %Check nonnegativity 
    W_T(W_T<=epsilon)=epsilon;
    %Normalization (if you need)
    W_T=normalize_factor(W_T,2);

    %% Feature Selection
    total_source = sum(S_diff);
    total_target = sum(T_diff);
    %total_rank = source_rank + target_rank;
    features = horzcat(W_S,W_T);
    % horizondal concat of features
    feature_list = horzcat(total_source,total_target);
    % now sort it based on value. 
    [~,I] = sort(feature_list,'descend');
    columns_to_update = I(1:cols_T);
    for j = 1:cols_T
        W(:,j) = features(:,columns_to_update(j));
        if columns_to_update(j) <= cols_S        
            feature_used(j) = 1;
            overall_diff(j) = total_source(columns_to_update(j));
        else
            feature_used(j) = 0;
            overall_diff(j) = total_target(columns_to_update(j)-cols_S);
        end
    end
    sum_feature = sum(feature_used);

%% Update H

    %if iter == 1
    U_in{1,1} = W;
    U_in{2,1} = H;
    tmpmat=mttkrp(V,U_in,1);
    grad= -(tmpmat-(W *Hmat));
    [~, Diff_c] = goWiter(grad,W,Hmat,rows_S, cols_S);
    difff = sum(sum(Diff_c));
    U_in{1,1} = W;
    U_in{2,1} = H;
    Hmat = (U_in{1,1}' *U_in{1,1})+epsilon;
    tmpmat=mttkrp(V,U_in,2);
    grad= -(tmpmat-(H *Hmat));
    [Hnew]=goHiter(grad,Hmat);
    if sum(sum(Hnew)) == 0
        fprintf('True H')
       [Hnew, ~] = goWiter2(grad,H,Hmat,size(H,1), cols_S);
    end
    H =  Hnew;
    %Check nonnegativity 
    H(H<=epsilon)=epsilon;
    %Normalization (if you need)

    H=normalize_factor(H,2);
    fprintf('Source feature used in Iteration %d is %d and Difference is %d\n',iter, sum_feature,sum(overall_diff));
end
