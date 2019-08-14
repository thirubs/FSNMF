%% Update and return difference in objective function
function [Wout,diffobj] = goWiter(GW,W,HH,n,k)
%si = zeros(n,k);
diffobj = zeros(n,k);
for i = 1 : n       
    for  j= 1:k % k = number of columns/features in source 
        s = (GW(i,j)+0)/(HH(j,j));
        s = W(i,j)-s;
        if( s< 0)
        	s=0;
        end
        s = s-W(i,j);
        %si(i,j) = s;
        diffobj(i,j) = (-1)*s*GW(i,j)-0.5*HH(j,j)*s*s;
        W(i,j) = W(i,j) + s;
     end
end



%% outputs
Wout = W; 
