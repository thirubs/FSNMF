function Wout = goHiter(GW,HH)
%{
for i = 1 : n
    for  j= 1:k 
        %nowidx = nowidx+1;
        %double s = GW[nowidx]/HH_d[j];
        s = (GW(i,j)+0)/(HH(j,j));
       % s= W(i,j)-s;
        %if( s< 0)
		%		s=0;
        %    end
		%	s = s-W(i,j);
        ss(i,j) = s;
        W(i,j) = W(i,j)+ss(i,j);
        w(i,j) = s;
     end
end
%}


W = GW/HH;
 if issparse(W)
                W = full(W);   % for the case R=1
 end
Wout = W;
