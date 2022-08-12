function [ p ] = find_p( A,i )
%FIND_P Summary of this function goes here
%   Detailed explanation goes here
    n=1;
    p=[];
    for m=1:size(A,1)
        if A(m,i)==1
            p(n)=m;
            n=n+1;
        end
    end
end

