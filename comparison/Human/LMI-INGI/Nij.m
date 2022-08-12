function [ nij ] = Nij( i,j,ktu,smatrix )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nij=0

    if k==1
        if smatrix(i,j)==1
            nij=1;
        end
        else
            nij=0;
    end
    if k==2
        for s=size(smatrix,1)
            if smatrix(i,s)==1 && smatrix(s,j)==1 && s~=i && s~=j
                nij=nij+1;
            end
        end
    end
    if k==3
        for s=size(smatrix,1)
            if smatrix(i,j)==1 && smatrix(s,j)==1 && s~=i && s~=j
                nij=nij+1;
            end
        end
    end
    
    if k==4
        for s=size(smatrix,1)
            if smatrix(i,s)==1 && smatrix(i,j)==1 && s~=i && s~=j
                nij=nij+1;
            end
        end
    end
    
    if k == 5
        for s=size(smatrix,1)
            if smatrix(i,s)==1 && smatrix(s,j)==1 && smatrix(i,j)==1 && s~=i && s~=j
                nij=nij+1;
            end
        end
    end
    if k == 6
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(i,s)==1 && smatrix(s,t)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 7
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(s,j)==1 && smatrix(i,s)==1 && smatrix(i,t)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 8
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(s,j)==1 && smatrix(t,s)==1 && smatrix(i,t)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
               end
            end
        end
    end
    if k == 9
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 10
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,t)==1 && smatrix(s,j)==1 && smatrix(t,j)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k ==11
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,i)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 12
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,s)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 13
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,j)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 14
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(i,t)==1 && smatrix(i,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 15
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,t)==1 && smatrix(s,j)==1 && smatrix(i,s)==1&& smatrix(j,t)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1;
                end
            end
        end
    end
    if k == 16
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1&& smatrix(t,i)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 17
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 18
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 19
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 20
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 21
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 22
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 23
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 24
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 25
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 26
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 27
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    if k == 28
        for s=size(smatrix,1)
            for t=size(smatrix,1)
                if smatrix(i,j)==1 && smatrix(s,j)==1 && smatrix(t,s)==1 && s~=i && s~=j && t~=i && t~=j
                    nij=nij+1
                end
            end
        end
    end
    
end

