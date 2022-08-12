simType = "seqSimilarity";
load(strcat('Datasets/',simType, '/SSmat.mat'))

pairtotal=zeros(nLNCRNA*nMIRNA,2);
m=1;n=1;
for i=1:nLNCRNA*nMIRNA
    pairtotal(i,1)=m;
    pairtotal(i,2)=n;
    n=n+1;
    if n>nMIRNA
        m=m+1;
        n=1;
    end
    if m>nLNCRNA
        break;
    end
end
pairstest=pairs;
for i=1:nLNCRNA*nMIRNA
    for j=1:pair_num
        if pairtotal(i,1)==pairstest(j,1) && pairtotal(i,2)==pairstest(j,2)
            pairtotal(i,1)=0;
            pairtotal(i,2)=0;
        end
    end
    
end
pairsN=pairtotal(find(pairtotal(:,1)~=0),:);
save (strcat('Datasets/',simType, '/pairsN.mat'))
