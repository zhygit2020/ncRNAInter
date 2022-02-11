simType = "seqSimilarity";

switch(simType)
    case "seqSimilarity"
        pairs = load('Datasets/seqSimilarity/association_pair.csv');
        MIRNASimilarityMatrix=load('Datasets/seqSimilarity/mirna_seq_similarity_matrix.csv');
        LNCRNASimilarityMatrix = load('Datasets/seqSimilarity/lncrna_seq_similarity_matrix.csv');
    case "profileSimilarity"
        pairs = load('Datasets/profileSimilarity/association_pair_expression_profile.csv');
        MIRNASimilarityMatrix=load('Datasets/profileSimilarity/mi_expression_profile_similarity_matrix.csv');
        LNCRNASimilarityMatrix = load('Datasets/profileSimilarity/lnc_expression_profile_similarity_matrix.csv');
    case "functionSimilarity"
        pairs = load('Datasets/functionSimilarity/lncRNA_miRNA_association.csv');
        MIRNASimilarityMatrix=load('Datasets/functionSimilarity/mirna_function_similairty_matrix.csv');
        LNCRNASimilarityMatrix = load('Datasets/functionSimilarity/lncrna_function_similairty_matrix.csv');
    otherwise
        fprintf('seqSimilarity, profileSimilarity, or functionSimilarity\n');
end


nMIRNA = size(MIRNASimilarityMatrix, 1);
nLNCRNA = size(LNCRNASimilarityMatrix, 1);
pair_num=size(pairs,1);

for ii=1:pair_num
    interactionmat(pairs(ii,1),pairs(ii,2))=1;
end

save(strcat('Datasets/',simType, '/interactionmat.mat'))

interaction=interactionmat;
TEST=interaction;
kr=mean(LNCRNASimilarityMatrix(:))*1.3;
kp=mean(MIRNASimilarityMatrix(:))*1.0;
MIRNASM=zeros(nMIRNA,nMIRNA);
LNCRNASM=zeros(nLNCRNA,nLNCRNA);

for i=1:nMIRNA
    for j=1:nMIRNA
        if MIRNASimilarityMatrix(i,j)>kp
            MIRNASM(i,j)=1;
        end
    end
end

for i2=1:nLNCRNA
    for j2=1:nLNCRNA
        if LNCRNASimilarityMatrix(i2,j2)>kr
            LNCRNASM(i2,j2)=1;
        end
    end
end

NIJMI=MIRNASM*MIRNASM;
NIJLNC=LNCRNASM*LNCRNASM;

tmatMI=diag(NIJMI*(ones(nMIRNA,nMIRNA)-eye(nMIRNA)))+1;
tmatLNC=diag(NIJLNC*(ones(nLNCRNA,nLNCRNA)-eye(nLNCRNA)))+1;

zeroLNC=zeros(nMIRNA,nLNCRNA);
zeroMI=zeros(nLNCRNA,nMIRNA);
SS=zeros(nLNCRNA,nMIRNA);
n=round(size(pairs,1)/5);

for cv=1:100
    
    Nnn=round(n*round(length(SS(:))-length(pairs)+n)/length(pairs));
    idx = randperm(size(pairs,1));
    idx=idx(1:n);

    pairs5fold = pairs;
    pairs5fold(idx,:)=[];
    load(strcat('Datasets/',simType, '/interactionmat.mat'));
    interaction=interactionmat;
    interaction(pairs(idx,1),pairs(idx,2))=0;
    M=interaction;
    
    for i=1:nMIRNA
        KKR=find_p(interaction,i);
        for j=1:size(KKR)
            zeroLNC(i,KKR(j))=1/tmatLNC(KKR(j));
        end
    end
    XM=zeroLNC*NIJLNC;
    for i=1:nLNCRNA
        KKP=find_p(interaction',i);
        for j=1:size(KKP)
            zeroMI(i,KKP(j))=1/tmatMI(KKP(j));
        end
    end
    XP=zeroMI*NIJMI;
    VM=XM'\interaction;
    VP=XP'\interaction';
    S=(XM'*VM+(XP'*VP)')/2;

    SS=SS+S;
  
end
SSmat=SS/100;
save(strcat('Datasets/',simType, '/SSmat.mat'))

