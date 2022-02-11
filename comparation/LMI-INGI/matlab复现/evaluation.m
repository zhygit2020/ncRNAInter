simType = "seqSimilarity";
load(strcat('Datasets/',simType, '/SSmat.mat'))

load(strcat('Datasets/',simType, '/pairsN.mat'))

trainM=SSmat(:);
train=sort(trainM,'descend');
thr=train(round(linspace(1,size(trainM,1),100)));


n=pair_num;
idx = randperm(size(pairsN,1));
idx=idx(1:n);
load(strcat('Datasets/',simType, '/interactionmat.mat'));
TEST=interactionmat;
for i=1:length(idx)
    TEST(pairsN(idx(i),1),pairsN(idx(i),2))=2;
end

testL=TEST(:);
numelpre=size(find(thr~=0),1);

Sen=zeros(1,numelpre); 
Spe=zeros(1,numelpre); 
Pre=zeros(1,numelpre);
Acc=zeros(1,numelpre);
ACC=zeros(1,numelpre);
TPR=zeros(1,numelpre);%REC, SEN
FPR=zeros(1,numelpre);
SPC=zeros(1,numelpre);%1-FPR
PPV=zeros(1,numelpre);%PRE
REC=zeros(1,numelpre);
NPV=zeros(1,numelpre);
FDR=zeros(1,numelpre);
MCC=zeros(1,numelpre);
F1=zeros(1,numelpre);
SPEC=zeros(1,numelpre);
tp=zeros(1,numelpre);
tn=zeros(1,numelpre);
np=zeros(1,numelpre);
nn=zeros(1,numelpre);

for I=1:numelpre   
    tp(I)=sum(trainM>=thr(I)&testL==1);
    tn(I)=sum(trainM<thr(I)&testL==2);
    np(I)=sum(trainM>=thr(I)&testL==2);
    nn(I)=sum(trainM<thr(I)&testL==1);
    TP=tp(I);TN=tn(I);FP=np(I);FN=nn(I);P=tp(I)+nn(I);N=tn(I)+np(I);
    
    TPR(I)=TP/P;
    FPR(I)=FP/N;
   
    NPV(I)=TN/(TN+FN);
    F1(I)=2*TP/(2*TP+FP+FN);
    if tp(I)+nn(I)==0
        Sen(I)=1;
        REC(I)=1;
    else
        Sen(I)=tp(I)/(tp(I)+nn(I));
        REC(I)=TP/(TP+FN);
    end
    
    if tn(I)+np(I)==0
        Spe(I)=1;
        SPEC(I)=1;
        SPC(I)=1;
    else
        Spe(I)=tn(I)/(tn(I)+np(I));
        SPEC(I)=TN/(TN+FP);
        SPC(I)=TN/N;
    end
    
    if tp(I)+np(I)==0
        Pre(I)=1;
        PPV(I)=1;
        FDR(I)=1;
    else
        Pre(I)=tp(I)/(tp(I)+np(I));
        PPV(I)=TP/(TP+FP);
        FDR(I)=FP/(FP+TP);
    end
    
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)==0
        MCC(I)=1;
    else
        MCC(I)=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    end
    
    if P+N==0
        ACC(I)=1;
    else
        ACC(I)=(TP+TN)/(P+N);
    end
  
end


AUC=abs(trapz(1-Spe,Sen));
AUPR=abs(trapz(Sen,Pre));

ACCm=sum(ACC)/length(ACC);
SPCm=sum(SPC)/length(SPC);
PPVm=sum(PPV)/length(PPV);
NPVm=sum(NPV)/length(NPV);
FDRm=sum(FDR)/length(FDR);
MCCm=sum(MCC)/length(MCC);
SPECm=sum(SPEC)/length(SPEC);
RECm=sum(REC)/length(REC);
Prem=sum(Pre)/length(Pre);
F1m=sum(F1)/length(F1);
    
figure;
plot(1-Spe,Sen);
axis([0 1.00 0 1.00]);
xlabel('1-Specificity');
ylabel('Sensitivity');

save(strcat('Datasets/',simType, '/Indicators.mat'))
