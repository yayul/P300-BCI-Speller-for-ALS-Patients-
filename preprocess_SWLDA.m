%% ECE 209
close all; clear all;

%% Preprocess Data
Fs = 256; %frequency
len = [51,255]; %length of epoch
K = 8;
%Filters
[b,a] = butter(4,[0.1 10]/(Fs/2)); %band-pass filter
d = designfilt('bandstopiir','FilterOrder',2, ... %notch filter
               'HalfPowerFrequency1',49,'HalfPowerFrequency2',51, ...
               'DesignMethod','butter','SampleRate',256);

target = cell(K,2,8);
for k = 1:K
    disp("Preprocessing for subject "+k+"...")
    load(['A0',int2str(k),'.mat']);
    y = data.y;
    Xf = filter(b,a,data.X); 
    Xf = filtfilt(d,Xf);

    for j = 1:2
        start = [];
        for i = 1:length(y)-1
            if (y(i+1) - y(i)) == j
                start = [start; i+1];
            end
        end

        for c = 1:8
            target{k}{j}{c} = [];
            for i = 1:length(start)
                if max(max(Xf(start(i):start(i)+len(2),:)))<70 &&...
                    min(min(Xf(start(i):start(i)+len(2),:)))>-70
                    target{k}{j}{c} = [target{k}{j}{c}, Xf(start(i):start(i)+len(2),c)]; %-mean(Xf(start(i)-51:start(i)-1,c))];
                end
            end
        end
    end
end
disp("Done")
%%
X = cell(K,2); X_mean = cell(K,2);
for k = 1:K
    for j = 1:2
        X{k}{j} = shiftdim(cat(3,target{k}{j}{:}),2); 
        X_mean{k}{j} = mean(X{k}{j},3);
    end
end
%% Plot data (check)   
plot(0:len(2),X_mean{8}{1}(2,:),0:len(2),X_mean{8}{2}(2,:));
xlabel("ms"); ylabel("\muv");
xticks(0:len(2)/10:len(2)); xticklabels(0:100:1000);
xlim([0 len(2)]); ylim([-5 5]);

%% Resampling
X_res = cell(size(X));
idx = len(1)+1:12:len(2)+1;
for k = 1:K
    for j = 1:2
        for i = 1:length(idx)-1
            X_res{k}{j}(:,i,:) = mean(X{k}{j}(:,idx(i)+1:idx(i+1),:),2);
        end
    end
end

%% SWLDA and Spatial Filtering
clear c
%acc_res_SF = zeros(K,7); score_res_SF = zeros(size(acc_res_SF));
%acc_res = zeros(K,7); score_res = zeros(size(acc_res));

for k = 1:K
    %X = [T_res{i}{1}; T_res{i}{2}];
    for j = 1:2
        labels{k}{j} = j*ones(length(X{k}{j}),1);
        c{k}{j} = cvpartition(labels{k}{j},'k',7);
    end
    disp("Classifying for subject "+k+"...");
    for i = 1:7 %k-fold cross validation
        %Raw
        %[X_test,X_train,Y_test,Y_train] = split(X{k},labels{k},c{k},i);
        %acc(k,i) = SWLDA(X_train,Y_train,X_test,Y_test)
        
        %Resampled
        [X_res_train,X_res_test,Y_res_train,Y_res_test] = split(X_res{k},labels{k},c{k},i); %Resampled
        [acc_res(k,i),score_res(k,i),C_res{k}{i}] = SWLDA(X_res_train,Y_res_train,X_res_test,Y_res_test)

        %Spatial Filtered
        %[X_SF_train, SF] = SpatialFilter(X_train, Y_train, 4);
        %X_SF_test = getSFData(X_test, SF);
        %acc_SF(k,i) = SWLDA(X_SF_train,Y_train,X_SF_test,Y_test)

        %Spatial Filtered + Resampled
        %[X_res_SF_train, SF] = SpatialFilter(X_res_train, Y_res_train, 2);
        %X_res_SF_test = getSFData(X_res_test, SF);
        %[acc_res_SF(k,i),score_res_SF(k,i),C_res_SF{k}{i}] = SWLDA(X_res_SF_train,Y_res_train,X_res_SF_test,Y_res_test)
    end
end

function [X_train,X_test,Y_train,Y_test] = split(X,labels,c,k)
    X_test = cat(3,X{1}(:,:,test(c{1},k)), X{2}(:,:,test(c{2},k)));
    X_train = cat(3,X{1}(:,:,~test(c{1},k)), X{2}(:,:,~test(c{2},k)));
    Y_train = [labels{1}(~test(c{1},k)); labels{2}(~test(c{2},k))];
    Y_test = [labels{1}(test(c{1},k)); labels{2}(test(c{2},k))];
end

function [acc,score,C] = SWLDA(X_train,Y_train,X_test,Y_test)
    [c,t,l] = size(X_train);
    X_train = reshape(X_train,[c*t,l])'; X_test = reshape(X_test,[c*t,length(X_test)])';
    mdl = stepwiseglm(X_train, Y_train,'constant','upper','linear','distr','Normal','NSteps',60,'PEnter',0.1,'PRemove',0.15);
    if (mdl.NumEstimatedCoefficients>1)
        inmodel = [];
        for i=2:mdl.NumEstimatedCoefficients
            inmodel = [inmodel str2num(mdl.CoefficientNames{i}(2:end))];
        end
    X_train = X_train(:,inmodel);
    X_test = X_test(:,inmodel);
    end
    Y_predict = classify(X_test,X_train,Y_train,'linear');
    acc = sum(Y_predict==Y_test)/length(Y_predict);
    [score,C] = f1_score(Y_predict,Y_test);
end

%Spatial Filter
function [SFData, spatialFilters, eigVal] = SpatialFilter(data, label, n)

    if ~exist('n', 'var') || isempty(n)
    n = 1; end

    spatialFilters = zeros(size(data, 1), 2 * n);

    data1 = data(:, :, label == 1);
    cov1 = zeros(size(data1, 1), size(data1, 1));
    for i = 1 : size(data1, 3)
        cov1 = cov1 + cov(squeeze(data1(:, :, i)).');
    end
    cov1 = cov1 / size(data1, 3);

    data2 = data(:, :, label == 2);
    cov2 = zeros(size(data2, 1), size(data2, 1));
    for i = 1 : size(data2, 3)
       cov2 = cov2 + cov(squeeze(data2(:, :, i)).');
    end
    cov2 = cov2 / size(data2, 3);

    %[eigVec, eigVal] = eigs(cov2 \ cov1);

    [eigVec,eigVal] = eig(cov2 \ cov1);
    [eigVal,I] = sort(diag(eigVal));
    eigVec = eigVec(:, I);
    
    for i = 1 : n
        spatialFilters(:, i) = eigVec(:, i);
        spatialFilters(:, n + i) = eigVec(:, end - i + 1);
    end
    
    SFData = getSFData(data, spatialFilters);
        
end

function SFData = getSFData(data, spatialFilters)
    SFData = zeros(size(spatialFilters, 2), size(data, 2), size(data, 3));
    
    for i = 1 : size(data,3)
        SFData(:, :, i) = spatialFilters.' * squeeze(data(:, :, i));
    end
end

function [score, C] = f1_score(predictedLabels, trueLabels)

    yHaT = predictedLabels;
    yval = trueLabels;

    tp = sum((yHaT == 2) & (yval == 2));
    fp = sum((yHaT == 2) & (yval == 1));
    fn = sum((yHaT == 1) & (yval == 2));

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    score = (2 * precision * recall) / (precision + recall);
    
    C = confusionmat(trueLabels, predictedLabels);
end