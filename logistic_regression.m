close all;
filePath = '/Users/yayulin/Documents/UCSD_courses/ECE_209/Project/dataset/';

nFBs = 1;
isEnsemble = 1;
fs = 256;
duration = 1000; % ms


n_sub = 8;

eeg = [];
labels = cell(1, n_sub);

dirty_flags = zeros(n_sub, 35);

NT_eeg = cell(1, n_sub);
T_eeg = cell(1, n_sub);

for i_sub = 1 : n_sub
    fileName = sprintf('A%02d.mat', i_sub);
    load([filePath fileName]);
    
    sample = data.X;
    StimType = data.y;
    StimClass = data.y_stim;
    trial = data.trial;

    testStartTime = trial(16);
    
    fc = 0.1;
    fs = 256;
    n = 4;
    [b,a] = butter(n,fc/(fs/2),'high');
    filtSample = filter(b, a, sample);

    fc = 10;
    [b,a] = butter(n,fc/(fs/2),'low');
    filtSample = filter(b, a, filtSample);

    wo = 50/(fs/2);  bw = wo/35;
    [b,a] = iirnotch(wo,bw);
    filtSample = filter(b, a, filtSample);

    start_NT = [];
    start_T = [];
   
    testStartIndexNT = -1;
    testStartIndexT = -1;
    testFlagNT = 0;
    testFlagT = 0;
   
    for i = 1:size(StimType, 1)-1
        if StimType(i) ~= StimType(i+1)
            if StimType(i+1) == 1
                start_NT = [start_NT; i+1];
                
                if i >= testStartTime && testFlagNT == 0
                    testStartIndexNT = size(start_NT, 1);
                    testFlagNT = 1;
                end
            elseif StimType(i+1) == 2
                start_T = [start_T; i+1];
                
                if i >= testStartTime && testFlagT == 0
                    testStartIndexT = size(start_T, 1);
                    testFlagT = 1;
                end
            end
        end
    end

    nonTarget = [];
    count_NT = 0;
    
    for i = 1:size(start_NT, 1)
        if max(max(filtSample(start_NT(i):start_NT(i)+255,:)))<70 &&...
                min(min(filtSample(start_NT(i):start_NT(i)+255,:)))>-70
            baseline = mean(filtSample(start_NT(i):start_NT(i)+50,2));
            nonTarget = cat(3, nonTarget, filtSample(start_NT(i):start_NT(i)+255, :).'-baseline);
            count_NT = count_NT + 1;
        else
            disp('dirty');
        end
    end

    target = [];
    count_T = 0;
    for i = 1:size(start_T, 1)
        if max(max(filtSample(start_T(i):start_T(i)+255,:)))<70 &&...
                min(min(filtSample(start_T(i):start_T(i)+255,:)))>-70
            baseline = -mean(filtSample(start_T(i):start_T(i)+50,2));
            target = cat(3, target, filtSample(start_T(i):start_T(i)+255,:).'-baseline);
            count_T = count_T + 1;
        else
            disp('dirty');
        end
    end
    
    eeg_tmp = cat(3, nonTarget, target);
    labels_tmp = [ones(1, count_NT) ones(1, count_T) * 2];
    
    NT_eeg{i_sub} = nonTarget;
    T_eeg{i_sub} = target;
    labels{i_sub} = labels_tmp;
    
    testStartNT(i_sub) = testStartIndexNT;
    testStartT(i_sub) = testStartIndexT;
end

%% Resample
len = 204;
idx = 0:12:len+1;
NT_res = cell(1, n_sub);
T_res = cell(1, n_sub);
for sub = 1:8
        for i = 1:length(idx)-1
            T_res{sub}(:,i,:) = mean(T_eeg{sub}(:,idx(i)+1:idx(i+1),:),2);
        end
end
for sub = 1:8
        for i = 1:length(idx)-1
            NT_res{sub}(:,i,:) = mean(NT_eeg{sub}(:,idx(i)+1:idx(i+1),:),2);
        end
end

NT_eeg = NT_res;
T_eeg = T_res;



%% 7 folds Logistic Regression
train_trials = 1 : 4200;

n_targret = 2;

n_fold = 7;

accuracy_raw = [];

for i_sub = 1 : n_sub
    lda_accs = [];
    true_labels = [];
    LR_estimate = [];
    LDA_estimate = [];
    
    start = 1;
    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
        valid_NT_ind = ceil((i_fold - 1) * NT_nTrial / n_fold + 1) : ceil(i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = ceil((i_fold - 1) * T_nTrial / n_fold + 1) : ceil(i_fold * T_nTrial / n_fold);
        
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        X = cat(3, fold_train_NT, fold_train_T);
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        
%         
        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        fold_accs = [];
        
        [coeff,score,latent,tsquared] = pca(X);
        feat_n = 28;
        feature = coeff(:,1:feat_n);
        % Logistic regression
        B = mnrfit(X,Y);
        
        % Fit discriminant analysis classifier
        Mdl = fitcdiscr(X,Y);
        
        % SVM
%         SVMModel = fitcsvm(X*feature,Y);
            

            % Test phase ---------------------------------------
            testdata = valids; 
            testdata = squeeze(reshape(testdata, 1, size(testdata, 1) *...
                size(testdata, 2), size(testdata, 3))).';
            
            pihat = mnrval(B,testdata);
            [M,estimated] = max(pihat,[],2);
             
            ypred = predict(Mdl,testdata);
%             ypred_SVM = predict(SVMModel,testdata*feature);

            % Evaluation ----------------------------------------
            isCorrect_LR = (estimated==valid_labels);
            isCorrect = (ypred==valid_labels);
%             isCorrect_SVM = (ypred_SVM==valid_labels);
            
            fold_accs = mean(isCorrect);
            fold_accs_LR = mean(isCorrect_LR);
%             fold_accs_SVM = mean(isCorrect_SVM);
    
            logistic_accs(i_fold) = mean(fold_accs); 
            logistic_accs_LR(i_fold) = mean(fold_accs_LR); 
%             logistic_accs_SVM(i_fold) = mean(fold_accs_SVM); 
            
            true_labels = [true_labels reshape(valid_labels,1,[])];
            LR_estimate = [LR_estimate reshape(estimated,1,[])];
            LDA_estimate = [LDA_estimate reshape(ypred,1,[])];
%             SVM_estimate = [SVM_estimate ypred_SVM];
    end
    [score_LDA, C_LDA] = f1_score(reshape(LDA_estimate,[],1), reshape(true_labels,[],1));
    [score_LR, C_LR] = f1_score(reshape(LR_estimate,1,[]), reshape(true_labels,1,[]));
%     [score_SVM, C_SVM] = f1_score(reshape(SVM_estimate,1,[]), reshape(true_labels,1,[]));
    i_sub
    feat_n
    fprintf('7 fold accuracy LDA = %2.2f%%\n', mean(logistic_accs)*100)
    fprintf('score = %2.2f\n', score_LDA); 
    disp(C_LDA);
    fprintf('7 fold accuracy LR = %2.2f%%\n', mean(logistic_accs_LR)*100)
    fprintf('score = %2.2f\n', score_LR);
    disp(C_LR);
%     fprintf('7 fold accuracy SVM = %2.2f%%\n', mean(logistic_accs_SVM)*100)
%     fprintf('score = %2.2f\n', score_SVM);
%     disp(C_SVM);
    accuracy_raw = [accuracy_raw mean(logistic_accs_LR)*100];
    
end


%% 7 folds SF Logistic Regression
train_trials = 1 : 4200;

n_targret = 2;

n_fold = 7;


accuracy_SF = [];

for i_sub = 1 : n_sub
    lda_accs = [];
    true_labels = [];
    LR_estimate = [];
    LDA_estimate = [];
    start = 1;
    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
       valid_NT_ind = ceil((i_fold - 1) * NT_nTrial / n_fold + 1) : ceil(i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = ceil((i_fold - 1) * T_nTrial / n_fold + 1) : ceil(i_fold * T_nTrial / n_fold);
         
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        X = cat(3, fold_train_NT, fold_train_T);
        [X, SF] = SpatialFilter(X, Y, 4);
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';      

        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        fold_accs = [];

            % Test phase ---------------------------------------
            testdata = valids;
            testdata = getSFData(testdata, SF);
            testdata = squeeze(reshape(testdata, 1, size(testdata, 1) * size(testdata, 2), size(testdata, 3))).';
           
        [coeff,score,latent,tsquared] = pca(X);
        
        feat_n = 40;
        feature = coeff(:,1:feat_n);
        
        % Logistic regression
        B = mnrfit(X*feature,Y);
        
        % Fit discriminant analysis classifier
        Mdl = fitcdiscr(X*feature,Y);
        ypred = predict(Mdl,testdata*feature);
            

            pihat = mnrval(B,testdata*feature);
            
            [M,estimated] = max(pihat,[],2);

            % Evaluation ----------------------------------------
            isCorrect_LR = (estimated==valid_labels);
            isCorrect = (ypred==valid_labels);
            fold_accs = mean(isCorrect);
            fold_accs_LR = mean(isCorrect_LR);
    
            logistic_accs(i_fold) = mean(fold_accs); 
            logistic_accs_LR(i_fold) = mean(fold_accs_LR); 
            
            true_labels = [true_labels reshape(valid_labels,1,[])];
            LR_estimate = [LR_estimate reshape(estimated,1,[])];
            LDA_estimate = [LDA_estimate reshape(ypred,1,[])];
    end
    [score_LDA, C_LDA] = f1_score(reshape(LDA_estimate,[],1), reshape(true_labels,[],1));
    [score_LR, C_LR] = f1_score(reshape(LR_estimate,1,[]), reshape(true_labels,1,[]));
    i_sub
    feat_n
    fprintf('7 fold accuracy LDA = %2.2f%%\n', mean(logistic_accs)*100)
    fprintf('score = %2.2f\n', score_LDA); 
    disp(C_LDA);
    fprintf('7 fold accuracy LR = %2.2f%%\n', mean(logistic_accs_LR)*100)
    fprintf('score = %2.2f\n', score_LR);
    disp(C_LR);
    accuracy_SF = [accuracy_SF mean(logistic_accs_LR)*100];
end

