close all;
filePath = '../data/';

nFBs = 1;
isEnsemble = 1;
fs = 256;
duration = 1000; % ms

n_sub = 8;

eeg = [];

dirty_flags = zeros(n_sub, 35);

NT_eeg = cell(1, n_sub);
T_eeg = cell(1, n_sub);
labels = cell(1, n_sub);

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
            nonTarget = cat(3, nonTarget, filtSample(start_NT(i):start_NT(i)+255, :).');
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
            target = cat(3, target, filtSample(start_T(i):start_T(i)+255,:).');
            count_T = count_T + 1;
        else
            disp('dirty');
        end
    end
    
    eeg_tmp = cat(3, nonTarget, target);
    labels_tmp = [ones(1, count_NT) ones(1, count_T) * 2];
    
    %eeg(i_sub, :, :, :) = eeg_tmp;
    NT_eeg{i_sub} = nonTarget;
    T_eeg{i_sub} = target;
    labels{i_sub} = labels_tmp;
    
    testStartNT(i_sub) = testStartIndexNT;
    testStartT(i_sub) = testStartIndexT;
end



%% LDA 7 folds

n_fold = 7;


for i_sub = 1 : n_sub
    lda_accs = [];
    
    estimateds = [];
    true_labels = [];
    

    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
        valid_NT_ind = round((i_fold - 1) * NT_nTrial / n_fold + 1 : i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = round((i_fold - 1) * T_nTrial / n_fold + 1 : i_fold * T_nTrial / n_fold);
        
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        X = cat(3, fold_train_NT, fold_train_T);
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        
        W = LDA(X,Y, [5, 1], 0.9);
        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        fold_accs = [];


        % Test phase ---------------------------------------
        testdata = valids; 
        testdata = squeeze(reshape(testdata, 1, size(testdata, 1) * size(testdata, 2), size(testdata, 3))).';

        L = [ones(size(testdata, 1), 1) testdata] * W';
        P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);

        [~, estimated] = max(P, [], 2);

        % Evaluation ----------------------------------------
        isCorrect = (estimated==valid_labels);
        fold_accs = mean(isCorrect);
        
        estimateds = [estimateds; estimated];
        true_labels = [true_labels; valid_labels];

        lda_accs(i_fold) = fold_accs;
        
    end
    fprintf('7 fold regular LDA accuracy = %2.2f%%\n', mean(lda_accs)*100);
    
    [f1, C] = f1_score(estimateds, true_labels);
    fprintf('7 fold regular LDA f1 score = %2.2f%%\n', f1 * 100);
    %disp(C);
    
    lda_sub_accs(i_sub) = mean(lda_accs);
end


%% SF LDA 7 folds


n_fold = 7;


for i_sub = 1 : n_sub
    
    lda_accs = [];
    
    estimateds = [];
    true_labels = [];
    
    start = 1;
    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
        valid_NT_ind = round((i_fold - 1) * NT_nTrial / n_fold + 1 : i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = round((i_fold - 1) * T_nTrial / n_fold + 1 : i_fold * T_nTrial / n_fold);
        
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        X = cat(3, fold_train_NT, fold_train_T);
        [X, SF] = SpatialFilter(X, Y, 4);
        %[X, SF] = xSpatialFilter(X, Y, 204, 50, 2);
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';      
        
        W = LDA(X,Y, [5 1], 0.8);
        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        % Test phase ---------------------------------------
        testdata = valids;
        testdata = getSFData(testdata, SF);
        %testdata = getSFData(testdata(:, 1 : 204, :), SF);
        testdata = squeeze(reshape(testdata, 1, size(testdata, 1) * size(testdata, 2), size(testdata, 3))).';

        L = [ones(size(testdata, 1), 1) testdata] * W';
        P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);

        [~, estimated] = max(P, [], 2);

        % Evaluation ----------------------------------------
        isCorrect = (estimated==valid_labels);
        fold_accs = mean(isCorrect);

        estimateds = [estimateds; estimated];
        true_labels = [true_labels; valid_labels];
        
        lda_accs(i_fold) = fold_accs; 
    end
    fprintf('7 fold SF LDA accuracy = %2.2f%%\n', mean(lda_accs)*100);
    
    [f1, C] = f1_score(estimateds, true_labels);
    fprintf('7 fold SF LDA f1 score = %2.2f%%\n', f1 * 100);
    %disp(C);
    sf_lda_sub_accs(i_sub) = mean(lda_accs);
end

%% Resample 7 folds LDA

n_resamplePoints = 12;


train_trials = 1 : 4200;

n_targret = 2;

n_fold = 7;


for i_sub = 1 : n_sub
    lda_accs = [];
    
    estimateds = [];
    true_labels = [];
    
    start = 1;
    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
        valid_NT_ind = round((i_fold - 1) * NT_nTrial / n_fold + 1 : i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = round((i_fold - 1) * T_nTrial / n_fold + 1 : i_fold * T_nTrial / n_fold);
        
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        X = cat(3, fold_train_NT, fold_train_T);
        
        resample_ind = 0 : n_resamplePoints : size(X, 2);
        
        resampleX = zeros(size(X, 1), size(resample_ind, 2), size(X, 3));
        
        for i = 1 : size(resample_ind, 2) - 1
            resampleX(:, i, :) = mean(X(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
        end
        
        X = resampleX;
        
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        
        W = LDA(X,Y, [5, 1], 0.05);
        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        % Test phase ---------------------------------------
        testdata = valids;
        
        resampleTestData = zeros(size(testdata, 1), size(resample_ind, 2), size(testdata, 3));
        
        for i = 1 : size(resample_ind, 2) - 1
            resampleTestData(:, i, :) = mean(testdata(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
        end
        
        testdata = resampleTestData;
        
        testdata = squeeze(reshape(testdata, 1, size(testdata, 1) * size(testdata, 2), size(testdata, 3))).';

        L = [ones(size(testdata, 1), 1) testdata] * W';
        P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);

        [~, estimated] = max(P, [], 2);
        
        estimateds = [estimateds; estimated];
        true_labels = [true_labels; valid_labels];
        % Evaluation ----------------------------------------
        isCorrect = (estimated==valid_labels);
        fold_accs = mean(isCorrect);

        lda_accs(i_fold) = fold_accs; 
    end
    fprintf('7 fold Resample LDA accuracy = %2.2f%%\n', mean(lda_accs)*100);
    
    [f1, C] = f1_score(estimateds, true_labels);
    fprintf('7 fold Resample LDA f1 score = %2.2f%%\n', f1 * 100);
    %disp(C);
    
    resample_lda_sub_accs(i_sub) = mean(lda_accs);
end


%% SF & Resample LDA 7 folds

train_trials = 1 : 4200;

n_targret = 2;

n_fold = 7;


for i_sub = 1 : n_sub
    lda_accs = [];

    estimateds = [];
    true_labels = [];
    
    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
        valid_NT_ind = round((i_fold - 1) * NT_nTrial / n_fold + 1 : i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = round((i_fold - 1) * T_nTrial / n_fold + 1 : i_fold * T_nTrial / n_fold);
        
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        X = cat(3, fold_train_NT, fold_train_T);
        [X, SF] = SpatialFilter(X, Y, 2);
        
        resample_ind = 0 : n_resamplePoints : size(X, 2);

        resampleX = zeros(size(X, 1), size(resample_ind, 2), size(X, 3));

        for i = 1 : size(resample_ind, 2) - 1
            resampleX(:, i, :) = mean(X(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
        end

        X = resampleX;
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
        
        W = LDA(X,Y, [5 1], 0.01);
        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        % Test phase ---------------------------------------
        testdata = valids;
        testdata = getSFData(testdata, SF);
    
        resampleTestData = zeros(size(testdata, 1), size(resample_ind, 2), size(testdata, 3));

        for i = 1 : size(resample_ind, 2) - 1
            resampleTestData(:, i, :) = mean(testdata(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
        end

        testdata = resampleTestData;
        testdata = squeeze(reshape(testdata, 1, size(testdata, 1) * size(testdata, 2), size(testdata, 3))).';

        L = [ones(size(testdata, 1), 1) testdata] * W';
        P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);

        [~, estimated] = max(P, [], 2);
                
        estimateds = [estimateds; estimated];
        true_labels = [true_labels; valid_labels];

        % Evaluation ----------------------------------------
        isCorrect = (estimated==valid_labels);
        fold_accs = mean(isCorrect);

        lda_accs(i_fold) = fold_accs; 
    end
    fprintf('7 fold SF & Resample accuracy = %2.2f%%\n', mean(lda_accs)*100);
    
    [f1, C] = f1_score(estimateds, true_labels);
    fprintf('7 fold SF & Resample LDA f1 score = %2.2f%%\n', f1 * 100);
    %disp(C);
    sf_resample_lda_sub_accs(i_sub) = mean(lda_accs);
end

%% Resample & SF LDA 7 folds

n_fold = 7;

for i_sub = 1 : n_sub
    lda_accs = [];

    estimateds = [];
    true_labels = [];
    
    for i_fold = 1 : n_fold

        NT_nTrial = size(NT_eeg{i_sub}, 3);
        fold_NT = NT_eeg{i_sub};
        valid_NT_ind = round((i_fold - 1) * NT_nTrial / n_fold + 1 : i_fold * NT_nTrial / n_fold);
        
        fold_train_NT = fold_NT;
        fold_train_NT(:, :, valid_NT_ind) = [];
        fold_test_NT = fold_NT(:, :, valid_NT_ind);
        
        T_nTrial = size(T_eeg{i_sub}, 3);
        fold_T = T_eeg{i_sub};
        valid_T_ind = round((i_fold - 1) * T_nTrial / n_fold + 1 : i_fold * T_nTrial / n_fold);
        
        fold_train_T = fold_T;
        fold_train_T(:, :, valid_T_ind) = [];
        fold_test_T = fold_T(:, :, valid_T_ind);
        
        Y = [ones(size(fold_train_NT, 3), 1); ones(size(fold_train_T, 3), 1) * 2];
        X = cat(3, fold_train_NT, fold_train_T);
         
        resample_ind = 0 : n_resamplePoints : size(X, 2);

        resampleX = zeros(size(X, 1), size(resample_ind, 2), size(X, 3));

        for i = 1 : size(resample_ind, 2) - 1
            resampleX(:, i, :) = mean(X(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
        end

        X = resampleX;
        
       [X, SF] = SpatialFilter(X, Y, 2);
        X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
        
        W = LDA(X,Y, [5 1], 0.001);
        
        valids = cat(3, fold_test_NT, fold_test_T);
        valid_labels = [ones(size(fold_test_NT, 3), 1); ones(size(fold_test_T, 3), 1) * 2];
        nTrials = size(valids, 3);

        % Test phase ---------------------------------------
        testdata = valids;         
        resampleTestData = zeros(size(testdata, 1), size(resample_ind, 2), size(testdata, 3));

        for i = 1 : size(resample_ind, 2) - 1
            resampleTestData(:, i, :) = mean(testdata(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
        end

        testdata = resampleTestData;
        testdata = getSFData(testdata, SF);
        
        testdata = squeeze(reshape(testdata, 1, size(testdata, 1) * size(testdata, 2), size(testdata, 3))).';

        L = [ones(size(testdata, 1), 1) testdata] * W';
        P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);

        [~, estimated] = max(P, [], 2);
                
        estimateds = [estimateds; estimated];
        true_labels = [true_labels; valid_labels];

        % Evaluation ----------------------------------------
        isCorrect = (estimated==valid_labels);
        fold_accs = mean(isCorrect);

        lda_accs(i_fold) = fold_accs; 
    end
    fprintf('7 fold SF & Resample accuracy = %2.2f%%\n', mean(lda_accs)*100);
    
    [f1, C] = f1_score(estimateds, true_labels);
    fprintf('7 fold SF & Resample LDA f1 score = %2.2f%%\n', f1 * 100);
    %disp(C);
    resample_sf_lda_sub_accs(i_sub) =  mean(lda_accs);
end

%% feature evaluate

n_sub = 1;

for i_sub = 1 : n_sub
    
    % NT, T x Train, Test
    
    NT_eeg_tmp = NT_eeg{i_sub};
    trains_NT = NT_eeg_tmp(:, :, train_trials_NT);
    T_eeg_tmp = T_eeg{i_sub};
    trains_T = T_eeg_tmp(:, :, train_trials_T);
    
    trains = cat(3, trains_NT, trains_T);
    train_labels = [ones(1, size(trains_NT, 3)) ones(1, size(trains_T, 3)) * 2].';
    
    tests_NT = NT_eeg_tmp(:, :, test_trials_NT);
    tests_T = NT_eeg_tmp(:, :, test_trials_T);
    
    tests = cat(3, tests_NT, tests_T);
    test_labels = [ones(1, size(tests_NT, 3)) + 2 ones(1, size(tests_T, 3)) * 2 + 2].';
    
    X = cat(3, trains, tests);
    X = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
    Y = [train_labels; test_labels];
    %X = tests;
    %Y = test_labels;

    score = feature_evaluate_and_scatter(X, Y);
    fprintf('Raw (nonTarget, target, train, test): ');
    disp(score);
    
    %--------------------------------------------
    
    NT_eeg_tmp = NT_eeg{i_sub};
    trains_NT = NT_eeg_tmp;
    T_eeg_tmp = T_eeg{i_sub};
    trains_T = T_eeg_tmp;
    
    trains = cat(3, trains_NT, trains_T);
    train_labels = [ones(1, size(trains_NT, 3)) ones(1, size(trains_T, 3)) * 2];
    
    
    
    Y = train_labels.';
    X = trains;
    
    X_raw = squeeze(reshape(X, 1, size(X, 1) * size(X, 2), size(X, 3))).';
    
    score = feature_evaluate_and_scatter(X_raw, Y);
    fprintf('raw: ');
    disp(score);
    
    
    resample_ind = 0 : n_resamplePoints : size(X, 2);
    resampleX = zeros(size(X, 1), size(resample_ind, 2), size(X, 3));
    
    for i = 1 : size(resample_ind, 2) - 1
        resampleX(:, i, :) = mean(X(:, resample_ind(i) + 1 : resample_ind(i + 1), :), 2);
    end

    resampleX = squeeze(reshape(resampleX, 1, size(resampleX, 1) * size(resampleX, 2), size(resampleX, 3))).';
    score = feature_evaluate_and_scatter(resampleX, Y);
    fprintf('Resample: ');
    disp(score);
    
    [SF_X, SF] = SpatialFilter(X, Y, 4);
    SF_X_reshape = squeeze(reshape(SF_X, 1, size(SF_X, 1) * size(SF_X, 2), size(SF_X, 3))).';
    
    score = feature_evaluate_and_scatter(SF_X_reshape, Y);
    fprintf('SF: ');
    disp(score);
    
    figure()
    hold on
    plot(mean(squeeze(trains(plot_channel, :, labels{i_sub} == 1)), 2), 'blue');
    plot(mean(squeeze(trains(plot_channel, :, labels{i_sub} == 2)), 2), 'red');
    hold off

    
    [xSF_X, SF, shiftedData] = xSpatialFilter(X, Y, 186, 70, 1);
    xSF_X_reshape = squeeze(reshape(xSF_X, 1, size(xSF_X, 1) * size(xSF_X, 2), size(xSF_X, 3))).';
    

    score = feature_evaluate_and_scatter(xSF_X_reshape, Y);
    fprintf('xSF: ');
    disp(score);
    
    figure();
    hold on
    plot(mean(squeeze(xSF_X(1, :, labels{i_sub} == 1)), 2), 'blue');
    plot(mean(squeeze(xSF_X(1, :, labels{i_sub} == 2)), 2), 'red');
    hold off
    
end
