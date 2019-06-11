figure;
for i_sub = 1:8

    fileName = sprintf('A%02d.mat', i_sub);
    load(fileName);
    
    sample = data.X;
    StimType = data.y;
    StimClass = data.y_stim;
    trial = data.trial;

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

    % NTarget = find(StimType==1);
    % Target = find(StimType==2);

    start_NT = [];
    start_T = [];
    for i = 1:length(StimType)-1
        if StimType(i) ~= StimType(i+1)
            if StimType(i+1) == 1
                start_NT = [start_NT; i+1];
            elseif StimType(i+1) == 2
                start_T = [start_T; i+1];
            end
        end
    end

    % cz mean
    nonTarget = [];
    count_NT = 0;
    for i = 1:length(start_NT)
        if max(filtSample(start_NT(i):start_NT(i)+255,2))<70 &&...
                min(filtSample(start_NT(i):start_NT(i)+255,2))>-70
            nonTarget = [nonTarget ...
                (filtSample(start_NT(i):start_NT(i)+255,2)-mean(filtSample(start_NT(i):start_NT(i)+50,2)))];
            count_NT = count_NT + 1;
        end
    end

    mean_NT = mean(nonTarget,2);

    target = [];
    count_T = 0;
    for i = 1:length(start_T)
        if max(filtSample(start_T(i):start_T(i)+255,2))<70 &&...
                min(filtSample(start_T(i):start_T(i)+255,2))>-70
            target = [target ...
                (filtSample(start_T(i):start_T(i)+255,2)-mean(filtSample(start_T(i):start_T(i)+50,2)))];
            count_T = count_T + 1;
        end
    end

    mean_T = mean(target,2);

    subplot(4,2,i_sub);
    plot(0:255,mean_T,0:255,mean_NT);
    xlabel("ms"); ylabel("\muv");
    xticks(0:25.5:255); xticklabels(0:100:1000);
    xlim([0 255]); ylim([-5 5]);
    if i_sub==2
        legend('Target','Non-target');
    end
    
    title(['Subject ' num2str(i_sub)]);
end


