function filtered_eeg = filter_lineNoise(eeg, fs, f)
    wo = f/(fs/2);  bw = wo/35;
    [b,a] = iirnotch(wo,bw);
    
    filtered_eeg = zeros(size(eeg));
    
    s = size(eeg);

    if size(s, 2) == 4
        for i = 1 : size(eeg, 1)
            for j = 1 : size(eeg, 2)
                for k = 1 : size(eeg, 4)
                    tmp = filtfilt(b, a, eeg(i, j, :, k));
                    filtered_eeg(i, j, :, k) = tmp(1, :); 
                end
            end
        end
        
    elseif size(s, 2) == 2
       for i = 1 : size(eeg, 1)
            tmp = filtfilt(b, a, eeg(i, :));
            filtered_eeg(i, :) = tmp; 
       end
    end
end