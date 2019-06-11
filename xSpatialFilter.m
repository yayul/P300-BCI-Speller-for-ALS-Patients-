function [SFData, spatialFilters, shiftedData, mean_offset] = xSpatialFilter(data, label, length, searchRange, n)

    if ~exist('n', 'var') || isempty(n)
    n = 1; end

    spatialFilters = zeros(size(data, 1), 2 * n);

    data1 = data(:, :, label == 1);
    cov1_set = zeros(size(data, 3), size(data1, 1), size(data1, 1), searchRange);   

    data2 = data(:, :, label == 2);
    cov2_set = zeros(size(data, 3), size(data2, 1), size(data2, 1), searchRange);
    
    for t = 1 : searchRange
        for i = 1 : size(data1, 3)
            cov1_set(i, :, :, t) = cov(squeeze(data1(:, t : t + length - 1, i)).');
        end
        
        for i = 1 : size(data2, 3)
            cov2_set(i, :, :, t) = cov(squeeze(data2(:, t : t + length - 1, i)).');
        end
    end
    
    origin_cov1 = zeros(size(data1, 1), size(data1, 1));
    start_cov1 = zeros(size(data1, 1), size(data1, 1));
    
    for i = 1 : size(data1, 3)    
        origin_cov1 = origin_cov1 + cov(squeeze(data1(:, :, i)).');
        start_cov1 = start_cov1 + cov(squeeze(data1(:, 1 : length, i)).');
    end
    
    origin_cov2 = zeros(size(data2, 1), size(data2, 1));
    start_cov2 = zeros(size(data2, 1), size(data2, 1));
    
    for i = 1 : size(data2, 3)
        origin_cov2 = origin_cov2 + cov(squeeze(data2(:, :, i)).');
        start_cov2 = start_cov2 + cov(squeeze(data2(:, 1 : length, i)).');
    end
    
    t1 = zeros(size(data1, 3), 1);
    
    for i = 1 : size(data1, 3)
        maxT = 0;
        maxVal = -1;
        
        current_cov1 = start_cov1;
        current_cov1 = current_cov1 - squeeze(cov1_set(i, :, :, 1));
        
        for t = 1 : searchRange
            tmp_cov1 = current_cov1 + squeeze(cov1_set(i, :, :, t));
            
            [~, eigVal] = eig(origin_cov2 \ tmp_cov1);
            eigVal = diag(eigVal);
            v_ratio = eigVal(1) / sum(eigVal(2:end));
            
            if v_ratio > maxVal
                maxVal = v_ratio;
                maxT = t;
            end
        end
        
        t1(i) = maxT;
        current_cov1 = current_cov1 + squeeze(cov1_set(i, :, :, maxT));
    end

    t2 = zeros(size(data2, 3), 1);
    
    for i = 1 : size(data2, 3)
        maxT = 0;
        maxVal = -1;
        
        current_cov2 = start_cov2;
        current_cov2 = current_cov2 - squeeze(cov2_set(i, :, :, 1));
        
        for t = 1 : searchRange
            tmp_cov2 = current_cov2 + squeeze(cov2_set(i, :, :, t));
            
            [~, eigVal] = eig(origin_cov1 \ tmp_cov2);
            eigVal = diag(eigVal);
            v_ratio = eigVal(1) / sum(eigVal(2:end));
            
            if v_ratio > maxVal
                maxVal = v_ratio;
                maxT = t;
            end
        end
        
        t2(i) = maxT;
        current_cov2 = current_cov2 + squeeze(cov2_set(i, :, :, maxT));
    end
    
    [eigVec,eigVal] = eigs(current_cov2 \ current_cov1);
    [eigVal,I] = sort(diag(eigVal));
    eigVec = eigVec(:, I);
    
    for i = 1 : n
        spatialFilters(:, i) = eigVec(:, i);
        spatialFilters(:, n + i) = eigVec(:, end - i + 1);
    end
    
    shiftedData = zeros(size(data, 1), length, size(data, 3));
    
    ind1 = find(label == 1);
    for i = 1 : size(ind1, 1)
        shiftedData(:, :, ind1(i)) = data(:, t1(i) : t1(i) + length - 1, ind1(i));
    end
    
    ind2 = find(label == 2);
    for i = 1 : size(ind2, 1)
        shiftedData(:, :, ind2(i)) = data(:, t2(i) : t2(i) + length - 1, ind2(i));
    end
    
    SFData = getSFData(shiftedData, spatialFilters);
    
    mean_offset = mean([t1; t2]);
end