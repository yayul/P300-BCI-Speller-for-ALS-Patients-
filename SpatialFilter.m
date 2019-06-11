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