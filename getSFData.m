function SFData = getSFData(data, spatialFilters)
    SFData = zeros(size(spatialFilters, 2), size(data, 2), size(data, 3));
    
    for i = 1 : size(data,3)
        SFData(:, :, i) = spatialFilters.' * squeeze(data(:, :, i));
    end
end
