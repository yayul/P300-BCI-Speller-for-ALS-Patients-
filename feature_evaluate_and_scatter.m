function [score] = feature_evaluate_and_scatter(data, labels)

    data_2d = tsne(data);
    
    unique_labels = unique(labels);
    
    figure();
    hold on;
    for i = 1 : size(unique_labels, 1)

        label = unique_labels(i);
        scatter(data_2d(labels == label, 1), data_2d(labels == label, 2), 'filled');
    end
    hold off;
    
    legend();
    
    s = silhouette(data, labels);
    
    score = mean(s);
end