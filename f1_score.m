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