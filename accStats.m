function [acc, sens, spec] = accStats( labels, preds )

acc = sum(labels == preds)/length(labels);
sens = sum(labels(labels == 1) == preds(labels == 1))/length(labels(labels == 1));
spec = sum(labels(labels == -1) == preds(labels == -1))/length(labels(labels == -1));


end

