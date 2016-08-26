function folds = kFold(n_subjects, n_folds)

step_size = ceil(n_subjects/n_folds);

for i = 1:n_folds
    
    start_index = ((i - 1) * step_size) + 1;
    stop_index = min((start_index + step_size - 1), n_subjects);
    fold_training_indices = 1:n_subjects;
    fold_training_indices(start_index:stop_index) = [];
    fold_testing_indices = start_index:stop_index;
    folds(i).training_indices = fold_training_indices;
    folds(i).testing_indices = fold_testing_indices;
    
    
end


end
