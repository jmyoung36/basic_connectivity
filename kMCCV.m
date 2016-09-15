function folds = kMCCV(n_subjects, n_repeats, test_frac)

for i = 1:n_repeats
    
    perm = randperm(n_subjects);
    n_test = round(test_frac * n_subjects);
    folds(i).training_indices = perm(n_test + 1:end);
    folds(i).testing_indices = perm(1:n_test);    
    
end

end
