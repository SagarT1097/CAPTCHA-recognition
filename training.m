% Assumes varables:
% train_data, validation_data - numerical 3D arrays of size [N x r x c] [1000 x 28 x 28] [10 x 325 x 435]
% train_labels, validation_labels - categorical vectors size N [10 x 3]
% Uses the user defined function 'FeatureExtraction' to compute features for each sample

data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,(2:4));

num_train = 1000;
num_validation = 200;
train_labels = {};
validation_labels = {};

train_patterns = [];
validation_patterns = [];

t = tic;
fprintf('Extracting Training Features...\n');
for i=1:num_train
    k = img_nrs(i);
    a = FeatureExtraction(imread(sprintf('Train/captcha_%04d.png', k)));
    if size(a) == 0
        i
    else
        for j=1:3
            train_patterns(end+1,:) = a(j,:,:);
            train_labels{end+1} = num2str(true_labels(i,j));
        end
    end
end
toc(t)

t = tic;
fprintf('Extracting Validation Features...\n');
for i=num_train+1:num_train+num_validation
    k = img_nrs(i);
    a = FeatureExtraction(imread(sprintf('Train/captcha_%04d.png', k)));
    if size(a) == 0
        i
    else
        for j=1:3
            validation_patterns(end+1,:) = a(j,:,:);
            validation_labels{end+1} = num2str(true_labels(i,j));
        end
    end
end
toc(t)

validation_labels = transpose(validation_labels);
train_labels = transpose(train_labels);

fprintf('Building model...\n');

% ADA BOOST 
tr = templateTree('MaxNumSplits',75);
Mdl = fitcensemble(double(train_patterns),train_labels, 'Learners',tr); 


save Mdl

fprintf('\nResubstitution error: %5.2f%%\n\n',100*resubLoss(Mdl));

if isa(Mdl,'classreg.learning.classif.ClassificationEnsemble')
	view(Mdl.Trained{1},'Mode','graph');
end

fprintf('Predicting validation set...\n');
t=tic;
validation_pred = predict(Mdl,validation_patterns);
training_pred=predict(Mdl,train_patterns);
toc(t);

accuracy = mean(cell2mat(validation_pred) == cell2mat(validation_labels));
fprintf('Validation accuracy: %5.2f%%\n',accuracy*100);

accuracy_Training = mean(cell2mat(training_pred) == cell2mat(train_labels));
fprintf('Training accuracy: %5.2f%%\n',accuracy_Training*100);

f=figure(2);
if (f.Position(3)<800)
	set(f,'Position',get(f,'Position').*[1,1,1.5,1.5]); %Enlarge figure
end
confusionchart(validation_labels, validation_pred, 'ColumnSummary','column-normalized', 'RowSummary','row-normalized');
title(sprintf('Validation accuracy: %5.2f%%\n',accuracy*100));
