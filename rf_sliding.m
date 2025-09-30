 % ---------------- CONFIG ----------------
dataFile = "31jan25training.xlsx";   % <-- change to your filename
outFile  = "rf_param_refined_multi_results.xlsx";

% ---------------- Load + Clean ----------------
% Read Excel with original headers preserved
data = readtable(dataFile, 'VariableNamingRule', 'preserve');

% Show variable names (for debugging)
disp("Available columns:")
disp(data.Properties.VariableNames)

% Detect Energy column (works even if renamed by MATLAB)
energyCols = data.Properties.VariableNames( ...
    contains(data.Properties.VariableNames, "Energy", 'IgnoreCase', true));
if isempty(energyCols)
    error("No column containing 'Energy' found in dataset!");
end
targetCol = energyCols{1};   % pick the first matching column
fprintf("Using target column: %s\n", targetCol);

% Ensure Date column is datetime
if any(strcmpi(data.Properties.VariableNames, "Date"))
    data.Date = datetime(data.Date, 'InputFormat','yyyy-MM-dd HH:mm:ss');
end

% Convert all non-Date columns to numeric if possible
for c = 1:width(data)
    if ~strcmpi(data.Properties.VariableNames{c}, "Date")
        if iscell(data{:,c})
            data{:,c} = strrep(data{:,c}, ',', '');
            data{:,c} = str2double(data{:,c});
        end
    end
end

% Drop rows with missing target
data = data(~isnan(data.(targetCol)), :);

% ---------------- Parameters ----------------
nEstimators = 500;     % RF trees
trainWindow = 21;      % sliding window days

% ---------------- Sliding Window Prediction ----------------
yTrue = [];
yPred = [];

for i = trainWindow+1:height(data)
    trainIdx = i-trainWindow:i-1;
    testIdx  = i;

    % Features: all except Date + target
    Xtrain = data{trainIdx, setdiff(data.Properties.VariableNames, ["Date", targetCol])};
    ytrain = data{trainIdx, targetCol};

    Xtest  = data{testIdx, setdiff(data.Properties.VariableNames, ["Date", targetCol])};
    ytest  = data{testIdx, targetCol};

    % Train Random Forest
    model = TreeBagger(nEstimators, Xtrain, ytrain, 'Method', 'regression');

    % Predict
    yhat = predict(model, Xtest);

    yTrue(end+1) = ytest;
    yPred(end+1) = yhat;
end

% ---------------- Evaluation ----------------
mae = mean(abs(yTrue - yPred));
fprintf("Final MAE: %.4f\n", mae);

% Save results
results = table((trainWindow+1:height(data))', yTrue', yPred', ...
    'VariableNames', {'Index','Actual','Predicted'});
writetable(results, outFile);
disp("Results saved to " + outFile)
