data=load("traindata.txt");
Y_data=data(:,9);
X_data=[ data(:,1:8) ones(926,1)];
folds=10;
mse_list=[];
theta_list =cell(1,10);
for i=1:10
   final_data=[polyFeatures(X_data(:,1),i) polyFeatures(X_data(:,2),i) polyFeatures(X_data(:,3),i) polyFeatures(X_data(:,4),i) polyFeatures(X_data(:,5),i) polyFeatures(X_data(:,6),i) polyFeatures(X_data(:,7),i) polyFeatures(X_data(:,8),i) X_data(:,9) Y_data];
   data_split=mat2cell(final_data,[93,93,93,93,93,93,92,92,92,92]);
   mse_sum=0;
   theta_sum=0;
    for j=1:10
        split_copy = data_split;
        validation_data=split_copy{i};
        split_copy(i) = [];
        training_data=cell2mat(split_copy);
        [m,n] = size(training_data);
        theta = trainLinearReg(training_data(:,1:n-1), training_data(:,n), i);
        theta_sum=theta_sum+theta;
        J = linearRegCostFunction(validation_data(:,1:n-1), validation_data(:,n),theta, i);
        mse_sum=mse_sum+J;
        
    end
    mse_avg=mse_sum/10;
    mse_list(i)=mse_avg;
    theta_avg=theta_sum./10;
    theta_list{1,i}=theta_avg;
    
    
end
[minimum_mse,i]=min(mse_list);
disp(theta_list{1,3});
X_data=load("testinputs.txt");
final_test_data=[polyFeatures(X_data(:,1),i) polyFeatures(X_data(:,2),i) polyFeatures(X_data(:,3),i) polyFeatures(X_data(:,4),i) polyFeatures(X_data(:,5),i) polyFeatures(X_data(:,6),i) polyFeatures(X_data(:,7),i) polyFeatures(X_data(:,8),i)  ones(103,1)];
predictions=(final_test_data*theta_list{1,i});
disp(predictions);
final_test_data_output=[X_data predictions];
fileID = fopen('test_output.txt','w');

fprintf(fileID,'\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n',final_test_data_output');

fclose(fileID);

