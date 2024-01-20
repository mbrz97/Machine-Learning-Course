clc
clear

% Taking any random data
X=randn(100,2);

% Adding noise to equation y = 3x1 + 7x2 + 4
Y=3*(X(:,1)+0.3*randn(100,1))+7*(X(:,2)+0.3*randn(100,1))+(4+0.3*randn(100,1));

% If data is not normalized, Zscore Normalization (Advised for SGD)
data=[zscore(X),Y];
data=[X,Y];

% 70% to training set + 30% to testing set 
train_set = data(1:70 ,:);
test_set = data(71:100, :);

% Define a symbolic varible for function plot
syms x1 x2

X_train=train_set(:,1:end-1); Y_train=train_set(:,end);
X_test=test_set(:,1:end-1); Y_test=test_set(:,end);

% Number of training instances
N=length(X_train)

% Number of testing instances
M=length(X_test)

% Learning Parameter, alpha = 0.1
% Append a vectors of one to X_train for calculating bias.
% Tolerence = 10^-5
X_train=[ones(N,1), X_train];

W=zeros(size(X_train,2),1);
W_old=ones(size(X_train,2),1);

while(norm(W_old-W) > 10^-5)
    W_old=W;
    W = W - 0.1/N*X_train'*(X_train*W - Y_train);
end
W

%Mean Square Error
predicted_values=[ones(length(X_test),1),X_test]*W;
mse3=sqrt(mean((predicted_values-Y_test).^2))

%Plot
figure
hold on
scatter3(X_test(:,1),X_test(:,2),Y_test)
fsurf(W(1)+W(2)*x1 +W(3)*x2)
xlabel('X_1')
ylabel('X_2')
zlabel('Y')
title('Regression Using Stochastic Gradient Descent')
view([-104.8 -5.2])
hold off