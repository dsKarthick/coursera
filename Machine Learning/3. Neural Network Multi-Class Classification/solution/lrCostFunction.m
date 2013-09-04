function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
sig = sigmoid(X*theta);

limit = size(X)(:,2);

reg = sum(theta'(:,2:limit) .* theta'(:,2:limit));

J = 1/(2 * m) .* sum((-1) .* y .* log(sig) - (1-y) .* log(1-sig)) + (lambda/(2*m)) .* reg;
grad = ((1/m) .* (sig-y)' * X) .+ ([0 (lambda/m).*theta'(:,2:end)]) ;

grad = grad(:);
end
