function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = X * theta;

J = (1/(m*2)) * sumsq(h-y) + (lambda/(2*m))*sumsq(theta(2:end));

grad = ((1/m) * X' * (h-y)) + ((lambda/m) * [0;theta(2:end)]);

grad = grad(:);
end
