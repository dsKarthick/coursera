function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

m = size(X, 1);
X = [ones(m,1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];

z3 = a2*Theta2';
a3 = sigmoid(a2*Theta2');

lookup_matrix = eye(num_labels);
Y = lookup_matrix(y,:);

reg = (lambda/(2*m)) * ((sum((Theta2(:,[2:end]) .^ 2)(:))) + (sum((Theta1(:,[2:end]) .^ 2)(:))));
J = (-1/m) * sum(sum((Y .* log(a3) + (1-Y) .* log(1 - a3)))(:)) + reg; 

% For each sample input, calculate error at 
%    d3 - ouput units
%    d2 - hidden units

error3 = a3 - Y;
error2 = (error3 * Theta2)(:,2:end) .* sigmoidGradient(z2);  

% We will use d3 and d2 to calculate the Delta 
Delta1 = (error2' * X)/m;
Delta2 = (error3' * a2)/m;

Theta1_grad = Delta1 + (lambda/m * [zeros(size(a2,2)-1,1) Theta1(:,2:end)]);
Theta2_grad = Delta2 + (lambda/m * [zeros(size(a3,2),1) Theta2(:,2:end)]);

 
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
