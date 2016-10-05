function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta1: 25x401; Theta2: 10x26

%Feedforward Propagation:
X = [ones(size(X,1),1) X]; % X: 5000x401 
a1 = X'; % a1: 401x5000

z2 = Theta1*a1; % z2: 25x5000
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2]; % a2: 26x5000

z3 = Theta2*a2; % z3: 10x5000
hTheta = sigmoid(z3);
a3 = hTheta; % a3: 10x5000

%Making the yk matrices of dimensions 10x1 for each example => 10x5000
yk = zeros(num_labels,m);
for i=1:m
    yk(y(i),i) = 1;
end

%Cost function (Not reguralised)
J = ( -yk.*log(hTheta) - (ones(size(yk))-yk).*log(ones(size(hTheta))-hTheta) );
J = sum(J(:))/m;

%Regularised Cost function
sum1 = 0; sum2 = 0;

for i=1:hidden_layer_size
    for j=2:(input_layer_size+1)
        sum1 = sum1 + Theta1(i,j)^2;
    end
end

for i=1:num_labels
    for j=2:(hidden_layer_size+1)
        sum2 = sum2 + Theta2(i,j)^2;
    end
end

J = J + (lambda/(2*m))*(sum1 + sum2);


%BackPropagation
d3 = a3 - yk; % d3: 10x5000
d2 = (Theta2(:,2:end))'*d3.*sigmoidGradient(z2); % d2: 25x5000

Theta1_grad = (Theta1_grad + d2*a1')/m; % Theta1_grad: 25x401
Theta2_grad = (Theta2_grad + d3*a2')/m; % Theta2_grad: 10x26

%Regularised:
regM = ones(size(Theta1)); regM(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda/m)*regM.*Theta1;

regM = ones(size(Theta2)); regM(:,1) = 0;
Theta2_grad = Theta2_grad + (lambda/m)*regM.*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
