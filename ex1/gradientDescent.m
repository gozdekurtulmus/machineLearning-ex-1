function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

    

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    
    
for iter = 1:num_iters
  root0 = 0;
  root1 = 0;
  hx = X*theta;
  for i=1:m
    root0 = root0 + (hx(i)- y(i))*X(i,1) ;
    root1 = root1 + (hx(i)- y(i))*X(i,2) ;
  endfor
  theta(1) = theta(1)- root0*alpha/m;
  theta(2) = theta(2)- root1*alpha/m;  
  
endfor

    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta)(1);



end
