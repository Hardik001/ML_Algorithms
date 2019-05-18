function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);



    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    for iter = 1:num_iters
	hypothesis=X*theta;
	error=hypothesis-y;
	errorsum=sum(error,1);
	temp=zeros(length(theta),1);
	temp(1)=theta(1)-(alpha*errorsum)/m;
	for i=2:length(theta)
		errorsumx=error.*X(:,i);
		errorsumx=sum(errorsumx,1);
		temp(i)=theta(i)-(alpha*errorsumx)/m;
	end
    %
	











    % ============================================================

    % Save the cost J in every iteration  
    theta  =temp;
    J_history(iter) = computeCostMulti(X, y, theta);

end


end
