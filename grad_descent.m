function [xmin,ymin] = grad_descent(xinit,yinit)
%f = @(x,y) x^5 * exp(-x^2 - y^2);

dfdx = @(x,y) 5*x^4*exp(- x^2 - y^2) - 2*x^6*exp(- x^2 - y^2);

dfdy = @(x,y) -2*x^5*y*exp(- x^2 - y^2);
x = xinit;
y = yinit;

% Set the learning rate
alpha = 0.0001;

% Set stop condition threshhold
e = 0.1e-4;

% Perform gradient descent
n_grad = norm([dfdx(x,y), dfdy(x,y)]);
while n_grad > e
    grad = [dfdx(x,y), dfdy(x,y)];
    n_grad = norm(grad);
    x = x - alpha * grad(1);
    y = y - alpha * grad(2);
end

xmin = x;
ymin = y;
