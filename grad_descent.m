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
while dfdx(x,y) > e && dfdy(x,y) > e
    x = x - alpha * dfdx(x,y);
    y = y - alpha * dfdy(x,y);
end

xmin = x;
ymin = y;
