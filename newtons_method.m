function [xmin,ymin] = newtons_method(xinit,yinit)
%f = @(x,y) x^5 * exp(-x^2 - y^2);

dfdx = @(x,y) 5*x^4*exp(- x^2 - y^2) - 2*x^6*exp(- x^2 - y^2);

dfdy = @(x,y) -2*x^5*y*exp(- x^2 - y^2);

d2fdx2 = @(x,y) 20*x^3*exp(- x^2 - y^2) - 22*x^5*exp(- x^2 - y^2) + 4*x^7*exp(- x^2 - y^2);

d2fdy2 = @(x,y) 4*x^5*y^2*exp(- x^2 - y^2) - 2*x^5*exp(- x^2 - y^2);

d2fdxdy = @(x,y) 4*x^6*y*exp(- x^2 - y^2) - 10*x^4*y*exp(- x^2 - y^2);

% Set the initial guess for x and y
x = xinit;
y = yinit;


% Set the learning rate
alpha = 0.01;

% Set stop condition threshhold
e = 0.1e-4;

% Perform Newton's method
while dfdx(x,y) > e && dfdy(x,y) > e
    grad = [dfdx(x,y); dfdy(x,y)];
    hess = [d2fdx2(x,y), d2fdxdy(x,y); d2fdxdy(x,y), d2fdy2(x,y)];

    delta = hess \ grad;
    x = x - alpha * delta(1);
    y = y - alpha * delta(2);
end

xmin = x;
ymin = y;
