function [xmin,ymin] = levmarq(xinit,yinit)
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

% Set the number of iterations
n = 10000;

% Perform Levenberg-Marquardt optimization
for i = 1:n
    grad = [dfdx(x,y); dfdy(x,y)];
    hess = [d2fdx2(x,y), d2fdxdy(x,y); d2fdxdy(x,y), d2fdy2(x,y)];

    % Update x and y using the gradient and Hessian
    delta = (hess + alpha * eye(2)) \ grad;
    x = x - delta(1);
    y = y - delta(2);
end

xmin = x;
ymin = y;
