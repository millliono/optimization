function [xmin,ymin] = grad_descent(xinit,yinit,method)
f = @(x,y) x^5 * exp(-x^2 - y^2);

dfdx = @(x,y) 5*x^4*exp(- x^2 - y^2) - 2*x^6*exp(- x^2 - y^2);

dfdy = @(x,y) -2*x^5*y*exp(- x^2 - y^2);
x = xinit;
y = yinit;

% Set stop condition threshhold
e = 0.1e-4;

if strcmp(method, "const")
    alpha = 0.0001;
    n_grad = norm([dfdx(x,y), dfdy(x,y)]);
    while n_grad > e
        grad = [dfdx(x,y), dfdy(x,y)];
        n_grad = norm(grad);
        x = x - alpha * grad(1);
        y = y - alpha * grad(2);
    end

elseif strcmp(method, "line_min")
    n_grad = norm([dfdx(x,y), dfdy(x,y)]);
    while n_grad > e
        grad = [dfdx(x,y), dfdy(x,y)];
        n_grad = norm(grad);
        alpha = linspace(0,10,1000);
        map = arrayfun(f, x - alpha * grad(1), y - alpha * grad(2));
        [~, idx] = min(map); 
        x = x - alpha(idx) * grad(1);
        y = y - alpha(idx) * grad(2);
    end

elseif strcmp(method, "armijo")
 
end

xmin = x;
ymin = y;
