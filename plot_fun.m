f = @(x,y) x.^5 .* exp(-x.^2 - y.^2);

x = -5:0.1:5;
y = -5:0.1:5;

[X,Y] = meshgrid(x,y);
Z = f(X,Y);

% Plot the function
mesh(X,Y,Z);