%Rosenbrock-Funktion

function [F,FX,FY,FXX,FYY,FXY] = rosenbrock(X,Y)


F = 100*(Y-X.^2).^2 + (1-X).^2;
FX = -2*(1-X)-400*X.*(-X.^2+Y);
FY = 200*(Y-X.^2);
FXX= 2 + 800*X.^2 - 400*(-X.^2 + Y);
FYY= 200*ones(size(F));
FXY= -400*X;

return