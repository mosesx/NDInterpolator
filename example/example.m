close all; clear all;
% NDInterpolator example matlab script. The script produces input for and
% uses the output of example.cpp.
%
% \author Marcel Rehberg
% \date 12.11.2010

% interpolation grid,
[X,Y]=meshgrid(linspace(-1.5,2,10),linspace(-.5,3,10));

% evaluation grid
[XX,YY]=meshgrid(linspace(-1.4,1.9,50),linspace(-.4,2.9,50));

% write grid files, remember NDInterpolator wants a matrix where
% each column is one point
ublasMatrixIO([X(:)';Y(:)'],'grid.dat');
ublasMatrixIO([XX(:)';YY(:)'],'eval.dat');

% evaluate rosenbrock and write data files
[F,FX,FY,FXX,FYY,FXY]=rosenbrock(X,Y);
[FF,FFX,FFY,FFXX,FFYY,FFXY]=rosenbrock(XX,YY);

ublasMatrixIO(F(:),'data.dat');
ublasMatrixIO([FX(:)';FY(:)'],'diffData.dat');

disp('Please execute a.out')
pause;

% load the output
FI=load('out.dat');
FIH=load('outH.dat');

FXI=load('outDiff.dat');
FXIH=load('outDiffH.dat');

FYYI=load('outDiff2.dat');
FYYIH=load('outDiff2H.dat');

FXYI=load('outDiffM.dat');
FXYIH=load('outDiffMH.dat');

figure(1)
subplot(2,2,1); surf(X,Y,F); title('original');
subplot(2,2,2); surf(XX,YY,reshape(FI,50,50)); title('Lagrange');
subplot(2,2,3); surf(XX,YY,reshape(FIH,50,50)); title('Hermite');
subplot(2,2,4); surf(XX,YY,abs(FF-reshape(FIH,50,50))./abs(FF)); title('rel error Hermite');

figure(2)
subplot(2,2,1); surf(X,Y,FX); title('FX original');
subplot(2,2,2); surf(XX,YY,reshape(FXI,50,50)); title('FX Lagrange');
subplot(2,2,3); surf(XX,YY,reshape(FXIH,50,50)); title('FX Hermite');
subplot(2,2,4); surf(XX,YY,abs(FFX-reshape(FXIH,50,50))./abs(FFX)); ...
    title('rel error Hermite');

figure(3)
subplot(2,2,1); surf(X,Y,FYY); title('FYY original');
subplot(2,2,2); surf(XX,YY,reshape(FYYI,50,50)); title('FYY Lagrange');
subplot(2,2,3); surf(XX,YY,reshape(FYYIH,50,50)); title('FYY Hermite');
subplot(2,2,4); surf(XX,YY,abs(FFYY-reshape(FYYIH,50,50))./abs(FFYY)); ...
    title('rel error Hermite');

figure(4)
subplot(2,2,1); surf(X,Y,FXY); title('FXY original');
subplot(2,2,2); surf(XX,YY,reshape(FXYI,50,50)); title('FXY Lagrange');
subplot(2,2,3); surf(XX,YY,reshape(FXYIH,50,50)); title('FXY Hermite');
subplot(2,2,4); surf(XX,YY,abs(FFXY-reshape(FXYIH,50,50))./abs(FFXY)); ...
    title('rel error Hermite');
