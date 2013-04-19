%Original Rosenbrock function
resOrig = load('res.dat')

%the Interpolation w.r.t. radial basis function

res = load('resIGeneralMultiquadric.dat')
figure(1)
surf(reshape(res,70,70))
figure(2)
surf(reshape(res-resOrig,70,70))

%Inverse Multiquadric
res1 = load('resIInverseMultiquadric.dat')
figure(3)
surf(reshape(res1,70,70))
figure(4)
surf(reshape(res1-resOrig,70,70))

%Gaussian 
res2 = load('resIGaussian.dat')
figure(5)
surf(reshape(res2,70,70))
figure(6)
surf(reshape(res2-resOrig,70,70))


