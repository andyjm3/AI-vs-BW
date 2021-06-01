addpath(pwd);

cd tests;
addpath(genpath(pwd));
cd ..;

cd thirdpartytools;
addpath(genpath(pwd));
cd ..;

% Experiments
% weighted least squares
test_linear_loss()
  
% Lyapunov equation
%test_lyapunov_loss()
  
% trace regression
%test_trace_regression()
