% Using CasADi to solve the classical Lokta Volterra Fishery problem
% The format of this code is modified from official examples from
% CasADi.org for better stratified clarity.
% Tong Zhao, Jan 2021

% Add Path for CasADi
addpath('E:\MATLAB_PLAYGROUND\CasADi\casadi-matlab')


clear all;close all;clc
import casadi.*

% Words for beginners: CasADi is a symbolic tool for optimization problems.
% OCP problems could be first expressed symblically, including model
% variables, model dynamic equations, objective function, initial
% conditions, constraints on state and control, etc. Then the symbolified
% OCP problem is converted into an NLP problem that solvers can deal with.
% OCP is a control problem, NLP is a math problem.

% The construction of f is for the construction of F(step update function
% of an RK4 step). f is the continuous form of dynamic, F is the discrete
% form dynamic in RK4 format.

% The discrete state dynamics is implemented by equality constraint g.

% Lotka-Volerra problem from Sundstrom (2009)
% x         - fish population
% u         - fishing rate per day
% w         - generalized optimization variable, includes {x,u}
% L         - cost function in one step form
% Q         - cost function in next step form
% f         - symbolic SX function (x0,p)->(xf,qf) for continuous dynamic
%   x0    	- symbol for start state on r.h.s
%   p     	- symbol for control on r.h.s
%   xf    	- symbol for end state on l.h.s
%   qf     	- symbol for objective on l.h.s
% F         - symbolic MX function (X0,U)->(X,Q) for each RK4-step
%   X0     	- symbol for start state on r.h.s
%   U      	- symbol for control on r.h.s
%   X      	- symbol for end state on l.h.s
%   Q     	- symbol for objective on r.h.s
% g         - (system) symbol for equality constraint
% x_start   - initial guess for state x distribution
% u_start   - initial guess for control u distribution
% w0        - initial gues for generalized opt variable w
% J         - cost function in all steps form

% This example applied the technique of "Lifting" initial conditions

%% OCP SYMBOLIC PROBLEM DEFINITION

T = 200; % Time horizon
N = 800; % number of control intervals

% Model parameters

% Declare model variables
x = SX.sym('x');
u = SX.sym('u');

% Model dynamic equations
xdot = [0.02*(1000*x-x*x)/1000 - u];

% Objective term: maximizing control (fishing yield)
L = -u;

% Initial condition for x
x0 = 250;

% Bounds on x
lbx = [0];
ubx = [1000];
    % Final state bounds on x
    lbx_fnl = 750;
    ubx_fnl = 1000;

% Bounds on u
lbu = 0;
ubu = 100;


%% PROBLEM PROCESSING TECHNIQUES: OCP TO NLP

% Formulate discrete time dynamics F
   % Fixed step Runge-Kutta 4 integrator
   M = 1; % RK4 steps per interval
   DT = T/N/M;  % length (in second) of an RK4 step
   % lower-case function for local use only, without names for i/o
   f = Function('f', {x, u}, {xdot, L});
   X0 = MX.sym('X0');
   U = MX.sym('U');
   X = X0;
   Q = 0;
   for j=1:M
       [k1, k1_q] = f(X, U);
       [k2, k2_q] = f(X + DT/2 * k1, U);
       [k3, k3_q] = f(X + DT/2 * k2, U);
       [k4, k4_q] = f(X + DT * k3, U);
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
   end
   % upper-case function for solver use, with names for i/o, T/N (sec)
   % time-span per step. Additionally the symbols for i/o are specified.
   F = Function('F', {X0, U}, {X, Q},char('x0','p'),char('xf','qf'));


% Initial guess for u
u_start = zeros(1,N);

% Get a feasible state trajectory as an initial guess, xk is a dummy var
xk = x0;
x_start = [xk];
for k=1:N
    ret = F('x0',xk, 'p',u_start(k));   % propagate one step
    xk = ret.xf;                        % obtain next state
    x_start = [x_start xk];             % concaternate states
end

% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
discrete = [];
J = 0;
g={};
lbg = [];
ubg = [];

% "Lift" initial conditions
w = {w{:} X0};      % initialize w with one entry (X0)
lbw = [lbw; x0];    % initial lower-bound for x
ubw = [ubw; x0];    % initial upper-bound for x
w0 = [w0; x_start(1)];  % initial guess of w
discrete = [discrete; 0];

% Formulate the NLP (in concaternating format) 
% States are "lifted" into decision variable w
% w includes: states X_0, X_1, ... X_N
%             controls U_0, U_1, ... U_(N-1)
% lbw/ubw includes: bounds for w in each step, could be used for uncoupled
%                   state constraint or control constraint
% g includes: coupled expressions of X and U
%             used for implementing equality constraint
% lbg/ubg includes: bounds for g in each step, not very useful
Xk = X0;
for k=1:N
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k-1)]);
    w   = {w{:} Uk};
    lbw = [lbw;lbu];
    ubw = [ubw;ubu];
    w0  = [w0;u_start(k)];
    
    % Integrate till the end of the interval
    Fk = F('x0',Xk,'p',Uk);
    Xk_end = Fk.xf;
    J=J+Fk.qf;

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k)], 1);
    w   = {w{:} Xk};
    
    % Add inequality constraint on w
    if k<N
        lbw = [lbw;lbx];
        ubw = [ubw;ubx];
    else
        lbw = [lbw;lbx_fnl];    % Add final state constraints
        ubw = [ubw;ubx_fnl];
    end
    w0  = [w0;x_start(:,k+1)];
    
    % Add equality constraint. Note: Xk is the symbol 'X_k', and Xk_end is
    % the expression of Xk in terms of earlier symbols.
    g   = {g{:} Xk_end-Xk};
    % Inequality constraint on state control combination
    lbg = [lbg; 0];
    ubg = [ubg; 0];
end

% Concatenate decision variables and constraint terms
w = vertcat(w{:});
g = vertcat(g{:});

% Create an NLP solver
nlp_prob = struct('f', J, 'x', w, 'g', g);
% nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, struct('discrete', discrete));
%nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});
nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); % Solve relaxed problem

% Plot the solution
tgrid = 0:T/N:T;
figure(1)
clf()
hold on

% Solve the NLP
sol = nlp_solver('x0',w0, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg);
w_opt = full(sol.x);
lam_w_opt = sol.lam_x;
lam_g_opt = sol.lam_g;
x0_opt = w_opt(1:2:end);
u_opt = w_opt(2:2:end);
plot(tgrid, x0_opt, '--');
stairs(tgrid, [nan; u_opt], '-.');
xlabel('t');
legend('x0','u');
grid('on');

