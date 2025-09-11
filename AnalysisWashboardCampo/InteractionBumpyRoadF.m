%****************************************************
% VEHICLE-ROAD INTERACTION WITH REAL PROFILE
%****************************************************
clear all; close all;

%========================
% VEHICLE PARAMETERS (quarter car model)
%========================
V = 5;              % Vehicle velocity [m/s]

% Amarok 2013 rear-right (starting guess)
m_1 = 60;            % Half mass of the axle [kg]
m_2 = 408;           % Half mass carried by rear axle [kg]
k_1 = 2.5e5;         % Tire stiffness [N/m]
k_2 = 1.2e2;         % Suspension stiffness [N/m]
c_1 = 6.0e4;         % Tire damping [Ns/m]
c_2 = 2.0e3;         % Suspension damping [Ns/m]

%**************
% General data
%**************
g = 9.8;                         
Dt = 0.001;                         
Dx = V*Dt;                          
n_res = 100;                       
n_plot = 1000;                       

%************************
% Matrices of the vehicle
%************************
M = [m_1,0; 0,m_2];                % Mass matrix
C = [c_1+c_2,-c_2;-c_2,c_2];       % Damping matrix
K = [k_1+k_2,-k_2;-k_2,k_2];       % Stiffness matrix

%*******************************
% LOAD REAL PROFILE FROM TXT
%*******************************
ProfileData = readmatrix('filtered_datos_perfil_4_Malchingui1.txt'); % 2 cols: x[m], y[mm]
X_full = ProfileData(:,1); 
Y_full = ProfileData(:,2)/1000;    % Convert mm to m

%========================
% Select window 60 < x < 120 m
%========================
x_min = 60; x_max = 70;
idx_window = (X_full >= x_min) & (X_full <= x_max);

X_window = X_full(idx_window);
Y_window = Y_full(idx_window);

%========================
% Interpolate to simulation steps
%========================
n_steps = floor((x_max - x_min)/Dx);
X_sim = linspace(x_min, x_max, n_steps);

% u_g is the road displacement
u_g = interp1(X_window, Y_window, X_sim, 'linear', 'extrap');

% u_gp is the derivative w.r.t. time
u_gp = [0 diff(u_g)/Dt];  % Forward difference approximation

%========================
% Plot field profile for validation
%========================
figure;
subplot(2,1,1)
plot(X_window, Y_window, 'b', 'LineWidth', 1.5); grid on;
xlabel('Distance [m]'); ylabel('Height [m]');
title('Field profile from measurements');

subplot(2,1,2)
plot(X_sim, u_g, 'r', 'LineWidth', 1.5); grid on;
xlabel('Distance [m]'); ylabel('Interpolated u_g [m]');
title('Interpolated profile for simulation');

%*************
% Vector f(t)
%*************
F = zeros(n_steps,2);
for i=1:n_steps-1
    F(i,1) = k_1*u_g(i) + c_1*u_gp(i) + m_1*g;
    F(i,2) = m_2*g;
end
F(n_steps,:) = F(n_steps-1,:);

%********************************
% Computation of the interaction
%********************************
U = zeros(n_steps,2);
X_rel = zeros(n_steps,1);
Reaction = zeros(n_steps,1);

U(1,1) = (m_1+m_2)*g/k_1; U(2,1) = U(1,1);  
U(1,2) = U(1,1)+m_2*g/k_2; U(2,2) = U(1,2);
X_rel(1) = U(1,1) - u_g(1);

for i=2:n_steps-1
    C = [c_1+c_2,-c_2;-c_2,c_2];
    K = [k_1+k_2,-k_2;-k_2,k_2];
    MC = M/Dt^2 + C/Dt;
    
    X_rel(i) = U(i,1) - u_g(i);
    Reaction(i) = k_1*X_rel(i) + c_1*(X_rel(i)-X_rel(i-1))/Dt;
    
    if X_rel(i) < 0
        F(i,1) = m_1*g;
        C = [c_2,-c_2;-c_2,c_2];
        K = [k_2,-k_2;-k_2,k_2];
        MC = M/Dt^2 + C/Dt;
        Reaction(i) = 0;
    end
    
    U(i+1,:) = F(i,:) - U(i,:)*K - (U(i-1,:) - 2*U(i,:))*M/Dt^2 + U(i,:)*C/Dt;
    U(i+1,:) = U(i+1,:) / MC;
end
U(n_steps,:) = U(n_steps-1,:);

%**********
% Plot simulation results
%**********
figure;
subplot(4,1,1); plot(X_sim, U(:,2)); title('Displacement of the vehicle'); grid on;
subplot(4,1,2); plot(X_sim, U(:,1)); title('Displacement of the axle'); grid on;
subplot(4,1,3); plot(X_sim, u_g); title('Road profile (simulation)'); grid on;
subplot(4,1,4); plot(X_sim, Reaction); title('Reaction'); grid on;

%*****************************
% Save results
%*****************************
Results = [X_sim(:), u_g(:), U(:,1), U(:,2), Reaction(:)];
dlmwrite('Road_Vehicle_Interaction.txt', Results, 'delimiter', '\t', 'precision', 6);

disp('End of simulation');
