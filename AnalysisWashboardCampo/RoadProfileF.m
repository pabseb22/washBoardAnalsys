%******************************************************
%                  GEOTECHNICS OF ROADS
%             ADVANCED ANALYSIS AND MODELLING
%         COMPUTATION OF THE ROAD PROFILE USING A
%        ISO SPECTRAL DESCRIPTION AND THE METHOD OF
%              SUPERPOSITION OF HARMONICS
%******************************************************
clear
%************************
% DATA OF THE PROBLEM
%************************
data = inputdlg({'kappa [m^3/cycle]'},...
    'ROAD PROFILE, Input data',[1 40],{''});
k=str2double(data(1,1));         % kappa [m^3/cycle])
%**********************************************
% Data for the spectral description of the road
%**********************************************
Omega_L=0.01;                      % Lower frequency limit [cycles/m]
Omega_U=10;                        % Upper frequency limit [cycles/m]
Omega_0=0.5*pi;                    % Reference spatial frequency [cycles/m]
Dx=0.002;                          % Sampling length [m]
N=1000;                            % Number of frequency bands
%***********************
% Profile of the road
%***********************
L_road=3/Omega_L;                  % Length of the sample [m]
DOmega=(Omega_U-Omega_L)/N;        % Width of each frequency band
n_steps=L_road/Dx;                 % Number of steps in the length L_road
u_g(1:n_steps)=0;
X=linspace(0,L_road,n_steps);
for i=1:N
    Omega_n=Omega_L+DOmega*(2*i-1)/2;
    w=2;
    if (Omega_n<=Omega_0)
        w=1.5;
    end
    S_zOmega=k*(Omega_0/Omega_n)^w;
    Z_n=sqrt(2*S_zOmega*DOmega);
    fi_n=rand*2*pi;
    for j=1:n_steps
        u_g(j)=u_g(j)+Z_n*sin(2*pi*Omega_n*X(j)+fi_n);
    end
end       
figure()
plot(X,u_g);
title('Profile of the road')
grid on
XX=X(:);U0=u_g(:);
Results=[XX,U0];
MyFile=['Road profile.txt'];
dlmwrite(MyFile,Results,'delimiter','\t','precision',6)
disp('End')





