%****************************************************
%                  GEOTECHNICS OF ROADS
%             ADVANCED ANALYSIS AND MODELLING
%          VEHICLE ROAD INTERACTION ON A BUMPY ROAD
%****************************************************
clear
%************************
% DATA OF THE PROBLEM
%************************
% Road
L_road=60;         % Length of the road [m]
d_bumps=10;        % Separation between bumps [m]
a_bumps=0.05;        % Amplitude of the bumps [m]
Lambda_b=1;       % Wave length of the bumps [m]
% Vehicle

V=5;         % Velocity of the vehicle [m/s]
m_1 = 60;           % Half mass of the axle [kg]
m_2 = 408;          % Half mass carried by rear axle [kg]
k_1 = 2.5e5;         % Tire stiffness [N/m]
k_2 = 1.2e2;         % Suspension stiffness [N/m]
c_1 = 6.0e4;         % Tire damping [Ns/m]
c_2 = 2.0e3;         % Suspension damping [Ns/m]
%**************
% General data
%**************
g=9.8;                             % Acceleration of the gravity [m/s^2]
Dt=0.0001;                         % Time step [s]
Dx=V*Dt;                           % Length step
n_res=100;                         % Number of time steps between results
n_plot=1000;                       % Number of time steps between plots
%************************
% Matrices of the vehicle
%************************
M=[m_1,0;0,m_2];                   % Mass matrix
C=[c_1+c_2,-c_2;-c_2,c_2];         % Damping matrix
K=[k_1+k_2,-k_2;-k_2,k_2];         % Stiffness matrix
%*******************************
% Profile of the road and bumps
%*******************************

n_steps=L_road/Dx;                 % Number of time and length steps
u_g(1:n_steps)=0;
X=linspace(0,L_road,n_steps);
nx_bumps=Lambda_b/Dx;              % Number of steps in each bump
n_bump=L_road/d_bumps-1;           % Number of bumps
u_bump(1:nx_bumps)=0;
for i=1:nx_bumps
    u_bump(i)=-a_bumps/2-a_bumps/2*sin(-pi/2+2*pi*i/nx_bumps);
end
for i=1:n_bump
    for j=1:nx_bumps
        u_g(i*d_bumps/Dx+j)=u_bump(j);
    end
end
%*************
% Vector f(t)
%*************
F(1:n_steps,1:2)=0;
for i=1:n_steps-1
    u_gp=(u_g(i+1)-u_g(i))/Dt;
    F(i,1)=k_1*u_g(i)+c_1*u_gp+m_1*g;
    F(i,2)=m_2*g;
end
F(n_steps,1:2)=F(n_steps-1,1:2);
%********************************
% Computation of the interaction
%********************************
U(1:n_steps,1:2)=0;
X_rel(1:n_steps)=0;
Reaction(1:n_steps)=0;
U(1,1)=(m_1+m_2)*g/k_1;U(2,1)=U(1,1);  %Initial values of the displacements
U(1,2)=U(1,1)+m_2*g/k_2;U(2,2)=U(1,2);
X_rel(1)=U(1,1)-u_g(1);
for i=2:n_steps-1
    C=[c_1+c_2,-c_2;-c_2,c_2];K=[k_1+k_2,-k_2;-k_2,k_2];
    MC=M/Dt^2+C/Dt;
    X_rel(i)=U(i,1)-u_g(i);
    Reaction(i)=k_1*X_rel(i)+c_1*(X_rel(i)-X_rel(i-1))/Dt;
    if X_rel(i)<0
        %  F(i,1)=k_1*u_g(i)+c_1*u_gp+m_1*g;
        F(i,1)=m_1*g;
        C=[c_2,-c_2;-c_2,c_2];
        K=[k_2,-k_2;-k_2,k_2];
        MC=M/Dt^2+C/Dt;
        Reaction(i)=0;
    end
    U(i+1,:)=F(i,:)-U(i,:)*K-(U(i-1,:)-2*U(i,:))*M/Dt^2+U(i,:)*C/Dt;
    U(i+1,:)=U(i+1,:)/MC;
end
U(n_steps,:)=U(n_steps-1,:);
%**********
% Plotting
%**********
figure()
subplot(4,1,1)
plot(X,U(1:n_steps,2));
title('Displacement of the vehicle')
grid on
subplot(4,1,2)
plot(X,U(1:n_steps,1));
title('Displacement of the axle')
grid on
subplot(4,1,3)
plot(X,u_g);
title('Profile of the road')
grid on
subplot(4,1,4)
plot(X,Reaction);
title('Reaction')
grid on
r=0.5;                         % Radius of the tire, only for plotting
number_plots=n_steps/n_plot;
x=-r:.01:r;
y=sqrt(r^2-x.^2);
n_points=(L_road/10)/Dx;              % Points in the plot
n_circles=n_steps/100;                % number of circles
Centers(1:100,1:2)=0;
figure()
for i=1:10
    subplot(5,2,i)
    plot(X((i-1)*n_points+1:i*n_points), ...
    -u_g((i-1)*n_points+1:i*n_points),'LineWidth',2)
    xlim([(i-1)*n_points*Dx i*n_points*Dx])
    ylim([-0.5 1.5])
    pbaspect([5 1 1])
    hold on
    for j=1:10
        centerx=X((i-1)*n_points+1+j*n_circles-1);
        centery=-U((i-1)*n_points+1+j*n_circles-1,1)+r;
        xcirc=x+centerx;ycirc1=y+centery;ycirc2=centery-y;
        plot(xcirc,ycirc1,'r');
        plot(xcirc,ycirc2,'r')
        %drawnow
        Centers((i-1)*10+j,1)=centerx;Centers((i-1)*10+j,2)=centery;
    end
end
XX=X(:);U0=u_g(:);U1=U(:,1);U2=U(:,2);R=Reaction(:);
Results=[XX,U0,U1,U2,R];
MyFile=['Road Vehicle Interaction.txt'];
dlmwrite(MyFile,Results,'delimiter','\t','precision',6)
MyFile=['CenterWheel.txt'];
dlmwrite(MyFile,Centers,'delimiter','\t','precision',6)
%*****************************
% Hystogram of reaction forces
%*****************************
edges=[0 0.001 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 50]*10;
NReact=histcounts(Reaction/1000,edges);
MyFile=['Hyst Reaction.txt'];
dlmwrite(MyFile,[edges,NReact],'delimiter','\t','precision',6)
disp('End')





