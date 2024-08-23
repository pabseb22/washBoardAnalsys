clc
clear all

Datos=load("Escalograma.mat"); % tiempo, x posicion laser, zg
xint=Datos.datos_esca (:,1);
yint=Datos.datos_esca (:,3);

tvec=linspace(xint(1),xint(end),length(xint));
tvec=tvec';
xk=interp1(xint,yint,tvec);
tvec=tvec'; xk=xk';
N=length(tvec);
% tvec=linspace(0,1,N);
% xk=sin(2*pi*4*tvec)+sin(2*pi*15*tvec);
Ts=tvec(2)-tvec(1);
Fs=1/Ts;


%fb = cwtfilterbank('SignalLength',numel(tvec),'FrequencyLimits',[1 50],'SamplingFrequency',Fs);


%[xk_cwt,f]=cwt(xk,Fs);
[xk_cwt,f]=cwt(xk,Fs,"FrequencyLimits",[1 20]);
figure 
plot(tvec,xk)

figure;
opengl software
SC = wscalogram('image',xk_cwt,'ydata',xk,'xdata',tvec,'scales',f);

%Figura Escalograma
figure
image("XData",tvec,"YData",f,"CData",abs(xk_cwt),"CDataMapping","scaled")
axis tight
colorbar
xlabel("Time (Seconds)","FontName","Times New Roman");
ylabel("Frequency (Hz)","FontName","Times New Roman");
title("Scalogram","FontName","Times New Roman")
set(axes1,'BoxStyle','full','CLim',[-15 15],'FontName','Times New Roman',...
'Layer','top');



% 
% xlim(axes1,[0 4.5]);
% box(axes1,'on');
% hold(axes1,'off');
% xlabel('x (m)','FontName','Times New Roman');
% ylabel('No. of passage','FontName','Times New Roman');
% zlabel('z (mm)','FontName','Times New Roman');
% ax.FontSize = 10
% set(axes1,'BoxStyle','full','CLim',[-15 15],'FontName','Times New Roman',...
%     'Layer','top');
% hcb=colorbar(axes1);
% hcb.Title.String = "z(mm)"
% title ('Soil Profile 2D');
% f = gcf;
% saveas(f,'SoilProfile2D.fig');













