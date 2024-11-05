% Limpia la ventana de comandos, elimina todas las variables del espacio de trabajo y cierra todas las figuras abiertas.
clc;
clear all;
close all;

% Lista de nombres de archivos para procesar.
file_names = {'Malchi_1_test_2_vel10.mat', 'Malchi_1_test_3_vel20.mat', 'Malchi_1_test_6_vel40.mat', 'Malchi_1_test1_vel60.mat'};

vel=[10,20,40,60]*0.277; %velocidad en Km/h pasamos m/s

% Constantes utilizadas en el procesamiento.
Resis = 240; % Valor de resistencia en ohmios.
MR = 500; % Constante del sensor.

Resultados = cell(length(file_names), 7);

% Procesa cada archivo en la lista.
for i = 1:length(file_names)
    % Carga los datos del acelerómetro del archivo actual.
    [time, laser, acelerometro1, acelerometro2] = load_accelerometer_data(file_names{i});

    % Preprocesa los datos del acelerómetro.
    [acelerometro1_centrada, acelerometro2_centrada, Fs] = preprocess_accelerometer_data(time, acelerometro1, acelerometro2);

    % Filtra los datos del acelerómetro.
    [acelerometro1_filtrado, acelerometro2_filtrado] = filter_accelerometer_data(acelerometro1_centrada, acelerometro2_centrada, Fs);

    % Integra los datos del acelerómetro para obtener velocidad y distancia.
    [vel1, distancia1, vel2, distancia2, d_laser,d_laser_smooth] = integrate_accelerometer_data(time, acelerometro1_filtrado, acelerometro2_filtrado, laser, Resis, MR);


    Resultados{i,1}=[time,acelerometro1_filtrado];Resultados{i,2}=[time,acelerometro2_filtrado];
    Resultados{i,3}=[time,vel1];Resultados{i,4}=[time,vel2];
    Resultados{i,5}=[time,distancia1];Resultados{i,6}=[time,distancia2];
    Resultados{i,7}=[time,d_laser];Resultados{i,8}=[time,d_laser_smooth];
    Resultados{i,9}=[time*vel(i),d_laser_smooth];
    % Visualiza los resultados.
    %plot_results(time, acelerometro1_centrada, acelerometro2_centrada, acelerometro1_filtrado, acelerometro2_filtrado, vel1, vel2, distancia1, distancia2, d_laser,d_laser_smooth);

    % Filtra la señal para el intervalo de tiempo de 0 a 5 segundos
    indices_tiempo = Resultados{i,9}(:,1) >= 9 & Resultados{i,9}(:,1) <= 14;
    tiempo_filtrado = Resultados{i,9}(indices_tiempo, 1);
    lectura_laser_filtrada = Resultados{i,9}(indices_tiempo, 2);

    [f, P1] = calcularYMostrarFFT(tiempo_filtrado, lectura_laser_filtrada);

    % Guarda los resultados de f y P1 en Resultados{i,10} en un formato adecuado
    Resultados{i,10} = {f, P1};

      % Guarda los resultados de f y P1 en Resultados{i,10} en un formato adecuado
     %Desf_eje=1.26;
     Desf_eje=0;
     Resultados{i,11} = [time*vel(i),d_laser_smooth-distancia1*1000]; % grabamos el perfil VIGA+Laser en mm
     Resultados{i,12} = [time*vel(i)+Desf_eje,distancia2*1000]; % grabamos el perfil EJE en mm
     dx=mean(diff(time*vel(i)));
     Resultados{i,13} = [time*vel(i),distancia1*1000-distancia2*1000];
        
end


%% Calculos de amplitudes
% amplitudes por velocidad


vel_plot=[10,20,40,60]

amplitude_acel_S1 = zeros(size(Resultados,1),1);
amplitude_acel_S2 = zeros(size(Resultados,1),1);
amplitude_d1md2 = zeros(size(Resultados,1),1);

for i = 1:size(Resultados,1)
    % Extract data for Acelerometro 1 and 2
    data_S1 = Resultados{i,1}(:,2);
    data_S2 = Resultados{i,2}(:,2);
    data_d1md2=Resultados{i,13}(:,2);

    % Use the function to calculate amplitudes
    amplitude_acel_S1(i) = calculate_amplitude(data_S1);
    amplitude_acel_S2(i) = calculate_amplitude(data_S2);
    amplitude_d1md2(i)=calculate_amplitude(data_d1md2)
end




figure
plot(vel_plot,amplitude_acel_S1)
hold on
plot(vel_plot,amplitude_acel_S2)

figure
plot(vel_plot,amplitude_d1md2)

%% Graficos comparativos
% Matrices para almacenar datos de todos los archivos
names_plot={'Vel=10 Km/h','Vel=20 Km/h','Vel=40 Km/h','Vel=60 Km/h'};

%%

figure(100);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

% Agrega datos a cada subplot
for i = 1:length(file_names)
    tiempo_plot=Resultados{i,1}(:,1);
    aceleration_plot=Resultados{i,1}(:,2);
    subplot(length(file_names), 1, i);
    plot(tiempo_plot, aceleration_plot, 'DisplayName', names_plot{i});
    xlabel('T (s)');
    ylabel('a (m/s^2)');
    title(['Archivo: ', names_plot{i}]);
    %xlim([0,10])
    %ylim([-4,4])
end

%Ajusta propiedades del gráfico principal
sgtitle('Aceleración Acelerómetro1 Todas las Velocidades');
%legend;

figure(102);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

% Agrega datos a cada subplot
for i = 1:length(file_names)
    tiempo_plot=Resultados{i,2}(:,1);
    aceleration_plot=Resultados{i,2}(:,2);
    subplot(length(file_names), 1, i);
    plot(tiempo_plot, aceleration_plot, 'DisplayName', names_plot{i});
    xlabel('T (s)');
    ylabel('a (m/s^2)');
    title(['Archivo: ', names_plot{i}]);
    xlim([0,15])
    ylim([-4,4])
end

%Ajusta propiedades del gráfico principal
sgtitle('Aceleración Acelerómetro2 Todas las Velocidades');
%legend;

figure(103);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

% Agrega datos a cada subplot
for i = 1:length(file_names)
    tiempo_plot=Resultados{i,5}(:,1);
    disp_plot=Resultados{i,5}(:,2)*1000;
    subplot(length(file_names), 1, i);
    plot(tiempo_plot, disp_plot, 'DisplayName', names_plot{i});
    xlabel('T (s)');
    ylabel('d (mm)');
    title(['Archivo: ', names_plot{i}]);
    xlim([0,15])
    %ylim([-80,80])
end

%Ajusta propiedades del gráfico principal
sgtitle('Desplazamiento Acelerómetro1 Todas las Velocidades');
%legend;

figure(104);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

% Agrega datos a cada subplot
for i = 1:length(file_names)
    tiempo_plot=Resultados{i,6}(:,1);
    disp_plot=Resultados{i,6}(:,2)*1000; % convertimos en milimetros para graficar
    subplot(length(file_names), 1, i);
    plot(tiempo_plot, disp_plot, 'DisplayName', names_plot{i});
    xlabel('T (s)');
    ylabel('d (mm)');
    title(['Archivo: ', names_plot{i}]);
    xlim([0,15])
    %ylim([-80,80])
end

%Ajusta propiedades del gráfico principal
sgtitle('Desplazamiento Acelerómetro2 Todas las Velocidades');
%legend;

figure(105);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

% Agrega datos a cada subplot
for i = 1:length(file_names)
    tiempo_plot=Resultados{i,7}(:,1);
    disp_plot=Resultados{i,7}(:,2);
    subplot(length(file_names), 1, i);
    plot(tiempo_plot, disp_plot, 'DisplayName', names_plot{i});
    xlabel('Tiempo (s)');
    ylabel('d (mm)');
    title(['Archivo: ', names_plot{i}]);
    xlim([0,15])
    %ylim([-5,5])
end

%Ajusta propiedades del gráfico principal
sgtitle('Señal Original Laser Todas las Velocidades');
%legend;

figure(106);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

% Agrega datos a cada subplot
for i = 1:length(file_names)
    tiempo_plot=Resultados{i,9}(:,1);
    disp_plot=Resultados{i,9}(:,2);
    subplot(length(file_names), 1, i);
    plot(tiempo_plot, disp_plot, 'DisplayName', names_plot{i});
    xlabel('Dist. (m)');
    ylabel('d (mm)');
    title(['Archivo: ', names_plot{i}]);
    xlim([10,11])
    %ylim([-5,5])
end

%Ajusta propiedades del gráfico principal
sgtitle('Posición vs Laser');
%legend;


% Asume que este bloque de código se repite para cada figura que creas (ejemplo para figura 107)
figure(107);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y
for i = 1:length(file_names)
    f_plot = Resultados{i,10}{1}; % Frecuencias
    P1_plot = Resultados{i,10}{2}; % Espectro de frecuencia
    
    ax = subplot(length(file_names), 1, i);
    plot(f_plot, P1_plot, 'DisplayName', names_plot{i});
    xlabel('Frecuencia (Hz)');
    ylabel('|P1(f)|');
    title(['Espectro FFT Archivo: ', names_plot{i}]);
    
    % Personalización de la fuente y posición de las etiquetas
    set(ax, 'FontName', 'Arial', 'FontSize', 10); % Cambia tipo y tamaño de fuente de los ejes
    ax.XAxisLocation = 'bottom'; % Mueve la etiqueta del eje X a la parte superior
    ax.YAxis.Label.Rotation = 0; % Hace la etiqueta del eje Y horizontal
    ax.YAxis.Label.HorizontalAlignment = 'right'; % Alinea la etiqueta del eje Y a la derecha
    ax.YAxis.Label.VerticalAlignment = 'bottom'; % Alinea la etiqueta del eje Y al borde superior interior
    
    xlim([0, 5]); % Ajusta según sea necesario
    ylim([0, max(P1_plot) * 1.1]); % Ajusta para mejorar la visualización
end

sgtitle('Espectro de Frecuencia FFT de Todas las Velocidades', 'FontName', 'Arial', 'FontSize', 10);

%%
figure(108);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

for i = 1:length(file_names)
    ax = subplot(length(file_names), 1, i);
    x_plot = Resultados{i,11}(:,1); %  x en (m)
    y_plot = Resultados{i,11}(:,2); % d (mm)
    plot(x_plot, y_plot, 'DisplayName', names_plot{i});
    hold on
    x2_plot = Resultados{i,12}(:,1); % x en (m)
    y2_plot = Resultados{i,12}(:,2); % d (mm)
    plot(x2_plot, y2_plot, 'DisplayName', names_plot{i});

    xlabel('x (m)');
    ylabel('d (mm)');
    title(['Distancias: ', names_plot{i}]);
    
    % Personalización de la fuente y posición de las etiquetas
    set(ax, 'FontName', 'Arial', 'FontSize', 10); % Cambia tipo y tamaño de fuente de los ejes
    ax.XAxisLocation = 'bottom'; % Mueve la etiqueta del eje X a la parte superior
    ax.YAxis.Label.Rotation = 0; % Hace la etiqueta del eje Y horizontal
    ax.YAxis.Label.HorizontalAlignment = 'right'; % Alinea la etiqueta del eje Y a la derecha
    ax.YAxis.Label.VerticalAlignment = 'bottom'; % Alinea la etiqueta del eje Y al borde superior interior
    
    xlim([10, 20]); % Ajusta según sea necesario
    %ylim([0, max(P1_plot) * 1.1]); % Ajusta para mejorar la visualización
    
    % % Guardar los datos
    save_path_1 = sprintf('datos_perfil_%d.txt', i);
    % save_path_2 = sprintf('datos_eje_%d.txt', i);
    writematrix([x_plot, y_plot], save_path_1, 'Delimiter', '\t');
    % writematrix([x2_plot, y2_plot], save_path_2, 'Delimiter', '\t');
end

sgtitle('Desplazmiento', 'FontName', 'Arial', 'FontSize', 10);



figure(109);
set(gcf, 'Color', 'w', 'Units', 'centimeters'); % Cambia unidades a centímetros y el fondo a blanco
%pos = get(gcf, 'Position'); % Obtiene la posición actual para conservar x e y
%set(gcf, 'Position', [pos(1) pos(2) 15 15]); % Establece nuevo ancho y alto manteniendo x, y

for i = 1:length(file_names)
    ax = subplot(length(file_names), 1, i);
    x_plot = Resultados{i,13}(:,1); %  x en (m)
    y_plot = Resultados{i,13}(:,2); % d (mm)
    plot(x_plot, y_plot, 'DisplayName', names_plot{i});

    xlabel('x (m)');
    ylabel('|d_1-d_2| (mm)');
    title(['Distancias: ', names_plot{i}]);
    
    % Personalización de la fuente y posición de las etiquetas
    set(ax, 'FontName', 'Arial', 'FontSize', 10); % Cambia tipo y tamaño de fuente de los ejes
    ax.XAxisLocation = 'bottom'; % Mueve la etiqueta del eje X a la parte superior
    ax.YAxis.Label.Rotation = 0; % Hace la etiqueta del eje Y horizontal
    ax.YAxis.Label.HorizontalAlignment = 'right'; % Alinea la etiqueta del eje Y a la derecha
    ax.YAxis.Label.VerticalAlignment = 'bottom'; % Alinea la etiqueta del eje Y al borde superior interior
    
    xlim([10, 20]); % Ajusta según sea necesario
    %ylim([-10, 10]); % Ajusta para mejorar la visualización
    
    % Guardar los datos
    save_path = sprintf('datos_d1md2_%d.txt', i);
    writematrix([x_plot, y_plot], save_path, 'Delimiter', '\t');
end

sgtitle('Desplazmiento', 'FontName', 'Arial', 'FontSize', 10);


%% funciones
% Define la función para cargar los datos del acelerómetro desde un archivo.
function [time, laser, acelerometro1, acelerometro2] = load_accelerometer_data(file_name)
    datos = load(file_name); % Carga el archivo.
    time = seconds(datos.data.Time); % Convierte el tiempo a segundos.
    laser = double(datos.data.cDAQ1Mod1_ai0); % Convierte los datos del láser a double. (voltios)
    acelerometro1 = double(datos.data.cDAQ1Mod1_ai1); % Convierte los datos del acelerómetro 1 a double. Acelerometro en viga de auto (Voltios)
    acelerometro2 = double(datos.data.cDAQ1Mod1_ai2); % Convierte los datos del acelerómetro 2 a double. Acelerometro en eje de auto en  (Voltios)
end

% Define la función para preprocesar los datos del acelerómetro.
function [acelerometro1_centrada, acelerometro2_centrada, Fs] = preprocess_accelerometer_data(time, acelerometro1, acelerometro2)
    acelerometro1 = acelerometro1 * 1000 / 0.997; % Convierte de mV a m/s^2.
    acelerometro2 = acelerometro2 * 1000 / 0.997; % Convierte de mV a m/s^2.

    acelerometro1_centrada = acelerometro1 - mean(acelerometro1); % Centra la señal.
    acelerometro2_centrada = acelerometro2 - mean(acelerometro2); % Centra la señal.
    
    Fs = 1 / (time(2) - time(1)); % Calcula la frecuencia de muestreo (Hz).
end

% Define la función para filtrar los datos del acelerómetro.
function [acelerometro1_filtrado, acelerometro2_filtrado] = filter_accelerometer_data(acelerometro1_centrada, acelerometro2_centrada, Fs)
    f_corte_highpass = 1; % Frecuencia de corte del filtro de paso alto (Hz).

    acelerometro1_filtrado = highpass(acelerometro1_centrada, f_corte_highpass, Fs); % Aplica el filtro de paso alto.
    acelerometro2_filtrado = highpass(acelerometro2_centrada, f_corte_highpass, Fs); % Aplica el filtro de paso alto.
end
% Define la función para integrar los datos del acelerómetro y calcular la distancia.
function [vel1, distancia1, vel2, distancia2, d_laser,d_laser_smooth] = integrate_accelerometer_data(time, acelerometro1_filtrado, acelerometro2_filtrado, laser, Resis, MR)
    % Integra las señales del acelerómetro para obtener la velocidad.
    vel1 = cumtrapz(time, acelerometro1_filtrado);
    vel2 = cumtrapz(time, acelerometro2_filtrado);

    % Filtra las señales de velocidad integradas.
    Fs = 1 / (time(2) - time(1)); % Frecuencia de muestreo (Hz).
    f_corte_highpass = 1; % Frecuencia de corte del filtro de paso alto (Hz).
    vel1 = highpass(vel1, f_corte_highpass, Fs); % Aplica el filtro de paso alto.
    vel2 = highpass(vel2, f_corte_highpass, Fs); % Aplica el filtro de paso alto.

    % Elimina el desplazamiento de CC.
    vel1 = vel1 - mean(vel1);
    vel2 = vel2 - mean(vel2);

    % Integra las señales de velocidad para obtener la distancia.
    distancia1 = cumtrapz(time, vel1);
    distancia2 = cumtrapz(time, vel2);

    % Filtra las señales de distancia integradas.
    %distancia1 = highpass(distancia1, f_corte_highpass, Fs); % Aplica el filtro de paso alto.
    %distancia2 = highpass(distancia2, f_corte_highpass, Fs); % Aplica el filtro de paso alto.

    % Elimina el desplazamiento de CC.
    distancia1 = distancia1 - mean(distancia1);
    distancia2 = distancia2 - mean(distancia2);
    
    % Calcula la distancia medida por el láser.
    I_laser = laser / Resis; % Calcula la corriente del láser.
    d_laser = (I_laser * 1000 - 4) / 16 * MR; % Calcula la distancia del láser.

    % Aplica el filtro pasa banda a la distancia del láser.
    % Define las frecuencias de corte para el filtro pasa banda.
    f_corte_bajo = 1; % Frecuencia de corte baja (Hz).
    Fs = 1 / (time(2) - time(1)); % Calcula la frecuencia de muestreo basada en el tiempo.

    % Aplica el filtro pasa banda.
    %d_laser = highpass(d_laser, f_corte_bajo,Fs);
    d_laser = lowpass(d_laser, f_corte_bajo,Fs);
    d_laser=d_laser-mean(d_laser);
 

    % Corrige los outliers reemplazándolos con la mediana
  
    d_laser_smooth = filloutliers(d_laser,"pchip");
end

% Define la función para visualizar los resultados.
function plot_results(time, acelerometro1_centrada, acelerometro2_centrada, acelerometro1_filtrado, acelerometro2_filtrado, vel1, vel2, distancia1, distancia2, d_laser,d_laser_smooth)
    % La visualización de las señales del acelerómetro originales está comentada para simplificar la presentación de resultados.

    % Visualiza las señales del acelerómetro filtradas.
    figure;
    subplot(3,3,1);
    plot(time, acelerometro1_filtrado);
    xlabel('Tiempo (s)');
    ylabel('Aceleracion (m/s^2)');
    title('Señal Acelerometro 1 Filtrada');

    subplot(3,3,2);
    plot(time, acelerometro2_filtrado);
    xlabel('Tiempo (s)');
    ylabel('Aceleracion (m/s^2)');
    title('Señal Acelerometro 2 Filtrada');

    % Visualiza las señales de velocidad integrada.
    subplot(3,3,4);
    plot(time, vel1);
    xlabel('Tiempo (s)');
    ylabel('Velocidad (m/s)');
    title('Velocidad Integrada Acelerometro 1');

    subplot(3,3,5);
    plot(time, vel2);
    xlabel('Tiempo (s)');
    ylabel('Velocidad (m/s)');
    title('Velocidad Integrada Acelerometro 2');

    % Visualiza las señales de distancia integrada.
    subplot(3,3,7);
    plot(time, distancia1*1000);
    xlabel('Tiempo (s)');
    ylabel('Distancia (mm)');
    title('Distancia Integrada Acelerometro 1');

    subplot(3,3,8);
    plot(time, distancia2*1000);
    xlabel('Tiempo (s)');
    ylabel('Distancia (mm)');
    title('Distancia Integrada Acelerometro 2');

    % Visualiza la distancia medida por el láser.
    subplot(3,3,6);
    plot(time, d_laser_smooth)
    xlabel('Tiempo (s)');
    ylabel('Distancia (mm)');
    title('Distancia Láser smooth');

    subplot(3,3,9);
    plot(time, d_laser);
    xlabel('Tiempo (s)');
    ylabel('Distancia (mm)');
    title('Distancia Láser');
end

function [f, P1] = calcularYMostrarFFT(tiempo, lectura, Fs)
    % Verifica si Fs es proporcionado
    if nargin < 3
        % Calcula la frecuencia de muestreo a partir de los datos de tiempo, si no se proporciona
        Fs = 1 / mean(diff(tiempo));
    end
    
    % Calcula la longitud óptima para la FFT usando nextpow2
    N = length(lectura); % Longitud original de la señal
    N2 = 2^nextpow2(N); % Encuentra la próxima potencia de 2
    
    % Calcula la FFT de la señal, rellenando con ceros hasta N2
    lectura_fft = fft(lectura, N2);
    
    % Calcula el espectro de frecuencia de la señal
    f = Fs*(0:(N2/2))/N2; % Vector de frecuencia ajustado para N2
    P2 = abs(lectura_fft/N2); % Doble lado del espectro ajustado para N2
    P1 = P2(1:N2/2+1); % Lado único del espectro
    P1(2:end-1) = 2*P1(2:end-1); % Compensa por solo tomar la mitad del espectro
    % % Visualizar el espectro de frecuencia
    % figure;
    % plot(f, P1) 
    % title('Espectro de Frecuencia de la Señal')
    % xlabel('Frecuencia (Hz)')
    % ylabel('|P1(f)|')
    % 
    % % Ajustar los límites de los ejes para una mejor visualización, si es necesario
    % xlim([0, Fs/2]);
end



function amplitude = calculate_amplitude(data)
    % Calculate mean and standard deviation
    mu = mean(data);
    sigma = std(data);

    % Compute Z-scores and filter out data points where abs(Z) > 3
    filtered_data = data(abs((data - mu) / sigma) <= 3);

    % Calculate amplitude or set to NaN if no valid data remains
    if ~isempty(filtered_data)
        amplitude = max(filtered_data) - min(filtered_data);
    else
        amplitude = NaN;
    end
end
