%--------------------------------------------------------------------------
% programa para plotar os graficos para a tese
%--------------------------------------------------------------------------
clear all; 
close all; clc;
%--------------------------------------------------------------------------
%leitura dos dados
%--------------------------------------------------------------------------
%
% Modelo
qModel_fortran = load('../workspace/KFS2Dv0.1/qModelExpA.out');
qModel_matlab = load('../workspace/KFS2Dv0.1/qAnalysisExpA.out');

qMf = qModel_fortran';
qMm = qModel_matlab';

x = 40;
y = x;
Dt= 60;
%
xy = x*y;
ponto = 7; % ponto escolhido empiricamente
%
% Modelo Fortran
qModF3D = reshape(qModel_fortran,x,y,Dt);
qModF3D_P = qModF3D(ponto,ponto,:);
qModF3D_Pvetor = qModF3D_P(:);
%
% Modelo Matlab
qModM3D = reshape(qModel_matlab,x,y,Dt);
qModM3D_P = qModM3D(ponto,ponto,:);
qModM3D_Pvetor = qModM3D_P(:);
%
figure(1)
plot(qModF3D_Pvetor,'b','linewidth',2); % Verdade
%hold on;
%plot(qModM3D_Pvetor,'r','linewidth',2); % Analise: FK

title('Rede Neural Auto (MPCA):  SW v HW - Variavel q');
grid on;
xlabel('Tempo');
ylabel('q(7,7)');
legend('Fortran','Matlab')
