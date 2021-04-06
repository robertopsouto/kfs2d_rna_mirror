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
qModel = load('./Empirico/result_2000_20160613/qModelExpA-norm.dat');
%
qAnaliseFK = load('./Empirico/result_2000_20160613/qFKExpA-norm.dat');
%
qAnaliseRNA_MPCA_SW = load('./FPGA/result_1500_20160621/Vm.txt');
qAnaliseRNA_MPCA_HW = load('./FPGA/result_1500_20160621/Vm_FPGA.txt');
%
qAnaliseRNA_Emp_SW = load('./FPGA/result_1500_20160621/Ve.txt');
qAnaliseRNA_Emp_HW = load('./FPGA/result_1500_20160621/Ve_FPGA.txt');
%
x = 40;
y = x;
Dt= 60;
%
xy = x*y;
ponto = 7; % ponto escolhido empiricamente
%
% Modelo
qMod3D = reshape(qModel,x,y,Dt);
qMod3D_P = qMod3D(ponto,ponto,:);
qMod3D_Pvetor = qMod3D_P(:);
%
% Analise: Filtro de Kalman
qAnaFK3D = reshape(qAnaliseFK ,x ,y ,Dt);
qAna3D_P = qAnaFK3D(ponto, ponto, :);
qAnaFK3D_Pvetor = qAna3D_P(:);
%
% Analise: RNA Autoconfigurada (MPCA) - Software
qAnaRNA_MPCA_SW3D = reshape(qAnaliseRNA_MPCA_SW, x, y, Dt);
qAnaRNA_MPCA_SW3D_P = qAnaRNA_MPCA_SW3D(ponto, ponto, :);
qAnaRNA_MPCA_SW_Pvetor = qAnaRNA_MPCA_SW3D_P(:);
%
% Analise: RNA Autoconfigurada (MPCA) - Hardware (FPGA)
qAnaRNA_MPCA_HW3D = reshape(qAnaliseRNA_MPCA_HW, x, y, Dt);
qAnaRNA_MPCA_HW3D_P = qAnaRNA_MPCA_HW3D(ponto, ponto, :);
qAnaRNA_MPCA_HW_Pvetor = qAnaRNA_MPCA_HW3D_P(:);
%
% Analise: RNA Empirica - Software
qAnaRNA_Emp_SW3D = reshape(qAnaliseRNA_Emp_SW, x, y, Dt);
qAnaRNA_Emp_SW3D_P = qAnaRNA_Emp_SW3D(ponto, ponto, :);
qAnaRNA_Emp_SW_Pvetor = qAnaRNA_Emp_SW3D_P(:);
%
% Analise: RNA Empirica - Hardware (FPGA)
qAnaRNA_Emp_HW3D = reshape(qAnaliseRNA_Emp_HW, x, y, Dt);
qAnaRNA_Emp_HW3D_P = qAnaRNA_Emp_HW3D(ponto, ponto, :);
qAnaRNA_Emp_HW_Pvetor = qAnaRNA_Emp_HW3D_P(:);
%
figure(1)
plot(qMod3D_Pvetor,'b','linewidth',2); % Verdade
hold on;
plot(qAnaFK3D_Pvetor,'r','linewidth',2); % Analise: FK
plot(qAnaRNA_MPCA_SW_Pvetor,'g','linewidth',2); % Analise: RNA - SW
plot(qAnaRNA_MPCA_HW_Pvetor,'y','linewidth',2); % Analise: RNA - HW
%title('Artificial Neural Network (MPCA):  Software versus Hardware');
grid on;
%xlabel('Time');
%ylabel('q(7,7)');
legend('True','Kalman Filter','ANN-MPCA(Softw)','ANN-MPCA(Hardw)')
%
figure(2)
plot(qMod3D_Pvetor,'b','linewidth',2); % Verdade
hold on;
plot(qAnaFK3D_Pvetor,'r','linewidth',2); % Analise: FK
plot(qAnaRNA_Emp_SW_Pvetor,'g','linewidth',2); % Analise: RNA - SW
plot(qAnaRNA_Emp_HW_Pvetor,'y','linewidth',2); % Analise: RNA - HW
%title('Artificial Neural Network (Empirical):  Software versus Hardware');
grid on;
%xlabel('Tie');
%ylabel('q(7,7)');
legend('True','Kalman Filter','ANN-Empirical(Softw)','ANN-Empirical(Hardw)')
%
figure(3)
plot(qMod3D_Pvetor,'b','linewidth',2); % Verdade
hold on;
plot(qAnaFK3D_Pvetor,'r','linewidth',2); % Analise: FK
plot(qAnaRNA_Emp_SW_Pvetor,'g','linewidth',2); % Analise: RNA - SW
plot(qAnaRNA_MPCA_SW_Pvetor,'y','linewidth',2); % Analise: RNA - SW
%title('Artificial Neural Network (Empirical):  Software versus Hardware');
grid on;
%xlabel('Tie');
%ylabel('q(7,7)');
legend('True','Kalman Filter','ANN-Empirical(Softw)','ANN-MPCA(Softw)')