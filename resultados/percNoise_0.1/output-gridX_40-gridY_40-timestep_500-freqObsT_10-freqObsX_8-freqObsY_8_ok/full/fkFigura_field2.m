%--------------------------------------------------------------------------
% programa para plotar os graficos para a tese
%--------------------------------------------------------------------------
clear all; 
close all; clc;
%-----------------------
%leitura dos dados
%-----------------------
qm = load('qModelExpA.out');
qa = load('qAnalysisExpA.out');
% dado da rede
qar = load('qAnalysisExpA_RNA.out');
%
ni = 40;
nj = ni;
nk = 500;
##%
##ninj = ni*nj;
##p = 7; % ponto
##%
##% dado do modelo
qm3D = reshape(qm,ni,nj,nk);
##qpm10 = qm3D(p,p,:);
##qpm10v = qpm10(:);
##%
##%valor estimado pelo Filtro de Kalman
qafk3D = reshape(qa,ni,nj,nk);
##qpa10 = qafk3D(p,p,:);
##qpa10v = qpa10(:);
##%
##%estimado pela Rede Neural
qr3D = reshape(qar,ni,nj,nk);
## qpr10 = qr3D(p,p,:);
## qpr10v = qpr10(:);
##%
figure(1)
contour(qm3D(:,:,30),'-k')
hold on
contour(qafk3D(:,:,30),':r')
#contour(qr3D(:,:,30),'--m')

##plot(qpm10v,'b','linewidth',0.5); hold on;
##plot(qpa10v,'r','linewidth',0.5);
##plot(qpr10v,'g','linewidth',0.5);
title('q variable'); grid on;
xlabel('X');
ylabel('Y');
legend('TRUE','KF')
##axis([1 500 -60 80])
##%print -depsc variavelqExpA.eps;
print -dpng -r240 -mono variavelqExpA_field_grid40_True_KF.png;
