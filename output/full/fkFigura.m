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
ni = 10;
nj = ni;
nk = 100;
%
ninj = ni*nj;
p = 7; % ponto
%
% dado do modelo
qm3D = reshape(qm,ni,nj,nk);
qpm10 = qm3D(p,p,:);
qpm10v = qpm10(:);
%
%valor estimado pelo Filtro de Kalman
qafk3D = reshape(qa,ni,nj,nk);
qpa10 = qafk3D(p,p,:);
qpa10v = qpa10(:);
%
%estimado pela Rede Neural
 qr3D = reshape(qar,ni,nj,nk);
 qpr10 = qr3D(p,p,:);
 qpr10v = qpr10(:);
%
figure(1)
plot(qpm10v,'b','linewidth',0.5); hold on;
plot(qpa10v,'r','linewidth',0.5);
plot(qpr10v,'g','linewidth',0.5);
title('variavel q'); grid on;
xlabel('tempo');
ylabel('q(7,7)');
legend('verdade','FK','RNA')
axis([1 100 -60 80])
%print -depsc variavelqExpA.eps;
print -dpng -r240 variavelqExpA.png;