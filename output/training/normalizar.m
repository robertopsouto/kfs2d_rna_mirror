%--------------------------------------------------------------------------
% programa para plotar os graficos para a tese
%--------------------------------------------------------------------------
clear all; 
close all; clc;
%-----------------------
%leitura dos dados
%-----------------------
qM = load('qModelExpA.out');
qO = load('qObservExpA.out');
qA = load('qAnalysisExpA.out');

valNormInf=-1.0;
valNormSup=+1.0;

qM_norm=(max(qM)*valNormInf-min(qM)*valNormSup+qM*(valNormSup-valNormInf))/(max(qM)-min(qM));
qO_norm=(max(qO)*valNormInf-min(qO)*valNormSup+qO*(valNormSup-valNormInf))/(max(qO)-min(qO));
qA_norm=(max(qA)*valNormInf-min(qA)*valNormSup+qA*(valNormSup-valNormInf))/(max(qA)-min(qA));

n=size(qM);
pontos_por_ciclo=25;
ciclos=n/pontos_por_ciclos;

train_range=ceil(0.60*ciclos)*pontos_por_clicos;
valid_init=train_range+1;
valid_end=train_range+ceil(0.10*ciclos)*pontos_por_clicos;
gen_init=valid_end+1;
gen_end=valid_end+ceil(0.30*ciclos)*pontos_por_clicos;;

fileID = fopen('x.txt','w');
fprintf(fileID,'      %8.5f',qM_norm(1:train_range));
fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('x.txt','a');
fprintf(fileID,'      %8.5f',qO_norm(1:train_range));
fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('y.txt','w');
fprintf(fileID,'      %8.5f',qA_norm(1:train_range));
fprintf(fileID,'\n');
fclose(fileID);

fileID = fopen('x_valid.txt','w');
fprintf(fileID,'      %8.5f',qM_norm(valid_init:valid_end));
fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('x_valid.txt','a');
fprintf(fileID,'      %8.5f',qO_norm(valid_init:valid_end));
fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('y_valid.txt','w');
fprintf(fileID,'      %8.5f',qA_norm(valid_init:valid_end));
fprintf(fileID,'\n');
fclose(fileID);


fileID = fopen('x_gen.txt','w');
fprintf(fileID,'      %8.5f',qM_norm(gen_init:gen_end));
fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('x_gen.txt','a');
fprintf(fileID,'      %8.5f',qO_norm(gen_init:gen_end));
fprintf(fileID,'\n');
fclose(fileID);
fileID = fopen('y_gen.txt','w');
fprintf(fileID,'      %8.5f',qA_norm(gen_init:gen_end));
fprintf(fileID,'\n');
fclose(fileID);
