clear all
close all
load ../output/training/qModelExpA.out;
load ../output/training/qObservExpA.out;
load ../output/training/qAnalysisExpA.out;
plot(qModelExpA)
hold on
plot(qObservExpA)
plot(qAnalysisExpA,'g')