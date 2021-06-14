Modelo de água rasa 2D com assimilação de dados por Filtro de Kalman, emulado por RNA.

Uso: \
```./run-KFS2d.sh assimType gridX gridY timeStep freqObsT freqObsX freqObsY percNoise neuronNumber ```

Uso com Filtro de Kalman: \
```./run-KFS2d.sh 1 10 10 60 10 2 2 0.2 10 ```

Uso com RNA: \
```./run-KFS2d.sh 2 10 10 60 10 2 2 0.2 10 ```





Versão OpenMP  (src/KFS2d.f90: em desenvolvimento)

```fortran
!$OMP PARALLEL           &
!$OMP DEFAULT(shared)    &
!$OMP PRIVATE(sX,i)     
do sY = 1, gridY
!$OMP DO
   do sX = 1, gridX
      tid = omp_get_thread_num()
      i = (sY-1)*gridX + sX
      vco(:,1,tid) = matmul(wqco(:,:),xANN(:,i))
      vco(:,1,tid) = vco(:,1,tid) - (bqco(:,1))
      yco(:,1,tid) = (1.d0 - DEXP(-vco(:,1,tid))) / (1.d0 + DEXP(-vco(:,1,tid)))
      vcs(:,1,tid) = matmul(wqcs(:,:), yco(:,1,tid))
      vcs(:,1,tid) = vcs(:,1,tid) - bqcs(:,1)
      yANN(:,i,tS) = (1.d0-DEXP(-vcs(:,1,tid)))/(1.d0+DEXP(-vcs(:,1,tid)))
      qGl(sX,sY) = (yANN(1,i,tS)*(qModelMax-qModelMin) + qModelMax + qModelMin)/2.0
   enddo
!$OMP END DO                
enddo
!$OMP END PARALLEL



```