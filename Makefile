#Compilador
#FC = /usr/local/bin/gfortran
FC = gfortran
#Opcoes de compilacao
FFLAGS = -c -O2 -fopenmp

#Opcoes de compilacao
FFLAGSOPT = -O2 -fopenmp 

#Bibliotecas
FFLIBS = -llapack -lblas -g -fcheck=all -Wall -fbacktrace -fopenmp
#FFLIBS = -I/scratch/cenapadrjsd/rpsouto/usr/local/spack/git/spack/opt/spack/linux-rhel7-ivybridge/gcc-8.3.0/openblas-0.3.13-mkwyvnyskpkkhdwiyrrnlsi2gypuw72z/include -L/scratch/cenapadrjsd/rpsouto/usr/local/spack/git/spack/opt/spack/linux-rhel7-ivybridge/gcc-8.3.0/openblas-0.3.13-mkwyvnyskpkkhdwiyrrnlsi2gypuw72z/lib -lopenblas -g -fcheck=all -Wall -fbacktrace

#Objetos
objects = kfsFunctions.o KFS2d.o

KFS2d:$(objects)
	$(FC) -o ./KFS2d $(objects) $(FFLIBS)

KFS2d.o:
	gfortran -O2 -c -g -fopenmp ./src/KFS2d.f90

kfsFunctions.o:
	gfortran -O2 -c -g -fopenmp ./src/kfsFunctions.f90

clean:
	rm -rf ./src/*.mod *.o KFS2d *.mod *.o *.out
