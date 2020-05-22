CC=g++
CFLAGS= -g -Wall -lm
FC = gfortran
FCFLAGS = -g -c -fdefault-real-8 -fbacktrace -fno-align-commons

Cpp: programa_6.cpp ran2.h funciones.h
	$(CC) $(CCFLAGS) -o $@ $<

Fortran: mosquitas2020_new.f90
	$(FC) $(FLFLAGS) -o $@ $<

clean:
	rm -f Cpp Fortran

