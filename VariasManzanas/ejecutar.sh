#!/bin/bash

echo ""
echo "------------------------------------------------------------------------"
echo "--------- Calculo la poblacion de mosquitas en una manzana -------------"
echo "------------------------------------------------------------------------"
echo "Programa mosquitas en C para 2 manzanas: mosquitas es una estructura con campos {VoM[], tacho[], DdV[], DdM[],Numanzana}"
echo ""
echo "Compilo y ejecuto"
make Cpp
echo ""
./Cpp


echo ""
echo "-------------------------------------------------------------------------"
echo "Programa mosquitas en Fortran para una manzana: mosquitas es una matriz de [nmax][4]"
echo ""
echo "Compilo y ejecuto"
make Fortran
echo ""
./Fortran

echo "Borro ejecutables"
make clean


