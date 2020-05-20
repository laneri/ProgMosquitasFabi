#!/bin/bash

echo ""
echo "------------------------------------------------------------------------"
echo "--------- Calculo la poblacion de mosquitas en una manzana -------------"
echo "------------------------------------------------------------------------"
echo "Programa mosquitas en C: mosquitas es un puntero a una estructura 'manzana' con campos {VoM[nmax], tacho[nmax], DdV[nmax], DdM[nmax]}"
echo ""
echo "Compilo y ejecuto"
make C
echo ""
./C


echo ""
echo "-------------------------------------------------------------------------"
echo "Programa mosquitas en Fortran: mosquitas es una matriz de [nmax][4]"
echo ""
echo "Compilo y ejecuto"
make Fortran
echo ""
./Fortran

echo "Borro ejecutables"
make clean


