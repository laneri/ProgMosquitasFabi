//JUNIO 2019
//esqueleto cuda prog mosquitas


#include<iostream>
#include<fstream>

#include "celdas.h"

#include "parametros.h" 

using namespace std;
//////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

    celdas Ciudad(NC,cohLen); //Creo una grilla Ciudad con NC manzanas y en cada manzana choLen cohortes de mosquitos
	
 
 	Ciudad.dinamica(DiasSimul,NC); //evoluciona el sistema un num DiasSimul
       
	
	
	return 0;
}

