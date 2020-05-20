#include<stdlib.h>
#include<stdio.h>

//------------------------------  declaración de parámetros que se mantienen fijos en mi programa  ----------------------
#define nmanzanas 1
#define nmax 80000	
#define inhem 5		//nro de hembras inhem=N
#define ntachito 5   	//nro de tachos
#define iovip 32        //nro de oviposición
#define ttotal 400	//400 tiempo
#define morhue 0.01   	//mortalidad de huevos
#define morlar 0.01     //mortalidad de larvas 
#define morpup 0.01   	//mortalidad de pupas
#define moracu 0.03	//morhue+morlar+morpup mortalidad total de las acuaticas  
#define morad 0.01 	//mortalidad diaria adultas
#define morpupad 0.17	//pupas que no se vuelven adultas
#define tpupad 17 	//pupas se vuelven adultas que pican a los 17 dias
#define tovip1 2	//tiempo entre dos oviposiciones (T=30)
#define tovip2a 3  	//tiempo entre dos oviposiciones (T=25)
#define tovip2b 4   	//tiempo entre dos oviposiciones (T=25)
#define tovip3 10	//tiempo ente dos oviposiciones (T=18)
#define sat 800		//numero maximo de huevos en cada tacho
#define prop 0.6    	//efectividad de la propaganda

//estructura

struct mosquitos
{
  int N;					//dimensión de cada array
  int *VoM; 					//vive (valor 1) o muere (valor 0)
  int *tacho;					//tacho en el que se encuentra
  int *DdV;					//días de vida 
  int *DdM; //día que muerte
  int *Numanzana;
//constructor
mosquitos(){
//void aloco_memoria(int *tach, manzana *mosquitas){
  N=nmax;  
  VoM = (int *) malloc(nmanzanas*nmax*sizeof(int )); 
  tacho = (int *) malloc(nmanzanas*nmax*sizeof(int ));
  DdV = (int *) malloc(nmanzanas*nmax*sizeof(int ));
  DdM = (int *) malloc(nmanzanas*nmax*sizeof(int ));
  Numanzana= (int *) malloc(nmanzanas*sizeof(int ));
}    
};


//****************************************************************************************************************************************
//                                              funciones necesarias 
//****************************************************************************************************************************************
