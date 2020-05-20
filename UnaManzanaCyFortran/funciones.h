#include<stdlib.h>
#include<stdio.h>

//------------------------------  declaración de parámetros que se mantienen fijos en mi programa  ----------------------

#define nmax 80000	
#define inhem 5		//nro de hembras inhem=N
#define ntachito 5   	//nro de tachos
#define iovip 32        //nro de oviposición
#define ttotal 400	//tiempo
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

typedef struct
{
  int N;					//dimensión de cada array
  int *VoM; 					//vive (valor 1) o muere (valor 0)
  int *tacho;					//tacho en el que se encuentra
  int *DdV;					//días de vida 
  int *DdM;					//día que muerte   
}manzana;


//****************************************************************************************************************************************
//                                              funciones necesarias 
//****************************************************************************************************************************************
// alocar memoria

void aloco_memoria(int *tach, manzana *mosquitas){

  mosquitas -> VoM = (int *) malloc(nmax*sizeof(int )); 
  mosquitas -> tacho = (int *) malloc(nmax*sizeof(int ));
  mosquitas -> DdV = (int *) malloc(nmax*sizeof(int ));
  mosquitas -> DdM = (int *) malloc(nmax*sizeof(int ));

  tach = (int *)malloc(nmax*sizeof(int));
}

// inicizalizar arrays a cero

void inicializar(manzana *mosquitas){

	for(int i=1;i<=nmax;i++){
		mosquitas -> VoM[i] = 0;
		mosquitas -> tacho[i] = 0;
		mosquitas -> DdV[i] = 0;
		mosquitas -> DdM[i] = 0;
	}

}


void incializar_vec(int N,int *vector){

	for(int i=1;i <= N;i++){   
	vector[i]=0; 			
	}
}

//evolución temporal
//int tiempo_en_dias(int tiempo, int tovip, long semilla){ esto no es correcto porque luego tomaba la direccion de la copia de semilla que no es la misma que la direccion de memoria del semilla original. Hay que pasarle el puntero a semilla long* sem y no su valor.
int tiempo_en_dias(int tiempo, int tovip, long* sem){
	if(tiempo < 80 || tiempo > 320) tovip=tovip1;
  	
		if(tiempo > 160 && tiempo < 240) tovip=tovip3;

  		if(tiempo >= 80 && tiempo <= 160){
    			if(ran2(sem) < 0.5){
    	 		tovip=tovip2a;}
    			else{
   	 		tovip=tovip2b;
   		 	}
	  	}

	  	if(tiempo >= 240 && tiempo <= 320){
	    		if(ran2(sem) < 0.5){
		 	tovip=tovip2a;}
	    		else{
		 	tovip=tovip2b;
	    		}
	  	}
return tovip;

}

//número de huevos por tacho

void huevos_x_tacho(int indice,int *tach,manzana *mosquitas){

	int j=0;
		for(int i=1;i <= indice; i++){ //ciclo que va desde i=1,...,5
   			if(mosquitas -> DdV[i] < tpupad && mosquitas -> VoM[i] == 1){ 
    			j = mosquitas -> tacho[i]; 
		    	tach[j]=tach[j]+1;
			}
		} 

	//printf("%d\n",tach[j]);
}


//muerte por vejez

void muerte_x_vejez(int indice, manzana *mosquitas){

	for(int i =1;i <= indice;i++){

		if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] >= mosquitas -> DdM[i])mosquitas -> VoM[i]=0;
	//printf("%d\n",mosquitas -> VoM[i]);

	} 
}

int poblacion_adulta(int ad,int indice,manzana *mosquitas){
  
 	for(int i =1;i <= indice; i++){
		if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] >= tpupad)ad=ad+1; //adultos tot
	}
return ad;
}

int poblacion_acuatica(int ac,int indice,manzana *mosquitas){
  
 	for(int i =1;i <= indice; i++){		
		if (mosquitas -> VoM[i] == 1 && mosquitas -> DdV[i] < tpupad)ac=ac+1; //acuaticos tot
	}
return ac;
}


//envejezco a la población 1 día

void envejecer_poblacion(int indice,manzana *mosquitas){
   	for(int i =1;i <= indice;i++){

	 	 if (mosquitas -> VoM[i] == 1)mosquitas -> DdV[i]= 1 + mosquitas -> DdV[i];
	//printf("%d\n",mosquitas -> DdV[i]);
	} 
}

