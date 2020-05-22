#define nmax 80000	
#define nmanzanas 2	//nro de manzanas
#define inhem 5		//nro de hembras
#define ntachito 5   	//nro de tachos
#define iovip 32        //nro de oviposición
#define ttotal 400	//tiempo
#define morhue 0.01   	//mortalidad de huevos
#define morlar 0.01     //mortalidad de larvas 
#define morpup 0.01   	//mortalidad de pupas
#define morad 0.01 	//mortalidad diaria adultas
#define moracu 0.03	//morhue+morlar+morpup;
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
  int *DdM; 					//día que muere
  int *Numanzana;				//nro de manzana

//constructor
 mosquitos()
{
	//void aloco_memoria(int *tach, mosquitos M){
  N=nmax;  
  VoM = (int *) malloc(nmanzanas*nmax*sizeof(int )); 
  tacho = (int *) malloc(nmanzanas*nmax*sizeof(int ));
  DdV = (int *) malloc(nmanzanas*nmax*sizeof(int ));
  DdM = (int *) malloc(nmanzanas*nmax*sizeof(int ));
  Numanzana= (int *) malloc(nmanzanas*sizeof(int ));
}    

};

// alocar memoria

//void aloco_memoria(int *tach){
//  tach = (int *)malloc(nmanzanas*nmax*sizeof(int)); //Aloco memoria para vector tach[nmax];
//}

//defino la temperatura segun la estacion y el tiempo entre oviposiciones
int tiempo_en_dias(int tiempo, int tovip, long *sem){

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
	//printf("%d\t %d\n",tiempo,tovip); 
}

void incializar_vec(int *tach){
	for(int i=1;i <= nmanzanas*nmax;i++){     //para ello inicializo el vector tacho[nmax] en 0
	tach[i]=0; 			
	}
}

void huevos_x_tacho(int indice, mosquitos M, int *tach){
	int j=0;
		for(int i=1;i <= indice; i++){ //ciclo que va desde i=1,...,5
   			if(M.DdV[i] < tpupad && M.VoM[i] == 1){ 
    			j=M.tacho[i]; 
		    	tach[j]=tach[j]+1;
			}
		} 

}

//mortalidades varias

void mortalidades(int indice, mosquitos M, long *sem){

	for(int i =1;i <= indice;i++){
		if (M.VoM[i] == 1 && M.DdV[i] < tpupad){ 
		  	 if (ran2(sem) < moracu)M.VoM[i]=0;  
		}

		if (M.VoM[i] == 1 && M.DdV[i] == tpupad){ 
	  	 	 if (ran2(sem) < morpupad)M.VoM[i]=0;  
	  	}

		if (M.VoM[i] == 1 && M.DdV[i] > tpupad){ 
	  	 	 if(ran2(sem) < morad)M.VoM[i]=0;  
		}

	//printf("%d\n",M.DdV[i]);
	}  //del indice 
}

// muerte por vejez

void muerte_x_vejez(int indice, mosquitos M){

	for(int i =1;i <= indice;i++){
		if (M.VoM[i] == 1 && M.DdV[i] >= M.DdM[i])M.VoM[i]=0;
	//printf("%d\n",M.VoM[i]);
	} 
}
//Descacharrado

void descacharro(int tiempo, int descach, long *semi, int indice, mosquitos M, int *tach){
   	if(tiempo%7 == 0 && tiempo > 20){

  		for(int itach=1;itach <= descach;itach++){
    		int ntach=ran2(semi)*ntachito;

			for(int i =1;i <= indice;i++){

  				if (M.VoM[i] == 1 && M.DdV[i] < tpupad && M.tacho[i] == ntach)M.VoM[i]=0;
				//printf("%d\n",M.VoM[i]);
			}

		}    
	
   	}
}

// cuento población

int poblacion_adultos(int ad, int indice, mosquitos M){
  	for(int i =1;i <= indice; i++){
		if (M.VoM[i] == 1 && M.DdV[i] >= tpupad)ad=ad+1; //adultos tot	
	}
return ad;
}

int poblacion_acuaticos(int ac, int indice, mosquitos M){
  	for(int i =1;i <= indice; i++){
		if (M.VoM[i] == 1 && M.DdV[i] < tpupad)ac=ac+1; //acuaticos tot
	}
return ac;
}

//envejezco población un día

void envejecer_poblacion(int indice, mosquitos M){
 
  	for(int i =1;i <= indice;i++){

	 	 if (M.VoM[i] == 1)M.DdV[i]=M.DdV[i]+1;
	//printf("%d\n",M.DdV[i]);
	} 
}
