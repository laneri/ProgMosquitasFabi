#define nmax 8000	//dimensión del array
#define nmanzanas 200	//nro de manzanas
#define inhem 5		//nro de hembras
#define ntachito 5   	//nro de tachos
#define ttotal 400	//tiempo
#define morhue 0.01   	//mortalidad de huevos
#define morlar 0.01     //mortalidad de larvas 
#define morpup 0.01   	//mortalidad de pupas
#define morad 0.01 	//mortalidad diaria adultas
#define moracu 0.03	//morhue+morlar+morpup;
#define morpupad 0.17	//pupas que no se vuelven adultas
#define tpupad1 9	//pupas se vuelven adultas a los 9 días en verano (desde oviposicion)****
#define tpupad2 13	//pupas se vuelven adultas a los 13 dias en otoño y primavera****
#define tpupad3 17  	//pupas se vuelven adultas a los 17 dias en invierno****
#define tovip1 2	//tiempo entre dos oviposiciones (T=30)
#define tovip2a 3  	//tiempo entre dos oviposiciones (T=25)
#define tovip2b 4   	//tiempo entre dos oviposiciones (T=25)
#define tovip3 29	//tiempo ente dos oviposiciones (T=18)
#define sat 800		//numero maximo de huevos en cada tacho
#define prop 0.6    	//efectividad de la propaganda

struct mosquitos
{
  int N;					//Nro de mosquitas
  int *VoM; 					//vive (valor 1) o muere (valor 0)
  int *tacho;					//tacho en el que se encuentra la hembra
  int *DdV;					//días de vida 
  int *DdM; 					//día que muere
  int *Numanzana;				//nro de manzana

//constructor
 mosquitos()
{
	//void aloco_memoria(int *tach, mosquitos M){
  N=nmax*nmanzanas;  
  VoM = (int *) malloc((N+1)*sizeof(int )); 
  tacho = (int *) malloc((N+1)*sizeof(int ));
  DdV = (int *) malloc((N+1)*sizeof(int ));
  DdM = (int *) malloc((N+1)*sizeof(int ));
  Numanzana= (int *) malloc((nmanzanas+1)*sizeof(int ));
}    

};

// Para luego calcular cuantos huevos pone la hembra en cada tacho en el que se encuentra
struct tachj
{
 int N;
 int *tach;

//constructor
tachj()
{	
  N=nmax*nmanzanas;
  tach = (int *) malloc((N+1)*sizeof(int ));
}
};

void inicializoMosquitos(mosquitos M){

 	for(int i=1;i<= inhem;i++){
	M.VoM[i] = 0; 
   	M.tacho[i] = 0;
   	M.DdV[i] = 0;
   	M.DdM[i] = 0;
	}
}

int dia_entre_oviposiciones(int dia){
		int tovip;
		if(dia < 80 || dia > 320)tovip=tovip3;  	
		if(dia >= 140 && dia <= 260)tovip=tovip1;
		if(dia > 80 && dia < 140)tovip=tovip2b;
	  	if(dia >= 260 && dia <= 320)tovip=tovip2a; 
	  
return tovip;
}

int dia_pupas_adultas(int dia){
		int tpupad;
		if(dia < 80 || dia > 320)tpupad=tpupad3;  	
		if(dia >= 140 && dia <= 260)tpupad=tpupad1;
  		if(dia > 80 && dia < 140)tpupad=tpupad2;
	  	if(dia >= 260 && dia <= 320)tpupad=tpupad2; 
return tpupad;
}

void conteo_huevosxT(int dia,int indice, int tpupad, mosquitos M, tachj T){

	for(int i=1;i <= indice; i++){ //ciclo que va desde i=1,...,5
		T.tach[i]=0;
   		if(M.DdV[i] < tpupad && M.VoM[i] == 1){ 
    		int j=M.tacho[i]; 
	    	T.tach[j]=T.tach[j]+1;
		}
	} 
}

void mortalidades(int indice, mosquitos M, int tpupad,long *seed){
	for(int i=1;i <= indice;i++){
		if (M.VoM[i] == 1 && M.DdV[i] < tpupad){if(ran2(seed) < moracu)M.VoM[i]=0;}
		if (M.VoM[i] == 1 && M.DdV[i] == tpupad){if(ran2(seed) < morpupad)M.VoM[i]=0;}
		if (M.VoM[i] == 1 && M.DdV[i] > tpupad){if(ran2(seed) < morad)M.VoM[i]=0;}  
	//printf("%d\n",M.DdV[i]);
	}  //del indice 
}

/*void nacimientos(int indice,mosquitos *M,tachj *T,int tovip,int tpupad,long *seed){
	for(int i =1;i <= indice;i++){
		if (T.tach[M.tacho[i]] < sat){ 
			if(M.VoM[i] == 1 && M.DdV[i] > tpupad){
				  if(M.DdV[i]%tovip == 0){
 				  int iovip=ran2(seed)*4+7; 
   					for(int ik=1;ik <= iovip;ik++){ 

 					indice=indice+1; 
 					M.VoM[indice]=1;
 					M.DdV[indice]=1;   
 					M.tacho[indice]=M.tacho[i]; 
	         			M.DdM[indice]=ran2(seed)*3+28; 
					int j=M.tacho[indice];
 					T.tach[j]=T.tach[j]+1;
   					}
  				   }    
			}  
   		} 

	//printf("%d\n",tach[M.tacho[i]]);
	} 

}
*/
void muerte_x_vejez(int indice, mosquitos M){

	for(int i=1;i <= indice;i++){if (M.VoM[i] == 1 && M.DdV[i] >= M.DdM[i])M.VoM[i]=0;} 
}

void descacharrado(int dia, long *seed, int descach,mosquitos M, int indice, int tpupad){
	if(dia%7 == 0 && dia > 150 && dia < 240){
  		for(int itach=0;itach < descach;itach++){
    		int ntach=ran2(seed)*ntachito + 1;
			for(int i=1;i <= indice;i++){
  				if (M.VoM[i] == 1 && M.DdV[i] < tpupad && M.tacho[i] == ntach)M.VoM[i]=0;
				//printf("%d\n",M.VoM[i]);
			}
		}    	
   	}
}

void poblacion_total(int dia,int indice, int tpupad,mosquitos M, int *poblacion){
  int ad=0;  //adultas total
  int ac=0;  //acuaticas vivas total

  	for(int i=1;i <= indice; i++){
		if (M.VoM[i] == 1 && M.DdV[i] >= tpupad)ad=ad+1; //adultos tot	
		if (M.VoM[i] == 1 && M.DdV[i] < tpupad)ac=ac+1; //acuaticos tot
	}

	poblacion[dia]=ac+ad;
}

void envejezco_poblacion(int indice, mosquitos M){

  	for(int i=1;i <= indice;i++){if (M.VoM[i] == 1)M.DdV[i]=M.DdV[i]+1;} 
}
