#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#include<thrust/device_vector.h>
#include<thrust/remove.h>
#include<thrust/find.h>
#include <thrust/fill.h>

#include "ran2.h"
#include "gpu_timer.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>

// macros utiles para CPU y GPU, y hacer desaparecer numeros magicos
#define Ninicial		    5
#define NUMEROTACHOS		5
#define MAXIMONUMEROBICHOS	80000

#define ESTADOMUERTO		1
#define ESTADOVIVO		    0
#define Ndias			    400

#define ntachito	    	5
#define morhue 			    0.01   	//mortalidad de huevos
#define morlar 			    0.01    //mortalidad de larvas 
#define morpup 			    0.01   	//mortalidad de pupas
#define morad 			    0.01 	//mortalidad diaria adultas
#define moracu 			    0.03	//morhue+morlar+morpup;
#define morpupad 		    0.17	//pupas que no se vuelven adultas
#define tpupad1 	    	9	    //pupas se vuelven adultas a los 9 días en verano (desde oviposicion)****
#define tpupad2 		    13	    //pupas se vuelven adultas a los 13 dias en otoño y primavera****
#define tpupad3 		    17  	//pupas se vuelven adultas a los 17 dias en invierno****
#define tovip1 			    2	    //tiempo entre dos oviposiciones (T=30)
#define tovip2a 		    3  	    //tiempo entre dos oviposiciones (T=25)
#define tovip2b			    4   	//tiempo entre dos oviposiciones (T=25)
#define tovip3 			    29	    //tiempo ente dos oviposiciones (T=18)
#define sat 			    800     //saturación de huevos por tacho
#define prop 			    0.6	    //efectividad de la propaganda

#define SEMILLAGLOBAL	    12345

long semilla = -975;  			//semilla para el generador de numeros aleatorios ran2()

using namespace std;


//defino la temperatura según la estación y el tiempo entre oviposiciones y de maduración de acuáticos	

__device__ __host__ int tiempo_entre_oviposiciones(int dia){
	int tovip;
	if(dia < 80 || dia > 320){tovip=tovip3;}	//<T>=18
  	if(dia >= 140 && dia <= 260){tovip=tovip1;}	//<T>=30
  	if(dia > 80 && dia < 140){tovip=tovip2b;}	//<T>=23
	if(dia >= 260 && dia <= 320){tovip=tovip2a;}	//<T>=23

	return tovip;}

__device__ __host__  int tiempo_pupas_adultas(int dia){
	int tpupad;
	if(dia < 80 || dia > 320){tpupad=tpupad3;}//pupas se vuelven adultas a los 17 dias en invierno
	if(dia >= 140 && dia <= 260){tpupad=tpupad1;}//pupas se vuelven adultas a los 9 días en verano (desde oviposicion)
  	if(dia > 80 && dia < 140){tpupad=tpupad2;}//pupas se vuelven adultas a los 13 dias en otoño y primavera
	if(dia >= 260 && dia <= 320){tpupad=tpupad2;}

	return tpupad;}	

/*__global__ void conteo_kernel(int *estado, int *edad, int *tacho, int *tach, int *N_mobil, int dia,int tpupad)
{
    
    int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x + 1;

	if(id<=N){	
	        //tach[id]=0; 
	   		if(edad[id] < tpupad && estado[id] == ESTADOVIVO){ 
	    		int j=tacho[id]; 
		    	tach[j]=tach[j] + 1;}
		
	}  
    
};*/
/*
__global__ void matar_kernel_eggs(int *estado, int *edad, int *tacho, int *N_mobil, int dia,int tpupad)
{
	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<N){		
	 	RNG philox;         
	    	RNG::ctr_type c={{}};
	    	RNG::key_type k={{}};
	    	RNG::ctr_type r;
	    	k[0]=id; 
	    	c[1]=dia;
	    	c[0]=SEMILLAGLOBAL; 
		
    		r = philox(c, k); 
     		double azar=(u01_closed_closed_32_53(r[0]));

		    if (edad[id] < tpupad){if(azar < moracu)estado[id]=ESTADOMUERTO;}
	}
};

__global__ void matar_kernel_pupas(int *estado, int *edad, int *tacho, int *N_mobil, int dia,int tpupad)
{
	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<N){		
	 	RNG philox;         
	    	RNG::ctr_type c={{}};
	    	RNG::key_type k={{}};
	    	RNG::ctr_type r;
	    	k[0]=id; 
	    	c[1]=dia;
	    	c[0]=SEMILLAGLOBAL; 
		
    		r = philox(c, k); 
     		double azar=(u01_closed_closed_32_53(r[0]));

		if (edad[id] == tpupad){if(azar < morpupad)estado[id]=ESTADOMUERTO;}

	}
};
__global__ void matar_kernel_adultas(int *estado, int *edad, int *tacho, int *N_mobil, int dia, int tpupad)
{
	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<N){		
	 	RNG philox;         
	    	RNG::ctr_type c={{}};
	    	RNG::key_type k={{}};
	    	RNG::ctr_type r;
	    	k[0]=id; 
	    	c[1]=dia;
	    	c[0]=SEMILLAGLOBAL; 
		
    		r = philox(c, k); 
     		double azar=(u01_closed_closed_32_53(r[0]));

		if (edad[id] > tpupad){if(azar < morad)estado[id]=ESTADOMUERTO;}

	}
};
*/
//muerte de las mosquitas por vejez
__global__ void matar_viejos_kernel(int *estado, int *edad, int *tacho, int *TdV,int *N_mobil,int dia)
{

	    int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){                     
	    	if(edad[id] >= TdV[id])estado[id]=ESTADOMUERTO;
	    }
};

__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *N_mobil,int dia, int ntach)
{

	int tpupad=tiempo_pupas_adultas(dia);

    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){
  		if (edad[id] < tpupad && tacho[id] == ntach)estado[id]=ESTADOMUERTO;    	
    	}
};

__global__ void envejecer_kernel(int *estado, int *edad,int *N_mobil)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){
  		    if (estado[id]== ESTADOVIVO)edad[id]= edad[id] + 1;    	
    	}
};

/////////////////////////////////////////////////////////////////////////////////////////
// Clase bichos: toda la info sobre todos los bichos, y sus funciones
struct bichos{

	thrust::device_vector<int> estado;  // Vivo o Muerto 0/1
	thrust::device_vector<int> edad;    // edad
	thrust::device_vector<int> tacho;   // Tacho en el que vive
	thrust::device_vector<int> TdV;     // tiempo de vida
	thrust::device_vector<int> manzana; //manzana

	thrust::device_vector<int> tach;	

	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)
	
	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_TdV;
	int *raw_tach;
	int *raw_N_mobil;
	int *raw_manzana;
				
	// constructor: N_ = bichos iniciales
	bichos(int N_)
	{

		// alocamos el maximo posible
		edad.resize(MAXIMONUMEROBICHOS);	
		tacho.resize(MAXIMONUMEROBICHOS);	
		estado.resize(MAXIMONUMEROBICHOS);
		TdV.resize(MAXIMONUMEROBICHOS);
		manzana.resize(MAXIMONUMEROBICHOS);
		tach.resize(MAXIMONUMEROBICHOS);

		// un entero en device
		N_mobil.resize(1);
		

		// inicializacion raw pointers
		raw_edad=thrust::raw_pointer_cast(edad.data());
		raw_tacho=thrust::raw_pointer_cast(tacho.data());
		raw_estado=thrust::raw_pointer_cast(estado.data());
		raw_TdV=thrust::raw_pointer_cast(TdV.data());
		raw_manzana=thrust::raw_pointer_cast(manzana.data());
		raw_N_mobil=thrust::raw_pointer_cast(N_mobil.data());
		raw_tach=thrust::raw_pointer_cast(tach.data());

		std::cout<<"VoM\tedad\tTdV\ttacho\tmanzana" << std::endl;

		// inicializacion totalmente random
		for(int i=0;i<N_;i++){
			edad[i]=ran2(&semilla)*5 + 12; 	//edad de 12 hasta 16 días 
			tacho[i]=i;			            //tacho en el que se encuentra la mosquita
			estado[i]=ESTADOVIVO; 		    //todos vivos inicialmente
			TdV[i]=ran2(&semilla)*3 + 28;	//tiempo de vida de 28 a 30
			manzana[i]=(int)(i/5);
		std::cout << estado[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << tacho[i] << "\t" << manzana[i] << std::endl;
		}		
		N_mobil[0]=N_;
	};
		
	void conteo_huevos(int dia, int tpupad){
	int N=N_mobil[0];
	
	/*fill(tach.begin(),tach.end(),0);
	
    //problemas con el kernel, probar segmentandolo en dos partes, uno que inicialice tach[] y otro que cuente los huevos:no funcionó
    conteo_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_tach,raw_N_mobil, dia,tpupad);	
	cudaDeviceSynchronize();*/

		for(int i=0;i < N; i++){
			tach[i]=0;
	   		if(edad[i] < tpupad && estado[i] == ESTADOVIVO){ 
	    		int j=tacho[i]; 
		    	tach[j]=tach[j] + 1;
			}
		}
	};
	
	void mortalidades_varias(int dia,int tpupad){

	int N=N_mobil[0];
	/*//problemas con los kernels para las mortalidades,tal vez tiene que ver con  la semilla generadora de los números random, porque afecta el resultad, o tal vez hay un tema con la condicion de carrera
    //  matar_kernel_eggs<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil, dia,tpupad);	
	//	cudaDeviceSynchronize();
	//	matar_kernel_pupas<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil, dia,tpupad);	
	//	cudaDeviceSynchronize();

	//	matar_kernel_adultas<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil, dia,tpupad);		
	//	cudaDeviceSynchronize();*/

		for(int i=0;i < N;i++){
		if (estado[i] == ESTADOVIVO && edad[i] < tpupad){if(ran2(&semilla) < moracu)estado[i]=ESTADOMUERTO;}
		if (estado[i] == ESTADOVIVO && edad[i] == tpupad){if(ran2(&semilla) < morpupad)estado[i]=ESTADOMUERTO;}
		if (estado[i] == ESTADOVIVO && edad[i] > tpupad){if(ran2(&semilla) < morad)estado[i]=ESTADOMUERTO;}  
		} 

	};

	// recorre los bichos, calcula el numero de nacidos por tacho
	void reproducir(int dia,int tpupad,int tovip){	

	int indice=N_mobil[0];
	
		for(int i =0;i <indice;i++){
			if (tach[tacho[i]] < sat){ 
				if(estado[i] == ESTADOVIVO && edad[i] > tpupad){
					  if(edad[i]%tovip == 0){
 					  int iovip=ran2(&semilla)*4+7; 
   						for(int ik=0;ik < iovip;ik++){ 
 						estado[indice]=ESTADOVIVO;
 						edad[indice]=1;   
 						tacho[indice]=tacho[i]; 
	         			TdV[indice]=ran2(&semilla)*3+28; 
						manzana[indice]=manzana[i];
						int j=tacho[indice];
 						tach[j]=tach[j] + 1;
 						indice=indice + 1; 
   						}
  					   }    
				}  
   			} 
		}

		// actualiza el numero de bichos si no se sobrepasa el maximo
		N_mobil[0]=indice;

	};

	void muerte_x_vejez(int dia){
		int N=N_mobil[0];

        //for(int i=0;i < N ;i++){if (estado[i] == ESTADOVIVO && edad[i] >= TdV[i])estado[i]=ESTADOMUERTO;} 
		matar_viejos_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_N_mobil, dia);
		cudaDeviceSynchronize();		
	};

	void descacharrado(int dia,int tpupad,int descach){
	int N=N_mobil[0];

		if(dia%7 == 0 && dia > 150 && dia < 240){
	  		for(int itach=0;itach < descach;itach++){
	    		int ntach=ran2(&semilla)*ntachito;
			    descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_N_mobil,dia,ntach);
				/*for(int i=0;i < N;i++){
	  				if (estado[i] == ESTADOVIVO && edad[i] < tpupad && tacho[i] == ntach)estado[i]=ESTADOMUERTO;
				}*/
			}    	
	   	}	
	};


	void envejecer(){
	int N=N_mobil[0];
	
	  	//for(int i=0;i < N;i++){if (estado[i] == ESTADOVIVO) edad[i] = edad[i]+1;}
    	envejecer_kernel<<<(N + 256-1)/256,256>>>(raw_estado,raw_edad,raw_N_mobil);
    	cudaDeviceSynchronize();
	};
	
	// Numero de bichos vivos
	int vivos(int dia){

	    int N=N_mobil[0];
	    int poblacion=0;
	  	for(int i=0;i < N; i++){
			if (estado[i] == ESTADOVIVO)poblacion=poblacion + 1;}

		return poblacion;
	};

	/*void recalcularN(){
		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),TdV.begin(),manzana.begin()));

		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];
		thrust::sort_by_key(estado.begin(), estado.end(),zip_iterator);		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(), estado.end(), ESTADOMUERTO);
		N_mobil[0]= iter-estado.begin();//me da la longitud del vector
    
	};*/
	/*int vivos(int dia){

		return N_mobil[0];
	};*/
	
	// Imprime el estado completo de los bichos (solo para debuging)
	void imprimir(int dia){
		int N=N_mobil[0];
				
		std::cout << "#dia=" << dia << ", #N=" << N_mobil[0] << "\n";
		std::cout << "estado\tedad\tTdV\ttacho\tmanzana\n";
		for(int i=0;i<N;i++){
			std::cout << estado[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << tacho[i] << "\t" << manzana[i] << std::endl;
		}
		std::cout << "\n";
	};
};


int main(){

	ofstream outfile;
    outfile.open("Poblacion_total.dat");

	int descach=round(ntachito*prop);
    gpu_timer Reloj_GPU;
	Reloj_GPU.tic();

	bichos mosquitas(Ninicial);

	for(int dia=1;dia<=Ndias;dia++){
	
		int tovip=tiempo_entre_oviposiciones(dia);
		int tpupad=tiempo_pupas_adultas(dia);

		mosquitas.conteo_huevos(dia,tpupad);
		mosquitas.mortalidades_varias(dia,tpupad);
	    mosquitas.reproducir(dia,tpupad,tovip);
	    mosquitas.muerte_x_vejez(dia);
		mosquitas.descacharrado(dia,tpupad,descach);
		mosquitas.envejecer();
		int vivas=mosquitas.vivos(dia);
		outfile << dia << "\t" << vivas << endl;
		//mosquitas.envejecer();
		//mosquitas.recalcularN();
    }//cierro dias
    double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
    printf("Tiempo en GPU: %lf minutos\n",t);
// close the opened file.
outfile.close();
}
