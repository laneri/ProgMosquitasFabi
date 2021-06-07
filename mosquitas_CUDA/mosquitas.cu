//*************************************************************************************************************
//                              Versión serializada del programa de mosquita para una manzana 
//                                              versión 2021 
//                                        Autora Fabiana Laguna
//*************************************************************************************************************
/*En la naturaleza, cada oviposicion son aprox. 64 huevos y la mitad son hembras. En este código solamente se modela la dinámica de las hembras. Si se quisiera agregar a los machos, se multiplica por dos. Para ello se considera:

1- el número de huevos que deposita la hembra  entre 10 y 35 por oviposicion (distribució uniforme).
2- la mortalidad diaria de huevos, pupas, larvas y adultas independiente de la Temperatura.
3- el 83% de las larvas pasan a adultas jovenes, es decir, mueren con probab 0.17 al pasar del agua al aire
4- el nro. de oviposiciones por hembra depende de la temperatura: 
        - 0 ovip. a 18 grados (es decir, que en invierno las hembras no ponen huevos)
        - 3 o 4 ovip. a 25 grados 
        - 6 ovip. a 30 grados.
        
5- la mortalidad de la hembra adulta entre los 27 y los 32 dias (distribución uniforme).
6- la maduración de la pupa entre los 17 y 19 días (distribución uniforme).
7- la efectividad de la campaña publicitaria a través del vaciado de tachos en los días de mas calor.
8- "las 4 estaciones", discretizando la curva de temperatura que se tiene para buenos aires (extraído del trabajo de Otero):
        - el día 1 es 1/07 en este programa (<T>=18°C durante los 1ros 80 dias)
9- el descacharrado se hace solo en dic, enero y febrero (día 150 al 240)
10- la hibernación de los huevos en invierno.
11- la mortalidad de las hembras adultas entre 28 y 30 dias (distribución uniforme).
12- la saturación de los tachos, es decir, un número maximo de huevos permitidos por tacho
13- la transferencia de tacho, es decir, cuando un tacho satura, la hembra busca otro tacho para depositar sus huevos.

Las condiciones iniciales para cada agente mosquita tiene cuatro propiedades
    -1 o 0 (si está viva o muerta)
    -edad (avanza de a 1 dia)
    -cohorte (tacho en el que vive)
    -tiempo en el que se vuelve adulta
    -dias que va a vivir

//*************************************************************************************************************
//                               Versión paralelizada del programa de mosquitas extendido a N manzanas
//                                                      version (2021) 
//                                                   Autora Ana A. Gramajo
//*************************************************************************************************************
Se extiende el código serializado de Fabiana, a uno paralelizado ya que se agrega 

    - la manzana donde se encuentra el cohorte en el que vive la mosquita a las condiciones iniciales.
    - la espacialidad, considerando que mosquita puede cambiar de manzana a 1ros vecinos para depositar sus huevos cuando se satura su cohorte original.
    
En esta nueva versión del código se puede elegir la distribución  de los cohortes por manzana:
    - distribución uniforme
    - distribución de Poisson.    
    
Los parámetros del código se ingresan a través del archivo parametros.h    
---------------------------------------------------------------------
*/

#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#include <random>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "ran2.h"
#include <cmath>
#include <thrust/device_vector.h>
#include "gpu_timer.h"
#include "parametros.h"

#include<stdio.h>


int tiempo_entre_oviposiciones(int dia){
	int t;
	//defino tiempos de oviposición y maduración en función de la temperatura	
	if(dia >= 1 && dia < 80) t=TOVIP3;      //<T>=18 
	if(dia >= 80 && dia <= 140)t=TOVIP2b;   //<T>=23
  	if(dia > 140 || dia < 260) t=TOVIP1;    //<T>=30
 	if(dia >= 260 && dia <= 320)t=TOVIP2a;  //<T>=27
 	if(dia > 320) t=TOVIP3;                 //<T>=18

	return t;}

__global__ void kernel_reproducir(int *estado, int *edad, int *tacho,int *TdV, int *pupacion, int *manzana, int *N_mobil, int dia, int tovip, int *nacidos)
{                                   

    	int indice=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;
		/*Si la mosquita esta viva, esta en edad adulta, en tiempo de oviposicion y vive en un tacho disponible entonces*/
  		if(id < indice && edad[id] > pupacion[id] && edad[id]%tovip == 0) 
  		{
			RNG philox;         
			RNG::ctr_type c={{}};
			RNG::key_type k={{}};
			RNG::ctr_type r;
			k[0]=id; 
			c[1]=dia;
			c[0]=SEMILLAGLOBAL; 
			
			r = philox(c, k); 
			double azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)

			/*Si el tacho en el que nacio tiene lugar entonces*/
				/*Antes estaba asi y andaba...*/
			int tach=tacho[id];     //tach es un entero que me indica el numero de tacho en el que esta cada mosquita
				

			int iovip=10+ (azar*25); //iovip es el numero de huevos que pone cada mosquita
						
			atomicAdd(nacidos+tach,iovip); /*sumo iovip HUEVOS en la posicion del vector nacidos (que tiene NTACHOS elementos)
				nacidos[0+tach] en el puntero al primer elemento del vector nacidos desplazado en tach elementos
				el vector nacidos tiene el numero de nacidos en cada tacho Ej: nacidos[0]=numero de nacidos en el tacho 1*/
				

  		}//cierro loop de hilos de mosquitas VIVAS   
};

//mortalidades varias	
__global__ void matar_kernel(int *estado, int *edad, int *tacho,int *pupacion, int *TdV,int *N_mobil, int dia)
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
    	double azar;

     	azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
     	if (edad[id] < pupacion[id]){if(azar < MORACU)estado[id]=ESTADOMUERTO;}

     	azar=(u01_closed_open_32_53(r[0]));
     	if (edad[id] == pupacion[id]){if(azar < MORPUPAD)estado[id]=ESTADOMUERTO;}

     	azar=(u01_closed_open_32_53(r[0]));
		if (edad[id] > pupacion[id]){if(azar < MORAD)estado[id]=ESTADOMUERTO;}     	

        //matar viejos
		if (edad[id] >= TdV[id]){estado[id]=ESTADOMUERTO;}
	}
};

__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *pupacion,int *N_mobil,int dia,int ntach)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){// esta condicion es igual a decir que la mosquita estรก viva
	        if (edad[id] < pupacion[id] && tacho[id] == ntach){
	        estado[id]=ESTADOMUERTO; // se mata la mosquita
	        }
    	}
};

//elimine el estado[id]=ESTADOVIVO, ya que al final sólo quedan las mosquitas vivas
__global__ void envejecer_kernel(int *estado, int *edad,int *pupacion,int *N_mobil,int dia)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

        if(id<N){
            if(dia < 80 || dia > 320){
                if(edad[id] > pupacion[id])edad[id]++;} //ADULTAS
  		    else{
  		        edad[id]++;}
    	}
};

//elimine el estado[id]=ESTADOVIVO, ya que al final sólo quedan las mosquitas vivas
__global__ void delay_kernel(int *N_mobil,int *Tau,int dia)
{
    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

        if(id<N){
            if(Tau[id] > 0)Tau[id]=Tau[id] - 1; //si Tau[id]=nTau=delay, entonces Tau[id]=nTau -1
    	}
};


struct transferirdetacho{
	int m;
	int tpupad;
	int* ptr;
	int cuantos;
	int dia;

	transferirdetacho(int m_,int tpupad_,int *ptr_,int cuantos_, int dia_):
	m(m_),tpupad(tpupad_),ptr(ptr_),cuantos(cuantos_), dia(dia_){};

	__device__ 
	int operator()(thrust::tuple<int,int> tup, int counter){
		int tachoactual=thrust::get<0>(tup);
		int edad=thrust::get<1>(tup);
		
		int tachonuevo=tachoactual;
        
		if(tachoactual==m && edad>=tpupad)
		{
			/* sortear nuevo tacho de la misma manzana*/
			RNG philox;         
			RNG::ctr_type c={{}};
			RNG::key_type k={{}};
			RNG::ctr_type r;
			k[0]=counter; 
			c[1]=dia;
			c[0]=SEMILLAGLOBAL; 

			r = philox(c, k); 
			float azar=(u01_closed_closed_32_53(r[0]));//generador de números aleatorios en el device entre [0,1]
			int indicedetachoelegido=int(azar*cuantos);

			
			//tachonuevo=tachoactual; // este es solo para test

			if(indicedetachoelegido<cuantos){
			tachonuevo=ptr[indicedetachoelegido];
			}			
			        
		}
		return tachonuevo;
	}
};

// NUEVO: otro functorcito usado para las estadisticas desagregadas

// struct para generar randoms uniformes
struct uniformRanInt{
	int dia;
	int medio;
	int ancho;
	uniformRanInt(int medio_, int ancho_, int dia_):medio(medio_),ancho(ancho_),dia(dia_)
	{};

	__device__ int operator()(int i)
	{
	 	RNG philox;         
	    RNG::ctr_type c={{}};
	    RNG::key_type k={{}};
	    RNG::ctr_type r;
	    k[0]=i; 
	    c[1]=dia;
	    c[0]=SEMILLAGLOBAL; 

		r = philox(c, k); 
     	float azar=(u01_closed_open_32_53(r[0]));//numero aleatorios entre [0,1)
		return int(medio+ancho*azar);
	}
};	

// otro functorcito usado para las estadisticas desagregadas
struct acuaticoeneltacho{
	int m;
    int t;
	acuaticoeneltacho(int m_, int t_):m(m_),t(t_){};
    
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int tach=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (tach==m && edad<t);
	}
};

struct aereaeneltacho
{
	int m;
    int t;
	aereaeneltacho(int m_, int t_):m(m_),t(t_){};   
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int tach=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
		return (tach==m && edad>t);
	}
};


struct acuaticoeneltachoANA{
	int m;

	acuaticoeneltachoANA(int m_):m(m_){};
    
	__device__ bool operator()(thrust::tuple<int,int,int> tupla)
	{
        int tacho=thrust::get<0>(tupla);
        int edad=thrust::get<1>(tupla);
        int pupacion=thrust::get<2>(tupla);
		return (tacho==m && edad < pupacion);
	}
};

//functorcito para contar adultos en la población
struct poblacion_1{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int edad=thrust::get<0>(tupla);//NUEVO: cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
		return (edad >= pupacion);
	}
};

// functorcito para contar acuáticos en la población
struct poblacion_2{
	__device__ bool operator()(thrust::tuple<int,int> tupla)
	{
        int edad=thrust::get<0>(tupla); //NUEVO:cambie el orden de la tupla
        int pupacion=thrust::get<1>(tupla);
		return (edad < pupacion);
	}
};

struct bichos{

	//defino arrays grandes en device la  info de cada mosquita
	thrust::device_vector<int> estado;  //viva o muerta
	thrust::device_vector<int> edad;    //edad de la mosquita   
	thrust::device_vector<int> tacho;   // numero de tacho en que se encuentra cada mosquita   
	thrust::device_vector<int> TdV;     //tiempo de vida de cada mosquita
	thrust::device_vector<int> pupacion; //dia de paso de pupa a adulta de cada mosquita
	thrust::device_vector<int> manzana; //nro. de manzana de cada mosquita

	thrust::device_vector<int> Tau;     //array para almacenar la disponibilidad de los tachos  

	// arrays medianos en device, numero_tachos elementos
	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho

	// arrays medianos en host, numero_manzanas elementos
	std::vector<std::vector<int> > tachos_por_manzana; //tachos_por_manzana[i]=vector de tachos de manzana i 
	std::vector<std::vector<int> > disponibilidad_de_tachos_por_manzana; //tachos_por_manzana[i]=vector de tachos disponibles de manzana i 

	// numero de tachos elementos
	std::vector<int> manzana_del_tacho;
	std::vector<int> disponibilidad_del_tacho;
	

	//array para almacenar nro de tachos por manzana
    thrust::device_vector<int> NroTachos;
	
	// array de un elemento = device variable
	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)

	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_TdV;
	int *raw_pupacion;
	int *raw_N_mobil;
	int *raw_manzana;
	int *raw_nacidos;

	int *raw_Tau;
	
	//constructor	
	bichos(int N_,long *semilla){

		// alocamos el maximo posible
		estado.resize(MAXIMONUMEROBICHOS);	
		tacho.resize(MAXIMONUMEROBICHOS);	
		edad.resize(MAXIMONUMEROBICHOS);
		pupacion.resize(MAXIMONUMEROBICHOS);
		TdV.resize(MAXIMONUMEROBICHOS);
		manzana.resize(MAXIMONUMEROBICHOS);

		Tau.resize(MAXIMONUMEROBICHOS);

		N_mobil.resize(1);
		
		tachos_por_manzana.resize(NUMEROMANZANAS); 
        disponibilidad_de_tachos_por_manzana.resize(NUMEROMANZANAS); 

		manzana_del_tacho.resize(MAXIMONUMEROBICHOS);
		disponibilidad_del_tacho.resize(MAXIMONUMEROBICHOS);

		NroTachos.resize(NUMEROMANZANAS);

		// nacidos en cada tacho, inicialmente 0
		nacidos.resize(NUMEROTACHOS);
		thrust::fill(nacidos.begin(),nacidos.end(),0);

		thrust::fill(estado.begin(),estado.end(),0);
		thrust::fill(edad.begin(),edad.end(),0);
		thrust::fill(tacho.begin(),tacho.end(),0);
		thrust::fill(pupacion.begin(),pupacion.end(),0);
		thrust::fill(TdV.begin(),TdV.end(),0);
		thrust::fill(manzana.begin(),manzana.end(),0);

		thrust::fill(Tau.begin(),Tau.end(),0);
		
		// inicializacion raw pointers
		raw_edad=thrust::raw_pointer_cast(edad.data());
		raw_tacho=thrust::raw_pointer_cast(tacho.data());
		raw_estado=thrust::raw_pointer_cast(estado.data());
		raw_TdV=thrust::raw_pointer_cast(TdV.data());
		raw_manzana=thrust::raw_pointer_cast(manzana.data());

		raw_Tau=thrust::raw_pointer_cast(Tau.data());

		raw_N_mobil=thrust::raw_pointer_cast(N_mobil.data());
		raw_pupacion=thrust::raw_pointer_cast(pupacion.data());
		raw_nacidos=thrust::raw_pointer_cast(nacidos.data());

        //para considerar una distribucion de Poisson de los tachos
        /*std::default_random_engine generator;
        std::poisson_distribution<int> distribution(65);

	    const int nrolls = 100000; // number of experiments
  	    const int nstars = 100;   // maximum number of stars to distribute

	    for (int i=0; i<nrolls; ++i) {
	    int number = distribution(generator);
	    if (number< N_) ++tacho[number];
	    }
	    */
	    
		//std::cout<<"VoM\ttacho\tedad\tTdV\ttpupad\tmanzana" << std::endl;

//*************************************************************************************************************
//                       condiciones iniciales para N_=NINICIAL ingresado en el archivo parametros.h
//*************************************************************************************************************
		std::cout << "*********************************************************************************************" << "\n";
		std::cout << "*************************  condiciones iniciales ************************************" << "\n";
		std::cout << "*********************************************************************************************" << "\n";
		std::cout << "indice i" <<"\ttachos[i] "<< "\tTau[i]"<<  "\t\tmanzana[i]  " << "\n";

		for(int i=0;i < N_;i++){
		    
    		tacho[i] = i;			 // tacho en el que se encuentra la mosquita
		    manzana_del_tacho[tacho[i]]=int (i/5);                              //5 tachos por manzana
    		//manzana_del_tacho[tacho[i]]=int(ran2(semilla)*NUMEROMANZANAS);     //le asigno al tacho una manzana al azar
		    manzana[i]=manzana_del_tacho[tacho[i]];                             //manzana en la que está el tacho i
		    tachos_por_manzana[manzana[i]].push_back(tacho[i]);                 //para tachos tengo en la manzana

		    disponibilidad_del_tacho[tacho[i]]=0;                               //disponibilidad del tacho i
		    Tau[i] = disponibilidad_del_tacho[tacho[i]];                        // 0 para disponible, distinto de 0 para no disponible
    		disponibilidad_de_tachos_por_manzana[manzana[i]].push_back(Tau[i]); //para tachos disponibles x manzana

		    //int tachosxmanzana=tachos_por_manzana[manzana[i]].size();           //nro de tachos x manzana
			    //if(tachosxmanzana <=9){                                        //pongo hasta 9 tachos por manzana
			    estado[i] = ESTADOVIVO; 	      		                        //todas vivas inicialmente
			    edad[i] = ran2(semilla)*7+19; 	                                //todas adultas al principio 
			    pupacion[i] = TPUPAD-2+(ran2(semilla)*5);                       //dia de pupacion (entre los 15 y 19 dias)
			    TdV[i] = ran2(semilla)*6+27 ;	                                //tiempo de vida de 27 a 32
	    		std::cout << i << "\t\t" <<tacho[i] << "\t\t" << Tau[i] << "\t\t" << manzana[i] << "\n";
			    //}
		}

		std::cout << "*********************************************************************************************" << "\n";
		std::cout << "  verificacion del llenado de tachos_por_manzana y de la disponibilidad_de_tachos por_manzana" << "\n";
		std::cout << "*********************************************************************************************" << "\n";

		// una verificacion del llenado de tachos_por_manzana
		int nmanzanas=tachos_por_manzana.size();
		for(int i=0;i<nmanzanas;i++){
			std::cout << "\n\n manzana " << i << "\ntachos " << "disponibilidad " << "\n";
			int ntachos=tachos_por_manzana[i].size();
			int contTach=0;
			for(int j=0;j<ntachos;j++){
				std::cout << (tachos_por_manzana[i])[j] << "\t " << (disponibilidad_de_tachos_por_manzana[i])[j] << "\n";//tacho que se encuentra en la manzana i
				contTach++;
				NroTachos[i]=contTach;                                                  //contador para contar nro de tachos por manzana
			}
			std::cout << "\nNro de tachos en la manzana\t" << contTach << "\n";
			std::cout << std::endl;
			std::cout << "-----------------------------------------------------------------------------------------------------" << "\n";			}
		
		std::cout << "inicializacion lista" << std::endl;
		N_mobil[0]=N_;
	};	

	// devuelve un numero de tacho random de la manzana m
	int nuevo_tacho_misma_manzana(int m){
		int ntachos=tachos_por_manzana[m].size();
		int r=int((rand()*1.0/RAND_MAX)*ntachos);		
		int nuevo_tacho=(tachos_por_manzana[m])[r];
		return nuevo_tacho;
	}
	
	// devuelve numero de manzana sorteada entre las cuatro manzanas vecinas de la manzana m
		int sorteo_manzana_vecina(int manz){
		// numero de las manzanas vecinas de una manzana
		std::vector<int> manzanas_vecinas(5);
				
		int x=manz%LADO;
		int y=int(manz/LADO);
		manzanas_vecinas[0]=(x-1+LADO)%LADO+LADO*y; //izqu
		manzanas_vecinas[1]=(x+1+LADO)%LADO+LADO*y; //derecha
		manzanas_vecinas[2]=LADO*((y-1+LADO)%LADO)+x; //abajo
		manzanas_vecinas[3]=LADO*((y+1+LADO)%LADO)+x; //arriba
		manzanas_vecinas[4]=manz; //centro
		
		int nvecinos=6; //cuento los vecinos y el centro
		int r=int((rand()*1.0/RAND_MAX)*nvecinos); //numero aleatorio entre 0 y 6
		//si sale 5 o 6 elijo el centro (es una forma trucha de darle mas probabilidad al centro)
		int manzana_sorteada;
		if(r>3)
			{
			manzana_sorteada=manz;
			}
			else{		
			manzana_sorteada=manzanas_vecinas[r];
		}
		return manzana_sorteada;
	}


	void mortalidades(int dia){

	int N=N_mobil[0];
	//mortalidades varias y muerte por vejez
	    matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_pupacion,raw_TdV,raw_N_mobil, dia);	
		cudaDeviceSynchronize();
	};

	void descacharrado(int dia,float *E,long *semilla){
	int N=N_mobil[0];
	
    	//int azar=1 + ran2(semilla)*14; //nro al azar entre [1,14]
		//if(dia%azar == 0 && dia > 120 && dia < 320){//para un vaciado de tachos entre 1 y 13 días

  		if(dia%7 == 0 && dia > 120 && dia < 320){//para un vaciado de tachos cada 7 días
  		
        if(dia==140){
  		std::cout << "*****************************************************************************" << "\n";
		std::cout << " Descacharrado para el día= "<< dia << "\n";
 		std::cout << "*****************************************************************************" << "\n";}

 		int nmanzanas=tachos_por_manzana.size(); //nro de manzanas
 		
   		    for(int i=0;i < nmanzanas;i++){

   		    int NroDescach=round(NroTachos[i]*E[i]);

            //chequeo
		    if(dia==140){
		    std::cout << "manzana: " << i << "\n";
		    std::cout << "indice sorteado |" << " tacho que se vacia "<<"\t|disponibilidad"<< "\n";}   

		    int ntachos=tachos_por_manzana[i].size();       //nro de tachos en cada manzana i
   			    for(int itach=0;itach < NroDescach;itach++){
   			        
   			    	int n=ran2(semilla)*ntachos;            //indice del tacho que se va a eliminar
   		 	    	int ntach=(tachos_por_manzana[i])[n];   //tacho que se elimina 
   		 	    	int nTau= 10;// + ran2(semilla)*30;      //delay de nTau días para la disponibilidad de ese tacho que se elimina

                    (disponibilidad_de_tachos_por_manzana[i])[n]=nTau;
   		 	    	disponibilidad_del_tacho[ntach]=(disponibilidad_de_tachos_por_manzana[i])[n];
   		 	    	Tau[ntach]=disponibilidad_del_tacho[ntach];
   		 	    	
   		 	    	//chequeo
                    if(dia==140){std::cout << "\t" << n << "\t|\t" << ntach << " \t\t|\t" << Tau[ntach]<<"\n";}
 
  				    descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado, raw_edad, raw_tacho, raw_pupacion, raw_N_mobil,dia,ntach);
  			        cudaDeviceSynchronize();
			    }//cierro for para eliminar los tachos
   		    }//cierro for para las manzanas
   		}//cierro if
   	};

    //nacimientos
	void reproducir(int dia,int tovip)
	{
	    
		int indice=N_mobil[0];
		//if(indice==0) {
		//	std::cout << "NO HAY MAS MOSQUITAS PARA REPRODUCIRSE" << std::endl; 	
		//exit(1);
		//}else{

		//nacimientos
		//antes de reproducir reinicializo en cero los nacidos en el paso anterior que ahora ya no son mas nacidos porque crecieron 
		thrust::fill(nacidos.begin(),nacidos.end(),0);

		// reproduce primero y luego ve si los tachos no están saturados para poner los huevos
		kernel_reproducir<<<(indice+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_pupacion,raw_manzana,raw_N_mobil,dia,tovip,raw_nacidos);
		cudaDeviceSynchronize();

			// despues de reproducir agrego todos los nacidos al final del array original, tacho a tacho
			int index=indice;
			for(int m=0;m<NUMEROTACHOS;m++){
                //si el tacho m está disponible
                
                if(Tau[m]==0){//NUEVO: Si el tacho m está disponible
                
				//calculo el número de acuaticos en cada tacho
				int antiguos=thrust::count_if(
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
					acuaticoeneltacho(m,TPUPAD)
				);

				//los nuevos vienen del kernel reproducir  
				int nuevos=nacidos[m];
				    //Ahora bien, si con los nuevos supero el maximo de huevos por tacho (SAT)
				    if(nuevos+antiguos>SAT){
				    nuevos=SAT-antiguos;	//ponen lo que pueden en el mismo tacho
					
				    /*NUEVO Para transferir de tacho*/
				    /*Muevo LOS ADULTOS a otro tacho de la misma manzana o de una manzana vecina*/
					int estamanzana = manzana_del_tacho[m];
					int manzanadeltacho = sorteo_manzana_vecina(estamanzana); //pone en la misma manzana o en una vecina
					//cuantos tachos tengo en la manzana? 
					int cuantos=(tachos_por_manzana[manzanadeltacho]).size();

					int* ptr_h=(tachos_por_manzana[manzanadeltacho]).data();

					thrust::device_vector<int> tachosDeLaManzana(cuantos);
					
				    	for(int k=0;k<cuantos;k++){
						tachosDeLaManzana[k]=ptr_h[k];
					    }					
					int* ptr_d=thrust::raw_pointer_cast(tachosDeLaManzana.data());
										
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
						thrust::make_zip_iterator(thrust::make_tuple(tacho.begin()+indice,edad.begin()+indice)),
						thrust::make_counting_iterator(0),
						tacho.begin(),
						transferirdetacho(m,TPUPAD,ptr_d,cuantos, dia)
					);

					//Una vez que se llenaron los tachos de la manzana, pone en las manzanas vecinas.


				    }//cierro if para transferencia de tacho
				/*HASTA ACÁ MAYOR PROBABILIDAD DE TRANSFERENCIA DE TACHO EN LA MISMA MANZANA Y MENOR PORB. DE TRANSFERENCIA DE MANZANA y TACHO */

				thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,ESTADOVIVO);	//nacen todas vivas       
				thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,1);		        //nacen con edad(dias) 0      
				thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	        //nacen en el tacho m

                thrust::fill(Tau.begin()+index,Tau.begin()+index+nuevos,disponibilidad_del_tacho[m]); //nacen en un tacho disponible

				// index en counting iteraror necesario para distintos randoms en cada tacho
				thrust::transform(
					thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),
					pupacion.begin()+index,uniformRanInt(15,5,dia)
				);
			
				
				thrust::transform(
					thrust::make_counting_iterator(index),thrust::make_counting_iterator(index+nuevos),
					TdV.begin()+index,uniformRanInt(27,6,dia)
				);

				thrust::fill(manzana.begin()+index,manzana.begin()+index+nuevos,manzana_del_tacho[m]);         //NUEVO
				index+=nuevos;		//actualizo el indice para me marque siempre en la ultima mosquita que nacio           
				}//cierro if para disponibilidd de tacho
			}//cierro for
		
		    // actualiza el indice movil hasta el ultimo bicho vivo
			if(index<MAXIMONUMEROBICHOS) {
				N_mobil[0]=index;
			}
			else{ ////satura la memoria reservada salgo del prog
				std::cout << "Demasiados Bichos!" << std::endl;
				exit(1);
			}	
				
	//	}//cierro else linea 523	
		
	};
	
    //Recalcular -> eliminar muertos y dejar vivos
    void recalcularN(){

		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),pupacion.begin(),TdV.begin(),manzana.begin()));
		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];
		thrust::sort_by_key(estado.begin(), estado.begin() + N,zip_iterator);		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(),estado.begin() + N, ESTADOMUERTO);
		N_mobil[0]= iter-estado.begin();//me da la longitud del vector
		//std::cout << "N_mobil " << N_mobil[0] <<std::endl;
	};
	
	//población total adultos + acuáticos
	int vivos(int dia){

	int N=N_mobil[0];

    int poblacion = thrust::count(estado.begin(), estado.begin() + N, ESTADOVIVO);
	return poblacion;
	};

    //población de acuáticos
	int acuaticos(int dia){

	int N=N_mobil[0];
	
    int ac= thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_2()
            );

		return ac;
	};

    //población de adultos
	int adultos(int dia){

	int N=N_mobil[0];
	//el predicado poblacion_1()corresponde a adultos
	    int ad=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),pupacion.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(edad.begin() +  N,pupacion.begin() + N)),
                poblacion_1()
            );

		return ad;
	};	

	//envejecer población
   	void envejecer(int dia){
	int N=N_mobil[0];
        envejecer_kernel<<<(N + 256-1)/256,256>>>(raw_estado,raw_edad,raw_pupacion,raw_N_mobil,dia);
        cudaDeviceSynchronize();
	}; 
	
	//disminuir el delay=nTau en 1 día para que el tacho eliminado vuelva a estar disponible
   	void delay(int dia){
	int N=N_mobil[0];
        delay_kernel<<<(N + 256-1)/256,256>>>(raw_N_mobil,raw_Tau,dia);
        cudaDeviceSynchronize();
        
        if(dia==140){std::cout << "chequeo de la disminución en 1 en el delay de disponibilidad de los tachos" << std::endl;}
        //chequeo 
        for(int i=0;i < N;i++){
			if(dia==140 && Tau[i]>0){
		        std::cout << "dia: " << dia <<"\t" << Tau[i] << std::endl;}
		}
		
		if(dia==140){
		std::cout << "*********************************************************************************************" << "\n";
		std::cout << "  verificacion del llenado de tachos_por_manzana y de la disponibilidad_de_tachos por_manzana" << "\n";
		std::cout << "*********************************************************************************************" << "\n";
		// una verificacion del llenado de tachos_por_manzana
		int nmanzanas=tachos_por_manzana.size();
		for(int i=0;i<nmanzanas;i++){
			int ntachos=tachos_por_manzana[i].size();
			for(int j=0;j<ntachos;j++){
				std::cout << (tachos_por_manzana[i])[j] << "\t " << (disponibilidad_de_tachos_por_manzana[i])[j] << "\n";//tacho que se encuentra en la manzana i
			}
			std::cout << std::endl;
			std::cout << "-----------------------------------------------------------------------------------------------------" << "\n";			}}
		
	}; 
	
};

int main(){

	FILE* archivo=NULL;
	char miarch[50];
	
    //alocamos memoria para el vector que almacena la efectividad por manzana
    float *E;//array cuyos elementos esla efectividad de propaganda en cada manzana
    E= (float *)malloc((NUMEROMANZANAS)*sizeof(float));    

    //para un descacharrado fijo en cada manzana
    for(int j=0;j<NUMEROMANZANAS;j++){E[j]=PROP;}

    //alocamos memoria para los vectores donde almaceno el nro de mosquitas por dia, y la suma de todas las poblaciones
    int *Poblacion;
    Poblacion= (int *)malloc((NDIAS+1)*sizeof(int));
    for(int i=1;i<=NDIAS;i++){Poblacion[i]=0;}



    
    //loop para el número de corridas con distinta semilla
    for(int seed=0;seed<NREALIZACIONES;seed++){
    std::cout << "nro de realizacion: "<< seed << "\n";

    long semilla=(long )time(NULL);    //incializamos semilla

    //para un descacharrado distinto en casa manzana, lo pongo dentro del loop para que varíe con la semilla
        //for(int j=0;j<NUMEROMANZANAS;j++){E[j]=0.4 + ran2(&semilla)*0.5;}
    
    gpu_timer Reloj_GPU;
    Reloj_GPU.tic();
    
    //inicializo
    bichos mosquitas(NINICIAL,&semilla);

    double treprod=0;
    double trecalc=0;
    double tdescacha=0;
    
        //calculo la población en cada iteración
	    for(int dia = 1; dia <= NDIAS; dia++){
	    //std::cout << "DIA" << dia << std::endl;
        int tovip=tiempo_entre_oviposiciones(dia);
	
	    //std::cout << "matar" << std::endl;
	    mosquitas.mortalidades(dia);
	    
	    gpu_timer Reloj_descacharrar;
	    Reloj_descacharrar.tic();
	    //std::cout << "descacharrar" << std::endl;
	    mosquitas.descacharrado(dia,E,&semilla); 
	    tdescacha= tdescacha+Reloj_descacharrar.tac()/60000; //de milisegundos -> minutos

	    gpu_timer Reloj_reproducir;
	    Reloj_reproducir.tic();
	    //std::cout << "reproducir" << std::endl;
	    mosquitas.reproducir(dia,tovip);
	    treprod= treprod+Reloj_reproducir.tac()/60000; //de milisegundos -> minutos
   

	    gpu_timer Reloj_recalcular;
	    Reloj_recalcular.tic();
	    //std::cout << "recalcular indice de mosquitas vivas" << std::endl;
	    mosquitas.recalcularN(); 
	    trecalc= trecalc+Reloj_recalcular.tac()/60000; //de milisegundos -> minutos
	
        //en esta versión del programa solo considero hembras
	    int vivas=mosquitas.vivos(dia);
	    int adultas=mosquitas.adultos(dia);
	    int acuaticas=mosquitas.acuaticos(dia);

	    //std::cout << "envejecer poblacion" << std::endl;
	    mosquitas.envejecer(dia);
	    mosquitas.delay(dia);
	    Poblacion[dia]= vivas;//guardo en un vector el nro de mosquitas para una determinada semilla
	    }//cierro loop para dias

       //tiempos de cáculo en GPU***************************************************		
       double t=Reloj_GPU.tac()/60000; //de milisegundos -> minutos
       printf("Tiempo de cálculo total para Población total de mosquitas en GPU: %lf minutos\n",t);
       printf("Tiempo en descacharrar: %lf minutos\n",tdescacha);
       printf("Tiempo en reproducir: %lf minutos\n",treprod);
       printf("Tiempo en recalcular: %lf minutos\n",trecalc);	
       //****************************************************************************
       std::cout << "\n";
        //miarch es un string de caracteres donde guardo el nombre del archivo cambiando la semilla con cada iteracion
    
	sprintf(miarch,"POBLACION_%d.txt",seed);
    	archivo=fopen(miarch,"w");
    	    for (int i=1;i<=NDIAS;i++){
		    fprintf(archivo,"%d\t%d \n",i,Poblacion[i]);}
	fclose(archivo); 
    }//cierro loop para el número de ITERACIONES
    
return 0;		

}// end for main
