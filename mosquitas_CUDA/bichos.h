#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
#include <cmath>
#include "parametros.h"
typedef r123::Philox2x32 RNG; // particular counter-based RNG

#define SEMILLAGLOBAL	123456789

//defino la temperatura según la estación y el tiempo entre oviposiciones y de maduración de acuáticos	

__host__ __device__	int tiempo_entre_oviposiciones(int dia){
	int tovip;
	if(dia < 80 || dia > 320){tovip=tovip3;}	//<T>=18
  	if(dia >= 140 && dia <= 260){tovip=tovip1;}	//<T>=30
  	if(dia >= 80 && dia < 140){tovip=tovip2b;}	//<T>=23
	if(dia > 260 && dia <= 320){tovip=tovip2a;}	//<T>=23

	return tovip;}

__host__ __device__	int tiempo_pupas_adultas(int dia){
	int tpupad;
	if(dia < 80 || dia > 320){tpupad=tpupad3;}//pupas se vuelven adultas a los 17 dias en invierno
	if(dia >= 140 && dia <= 260){tpupad=tpupad1;}//pupas se vuelven adultas a los 9 días en verano (desde oviposicion)
  	if(dia >= 80 && dia < 140){tpupad=tpupad2;}//pupas se vuelven adultas a los 13 dias en otoño y primavera
	if(dia > 260 && dia <= 320){tpupad=tpupad2;}

	return tpupad;}	
	
// El kernel de reproduccion
// Cada nacido va a parar a un array contador, "nacidos", que cuenta los nuevos por tacho 
// Todavia no se agregan al array de bichos, eso se hace luego.
__global__ void reproducir_kernel(int *estado, int *edad, int *tacho, int *N_mobil, int *nacidos, int dia, int tpupad, int *resto){

	int N=N_mobil[0];	                          //número de mosquitas
	int id = blockIdx.x*blockDim.x + threadIdx.x; 	  //el índice comienza desde cero

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
   	    int tach=tacho[id];	
	    int eggs=azar*3+7;

    		//si la mosquita está viva, se vuelve adulta y resto=0
	        if(estado[id] == ESTADOVIVO && edad[id] > tpupad && resto[id]==0){
        	atomicAdd(nacidos+tach,eggs);}         //la mosquita pone entre 7 y 10 huevos en el tacho
	}
};


// kernel asesino: mata mosquitas con cierta probabilidad
// Idea: cada hilo decide a que bicho va a matar, cambiando su estado 0->1.
// No se actualiza aun el numero de bichos, para eso hace falta reordenar y sacar los muertos
__global__ void matar_kernel(int *estado, int *edad, int *tacho, int *N_mobil,int tpupad, int dia)
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

		if (estado[id] == ESTADOVIVO && edad[id] < tpupad){if(azar < moracu)estado[id]=ESTADOMUERTO;}
		if (estado[id] == ESTADOVIVO && edad[id] == tpupad){if(azar < morpupad)estado[id]=ESTADOMUERTO;}
		if (estado[id] == ESTADOVIVO && edad[id] > tpupad){if(azar < morad)estado[id]=ESTADOMUERTO;} 
 
	}
};


//muerte de las mosquitas por vejez
__global__ void matar_viejos_kernel(int *estado, int *edad, int *tacho, int *TdV,int *N_mobil,int dia)
{

	    int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){                       
	    	if(estado[id]==ESTADOVIVO && edad[id]>=TdV[id])estado[id]=ESTADOMUERTO;
	    }
};

//muerte de los acuáticos ( < 17 días) por descacharrado de los tachos

__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *N_mobil,int tpupad,int dia, int ntach)
{

    	int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){	
  		    if (estado[id]==ESTADOVIVO && edad[id] < tpupad && tacho[id] == ntach)estado[id]=ESTADOMUERTO;    	
    	}
};

// un functorcito usado para las estadisticas desagregadas
struct iguala{
	int m;
	iguala(int m_):m(m_){};

	__device__ bool operator()(int man)
	{
		return man==m;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////
// Clase bichos: toda la info sobre todos los bichos, y sus funciones
struct bichos{

	thrust::device_vector<int> estado;  // Vivo o Muerto 0/1
	thrust::device_vector<int> edad;    // edad de la mosquita
	thrust::device_vector<int> tacho;   // tacho donde vive la mosquita
	thrust::device_vector<int> nacidos; // nacidos por tacho
	thrust::device_vector<int> TdV;     // tiempo de vida de la mosquita
	thrust::device_vector<int> manzana; // manzana donde se encuentra el tacho
	
	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)
	
	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_N_mobil;
	int *raw_nacidos;
	int *raw_TdV;
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

		// un entero en device
		N_mobil.resize(1);
		
		// nacidos en cada tacho, inicialmente 0
		nacidos.resize(NUMEROTACHOS);
		fill(nacidos.begin(),nacidos.end(),0);
	
		// numero de vivos inicial
		N_mobil[0]=N_;

		// inicializacion raw pointers
		raw_edad=thrust::raw_pointer_cast(edad.data());
		raw_tacho=thrust::raw_pointer_cast(tacho.data());
		raw_estado=thrust::raw_pointer_cast(estado.data());
		raw_TdV=thrust::raw_pointer_cast(TdV.data());
		raw_manzana=thrust::raw_pointer_cast(manzana.data());

		raw_N_mobil=thrust::raw_pointer_cast(N_mobil.data());
		raw_nacidos=thrust::raw_pointer_cast(nacidos.data());
		
		// inicializacion totalmente random: 5 tachos por manzana, una mosquita por tacho

		for(int i=0;i<N_;i++){
			edad[i]= rand()%5 + 12; // edad de la mosquita i entre 12 y 16 dias
			tacho[i]=i;		// tacho en la que se encuentra la mosquita i 
			estado[i]=0; 		// todas las mosquitas vivas inicialmente
			TdV[i]=rand()%3 + 28;	// tiempo de vida de la mosquita i entre 28 y 30 dias
			int j=tacho[i];
			manzana[i]=j;		// manzana donde se encuentra el tacho[i]
		}//cierro loop para i		

	};//cierro costructor

	// recorre los bichos y los mata con cierta probabilidad funcion de la edad, actualizando estados
	//mortalidades varias
	void matar(int dia){
		int N=N_mobil[0];

		int tpupad=tiempo_pupas_adultas(dia);

		matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil,tpupad, dia);		
	};

	// recorre los bichos, calcula el numero de nacidos por tacho
	void reproducir(int dia){
	    	int N=N_mobil[0];

    	    	int tpupad=tiempo_pupas_adultas(dia);
	    	int tovip=tiempo_entre_oviposiciones(dia);

		//declaro dos arrays de igual dimension, para almacenar el resto y el otro cuyos elementos es tovip.	    
	    	thrust::device_vector<int> resto(MAXIMONUMEROBICHOS);    
            	thrust::device_vector<int> t(MAXIMONUMEROBICHOS);       
	    
            	// fill t with tovip
            	thrust::fill(t.begin(), t.end(), tovip);

            	// compute resto = edad[i]%t[i] 
            	thrust::transform(edad.begin(), edad.end(), t.begin(), resto.begin(), thrust::modulus<int>());

		auto zip_iteratorR=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),estado.begin()));

		thrust::sort_by_key(resto.begin(), resto.begin()+N,zip_iteratorR);// ordena los elementos del array "resto" de menor a mayor (de 0 a N)		
	      
       	   	/*for(int i=0;i<N;i++){
            		if(dia==70){std::cout << estado[i] << "\t" << edad[i] << "\t" << tpupad << "\t" << tovip << "\t" << resto[i] << std::endl;}//bien
            	}*/

	    	int *raw_resto=thrust::raw_pointer_cast(resto.data());
	    
        	// reproduce, calculando nacidos por tacho
		reproducir_kernel<<<(N+256-1)/256,256>>>
		(raw_estado,raw_edad,raw_tacho,raw_N_mobil, raw_nacidos, dia,tpupad,raw_resto);
         
		cudaDeviceSynchronize();

		// agrega todos los nacidos al final del array original, tacho a tacho
		int index=N;
		for(int m=0;m<NUMEROTACHOS;m++){
			//std::cout << "listo nacidos " << std::endl; 
			int nuevos=nacidos[m];
			if(tacho[m]+nacidos[m] > sat)nuevos=sat;            			//cond. saturacion (800)
			thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,0);	//todos vivos(0)	
			thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,0);		//todos nacen el dia 0	
			thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	//en el tacho m
			thrust::fill(TdV.begin()+index,TdV.begin()+index+nuevos,rand()%3+28); 	//con tiempo de vida 28 a 30 
			index+=nuevos;	
			//  if(dia==15){std::cout << nuevos << "\t" << index << "\t"<< tacho[m] << std::endl;}
		
		}

		// actualiza el numero de bichos si no se sobrepasa el maximo
		if(index<MAXIMONUMEROBICHOS) {
			N_mobil[0]=index; //actualizo el número de bichos
		}
		// caso contrario satura al maximo
		else{
			std::cout << "Demasiados Bichos!" << std::endl;
			 N_mobil[0]=MAXIMONUMEROBICHOS-1;	
		}
	   
	};


	//mortalidades por llegar a viejas 
	void matar_viejos(int dia){
		int N=N_mobil[0];
		matar_viejos_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_N_mobil,dia);		
	};

	//mortalidades por llegar a viejas 
	void descacharrar_tacho(int dia){
		int N=N_mobil[0];
		int tpupad=tiempo_pupas_adultas(dia);
		int descach=std::round(NUMEROTACHOS*0.5);//defino el nro. de tachos a descacharrar

		if(dia%7 == 0 && dia > 150 && dia < 240){
  			for(int itach=0;itach < descach;itach++){
    			int ntach=rand()%NUMEROTACHOS;	//sorteo cuáles son esos tachos y se los paso al kernel
			descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil,tpupad, dia,ntach);}} 
	};

	// reordenando para poner los muertos al fondo, podemos calcular el numero de vivos, y actualizar N_mobil
	void recalcularN(){
		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin()));

		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];

		thrust::sort_by_key(estado.begin(), estado.begin()+N,zip_iterator);// ordena de menor a mayor (de 0 a 1), los vivos primeros(0) y los muertos al final del array (1)		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(), estado.begin()+N, ESTADOMUERTO);//encuentra en el array el estado 1
		N_mobil[0]= iter-estado.begin();//lo quita del array
	};

	// Envejezco a toda la población un día (aumento en 1): recorre los bichos y les aumenta la edad en una unidad
	void envejecer(){
		int N=N_mobil[0];
		using namespace thrust::placeholders;
		thrust::transform(edad.begin(),edad.begin()+N, edad.begin(),_1+1);
	};

	// Imprime el estado completo de los bichos (solo para debuging)
	void imprimir(int dia){
		int N=N_mobil[0];

    	int tpupad=tiempo_pupas_adultas(dia);
	    int tovip=tiempo_entre_oviposiciones(dia);
	    
		std::cout << "#dia=" << dia << ", #N=" << N_mobil[0] <<  ", tovip=" << tovip <<  ", tpupad=" << tpupad << "\n";
		/*std::cout << "m\testado\tedad\ttacho\tTdV\n";
		int m=0;
		for(int i=0;i<N;i++){
			std::cout << m << "\t" << estado[i] << "\t" << edad[i] << "\t" << tacho[i] << "\t" << TdV[i] << std::endl;   
		m++;	
		}
		std::cout << m << std::endl;*/
		std::cout << "\n" << std::endl;
	};

	// Estadisticas de distinto tipo sobre el array de bichos
	void imprimir_estadisticas(){
		int N=N_mobil[0];
		
		std::cout << "hay " << N << " mosquitas en total" << std::endl;
		
		for(int m=0;m<NUMEROTACHOS;m++){
			std::cout << "hay " << 
			thrust::count_if(tacho.begin(),tacho.begin()+N,iguala(m))
			<< " en el tacho " << m << std::endl;				
		}
		std::cout << "\n" << std::endl;

		for(int i=0;i<MAXIMAEDAD;i++){
			std::cout << "hay " << 
			thrust::count_if(edad.begin(),edad.begin()+N,iguala(i))
			<< " con edad " << i << std::endl;				
		}

		std::cout << "\n" << std::endl;

		std::cout << "hay " << NUMEROTACHOS << " tachos en total" << std::endl;

		for(int i=0;i<NUMEROMANZANAS;i++){
			std::cout << "hay " << 
			thrust::count_if(manzana.begin(),manzana.begin()+NUMEROTACHOS,iguala(i))
			<< " en la manzana " << i << std::endl;				
		}


	}

	// Numero de bichos vivos
	int vivos(int dia){
	    return N_mobil[0];
    };

	void avanza_dia(int dia){
		matar(dia);			        //mortalidades varias
		reproducir(dia); 		    //nacimientos - descomento y se pincha todo
		matar_viejos(dia);		    //mortalidad por vejez
		descacharrar_tacho(dia);	//descacharrar tacho
		envejecer();			    //envejecer un dia la poblacion
		recalcularN();
	};

};



