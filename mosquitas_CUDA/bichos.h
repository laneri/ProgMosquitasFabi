#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]

#include <cmath>
#include "parametros.h"


typedef r123::Philox2x32 RNG; // particular counter-based RNG

#include "ran2.h"

// macros utiles para CPU y GPU, y hacer desaparecer numeros magicos
#define Ninicial		20
#define NUMEROTACHOS		20
#define NUMEROMANZANAS		4
#define MAXIMONUMEROBICHOS	8000
#define NUMERODEESTADOS		2
#define NUMERODEHUEVOS		2
#define MAXIMAEDAD		10

#define ESTADOMUERTO		1
#define ESTADOVIVO		0
#define Ndias			400

#define morhue 			0.01   	//mortalidad de huevos
#define morlar 			0.01    //mortalidad de larvas 
#define morpup 			0.01   	//mortalidad de pupas
#define morad 			0.01 	//mortalidad diaria adultas
#define moracu 			0.03	//morhue+morlar+morpup;
#define morpupad 		0.17	//pupas que no se vuelven adultas
#define tpupad1 		9	//pupas se vuelven adultas a los 9 días en verano (desde oviposicion)****
#define tpupad2 		13	//pupas se vuelven adultas a los 13 dias en otoño y primavera****
#define tpupad3 		17  	//pupas se vuelven adultas a los 17 dias en invierno****
#define tovip1 			2	//tiempo entre dos oviposiciones (T=30)
#define tovip2a 		3  	//tiempo entre dos oviposiciones (T=25)
#define tovip2b			4   	//tiempo entre dos oviposiciones (T=25)
#define tovip3 			30	//tiempo ente dos oviposiciones (T=18)
#define sat 			800     //saturación de huevos por tacho
#define prop 			0.5	//efectividad de la propaganda

// cambiar si se quiere tener distintos numeros cada corrida
#define SEMILLAGLOBAL	12345

long semilla = -975;  			//semilla para el generador de numeros aleatorios ran2()

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
// Idea: cada hilo un bicho, que se reproduce con cierta Prob.
// Cada nacido va a parar a un array contador, "nacidos", que cuenta los nuevos por tacho 
// Todavia no se agregan al array de bichos, eso se hace luego.
__global__ void reproducir_kernel(int *estado, int *edad, int *tacho, int *N_mobil, int *nacidos, int dia){

	int tpupad=tiempo_pupas_adultas(dia);
	int tovip=tiempo_entre_oviposiciones(dia);

	int N=N_mobil[0];	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
//si la mosquita esta viva y esta en edad adulta (puede volar) entonces pone huevos cada tovip dias	
	if(id<N){
		if(estado[id] == ESTADOVIVO && edad[id] > tpupad && edad[id]%tovip == 0){
			int tach=tacho[id];	//pongo todos los huevos en el mismo tacho     		
	     		atomicAdd(nacidos+tach,NUMERODEHUEVOS);
			}

	}
}


// kernel asesino
// Idea: cada hilo decide a que bicho va a matar, cambiando su estado 0->1.
// No se actualiza aun el numero de bichos, para eso hace falta reordenar y sacar los muertos
__global__ void matar_kernel_eggs(int *estado, int *edad, int *tacho, int *N_mobil, int dia)
{

	int tpupad=tiempo_pupas_adultas(dia);

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

		//double prob_morir=edad[id]*1.0/MAXIMAEDAD;

		//MORTALIDADADES VARIAS (morirse antes de ser vieja)
		//if (estado[id] == ESTADOVIVO && edad[id] < tpupad){ 
		//  	 if (azar < prob_morir)estado[id]=ESTADOMUERTO;} 
		if (estado[id] == ESTADOVIVO && edad[id] < tpupad){if(azar < moracu)estado[id]=ESTADOMUERTO;} //muere en estado acuatico
		if (estado[id] == ESTADOVIVO && edad[id] == tpupad){if(azar < morpupad)estado[id]=ESTADOMUERTO;} 
		if (estado[id] == ESTADOVIVO && edad[id] > tpupad){if(azar < morad)estado[id]=ESTADOMUERTO;} //muere en estado adulto
		
 

	}
};

__global__ void matar_kernel_pupas(int *estado, int *edad, int *tacho, int *N_mobil, int dia)
{
	int tpupad=tiempo_pupas_adultas(dia);

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
__global__ void matar_kernel_adultas(int *estado, int *edad, int *tacho, int *N_mobil, int dia)
{
	int tpupad=tiempo_pupas_adultas(dia);

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

//muerte de las mosquitas por vejez
__global__ void matar_viejos_kernel(int *estado, int *edad, int *tacho, int *TdV,int *N_mobil,int dia)
{

	    int N=N_mobil[0];
	    int id = blockIdx.x*blockDim.x + threadIdx.x;

	    if(id<N){                     
	    	if(edad[id] >= TdV[id])estado[id]=ESTADOMUERTO;
	    }
};

//muerte de los acuáticos ( < 17 días) por descacharrado de los tachos

__global__ void descacharrado_kernel(int *estado, int *edad, int *tacho, int *N_mobil,int dia, int ntach)
{

	int tpupad=tiempo_pupas_adultas(dia);

    	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<N){
  		if (edad[id] < tpupad && tacho[id] == ntach)estado[id]=ESTADOMUERTO;    	
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

/////////////////////////////////////////////////////////////////////////////////////////
// Clase bichos: toda la info sobre todos los bichos, y sus funciones
struct bichos{

	thrust::device_vector<int> estado;  // Vivo o Muerto 0/1

	thrust::device_vector<int> edad;    // tiene num de mosqu elementos y los valores van de 0 a MAXIMAEDAD
	thrust::device_vector<int> tacho;   // 0 a NUMEROTACHOS
	thrust::device_vector<int> TdV; //tiempo de vida
	thrust::device_vector<int> manzana; //manzana

	thrust::device_vector<int> nacidos; // tiene el num de tachos elementos, numero de nacidos por tacho
	
	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)
	
	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_tacho;
	int *raw_estado;
	int *raw_TdV;
	int *raw_N_mobil;
	int *raw_manzana;
	int *raw_nacidos;
			
	// constructor: N_ = bichos iniciales
	bichos(int N_)
	{
        std::cout << "comenzando inicializacion para " << MAXIMONUMEROBICHOS << " bichos" << std::endl;

            // alocamos el maximo posible
        edad.resize(MAXIMONUMEROBICHOS);
        std::cout << "resizes hechos" << std::endl;

        tacho.resize(MAXIMONUMEROBICHOS);	
        estado.resize(MAXIMONUMEROBICHOS);
        TdV.resize(MAXIMONUMEROBICHOS);

		//manzana	
		manzana.resize(MAXIMONUMEROBICHOS);
		fill(manzana.begin(),manzana.end(),0);


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
		
		std::cout<<"VoM\tedad\tTdV\ttacho\tmanzana" << std::endl;

		// inicializacion totalmente random
		for(int i=0;i<N_;i++){
			edad[i]=ran2(&semilla)*5 + 12; 	//edad de 12 hasta 16 días 
			tacho[i]=i;		//tacho en el que se encuentra la mosquita
			estado[i]=0; 		//todos vivos inicialmente
			TdV[i]=ran2(&semilla)*3 + 28;	//tiempo de vida de 28 a 30
			manzana[i]=(int) i/5; 	// 5 tachos por manzana
		std::cout << estado[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << tacho[i] << "\t" << manzana[i] << std::endl;
		}		


        std::cout << "inicializacion lista" << std::endl;
        
    };

	//defino la temperatura segun la estacion y el tiempo entre oviposiciones y de maduración de acuaticos	
	int tiempo_entre_oviposiciones(int dia){
		int tovip;
		if(dia < 80 || dia > 320){tovip=tovip3;}	//<T>=18
  		if(dia >= 140 && dia <= 260){tovip=tovip1;}	//<T>=30
  		if(dia > 80 && dia < 140){tovip=tovip2b;}	//<T>=23
	  	if(dia >= 260 && dia <= 320){tovip=tovip2a;}	//<T>=23
	return tovip;}

	int tiempo_pupas_adultas(int dia){
		int tpupad;
		if(dia < 80 || dia > 320){tpupad=tpupad3;}//pupas se vuelven adultas a los 17 dias en invierno
		if(dia >= 140 && dia <= 260){tpupad=tpupad1;}//pupas se vuelven adultas a los 9 días en verano (desde oviposicion)
  		if(dia > 80 && dia < 140){tpupad=tpupad2;}//pupas se vuelven adultas a los 13 dias en otoño y primavera
	  	if(dia >= 260 && dia <= 320){tpupad=tpupad2;}
	return tpupad;}	


	// recorre los bichos, calcula el numero de nacidos por tacho
	void reproducir(int dia){
		int N=N_mobil[0]; //este es el indice que me indica donde termina el array de mosquitas vivas, se acuatila en el kernel reproducir

		// no hay nuevos nacidos
		thrust::fill(nacidos.begin(),nacidos.end(),0);
		
		// reproduce, calculando nacidos por tacho
		reproducir_kernel<<<(N+256-1)/256,256>>>
		(raw_estado,raw_edad,raw_tacho,raw_N_mobil, raw_nacidos, dia);	


		cudaDeviceSynchronize();

		// agrega todos los nacidos al final del array original, tacho a tacho
		int index=N;
        bool nosaturo=1;
		for(int m=0;m<NUMEROTACHOS;m++){
			//std::cout << "listo nacidos " << std::endl; 

 
            //antiguos=thrust::count_if(tacho.begin(),tacho.begin()+N,iguala(m));

            int antiguos=thrust::count_if(
                thrust::make_zip_iterator(thrust::make_tuple(tacho.begin(),edad.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(tacho.end(),edad.end())),
                acuaticoeneltacho(m,tpupad)
            );
            
            int nuevos=nacidos[m];
	//		if(tacho[m]+nacidos[m] < 800){ 						//cond. saturacion 
            nosaturo=((index+nuevos) < MAXIMONUMEROBICHOS);
            if((antiguos+nuevos) < 800 && nosaturo==1){ 						//cond. saturacion 
            thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,0);	//todos vivos(0)	
			thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,0);		//todos nacen con edad 0	
			thrust::fill(tacho.begin()+index,tacho.begin()+index+nuevos,m); 	//en el tacho m
			thrust::fill(TdV.begin()+index,TdV.begin()+index+nuevos,rand()%3+28); // con tiempo de vida 28 a 30
			index+=nuevos;		//actualizo el indice para me marque siempre en la ultima mosquita que nacio 

			}
		}
	// problema si esto satura saturan los tachos mas grandes, ver como cambiar esto por ejemplo llnando tachos al azar
        if(nosaturo==0) std::cout << "algunos tachos no se rellenaron por saturacion del total de mosquitas" << std::endl;
	
		// actualiza el numero de bichos si no se sobrepasa el maximo
		if(index<MAXIMONUMEROBICHOS) {
			N_mobil[0]=index;
		}
		// caso contrario satura al maximo
		else{
			//std::cout << "Demasiados Bichos!" << std::endl;
			 N_mobil[0]=MAXIMONUMEROBICHOS-1;	
		}
	};

	// recorre los bichos y los mata con cierta probabilidad funcion de la edad, actualizando estados
	void mortalidades_varias(int dia){
		int N=N_mobil[0];

		matar_kernel_eggs<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil, dia);	
		cudaDeviceSynchronize();

		matar_kernel_pupas<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil, dia);	
		cudaDeviceSynchronize();

		matar_kernel_adultas<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil, dia);		
		cudaDeviceSynchronize();
	};

	//mortalidades por llegar a viejas 
	void descacharrar_tacho(int dia){

		int N=N_mobil[0];
		int descach=std::round(NUMEROTACHOS*prop);//defino el nro. de tachos a descacharrar

		if(dia%7 == 0 && dia > 150 && dia < 240){
  			for(int itach=0;itach < descach;itach++){
    			int ntach=ran2(&semilla)*NUMEROTACHOS;	//sorteo cuáles son esos tachos y se los paso al kernel
			descacharrado_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_N_mobil,dia,ntach);}
		} 
	};

	//mortalidades por llegar a viejas 
	void matar_viejos(int dia){
		int N=N_mobil[0];

		matar_viejos_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_tacho,raw_TdV,raw_N_mobil, dia);
		cudaDeviceSynchronize();		
	};

	// reordenando para poner los muertos al fondo, podemos calcular el numero de vivos, y actualizar N_mobil
	void recalcularN(){
		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),tacho.begin(),TdV.begin(),manzana.begin()));


		// ordenamos segun estado 0-vivo, 1-muerto
		int N=N_mobil[0];
		thrust::sort_by_key(estado.begin(), estado.begin()+N,zip_iterator);		
	
		// y ahora determinamos la posicion del primer muerto = N_mobil
		auto iter=thrust::find(estado.begin(), estado.begin()+N, ESTADOMUERTO);
		N_mobil[0]= iter-estado.begin();
	};

	// recorre los bichos y les aumenta la edad en una unidad
	void envejecer(){
		int N=N_mobil[0];
		using namespace thrust::placeholders;
		thrust::transform(edad.begin(),edad.begin()+N, edad.begin(),_1+1);
	};

	// Imprime el estado completo de los bichos (solo para debuging)
	void imprimir(int dia){
		int N=N_mobil[0];
				
		std::cout << "#dia=" << dia << ", #N=" << N_mobil[0] << "\n";
		std::cout << "estado\tedad\tTdV\ttacho\tmanzana\n";
		for(int i=0;i<N;i++){
			std::cout << estado[i] << "\t" << edad[i] << "\t" << TdV[i] << "\t" << tacho[i] << "\t" << manzana[i] << std::endl;
		}
		std::cout << std::endl;
	};

	// Estadisticas de distinto tipo sobre el array de bichos
	void imprimir_estadisticas(){
		int N=N_mobil[0];
		
		std::cout << "hay " << N << " mosquitas en total" << std::endl;
		
		for(int m=0;m<NUMEROTACHOS;m++){
			std::cout << "hay " << 
			thrust::count_if(tacho.begin(),tacho.begin()+N,iguala(m))
			<< " en la tacho " << m << std::endl;				
		}

		std::cout << std::endl;

		for(int i=0;i<MAXIMAEDAD;i++){
			std::cout << "hay " << 
			thrust::count_if(edad.begin(),edad.begin()+N,iguala(i))
			<< " con edad " << i << std::endl;				
		}

		std::cout<< std::endl;

		std::cout << "hay " << NUMEROTACHOS << " tachos en total" << std::endl;

		for(int i=0;i<NUMEROMANZANAS;i++){
			std::cout << "hay " << 
			thrust::count_if(manzana.begin(),manzana.begin()+NUMEROTACHOS,iguala(i))
			<< " en la manzana " << i << std::endl;				
		}
	}

	// Numero de bichos vivos
	int vivos(){
		return N_mobil[0];
	};

	void avanza_dia(int dia)
    {
        std::cout << "reproducir" << std::endl;
		reproducir(dia); 		//nacimientos

        std::cout << "matar" << std::endl;
        matar(dia);			//mortalidades varias

        matar_viejos(dia);		//mortalidad por vejez
		descacharrar_tacho(dia);	//descacharrar tacho
		envejecer();			//envejecer un dia la poblacion

		recalcularN();
	};

};

