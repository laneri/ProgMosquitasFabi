#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG

//etapas de la mosquita: huevo -> larva (5días) -> pupa (7días)-> mosquita adulta (vive 11 dias)

// macros utiles para CPU y GPU, y hacer desaparecer numeros magicos
#define MAXIMAEDAD		31 	//dias que vive la mosquita	
#define NUMEROMANZANAS		10	
#define MAXIMONUMEROBICHOS	4000 	//maximo de huevos por manzana
#define NUMERODEESTADOS		2	//vivo(0) o muerto (1)
#define NUMERODEHUEVOS		10 	//número de huevos por oviposicion	

#define ESTADOMUERTO	1
#define ESTADOVIVO	0

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

// cambiar si se quiere tener distintos numeros cada corrida
#define SEMILLAGLOBAL	12345



// El kernel de reproduccion
// Idea: cada hilo un bicho, que se reproduce con cierta Prob.
// Cada nacido va a parar a un array contador, "nacidos", que cuenta los nuevos por manzana 
// Todavia no se agregan al array de bichos, eso se hace luego.
__global__ void reproducir_kernel(int *estado, int *edad, int *manzana, int *N_mobil, int *nacidos, int tpupad, int tovip, int dia){

	int N=N_mobil[0];	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id<N){
		if(estado[id] == ESTADOVIVO && edad[id] > tpupad){
			if(edad[id]%tovip == 0){
		 	RNG philox;   //counter-based random number generator      
	    		RNG::ctr_type c={{}};//counter
	    		RNG::key_type k={{}};//key
	    		RNG::ctr_type r;
	    		k[0]=id; //user supplied seed
	    		c[1]=dia;// some loop-dependent application variable
	    		c[0]=SEMILLAGLOBAL; // another loop-dependent application variable 
		
    			r = philox(c, k); //On each iteration,r contains an array of 2 32-bit random values that will not be repeated by any other call to rng as long as c and k are not reused.

     			double azar=(u01_closed_closed_32_53(r[0]));//numeros random entre [0,1]
			int iovip= azar*3 + 7;  		    //entre 7 y 10 huevos pone la mosquita id
			int manz=manzana[id];  		
	     		atomicAdd(nacidos+manz,iovip);	//ordena el número de huevos en el array de nacidos		
			} 	
		}
	}
}


// kernel asesino: mata mosquitas con cierta probabilidad antes de volverse vieja
// Idea: cada hilo decide a que bicho va a matar, cambiando su estado 0->1.
// No se actualiza aun el numero de bichos, para eso hace falta reordenar y sacar los muertos
__global__ void matar_kernel(int *estado, int *edad, int *manzana, int *N_mobil,int tpupad, int dia)
{

	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<N){		
	 	RNG philox;   //counter-based random number generator      
	    	RNG::ctr_type c={{}};//counter
	    	RNG::key_type k={{}};//key
	    	RNG::ctr_type r;
	    	k[0]=id; //user supplied seed
	    	c[1]=dia;// some loop-dependent application variable
	    	c[0]=SEMILLAGLOBAL; // another loop-dependent application variable 
		
    		r = philox(c, k); //On each iteration,r contains an array of 2 32-bit random values that will not be repeated by any other call to rng as long as c and k are not reused.

     		double azar=(u01_closed_closed_32_53(r[0]));//numero random entre [0,1]

		//MORTALIDADADES VARIAS (morirse antes de ser vieja)
		if (estado[id] == ESTADOVIVO && edad[id] < tpupad){ 
		  	 if (azar < moracu)estado[id]=ESTADOMUERTO;} 
 
		if (estado[id] == ESTADOVIVO && edad[id] == tpupad){ 
	  	 	 if (azar < morpupad)estado[id]=ESTADOMUERTO;}

		if (estado[id] == ESTADOVIVO && edad[id] > tpupad){ 
	  	 	 if(azar < morad)estado[id]=ESTADOMUERTO;}

	}
};
//muerte de las mosquitas por vejez
__global__ void matar_viejos_kernel(int *estado, int *edad, int *manzana, int *N_mobil,int dia)
{

	int N=N_mobil[0];
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<N){	
		RNG philox;   //counter-based random number generator      
	    	RNG::ctr_type c={{}};//counter
	    	RNG::key_type k={{}};//key
	    	RNG::ctr_type r;
	    	k[0]=id; //user supplied seed
	    	c[1]=dia;// some loop-dependent application variable
	    	c[0]=SEMILLAGLOBAL; // another loop-dependent application variable 
		
    		r = philox(c, k); //On each iteration,r contains an array of 2 32-bit random values that will not be repeated by any other call to rng as long as c and k are not reused.

     		double tDvida= 28 + (u01_closed_closed_32_53(r[0]))*2;//tiempo de vida entre 28 y 30 dias 	
		if (estado[id] == ESTADOVIVO && edad[id] >= tDvida){ 
		estado[id]=ESTADOMUERTO;}
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
	thrust::device_vector<int> edad;    // 0 a MAXIMAEDAD
	thrust::device_vector<int> manzana; // 0 a NUMEROMANZANAS
	thrust::device_vector<int> nacidos; // nacidos por manzana

	thrust::device_vector<int> N_mobil; // Numero de bichos fluctuante (1 elemento)
	
	// punteros crudos a los arrays para pasarselos a kernels
	int *raw_edad;
	int *raw_manzana;
	int *raw_estado;
	int *raw_N_mobil;
	int *raw_nacidos;
			
	// constructor: N_ = bichos iniciales
	bichos(int N_)
	{

		// alocamos el maximo posible
		edad.resize(MAXIMONUMEROBICHOS);	
		manzana.resize(MAXIMONUMEROBICHOS);	
		estado.resize(MAXIMONUMEROBICHOS);	
		
		// un entero en device
		N_mobil.resize(1);
		
		// nacidos en cada manzana, inicialmente 0
		nacidos.resize(NUMEROMANZANAS);
		fill(nacidos.begin(),nacidos.end(),0);
	
		// numero de vivos inicial
		N_mobil[0]=N_;

		// inicializacion raw pointers
		raw_edad=thrust::raw_pointer_cast(edad.data());
		raw_manzana=thrust::raw_pointer_cast(manzana.data());
		raw_estado=thrust::raw_pointer_cast(estado.data());
		raw_N_mobil=thrust::raw_pointer_cast(N_mobil.data());
		raw_nacidos=thrust::raw_pointer_cast(nacidos.data());
		
		// inicializacion totalmente random
		for(int i=0;i<N_;i++){
			edad[i]= rand()%MAXIMAEDAD;  	// edad de la mosquita i entre 0 y 30
			manzana[i]=rand()%NUMEROMANZANAS;//manzana en la que se encuentra la mosquita i entre 0 y 10
			estado[i]=0; 			//todas las mosquitas vivas inicialmente
		}		
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

	// recorre los bichos, calcula el numero de nacidos por manzana
	void reproducir(int dia){
		int N=N_mobil[0];

		int tovip=tiempo_entre_oviposiciones(dia);
		int tpupad=tiempo_pupas_adultas(dia);
		
		// reproduce, calculando nacidos por manzana
		reproducir_kernel<<<(N+256-1)/256,256>>>
		(raw_estado,raw_edad,raw_manzana,raw_N_mobil, raw_nacidos, dia,tpupad,tovip);	


		cudaDeviceSynchronize();

		// agrega todos los nacidos al final del array original, manzana a manzana
		int index=N;
		for(int m=0;m<NUMEROMANZANAS;m++){
			//std::cout << "listo nacidos " << std::endl; 
			int nuevos=nacidos[m];
			thrust::fill(estado.begin()+index,estado.begin()+index+nuevos,0);		
			thrust::fill(edad.begin()+index,edad.begin()+index+nuevos,0);		
			thrust::fill(manzana.begin()+index,manzana.begin()+index+nuevos,m);
			index+=nuevos;		
		}
	
		// actualiza el numero de bichos si no se sobrepasa el maximo
		if(index<MAXIMONUMEROBICHOS) {
			N_mobil[0]=index;
		}
		// caso contrario satura al maximo
		else{
			std::cout << "Demasiados Bichos!" << std::endl;
			 N_mobil[0]=MAXIMONUMEROBICHOS-1;	
		}
	};

	// recorre los bichos y los mata con cierta probabilidad funcion de la edad, actualizando estados
	//mortalidades varias
	void matar(int dia){
		int N=N_mobil[0];

		int tpupad=tiempo_pupas_adultas(dia);

		matar_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_manzana,raw_N_mobil,tpupad, dia);		
	};

	//mortalidades por llegar a viejas 
	void matar_viejos(int dia){
		int N=N_mobil[0];

		matar_viejos_kernel<<<(N+256-1)/256,256>>>(raw_estado,raw_edad,raw_manzana,raw_N_mobil, dia);		
	};

	// reordenando para poner los muertos al fondo, podemos calcular el numero de vivos, y actualizar N_mobil
	void recalcularN(){
		auto zip_iterator=
		thrust::make_zip_iterator(thrust::make_tuple(edad.begin(),manzana.begin()));

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
				
		std::cout << "#dia=" << dia << ", #N=" << N_mobil[0] << "\n";
		std::cout << "estado\tedad\tmanzana\n";
		for(int i=0;i<N;i++){
			std::cout << estado[i] << "\t" << edad[i] << "\t"<< "\t" << manzana[i] << std::endl;
		}
		std::cout << std::endl;
	};

	// Estadisticas de distinto tipo sobre el array de bichos
	void imprimir_estadisticas(){
		int N=N_mobil[0];
		
		std::cout << "hay " << N << " mosquitas en total" << std::endl;
		
		for(int m=0;m<NUMEROMANZANAS;m++){
			std::cout << "hay " << 
			thrust::count_if(manzana.begin(),manzana.begin()+N,iguala(m))
			<< " en la manzana " << m << std::endl;				
		}
		std::cout << "\n" << std::endl;

		for(int i=0;i<MAXIMAEDAD;i++){
			std::cout << "hay " << 
			thrust::count_if(edad.begin(),edad.begin()+N,iguala(i))
			<< " con edad " << i << std::endl;				
		}
		// etc....
	}

	// Numero de bichos vivos
	int vivos(){
		return N_mobil[0];
	};
	//prestar atencion a orden!
	void avanza_dia(int dia){
		reproducir(dia); //nacimientos
		matar(dia);	//mortalidades varias
		matar_viejos(dia);//mortalidad por vejez
		envejecer();	//envejecer
		recalcularN();
	};

};


