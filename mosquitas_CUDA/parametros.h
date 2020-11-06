/*
Macros utiles para CPU y GPU
Relaciones a tener en cuenta para modificar parámetros
- NUMEROTACHOS=Ninicial
*/
#define NINICIAL 		    7		//número inicial de mosquitos para codigo GPU	
#define NUMEROTACHOS		7		//máximo número de tachos 
#define NUMERODEHUEVOS		10 		//número de huevos por oviposicion	
#define MAXIMONUMEROBICHOS	80000	//número maximo de huevos

#define ESTADOMUERTO    	1
#define ESTADOVIVO	        0
#define NDIAS			    700    //cantidad de dias en el año

#define MORHUE              0.01   	//mortalidad de huevos
#define MORLAR              0.01    //mortalidad de larvas 
#define MORPUP              0.01   	//mortalidad de pupas
#define MORAD               0.01 	//mortalidad diaria adultas
#define MORACU              0.03	//morhue+MORLAR+MORPUP;
#define MORPUPAD            0.17	//pupas que no se vuelven adultas
#define TOVIP1              2   	//tiempo entre dos oviposiciones (T=30)
#define TOVIP2a             3     	//tiempo entre dos oviposiciones (T=25)
#define TOVIP2b             4   	//tiempo entre dos oviposiciones (T=25)
#define TOVIP3              30  	//tiempo ente dos oviposiciones (T=18)
#define SAT                 800	    //numero maximo de huevos en cada tacho
#define PROP                0.3	    //efectividad de la propaganda


#define TPUPAD  		    17  	//pupas se vuelven adultas a los 17 dias en invierno****

#define SEMILLAGLOBAL	    22344888
//#define SEMILLAGLOBAL	    33455999

long semilla = -975;  			//semilla para el generador de numeros aleatorios ran2()
