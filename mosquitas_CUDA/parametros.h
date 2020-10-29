/*
Macros utiles para CPU y GPU
Relaciones a tener en cuenta para modificar parรกmetros
- NUMEROTACHOS=Ninicial
*/

#define NINICIAL	    	5
#define NUMEROTACHOS		5
#define MAXIMONUMEROBICHOS	80000

#define ESTADOMUERTO		1
#define ESTADOVIVO		    0
#define NDIAS			    400

#define MORHUE 			    0.01   	//mortalidad de huevos
#define MORLAR 			    0.01    //mortalidad de larvas 
#define MORPUP 			    0.01   	//mortalidad de pupas
#define MORAD 			    0.01 	//mortalidad diaria adultas
#define MORACU 			    0.03	//morhue+morlar+morpup;
#define MORPUPAD 		    0.17	//pupas que no se vuelven adultas
#define TPUPAD	 		    17  	//pupas se vuelven adultas a los 17 dias en invierno****
#define TOVIP1 			    2	    //tiempo entre dos oviposiciones (T=30)
#define TOVIP2a 		    3  	    //tiempo entre dos oviposiciones (T=25)
#define TOVIP2b			    4   	//tiempo entre dos oviposiciones (T=25)
#define TOVIP3 			    30	    //tiempo ente dos oviposiciones (T=18)
#define SAT 			    800     //saturaciรณn de huevos por tacho
#define PROP 			    0.6	    //efectividad de la propaganda

#define TPUPAD  		    17  	//pupas se vuelven adultas a los 17 dias en invierno****

//#define SEMILLAGLOBAL	   	22399989
#define SEMILLAGLOBAL	   	123456789
//#define SEMILLAGLOBAL	   	975

long semilla = -739;  			//semilla para el generador de numeros aleatorios ran2()

