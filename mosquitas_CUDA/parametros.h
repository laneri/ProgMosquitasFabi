/*
Macros utiles para CPU y GPU


Relaciones a tener en cuenta para modificar parámetros

- (NUMERODEMAZANAS/NUMEROTACHOS) = 5
- NUMEROTACHOS=Ninicial
- MAXIMONUMEROBICHOS = NUMEROMANZANAS*4000

*/
#define Ninicial 		320		//número inicial de mosquitos para codigo GPU	
#define MAXIMAEDAD		31 		//dias que vive la mosquita	
#define NUMEROTACHOS		320		//máximo número de tachos 
#define NUMEROMANZANAS		1600		//número de manzanas
#define MAXIMONUMEROBICHOS	1280000	//número maximo de huevos
#define NUMERODEESTADOS		2		//vivo(0) o muerto (1)
#define NUMERODEHUEVOS		10 		//número de huevos por oviposicion	

#define ESTADOMUERTO	1
#define ESTADOVIVO	0


#define nmax 800000	//dimensión del array
#define inhem 5   	
#define ntachito 5	//numero de tachos por manzana
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
#define prop 0.6	//efectividad de la propaganda

