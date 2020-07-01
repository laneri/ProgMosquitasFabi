Cada mosquita en el programa tiene definido:
su estado               -> 0(vivo) o 1(muerto)
su edad                 -> entre 0 y 30 dias (aleatorio) (es decir, tenemos huevo,larva,pupa,adulta)
tacho en el que vive    -> (aleatorio)

condiciones iniciales: dia=0 y N=5 (# de mosquitas)

La función test1() calcula por dia:
1) el número total de mosquitas después de reproducirse
2) el número de mosquitas que se mueren con alguna probabilidad
3) el número de mosquitas que se mueren por viejas
4) el número de mosquitas después de descacharrar un tacho
5) envejecimiento de 1 día para la edad de las mosquitas
6) se remueven las mosquitas muertas
7) estadística: cuantas mosquitas hay por tacho,cuantas mosquitas hay por determinada edad y cuando tachos hay por manzana. 

La función test2() calcula la población de mosquitas de todos los tachos en un año (400 días)

