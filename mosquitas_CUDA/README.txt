para ejecutar el Makefile escribir en la terminal:

1) make bichos 
2) make mosquitas 
3) make submit 

en el output vamos a ver:  

4) un archivo poblacion.o# 
haciendo un cat en poblacion.o# se puede ver las condiciones iniciales de los códigos en C++ y CUDA y el tiempo de cálculo

5) los archivos 
	-Poblacion_total_GPU.dat
	-Poblacion_total_CPU.dat

Para graficar los resultados en gnuplot escribir:

pl 'Poblacion_total_CPU.dat' u 1:2 w l lw 5 title 'CPU','Poblacion_total_GPU.dat' u 1:2 w l lw 2 title 'GPU', 'mosquitas_totales.dat' u 1:4 w l lw 6 title 'Fortran'
