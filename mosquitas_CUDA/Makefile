APP?=bichos

THRUSTPATH=.
#THRUSTPATH=/share/apps/icnpg/common/thrust-master

RANDOM123=-I./include/

all: ${APP}

${APP}: $(APP).cu $(APP).h 
	nvcc -std=c++11 -O2 -o $@ $< -Wno-deprecated-gpu-targets $(RANDOM123)	

clean:
	rm -f $(APP)

run: $(APP)	
	./$(APP)

submit:	$(APP)
	qsub -N $(APP) jobGPU ./$(APP) 
	#watch qstat

