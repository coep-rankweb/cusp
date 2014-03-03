FILE = web
FPATH = ~/Documents/save_28_02/$(FILE).mtx
#FILE = simple
#FPATH = market/$(FILE).mtx
STATS = stats_$(FILE)

BETA = 0.85

ARGS = $(FPATH) $(BETA)
gpu:
	nvcc -arch=sm_20 thrust_defined.cu -o thrust_gpu
	time ./thrust_gpu $(ARGS) > b_gpu_rank
cpu:
	nvcc thrust_defined.cu -DCPU -o thrust_cpu
	time ./thrust_cpu $(ARGS) > b_cpu

prof:
	nvcc -arch=sm_20 --profile thrust_defined.cu -o thrust_gpu
	nvprof ./thrust_gpu $(ARGS) > $(STATS)
hello:
	echo "kjfhes;lkfhdl;ksajfhdlksajfhdolsi"
	touch abc
