FILE = web
#FPATH = ~/Documents/save_11_03/$(FILE).mtx
FPATH = ~/kernel_panic/core/spyder/data/$(FILE).mtx
#FILE = simple
#FPATH = market/$(FILE).mtx
STATS = stats_$(FILE)

BETA = 0.75

ARGS = $(FPATH) $(BETA)
gpu:
	nvcc -arch=sm_20 thrust_defined.cu -o thrust_gpu
	time ./thrust_gpu $(ARGS) >> b_gpu_dia
new_gpu:
	#nvcc -arch=sm_20 diag.cu -o thrust_gpu
	time ./thrust_gpu $(ARGS) > a_gpu
new_cpu:
	nvcc diag.cu -DCPU -o thrust_cpu
	time ./thrust_cpu $(ARGS) > a_cpu
cpu:
	nvcc thrust_defined.cu -DCPU -o thrust_cpu
	time ./thrust_cpu $(ARGS) > b_cpu

prof:
	nvcc -arch=sm_20 --profile thrust_defined.cu -o thrust_gpu
	nvprof ./thrust_gpu $(ARGS) > $(STATS)
hello:
	echo "kjfhes;lkfhdl;ksajfhdlksajfhdolsi"
	touch abc
