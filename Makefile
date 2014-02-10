FILE = web
FPATH = ../spyder/data/$(FILE).mtx
STATS = stats_$(FILE)
gpu:
	nvcc -arch=sm_20 thrust_defined.cu -o thrust_gpu
	time ./thrust_gpu $(FPATH) > b_gpu
cpu:
	nvcc thrust_defined.cu -DCPU -o thrust_cpu
	time ./thrust_cpu $(FPATH) > b_cpu

prof:
	nvcc -arch=sm_20 --profile thrust_defined.cu -o thrust_gpu
	nvprof ./thrust_gpu $(FPATH) > $(STATS)
