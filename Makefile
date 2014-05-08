FILE = web
#FPATH = ~/Documents/save_01_04/$(FILE).mtx
FPATH = /home/nvidia/Apr15_dump/web.mtx
#FPATH = ~/kernel_panic/core/spyder/data/$(FILE).mtx
#FILE = simple
#FPATH = market/$(FILE).mtx
STATS = stats_$(FILE)

BETA = 0.85

ARGS = $(FPATH) $(BETA)
gpu:
	nvcc -arch=sm_20 thrust_defined.cu -o thrust_gpu
	time ./thrust_gpu $(ARGS) >> b_gpu_dia

new_gpu:
	#nvcc -arch=sm_20 page_rank.cu -o thrust_gpu
	time ./thrust_gpu $(ARGS) > a_gpu_$(BETA)

new_cpu:
	nvcc page_rank.cu -DCPU -o thrust_cpu
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


aggreg_gpu:
	./thrust_gpu $(ARGS) 2>> aggregate_gpu_$(BETA)

aggreg_cpu:
	echo $(FPATH) >> aggregate_cpu_$(BETA)
	head -3 $(FPATH) >> aggregate_cpu_$(BETA)
	./thrust_cpu $(ARGS) 2>> aggregate_cpu_$(BETA)

agg:
	echo ~/Documents/save_11_03/web.mtx >> aggregate_cpu_$(BETA)_475633
	head -3 ~/Documents/save_11_03/web.mtx >> aggregate_cpu_$(BETA)_475633
	./thrust_cpu ~/Documents/save_11_03/web.mtx 0.85 2>> aggregate_cpu_$(BETA)_475633
	echo ~/Documents/save_28_02/web.mtx >> aggregate_cpu_$(BETA)_350045
	head -3 ~/Documents/save_28_02/web.mtx >> aggregate_cpu_$(BETA)_350045
	./thrust_cpu ~/Documents/save_28_02/web.mtx 0.85 2>> aggregate_cpu_$(BETA)_350045
