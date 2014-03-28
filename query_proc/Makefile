FLAG = CPU

search: page_rank.o mongo_def.o search.o
	nvcc mongo_def.o page_rank.o search.o -o search -lmongoc

search.o: search.cu mongo_def.h cuda_def.h
	nvcc -c --save-temps search.cu -arch=sm_20 -DMONGO_HAVE_UNISTD

page_rank.o: page_rank.cu cuda_def.o
	nvcc -c cuda_def.o page_rank.cu -arch=sm_20

mongo_def.o: mongo_def.h mongo_def.cpp
	g++ -L/usr/local/lib -DMONGO_HAVE_UNISTD -g -c -Wno-write-strings mongo_def.cpp

cuda_def.o: cuda_def.cu cuda_def.h
	nvcc -c cuda_def.cu -D$(FLAG) -arch=sm_20

clean:
	rm -f *.o search
