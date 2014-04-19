#FLAG = -DCPU
FLAG =

classical: page_rank.o mongo_def.o classical.o 
	nvcc page_rank.o mongo_def.o classical.o $(FLAG) -o classical -lmongoc

intelligent: page_rank.o mongo_def.o search.o
	nvcc mongo_def.o page_rank.o search.o $(FLAG) -o intelligent -lmongoc

classical.o: classical.cu mongo_def.h page_rank.h
	nvcc -c classical.cu $(FLAG) -arch=sm_20 -DMONGO_HAVE_UNISTD

page_rank.o: page_rank.cu
	nvcc -c page_rank.cu $(FLAG) -arch=sm_20

mongo_def.o: mongo_def.h mongo_def.cpp
	g++ -L/usr/local/lib -DMONGO_HAVE_UNISTD -g -c -Wno-write-strings mongo_def.cpp

search.o: search.cu mongo_def.h page_rank.h
	nvcc -c search.cu $(FLAG) -arch=sm_20 -DMONGO_HAVE_UNISTD

clean:
	rm -f *.o search
