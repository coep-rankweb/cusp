a.out: porter.o similarity.o home.cu defines.h
	nvcc home.cu porter.o similarity.o -arch=sm_20 -lhiredis -lm
porter.o: porter.cpp
	g++ -c porter.cpp
similarity.o: similarity.cpp defines.h
	g++ -c similarity.cpp
clean:
	rm -f *.o

mongo:
	g++ test_mongo.cpp -pthread -lmongoclient -lboost_thread-mt -lboost_filesystem -lboost_program_options -lboost_system
