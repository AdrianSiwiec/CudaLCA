NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-arch sm_50 -O2 -Xptxas -dlcm=ca

CXX=g++
CXXFLAGS=-std=c++11 -O2 -fno-stack-protector

all: cudaSimpleLCA.e cpuSimpleLCA.e generateSimple.e

cpuSimpleLCA.e: cpuSimpleLCA.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

cudaSimpleLCA.e: cudaSimpleLCA.cu commons.o cudaCommons.o
	$(NVCC) $^ -o $@

generateSimple.e: generateSimple.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: all clean

clean:
	rm -f *.o *.e