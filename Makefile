NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-std=c++11 -arch sm_50 -O2 -Xptxas -dlcm=ca --expt-extended-lambda -I ../moderngpu/src

CXX=g++
CXXFLAGS=-std=c++11 -O2 -fno-stack-protector 

all: cudaInlabelLCA.e cudaSimpleLCA.e cpuSimpleLCA.e generateSimple.e

hello.e: hello.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

cpuSimpleLCA.e: cpuSimpleLCA.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

cudaInlabelLCA.e: cudaInlabelLCA.cu commons.o cudaCommons.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

cudaSimpleLCA.e: cudaSimpleLCA.cu commons.o cudaCommons.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

generateSimple.e: generateSimple.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: all clean

clean:
	rm -f *.o *.e