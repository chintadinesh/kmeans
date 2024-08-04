# Define the compiler
CXX = clang++

# Define the compiler
NVCC = /usr/local/cuda/bin/nvcc

# Define compiler flags
CXXFLAGS = -Wall -std=c++17

# Define nvcc compiler flags
NVCCFLAGS = -std=c++17

# Define the output executable
TARGET = kmeans
# Find all source files
SRCS = $(wildcard *.cpp)

# Find all CUDA source files
CUSRCS = $(wildcard *.cu)

# Define the object files
OBJS = $(SRCS:.cpp=.o)

# Define the object files
CUOBJS = $(CUSRCS:.cu=.o)

# Directories
INCLUDES := -I/usr/local/cuda/include
LIBDIRS := -L/usr/local/cuda/lib64
CULIBS := -lcudart -lcuda

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS) $(CUOBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(CUOBJS) $(CULIBS)

# Rule to compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cu files into .o files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to clean up build files
clean:
	rm -f $(TARGET) $(OBJS) $(CUOBJS)

run:
	@./kmeans -k 16 -d 16 -i input/random-n2048-d16-c16.txt 

runc:
	@./kmeans -k 16 -d 16 -i input/random-n2048-d16-c16.txt -c

run1: run
run2:
	@./kmeans -k 3 -i input/random-n2048-d16-c16.txt

.PHONY: all clean