# Define the compiler
CXX = clang++
# Define compiler flags
CXXFLAGS = -Wall -std=c++17 -g
# Define the output executable
TARGET = kmeans_cpu
# Find all source files
SRCS = $(wildcard *.cpp)
# Define the object files
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean up build files
clean:
	rm -f $(TARGET) $(OBJS)

run:
	@./kmeans_cpu -k 16 -d 16 -i input/random-n2048-d16-c16.txt 

runc:
	@./kmeans_cpu -k 16 -d 16 -i input/random-n2048-d16-c16.txt -c

run1: run
run2:
	@./kmeans_cpu -k 3 -i input/random-n2048-d16-c16.txt

.PHONY: all clean