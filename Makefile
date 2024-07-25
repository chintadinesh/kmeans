# Define the compiler
CXX = clang++
# Define compiler flags
CXXFLAGS = -Wall -g -std=c++17
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

.PHONY: all clean