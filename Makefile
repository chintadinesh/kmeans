SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
BIN_DIR = bin
INC_DIR = inc
APP_DIR = app

CC = g++ -std=c++17

SRCS = $(wildcard $(SRC_DIR)/*.c*)
$(info SRCS : $(SRCS))

APPS = $(wildcard $(APP_DIR)/*.c*) 
$(info APPS : $(APPS))

LIBS = $(patsubst $(SRC_DIR)/%.cpp,$(LIB_DIR)/%.o,$(SRCS))
$(info LIBS : $(LIBS))

OBJS = $(patsubst $(APP_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(APPS))
$(info OBJS : $(OBJS))

BINS = $(patsubst $(OBJ_DIR)/%.o,$(BIN_DIR)/%,$(OBJS))
$(info BINS : $(BINS))

all: $(BINS) $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR) 


$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(LIBS) 
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(INC_DIR) -o $@ $^ 

$(LIB_DIR)/%.o: $(SRC_DIR)/%.c* 
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

$(OBJ_DIR)/%.o: $(APP_DIR)/%.c* 
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

$(LIB_DIR):
	mkdir $(LIB_DIR)

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f $(BIN_DIR)/* $(OBJ_DIR)/* $(LIB_DIR)/*

run:
	bin/kmeans_cpu