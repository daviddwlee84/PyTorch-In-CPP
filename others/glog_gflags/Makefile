CXX = g++
CXXFLAGS = -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lglog -lgflags
DEFINES = -DGLOG_USE_GLOG_EXPORT

TARGET = glog_gflags_makefile
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(DEFINES) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)
