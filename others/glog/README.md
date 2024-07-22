# Compiling GLog

## With CMake

- [Overview - Google Logging Library](https://google.github.io/glog/stable/)
- [Adjusting Output - Google Logging Library](https://google.github.io/glog/stable/flags/?h=logtostderr#using-command-line-parameters-and-environment-variables)

```bash
take build
cmake ..
make

./glog_cmake
ls /tmp/glog_cmake*

GLOG_logtostderr=1 ./glog_cmake
mkdir logs
GLOG_log_dir=`pwd`/logs ./glog_cmake
ls logs
```

## With g++

```bash
g++ main.cpp -o glog_g++ -L/usr/local/lib -lglog -I/usr/local/include -DGLOG_USE_GLOG_EXPORT
./glob_g++
```

## With Makefile

```bash
make
./glob_makefile
```
