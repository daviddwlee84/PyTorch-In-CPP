#include <glog/logging.h>

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "glog INFO";
    LOG(WARNING) << "glog WARNING";
    LOG(ERROR) << "glog ERROR";
    DLOG(ERROR) << "glob Debug ERROR";
}