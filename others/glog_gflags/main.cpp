#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(name, "World", "Name to greet");

int main(int argc, char *argv[])
{
    // Initialize gflags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Initialize glog
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1; // Log to stderr instead of files

    // Use glog for logging
    LOG(INFO) << "Hello " << FLAGS_name;

    // Shutdown glog
    google::ShutdownGoogleLogging();

    // Shutdown gflags
    gflags::ShutDownCommandLineFlags();
    return 0;
}
