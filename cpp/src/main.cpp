#include "PyInterface.h"

int main() {
#ifdef __linux__
    auto py = new il::PyInteractiveLearningFloat(0, "/mnt/c/Users/ok261/Documents/OK/Projects/HyperplanesVsANN/cpp/hnsw/src/hnswvbs_48M_500ef_l2.bin", 0, 1000, 1000);
    // auto py2 = new il::PyInteractiveLearningFloat(1, "/mnt/c/Users/ok261/Documents/OK/Projects/HyperplanesVsANN/cpp/annoy/src/annoyvbs_200t_400000sk_l2.bin", 0, 1000, 5000);
    delete py;
    // delete py2;
#endif //__linux__
#ifdef _WIN32
    auto py = new il::PyInteractiveLearningFloat(0, "C:/Users/ok261/Documents/OK/Projects/HyperplanesVsANN/cpp/hnsw/src/hnswlsc_48M_500ef.bin", 0, 1000, 1000);
#endif // _WIN32

    return 0;
};