#ifndef _HNSWHELPER
#define _HNSWHELPER

#include "hnsw/hnswlib.h"

class HNSWHelper {
    public:
    /// TODO: Add extra argument for space choice
    HNSWHelper(const char* index_path, int metric, int dim, int ef);

    ~HNSWHelper();

    std::vector<uint32_t> get_suggestions(float* qpt, int k);

    std::vector<uint32_t> get_suggestions(float* qpt, int k, int ef);

    private:
    int _ef;
    int _metric;
    int _dim;
    hnswlib::SpaceInterface<float>* space;
    hnswlib::HierarchicalNSW<float>* _indx;
};

#endif //_HNSWHELPER