#ifndef _IVFHELPER
#define _IVFHELPER

#include <faiss/IndexIVFFlat.h>

class IVFHelper {
    public:
    IVFHelper(const char* index_path, int metric, int dim, int nprobe);

    ~IVFHelper();

    std::vector<uint32_t> get_suggestions(float* qpt, int k);

    std::vector<uint32_t> get_suggestions(float* qpt, int k, int nprobe);

    private:
    int _nprobe;
    int _metric;
    int _dim;

    faiss::IndexIVFFlat* _indx;
    
};

#endif //_IVFHELPER