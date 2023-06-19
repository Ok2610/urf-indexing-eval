#include "HnswHelper.h"

HNSWHelper::HNSWHelper(const char* index_path, int metric, int dim, int ef) {
    _metric = metric;
    _dim = dim;
    _ef = ef;
    if (_metric == 0) {
        std::cout << "(HNSW) Loading index with IP from file "  << index_path << std::endl;
        space = new hnswlib::InnerProductSpace(_dim);
        _indx = new hnswlib::HierarchicalNSW<float>(space, index_path);
        std::cout << "(HNSW) Max elements in index " << _indx->max_elements_ << std::endl;
        _indx->setEf(_ef);
    } else if (metric == 1) {
        std::cout << "(HNSW) Loading index with L2" << std::endl;
        space = new hnswlib::L2Space(_dim);
        _indx = new hnswlib::HierarchicalNSW<float>(space, index_path);
        _indx->setEf(_ef);
    }
};

HNSWHelper::~HNSWHelper() {
    delete space;
    delete _indx;
};

std::vector<uint32_t> HNSWHelper::get_suggestions(float* qpt, int k) {
    // std::cout << "(HNSW) Calling searchKnn to get " << k << " items" << std::endl;
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = _indx->searchKnn(qpt, k);

    // NOTE!! The resulting priority queue is descending so the top item is the last element
    auto ids = std::vector<uint32_t>(result.size());
    for (int i = 0; i < k; i++) {
        hnswlib::labeltype label = result.top().second;
        // Insert from the back
        ids[k-1-i] = label;
        result.pop();
    }

    std::cout << "Distance Computations: " << _indx->metric_distance_computations << std::endl;

    return ids;
};

std::vector<uint32_t> HNSWHelper::get_suggestions(float* qpt, int k, int ef) {
    _indx->setEf(ef);
    // std::cout << "(HNSW) ef set to " << ef << "... Calling searchKnn" << std::endl;
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = _indx->searchKnn(qpt, k);

    // NOTE!! The resulting priority queue is descending so the top item is the last element
    auto ids = std::vector<uint32_t>(result.size());
    for (int i = 0; i < k; i++) {
        hnswlib::labeltype label = result.top().second;
        // Insert from the back
        ids[k-1-i] = label;
        result.pop();
    }
    
    std::cout << "Distance Computations: " << _indx->metric_distance_computations << std::endl;

    return ids;
};
