#include "IVFHelper.h"
#include <faiss/index_io.h>
#include <iostream>

IVFHelper::IVFHelper(const char* index_path, int metric, int dim, int nprobe) {
    _metric = metric;
    _dim = dim;
    _nprobe = nprobe;

    if (_metric == 0) {
        std::cout << "(IVF) Loading IVF index with IP..." << std::endl;
        _indx = (faiss::IndexIVFFlat*) faiss::read_index(index_path);
        _indx->metric_type = faiss::METRIC_INNER_PRODUCT;
    } else if (metric == 1) {
        std::cout << "(IVF) Loading IVF index with L2..." << std::endl;
        _indx = (faiss::IndexIVFFlat*) faiss::read_index(index_path);
        _indx->metric_type = faiss::METRIC_L2;
    } else {
        std::cout << "(IVF) Unknown metric specified!" << std::endl;
        return;
    }
    _indx->nprobe = _nprobe;
}

IVFHelper::~IVFHelper() {
    delete _indx;
}

std::vector<uint32_t> IVFHelper::get_suggestions(float* qpt, int k) {
    auto topIds = new faiss::idx_t[k];
    auto topDistances = new float[k];
    _indx->search(1, qpt, k, topDistances, topIds);

    auto res = std::vector<uint32_t>(k);
    for (int i = 0; i < k; i++) {
        if (topIds[i] == -1) {
            res.resize(i);
            break;
        }
        res[i] = static_cast<uint32_t>(topIds[i]);
    }

    std::cout << "ndis: " << faiss::indexIVF_stats.ndis << std::endl;
    std::cout << "nq: " << faiss::indexIVF_stats.nq << std::endl;
    std::cout << "Distance Computations: " << faiss::indexIVF_stats.ndis + _indx->nlist << std::endl;

    return res;
}


std::vector<uint32_t> IVFHelper::get_suggestions(float* qpt, int k, int search_par) {
    auto topIds = new faiss::idx_t[k];
    auto topDistances = new float[k];
    _indx->nprobe = search_par;
    _indx->search(1, qpt, k, topDistances, topIds);

    auto res = std::vector<uint32_t>(k);
    for (int i = 0; i < k; i++) {
        if (topIds[i] == -1) {
            res.resize(i);
            break;
        }
        res[i] = static_cast<uint32_t>(topIds[i]);
    }

    std::cout << "ndis: " << faiss::indexIVF_stats.ndis << std::endl;
    std::cout << "nq: " << faiss::indexIVF_stats.nq << std::endl;
    std::cout << "Distance Computations: " << faiss::indexIVF_stats.ndis + _indx->nlist << std::endl;

    return res;
}