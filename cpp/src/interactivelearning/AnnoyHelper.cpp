#include "AnnoyHelper.h"

AnnoyHelper::AnnoyHelper(int dim, int search_k) {
    _dim = dim;
    _search_k = search_k;
}

AnnoyHelper::~AnnoyHelper() {}

std::vector<uint32_t> AnnoyHelper::get_suggestionsDot(
    AnnoyIndex<int, float, DotProduct, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy, float* qpt, int k) {

    std::vector<int> topItems;
    std::vector<float> topDistances;
    annoy->get_nns_by_vector(qpt, k, _search_k, &topItems, &topDistances);

    auto res = std::vector<uint32_t>(topItems.size(),0);
    for (size_t i = 0; i < topItems.size(); i++) {
        res[i] = static_cast<uint32_t>(topItems[i]);
    }
    return res;
}

std::vector<uint32_t> AnnoyHelper::get_suggestionsDot(
    AnnoyIndex<int, float, DotProduct, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy,
    float* qpt, int k, int search_k) {

    std::vector<int> topItems;
    std::vector<float> topDistances;
    annoy->get_nns_by_vector(qpt, k, search_k, &topItems, &topDistances);

    auto res = std::vector<uint32_t>(topItems.size(),0);
    for (size_t i = 0; i < topItems.size(); i++) {
        res[i] = static_cast<uint32_t>(topItems[i]);
    }
    return res;
}


std::vector<uint32_t> AnnoyHelper::get_suggestionsL2(
    AnnoyIndex<int, float, Euclidean, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy, float* qpt, int k) {

    std::vector<int> topItems;
    std::vector<float> topDistances;
    annoy->get_nns_by_vector(qpt, k, _search_k, &topItems, &topDistances);

    auto res = std::vector<uint32_t>(topItems.size(),0);
    for (size_t i = 0; i < topItems.size(); i++) {
        res[i] = static_cast<uint32_t>(topItems[i]);
    }
    return res;
}

std::vector<uint32_t> AnnoyHelper::get_suggestionsL2(
    AnnoyIndex<int, float, Euclidean, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy,
    float* qpt, int k, int search_k) {

    std::vector<int> topItems;
    std::vector<float> topDistances;
    annoy->get_nns_by_vector(qpt, k, search_k, &topItems, &topDistances);

    auto res = std::vector<uint32_t>(topItems.size(),0);
    for (size_t i = 0; i < topItems.size(); i++) {
        res[i] = static_cast<uint32_t>(topItems[i]);
    }
    return res;
}