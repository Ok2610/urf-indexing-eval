#ifndef _ANNOYHELPER
#define _ANNOYHELPER

#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

using namespace Annoy;

class AnnoyHelper {
    public:
    AnnoyHelper(int dim, int search_k);

    ~AnnoyHelper();

    std::vector<uint32_t> get_suggestionsDot(
        AnnoyIndex<int, float, DotProduct, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy,
        float* qpt, int k);

    std::vector<uint32_t> get_suggestionsDot(
        AnnoyIndex<int, float, DotProduct, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy,
        float* qpt, int k, int search_k);

    std::vector<uint32_t> get_suggestionsL2(
        AnnoyIndex<int, float, Euclidean, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy,
        float* qpt, int k);

    std::vector<uint32_t> get_suggestionsL2(
        AnnoyIndex<int, float, Euclidean, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>*& annoy,
        float* qpt, int k, int search_k);

    private:
    int _search_k;
    int _dim;
};

#endif //_ANNOYHELPER