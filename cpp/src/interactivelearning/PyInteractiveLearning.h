#include "HnswHelper.h"
#include "AnnoyHelper.h"
#include "IVFHelper.h"
#include "exq/ExqClassifier.h"

namespace il {
    struct IndexCtrl {
        HNSWHelper* hnsw_helper;
        AnnoyHelper* annoy_helper;
        IVFHelper* ivf_helper;
        AnnoyIndex<int, float, Annoy::DotProduct, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoy_index_dot;
        AnnoyIndex<int, float, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoy_index_L2;
    };

    struct PyInteractiveLearning {
        int _mtype;
        int _indx_type;
        int _dim;
        int _metric;
        IndexCtrl _indx;
        exq::ExqClassifier* _classifier;

        virtual std::vector<uint32_t> get_suggestions(int k, int search_par) = 0;
    };

    struct PyInteractiveLearningFloat : PyInteractiveLearning {
        PyInteractiveLearningFloat(int indx_type, const char* indx_path, int metric, int dim, int search_par) {
            _mtype = 0;
            _indx_type = indx_type;
            _dim = dim;
            _metric = metric;
            _classifier = new exq::ExqClassifier(dim);

            if (_indx_type == 0) {
                _indx.hnsw_helper = new HNSWHelper(indx_path, metric, dim, search_par);
            } else if (_indx_type == 1) {
                _indx.annoy_helper = new AnnoyHelper(dim, search_par);
                if (_metric == 0) {
                    std::cout << "Loading Annoy Index with IP" << std::endl;
                    _indx.annoy_index_dot = new AnnoyIndex<int, float, DotProduct, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>(_dim);
                    _indx.annoy_index_dot->load(indx_path);
                } else if (_metric == 1) {
                    std::cout << "Loading Annoy Index with L2" << std::endl;
                    _indx.annoy_index_L2 = new AnnoyIndex<int, float, Euclidean, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>(_dim);
                    _indx.annoy_index_L2->load(indx_path);
                }
            } else if (_indx_type == 2) {
                _indx.ivf_helper = new IVFHelper(indx_path, metric, dim, search_par);
            } else {
                const char* err_msg = "Undefined Index Choice " + static_cast<char>(_indx_type);
                PyErr_SetString(PyExc_RuntimeError, err_msg);
            }
        };

        std::vector<uint32_t> get_suggestions(int k, int search_par) {
            if (_indx_type == 0) {
                if (search_par == -1)
                    return _indx.hnsw_helper->get_suggestions(_classifier->get_weights().data(), k);
                else
                    return _indx.hnsw_helper->get_suggestions(_classifier->get_weights().data(), k, search_par);
            } else if (_indx_type == 1) {
                if (search_par == -1) {
                    if (_metric == 0)
                        return _indx.annoy_helper->get_suggestionsDot(_indx.annoy_index_dot, _classifier->get_weights().data(), k);
                    else
                        return _indx.annoy_helper->get_suggestionsL2(_indx.annoy_index_L2, _classifier->get_weights().data(), k);
                } else {
                    if (_metric == 0)
                        return _indx.annoy_helper->get_suggestionsDot(_indx.annoy_index_dot, _classifier->get_weights().data(), k, search_par);
                    else
                        return _indx.annoy_helper->get_suggestionsL2(_indx.annoy_index_L2, _classifier->get_weights().data(), k, search_par);
                }

            } else if (_indx_type == 2) {
                if (search_par == -1)
                    return _indx.ivf_helper->get_suggestions(_classifier->get_weights().data(), k);
                else
                    return _indx.ivf_helper->get_suggestions(_classifier->get_weights().data(), k, search_par);
            }
        };

        ~PyInteractiveLearningFloat() {
            if (_indx_type == 0) {
                delete _indx.hnsw_helper;
            } else if (_indx_type == 1) {
                if (_metric == 0) {
                    _indx.annoy_index_dot->unload(); 
                    delete _indx.annoy_index_dot;
                } else if (_metric == 1) {
                    _indx.annoy_index_L2->unload(); 
                    delete _indx.annoy_index_L2;
                }
                delete _indx.annoy_helper;
            } else if (_indx_type == 2) {}

            delete _classifier;
        };
    };

    struct PyInteractiveLearningR64 : PyInteractiveLearning {
        HNSWHelper* hnsw_helper;

        std::vector<uint32_t> get_suggestions(int k, int search_par) {
            std::vector<uint32_t> res;
            return res;
        }
    };
}