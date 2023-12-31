#include "ExqClassifier.h"

using namespace exq;
using std::cout;
using std::endl;

using cv::Ptr;
using cv::ml::TrainData;

ExqClassifier::ExqClassifier(int totalFeats) {
    _svm = SVMSGD::create();
    reset_classifier();
//    _svm->setSvmsgdType(SVMSGD::ASGD);
//    _svm->setOptimalParameters();
//    _svm->setMarginType(SVMSGD::HARD_MARGIN);
//    _svm->setMarginRegularization(0.0001);
//    _svm->setInitialStepSize(0.0001);
//    _svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 0.01));
    _totalFeats = totalFeats;
}

ExqClassifier::~ExqClassifier() {
    if (_svm->isTrained()) {
        _weights.clear();
        _svm->clear();
        _svm.release();
    }
}

void ExqClassifier::reset_classifier() {
    if (_svm->isTrained()) {
        _weights.clear();
        _svm->clear();
        _svm.release();
        _svm = Ptr<SVMSGD>();
        _svm = SVMSGD::create();
    }
    _svm->setSvmsgdType(SVMSGD::ASGD);
    _svm->setOptimalParameters();
    _svm->setMarginType(SVMSGD::HARD_MARGIN);
    _svm->setMarginRegularization(0.01);
    _svm->setInitialStepSize(0.01);
    _svm->setStepDecreasingPower(0.75);
    _svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT, 1000, 1));
}

std::vector<float> ExqClassifier::train(vector<vector<float>> data, vector<float> labels) {
#if defined(DEBUG) || defined(DEBUG_TRAIN)
    for (int i = 0; i < (int)data.size(); i++) {
        for (int j = 0; j < (int)data[i].size(); j++) {
            if (data[i][j] > 0.0) {
                cout << "Item " << i << " Feature (" << j << "," << data[i][j] << ")" << endl;
            }
        }
        cout << "Label " << labels[i] << endl;
    }
#endif
    //Controller calculates scores and creates the 2D data vector
    int rows = data.size();
    int cols = data[0].size();
    cv::Mat labelsMat(rows, 1, CV_32FC1);
    cv::Mat dataMat(rows, cols, CV_32F);
    for (int i = 0; i < rows; i++) {
        labelsMat.at<float>(i,0) = labels[i];
        for (int j = 0; j < cols; j++) {
            dataMat.at<float>(i,j) = data[i][j];
        }
    }

#if defined(DEBUG_EXTRA) || defined(DEBUG_TRAIN_EXTRA)
    for (int i = 0; i < rows; i++) {
        cout << "Item " << i << endl;
        for (int j = 0; j < cols; j++) {
            cout << dataMat.at<float>(i,j) << ",";
        }
        cout << endl;
    }
#endif

    _svm->train(dataMat, cv::ml::ROW_SAMPLE, labelsMat);
    cv::Mat sv = _svm->getWeights();
    rows = sv.rows;
    cols = sv.cols;
#if defined(DEBUG) || defined(DEBUG_TRAIN)
    cout << "Number of support vectors: " << rows << endl;
    cout << "Number of features in support vectors: " << cols << endl;
#endif
    _weights.resize(cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            _weights[c] += sv.at<float>(r,c);
        }
    }

#if defined(DEBUG) || defined(DEBUG_TRAIN)
    cout << "(ExqClassifier) Non zero weights: " << endl;
    for (int i = 0; i < (int)_weights.size(); i++) {
        if (_weights[i] != 0.0) {
            cout << i << " " << _weights[i] << endl;
        }
    }
    cout << "bias: " << _svm->getShift() << endl;
#endif
    return _weights;
}

vector<float> ExqClassifier::get_weights() {
    return _weights;
}

double ExqClassifier::get_bias() {
    return _svm->getShift();
}
