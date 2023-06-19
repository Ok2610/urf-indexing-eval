//
// Classifier designed for Exquisitor
// May create an interface/superclass of this, to allow additional classifier solutions.
//

#ifndef EXQUISITOR_EXQCLASSIFIER_H
#define EXQUISITOR_EXQCLASSIFIER_H

#include <opencv2/ml.hpp>

namespace exq {

    using std::vector;
    using cv::Ptr;
    using cv::ml::SVMSGD;
    using cv::ml::TrainData;

    class ExqClassifier {
    public:
        ExqClassifier(int totalFeats);

        ~ExqClassifier();

        void reset_classifier();

        vector<float> train(vector<vector<float>> data, vector<float> labels);

        vector<float> get_weights();

        double get_bias();

        int get_total_feats();

    protected:
        int _totalFeats;
        Ptr<SVMSGD> _svm;
        vector<float> _weights;

    }; //End of class ExqClassifier

    inline int ExqClassifier::get_total_feats() { return _totalFeats; }

} //End of namespace exq

#endif //EXQUISITOR_EXQCLASSIFIER_H
