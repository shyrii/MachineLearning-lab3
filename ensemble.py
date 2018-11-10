import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.week_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        # 分类器的数组
        self.classifiers = self.n_weakers_limit * [0]
        # 分类器的权重
        self.classifier_w = self.n_weakers_limit * [0]
        # 样本权重
        feature_w = np.ones((X.shape[0])) / X.shape[0]

        for i in range(self.n_weakers_limit):
            self.classifiers[i] = self.week_classifier(max_depth=2)
            self.classifiers[i].fit(X, y, sample_weight=feature_w)

            prediction = self.classifiers[i].predict(X)
            prediction = prediction.reshape(prediction.shape[0], 1)

            # 计算错误率
            e = 0
            for j in range(prediction.shape[0]):
                if (y[j][0] != prediction[j][0]):
                    e += feature_w[j]
            print("e=", e)

            # 更新分类器权重
            if(e <= 0.5):
                self.classifier_w[i] = 1 / 2 * np.log((1 - e) / e)
            else:
                self.classifier_w[i] = 0

            print(self.classifier_w[i])

            #   更新样本权重
            for k in range(feature_w.shape[0]):
                feature_w[k] = feature_w[k] * np.exp(-self.classifier_w[i] * y[k][0] * prediction[k][0])
            feature_w = feature_w / np.sum(feature_w)
            # print(feature_w)

        pass


    def predict_scores(self, X, y):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = 0
        for i in range(self.n_weakers_limit):
            scores = scores + self.classifier_w[i] * self.classifiers[i].score(X, y)

        return scores
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        f = np.zeros((X.shape[0], 1))
        for i in range(self.n_weakers_limit):
            f = f + self.classifier_w[i] * self.classifiers[i].predict(X).reshape((X.shape[0], 1))
        print(f)
        f = np.sign(f)
        return f
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
