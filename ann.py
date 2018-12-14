##
# Addison Womack Short SL Project - ANN Code
#

import numpy as np

import csv

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from joblib import load, dump

# import provided graphing function from TA Ryan
from reliability_curve_stuff import plot_reliability_curve

from sklearn import metrics


def plot_roc(actual_data_class_values, predicted_probability_values, color, label):
    fpr, tpr, _ = metrics.roc_curve(actual_data_class_values, predicted_probability_values)
    plt.plot(fpr, tpr, label=label, color=color)
    plt.plot([0, 1], [0, 1], '--', linewidth=2, color='tab:gray')
    plt.title("ANN ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderpad=0.5)


def make_plots(actual_data_class_values, predicted_probability_values, title):
    fpr, tpr, _ = metrics.roc_curve(actual_data_class_values, predicted_probability_values)
    auc_score = metrics.roc_auc_score(actual_data_class_values, predicted_probability_values)
    print("AUC is " + str(auc_score))
    plt.plot(fpr, tpr, label="line indentifying label")
    plt.plot([0, 1], [0, 1], '--', linewidth=2, color='tab:gray')
    plt.title(title + " ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    plot_reliability_curve(actual_data_class_values, predicted_probability_values)
    plt.title(title + " Reliability Curve")
    plt.show()

def csv_to_matrix(filename):
    row_count = 0
    number_of_attributes = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row_count == 0:
                number_of_attributes = len(row)
            row_count += 1
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x = np.zeros(shape=(row_count, number_of_attributes - 1))
        y = np.zeros(shape=(row_count, 1))
        is_first = True
        row_count = 0
        for row in csv_reader:
            if is_first:
                is_first = False
            else:
                for i in range(0, number_of_attributes):
                    if i == (number_of_attributes - 1):
                        y[row_count] = row[i]
                    else:
                        x[row_count][i] = row[i]
                row_count += 1
        return x, y, row_count


def getCrossEntropy(model, x, y, number_of_rows, true_probability = 0.5):
    cross_entropy = 0.0

    num_true_positives = 0
    num_false_positives = 0
    num_true_negatives = 0
    num_false_negatives = 0

    predictions = np.zeros((number_of_rows, 1))

    for c in range(0, number_of_rows):
        current_sample = x[:][c].reshape(1, -1)
        y_predicted = model.predict(current_sample)
        if y_predicted >= 0.99999:
            y_predicted = 0.99999
        elif y_predicted < 0.00001:
            y_predicted = 0.00001

        predictions[c] = y_predicted

        y_actual = y[c]
        if y_actual >= 0.99999:
            y_actual = 0.99999

        cross_entropy = cross_entropy + (y_actual * np.log2(y_predicted)) + ((1 - y_actual) * np.log2(1 - y_predicted))

        if y_actual >= 0.9999:
            if y_predicted > true_probability:
                num_true_positives += 1
            else:
                num_false_negatives += 1
        else:
            if y_predicted > true_probability:
                num_false_positives += 1
            else:
                num_true_negatives += 1

    return cross_entropy / (-1.0 * number_of_rows), num_true_positives, num_false_positives, num_true_negatives, num_false_negatives, predictions


# definition of the first derivative of the sigmoid function
def sigmoid_derivative(sigmoid_function):
    return (1.0 - sigmoid_function) * sigmoid_function


# definition of a sigmoid function
def sigmoid(x):
    x[x > 500] = 500
    x[x < -500] = -500

    return 1.0 / (1 + (np.exp(-x)))


def train(ann_model, number_of_iterations=100):
    for t in range(number_of_iterations):
        ann_model.feed_forward()
        ann_model.back_propagation()


# Neural Network with 1 hidden layer
class NeuralNetwork:
    def __init__(self, x, y, num_nodes_layer_1=14, alpha=0.0005):
        # training input predictor variables
        self.input = x

        # training input expected value
        self.y = y

        # first (and only) hidden layer for this neural net
        self.hidden_layer_1 = np.ones((num_nodes_layer_1, 1))

        # output layer
        self.output = np.zeros(self.y.shape)

        # weights for input layer
        self.input_weights = np.random.rand(self.input.shape[1], num_nodes_layer_1)

        # weights for hidden layer
        self.hidden_weights = np.random.rand(num_nodes_layer_1, 1)

        # learning rate for this neural network
        self.alpha = alpha

    def predict(self, observation):
        hidden_layer1 = sigmoid(np.dot(observation, self.input_weights))
        return sigmoid(np.dot(hidden_layer1, self.hidden_weights))

    def feed_forward(self):
        self.hidden_layer_1 = sigmoid(np.dot(self.input, self.input_weights))
        self.output = sigmoid(np.dot(self.hidden_layer_1, self.hidden_weights))

    def back_propagation(self):
        # difference of actual observed y and output layer
        y_difference = self.y - self.output
        output_derivative = sigmoid_derivative(self.output)

        # Gradientk(2) = (Yactual - Ypredicted) * g'(Ak)
        gradientK2 = (y_difference * output_derivative)

        hidden_layer_update = np.dot(self.hidden_layer_1.T, gradientK2)

        # Gradientj(1) = g'(Aj) * Sum[Wjk(2) * Gradientk(2)]
        gradientj1 = (sigmoid_derivative(self.hidden_layer_1) * np.dot(gradientK2, self.hidden_weights.T))

        input_layer_update = np.dot(self.input.T, gradientj1)

        # incremental update rule ( Wjk <- Wjk + alpha(Sum (Zj Deltak)))
        # Wjk(2) = Wjk(2) + alpha * (Zj)Gradientk(2)
        self.input_weights += (self.alpha * input_layer_update)

        # Wij(1) = Wij(1) + alpha * (Xi)Gradientj(1)
        self.hidden_weights += (self.alpha * hidden_layer_update)


# Neural network with 2 hidden layers
class NeuralNetworkV2:
    def __init__(self, x, y, num_nodes_layer_1=14, num_nodes_layer_2=7, alpha=0.001):
        # training input predictor variables
        self.input = x

        # training input expected value
        self.y = y

        # first hidden layer
        self.hidden_layer_1 = np.ones((num_nodes_layer_1, num_nodes_layer_2))

        # second hidden layer
        self.hidden_layer_2 = np.ones((num_nodes_layer_2, 1))

        # output layer
        self.output = np.zeros(self.y.shape)

        # weights for input layer
        self.input_layer_weights = np.random.rand(self.input.shape[1], num_nodes_layer_1)

        # weights for first hidden layer
        self.hidden_layer_1_weights = np.random.rand(num_nodes_layer_1, num_nodes_layer_2)

        # weights for third hidden layer
        self.hidden_layer_2_weights = np.random.rand(num_nodes_layer_2, 1)

        # learning rate for this model
        self.alpha = alpha

    def predict(self, observation):
        hidden_layer1 = sigmoid(np.dot(observation, self.input_layer_weights))
        hidden_layer2 = sigmoid(np.dot(hidden_layer1, self.hidden_layer_1_weights))
        return sigmoid(np.dot(hidden_layer2, self.hidden_layer_2_weights))

    def feed_forward(self):
        self.hidden_layer_1 = sigmoid(np.dot(self.input, self.input_layer_weights))
        self.hidden_layer_2 = sigmoid(np.dot(self.hidden_layer_1, self.hidden_layer_1_weights))
        self.output = sigmoid(np.dot(self.hidden_layer_2, self.hidden_layer_2_weights))

    def back_propagation(self):
        # difference of expected y and predicted y
        y_difference = self.y - self.output
        output_derivative = sigmoid_derivative(self.output)

        # GradientL(3) = (Yactual - Ypredicted) * g'(Ak)
        gradientL3 = (y_difference * output_derivative)

        layer2update = np.dot(self.hidden_layer_2.T, gradientL3)

        # Gradientk(2) = g'(Aj) * Sum[Wjk(2) * Gradientk(2)]
        gradientK2 = (sigmoid_derivative(self.hidden_layer_2) * np.dot(gradientL3, self.hidden_layer_2_weights.T))

        layer1update = np.dot(self.hidden_layer_1.T, gradientK2)

        # Gradientj(1) = g''(Aj) * Sum[Wjk(2) * Gradientk(2)]
        gradientj1 = (sigmoid_derivative(self.hidden_layer_1) * np.dot(gradientK2, self.hidden_layer_1_weights.T))

        d_weights1 = np.dot(self.input.T, gradientj1)

        # incremental update rule ( Wjk <- Wjk + alpha(Sum (Zj Deltak)))
        # Wjk(2) = Wjk(2) + alpha * (Zj)Gradientk(2)
        self.input_layer_weights += (self.alpha * d_weights1)

        # Wij(1) = Wij(1) + alpha * (Xi)Gradientj(1)
        self.hidden_layer_1_weights += (self.alpha * layer1update)

        self.hidden_layer_2_weights += (self.alpha * layer2update)

def outputContingencyAndCE(model, x, y, num, true_probability=0.5):
    ce, tp, fp, tn, fn, predictions = getCrossEntropy(model, x, y, num, true_probability)

    print("CE is " + str(ce))
    print("num true positives = " + str(tp))
    print("num false positives = " + str(fp))
    print("num true negatives = " + str(tn))
    print("num false negatives = " + str(fn))
    print("Accuracy is " + str((1.0 * (tp + tn)) / num))

    return predictions


if __name__ == "__main__":

    validationX, validationY, numValidation = csv_to_matrix("validationSet.csv")
    testingX, testingY, numTesting = csv_to_matrix("testingSet.csv")
    trainingX, trainingY, numTraining = csv_to_matrix("trainingSet.csv")

    # model for comparison (multi-layer perceptron from Sci-Kit)
    model = MLPClassifier(max_iter=3000) #load('mlp.joblib')
    model.fit(trainingX, trainingY.ravel())
    dump(model, 'mlp.joblib')

    comparisonModelPredictions = np.ones((numTesting, 1))
    for c in range(0, numTesting):
        currentSample = testingX[:][c].reshape(1, -1)
        comparisonModelPredictions[c] = model.predict_proba(currentSample)[0][1]

    make_plots(testingY, comparisonModelPredictions, "SciKit Learn Comparison (Multi-Layer Perceptron)")

    nn = NeuralNetwork(trainingX, trainingY, alpha=0.0005) # load('oneLayer.joblib')

    train(nn, 3000)

    for a in range(5, 10):

        averageCE = 0.0
        averageAcc = 0.0

        print("-------------- for alpha of " + str(a / 10000.0) + "-----------")
        for i in range(5, 10):

            nn = NeuralNetwork(trainingX, trainingY, alpha=(a / 10000.0))
            train(nn, 1000)
            print("For validation:")
            ce, tp, fp, tn, fn, currentPredictions = getCrossEntropy(nn, validationX, validationY, numValidation, 0.5)
            accuracy = (1.0 * (tp + tn)) / numValidation

            averageCE = averageCE + (ce / 5.0)
            averageAcc = averageAcc + (accuracy / 5.0)

        print("Average CE is " + str(averageCE))
        print("Average Acc. is " + str(averageAcc))

    print("For training:")
    outputContingencyAndCE(nn, trainingX, trainingY, numTraining, 0.5)

    print("For validation:")
    outputContingencyAndCE(nn, validationX, validationY, numValidation, 0.5)

    testingY = np.delete(testingY, numTesting - 1)

    print("For Testing:")
    oneLayerPredictions = outputContingencyAndCE(nn, testingX, testingY, numTesting, 0.5)

    make_plots(testingY, oneLayerPredictions, "One Layer ANN (10th of an inch of rain)")

    dump(nn, 'oneLayer.joblib')

    # two layer NN pull in upsampled training set
    trainingX, trainingY, numTraining = csv_to_matrix("augmentedTrainingSet.csv")

    nn = NeuralNetworkV2(trainingX, trainingY, alpha=0.05) #load('twoLayer.joblib')

    print("For Testing:")
    twoLayerPredictions = outputContingencyAndCE(nn, testingX, testingY, numTesting, 0.5)

    make_plots(testingY, twoLayerPredictions)

    train(nn, 50000)

    print("For training:")
    outputContingencyAndCE(nn, trainingX, trainingY, numTraining, 0.5)

    print("For validation:")
    outputContingencyAndCE(nn, validationX, validationY, numValidation, 0.5)

    dump(nn, 'twoLayer.joblib')

    # read in third of inch files
    validationX, validationY, numValidation = csv_to_matrix("secondValidationSet.csv")
    testingX, thirdInchTestingY, thirdInchNumTesting = csv_to_matrix("secondTestingSet.csv")
    trainingX, trainingY, numTraining = csv_to_matrix("secondTrainingSet.csv")

    thirdInchTestingY = np.delete(thirdInchTestingY, thirdInchNumTesting - 1)

    nn = NeuralNetwork(trainingX, trainingY, alpha=0.0005) #load('thirdOfInchOneLayer.joblib')

    train(nn, 3000)

    dump(nn, 'thirdOfInchOneLayer.joblib')

    print("Third of Inch For Testing:")
    thirdInchPredictions = outputContingencyAndCE(nn, testingX, testingY, numTesting, 0.5)

    make_plots(testingY, thirdInchPredictions, "One Layer ANN (3rd of an inch of rain)")

    plot_roc(testingY, comparisonModelPredictions, color='tab:red', label='MLP')
    plot_roc(thirdInchTestingY, thirdInchPredictions, color='tab:blue', label='ANN: Third Inch Data')
    plot_roc(testingY, oneLayerPredictions, color='tab:green', label='ANN: Tenth of Inch Data')
    plt.show()
