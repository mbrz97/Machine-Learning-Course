import math
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticActivation:
    def Activate(self, value):
        return (1.0 / ((1 + math.pow(math.e, -value))))

    def Deactivate(self, value):
        return (value * ((1 - value)))

    def GetName(self):
        return "sigmoid"


class Neuron:
    def __init__(self, d):
        self._Weights = None
        self._WeightsPrevious = None
        self._Value = None
        self._PartialDelta = None
        self.Weights = [random.uniform(0, 1) for _ in range(d)]
        self.WeightsPrevious = [random.uniform(0, 1) for _ in range(d)]

    @property
    def Weights(self):
        return self._Weights

    @Weights.setter
    def Weights(self, value):
        self._Weights = value

    @property
    def WeightsPrevious(self):
        return self._WeightsPrevious

    @WeightsPrevious.setter
    def WeightsPrevious(self, value):
        self._WeightsPrevious = value

    def UpdateWeight(self, position, newValue):
        self.WeightsPrevious[position] = self.Weights[position]
        self.Weights[position] = newValue

    @property
    def Value(self):
        return self._Value

    @Value.setter
    def Value(self, value):
        self._Value = value

    @property
    def PartialDelta(self):
        return self._PartialDelta

    @PartialDelta.setter
    def PartialDelta(self, value):
        self._PartialDelta = value


class MLPClassifier:
    def __init__(self, layerSizes, randomSeed=None, verbose=False, learningRate=0.05):
        self.network = []
        self.totalError = 0

        if randomSeed != None:
            random.seed(randomSeed)

        self.activation = LogisticActivation()

        first = [0 for _ in range(layerSizes[0])]
        i = 0
        while i < len(first):
            first[i] = Neuron(0)
            i += 1
        self.network.append(first)

        i = 1
        while i < len(layerSizes):
            next = [0 for _ in range(layerSizes[i])]
            j = 0
            while j < len(next):
                next[j] = Neuron(layerSizes[(i - 1)])
                j += 1
            self.network.append(next)
            i += 1

        self.bias = [0 for _ in range(len(layerSizes))]
        i = 0
        while i < len(self.bias):
            self.bias[i] = random.uniform(0, 1)
            i += 1
        self.verbose = verbose
        self.learningRate = learningRate

    def Fit(self, X, y, numberOfEpochs=1):
        e = 0
        while e < numberOfEpochs:
            i = 0
            while i < len(X):
                self.Forward(X[i])
                totalError = self.CalculateError(y[i])
                self.BackwardOutputLayer(y[i])
                self.BackwardHiddenLayer()
                i += 1
            self.Print("epoch: {}. total error: {}".format(e, totalError))
            e += 1

    def Predict(self, X):
        predictions = []
        i = 0
        while (i < len(X)):
            p = self.Forward(X[i])
            predictions.append(p)
            i += 1
        return predictions

    def Forward(self, inputVector):
        i = 0
        while i < len(inputVector):
            self.network[0][i].Value = inputVector[i]
            i += 1

        L = 1
        while L < len(self.network):
            xi = 0
            while xi < len(self.network[L]):
                neuron = self.network[L][xi]
                z = 0
                xj = 0
                while xj < len(self.network[(L - 1)]):
                    sn = self.network[(L - 1)][xj]
                    w = neuron.Weights[xj]
                    x = sn.Value
                    z += (x * w)
                    xj += 1
                z += self.bias[L]
                neuron.Value = self.activation.Activate(z)
                xi += 1
            L += 1

        last = self.network[(len(self.network) - 1)]
        result = [0 for _ in range(len(last))]
        i = 0
        while i < len(result):
            result[i] = last[i].Value
            i += 1
        return result

    def Print(self, m):
        if not self.verbose:
            return
        print(m)

    def BackwardOutputLayer(self, y):
        outputLayer = self.network[-1]
        i = 0
        while i < len(outputLayer):
            lastHiddenLayer = self.network[(len(self.network) - 2)]
            error = -(y[i] - outputLayer[i].Value)
            delta = (error * self.activation.Deactivate(outputLayer[i].Value))
            outputLayer[i].PartialDelta = delta
            j = 0
            while j < len(lastHiddenLayer):
                d = (delta * lastHiddenLayer[j].Value)
                newWeight = (outputLayer[i].Weights[j] -
                             ((self.learningRate * d)))
                outputLayer[i].UpdateWeight(j, newWeight)
                j += 1
            i += 1

    def BackwardHiddenLayer(self):
        i = (len(self.network) - 2)
        while i >= 1:
            layer = self.network[i]
            outputLayer = self.network[(len(self.network) - 1)]
            inputLayer = self.network[(i - 1)]
            ni = 0
            while ni < len(layer):
                node = layer[ni]
                sumPartial = 0
                mi = 0
                while mi < len(outputLayer):
                    sumPartial += (outputLayer[mi].WeightsPrevious[ni]
                                   * outputLayer[mi].PartialDelta)
                    mi += 1
                wi = 0
                while wi < len(node.Weights):
                    delta = (
                            (sumPartial * self.activation.Deactivate(node.Value)) * inputLayer[wi].Value)
                    newWeight = (node.Weights[wi] -
                                 (self.learningRate * delta))
                    node.UpdateWeight(wi, newWeight)
                    wi += 1
                ni += 1
            i -= 1

    def CalculateError(self, y):
        last = self.network[(len(self.network) - 1)]
        aL = [0 for _ in range(len(last))]
        i = 0
        while i < len(aL):
            aL[i] = last[i].Value
            i += 1
        cost = 0
        x = 0
        while x < len(y):
            cost += (math.pow(abs((y[x] - aL[x])), 2) / 2)
            x += 1
        return cost


def box(xs):
    return [[x] for x in xs]


# Load the data into a pandas dataframe
df = pd.read_csv('breastcancer.csv')

# Drop the 32th column (NaN values)
df = df.drop(df.columns[32], axis=1)

# Split the data into training and test sets
X = df.drop(['id', 'diagnosis'], axis=1)
ss = StandardScaler()
X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
print(X)

# Encode the diagnosis column (M = 1, B = 0)
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Train the MLP classifier
mlp = MLPClassifier(layerSizes=[30, 10, 10, 10, 1], verbose=False)
mlp.Fit(X_train.values.tolist(), box(
    y_train.values.tolist()), numberOfEpochs=100)

# Predict the labels of the test set
y_pred = mlp.Predict(X_test.values.tolist())
y_pred = [0 if y[0] <= 0.5 else 1 for y in y_pred]

for i in range(len(y_pred)):
    print(y_pred[i])

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, pd.DataFrame(y_pred))
print("Accuracy:", accuracy)
