import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import csr_matrix, hstack, vstack
import graph_tool.all as gt
from sklearn import cross_validation
from sklearn import metrics

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold

def distance(g, inV, outV):
    return gt.shortest_distance(g, source=g.vertex(inV), target=g.vertex(outV))

def weightedDistance(g, weightsG, inV, outV):
    return gt.shortest_distance(g, weights=weightsG, source=g.vertex(inV), target=g.vertex(outV))

def reverseDistance(g, inV, outV):
    return gt.shortest_distance(gt.GraphView(g, reversed=True, directed=True), source=g.vertex(inV), target=g.vertex(outV))

def getDistancesFromOutput(g, outV):
    return gt.shortest_distance(gt.GraphView(g, reversed=True, directed=True), source=g.vertex(outV))

# train_set = np.genfromtxt('txTripletsCounts.txt')
# np.save("txTripletsCounts", train_set.astype('int64', casting='unsafe'))
# test_set = np.genfromtxt('testTriplets.txt')
# np.save("testTriplets", test_set.astype('int64', casting='unsafe'))

train_set = np.load("txTripletsCounts.npy").astype('int64', casting='unsafe')
test_set = np.load("testTriplets.npy").astype('int64', casting='unsafe')

graph = gt.Graph()
graph.add_edge_list(train_set[:,[0,1]])

# data = np.ones((train_set.shape[0])) == 1.0
data = train_set[:, 2]
columns = train_set[:,0]
rows = train_set[:,1]

relMatrix = csr_matrix((data, (columns, rows)), shape=(444075, 444075))

def predictForOutput(addr, relMatrix):

    indexes = np.arange(444075)

    distances = csr_matrix(np.array(getDistancesFromOutput(graph, addr).a))
    relMatrix = csr_matrix(hstack((relMatrix, np.transpose(distances))))

    fromAddresses = test_set[test_set[:, 1] == addr][:, 0]
    columnIndexes = indexes[indexes != addr]
    rowIndexes = np.delete(indexes, fromAddresses)
    train_x = relMatrix[:, columnIndexes][rowIndexes, :]
    train_y = relMatrix[:, addr][rowIndexes, :].toarray().ravel() >= 1

    # ch2 = SelectKBest(f_regression, 10000)
    vt_zero_columns = VarianceThreshold()
    train_x = vt_zero_columns.fit_transform(train_x, train_y)
    clf = MultinomialNB()
    # clf = KNeighborsClassifier(n_neighbors=5)
    # clf = LinearSVC(class_weight="auto")
    # clf = SGDClassifier(class_weight = "auto", loss = "squared_hinge")
    clf.fit(train_x, train_y)
    test_x = relMatrix[:, columnIndexes][fromAddresses, :]
    test_x = vt_zero_columns.transform(test_x)    
    # predicted = clf.predict(test_x)
    predicted = clf.predict_proba(test_x)
    if True in clf.classes_:
        true_column = np.where(clf.classes_ == True)[0][0]
        predicted = predicted[:, true_column]
    else:
        predicted = np.zeros(len(test_x))
    # test_y = test_set[test_set[:, 1] == addr][:, 2] >= 1
    # print(clf.score(test_x, test_y))
    # print(metrics.confusion_matrix(test_y, predicted))
    # print(test_y)
    return predicted

# print(predictForOutput(51, relMatrix))
# print(predictForOutput(239761, relMatrix))
predictions = []
output_addresses = np.unique(test_set[:,1])
for i, address in enumerate(output_addresses):
    print("Computed for " + str(i))
    print("Address: " + str(address))
    predictions.append(predictForOutput(address, relMatrix))

    # predictions = np.load("multinomialNB_predictions.npy")
predictions = np.hstack(predictions)
# np.save("KNN_prob_predictions.npy", predictions)
np.save("multinomialNB_prob_predictions.npy", predictions)
# predictions = np.load("multinomialNB_prob_predictions.npy")

actual = np.hstack([test_set[test_set[:, 1] == address][:, 2] >= 1
                    for address in output_addresses])

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(actual, predictions)
roc_auc = auc(fpr, tpr)

# Pltot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()






