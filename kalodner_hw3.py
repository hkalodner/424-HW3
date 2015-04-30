import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import csr_matrix, hstack, vstack
import graph_tool.all as gt
from sklearn import cross_validation
from sklearn import metrics

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
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

def getDistancesToOutput(g, outV):
    return gt.shortest_distance(g, source=g.vertex(outV))

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
    # clf = MultinomialNB()
    # clf = KNeighborsClassifier(n_neighbors=5)
    clf = LinearSVC(class_weight="auto")
    # clf = LogisticRegression(class_weight="auto")
    # clf = SGDClassifier(class_weight = "auto", loss = "squared_hinge")
    clf.fit(train_x, train_y)
    test_x = relMatrix[:, columnIndexes][fromAddresses, :]
    test_x = vt_zero_columns.transform(test_x)    
    predicted = clf.predict(test_x)
    # predicted = clf.predict_proba(test_x)
    # if True in clf.classes_:
    #     true_column = np.where(clf.classes_ == True)[0][0]
    #     predicted = predicted[:, true_column]
    # else:
    #     predicted = np.zeros(test_x.shape[0])
    # test_y = test_set[test_set[:, 1] == addr][:, 2] >= 1
    # print(clf.score(test_x, test_y))
    # print(metrics.confusion_matrix(test_y, predicted))
    # print(test_y)
    return predicted

output_addresses = np.unique(test_set[:,1])
# print(predictForOutput(51, relMatrix))
# print(predictForOutput(239761, relMatrix))
# predictions = []

# for i, address in enumerate(output_addresses):
#     print("Computed for " + str(i))
#     print("Address: " + str(address))
#     predictions.append(predictForOutput(address, relMatrix))


# predictions = np.hstack(predictions)
# np.save("KNN_prob_predictions.npy", predictions)
# np.save("multinomialNB_prob_predictions.npy", predictions)
# np.save("SVC_predictions.npy", predictions)
# predictions = np.load("multinomialNB_predictions.npy")
# predictions = np.load("multinomialNB_prob_predictions.npy")

# actual = np.hstack([test_set[test_set[:, 1] == address][:, 2] >= 1
#                     for address in output_addresses])


# print(classification_report(actual, predictions))

# Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(actual, predictions)
# roc_auc = auc(fpr, tpr)

# nb_fpr, nb_tpr, nb_thresholds = roc_curve(actual, nb_predictions)
# nb_roc_auc = auc(nb_fpr, nb_tpr)

# knn_fpr, knn_tpr, knn_thresholds = roc_curve(actual, knn_predictions)
# knn_roc_auc = auc(knn_fpr, knn_tpr)

# # Pltot ROC curve
# plt.clf()
# # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot(knn_fpr, knn_tpr, label='KNN (area = %0.2f)' % knn_roc_auc)
# plt.plot(nb_fpr, nb_tpr, label='NB (area = %0.2f)' % nb_roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# transaction_counts = np.zeros(444075)
# for row in train_set:
#     transaction_counts[row[0]] += row[2]

# num_transactions = transaction_counts

def predictLessFeatures(addr, relMatrix):

    indexes = np.arange(444075)

    distances = np.array(getDistancesFromOutput(graph, addr).a)
    distances_output_sending = np.array(getDistancesToOutput(graph, addr).a)
    train_data = np.transpose(np.vstack((distances,
                                         num_transactions,
                                         distances_output_sending)))
    fromAddresses = test_set[test_set[:, 1] == addr][:, 0]
    rowIndexes = np.delete(indexes, fromAddresses)
    train_x = train_data[rowIndexes, :]
    train_y = relMatrix[:, addr][rowIndexes, :].toarray().ravel() >= 1

    # clf = MultinomialNB()
    # clf = KNeighborsClassifier(n_neighbors=5)
    # clf = LinearSVC(class_weight="auto")
    clf = LogisticRegression(class_weight="auto")
    # clf = SGDClassifier(class_weight = "auto", loss = "squared_hinge")
    clf.fit(train_x, train_y)
    test_x = train_data[fromAddresses, :]
    predicted = clf.predict(test_x)
    # predicted = clf.predict_proba(test_x)
    # if True in clf.classes_:
    #     true_column = np.where(clf.classes_ == True)[0][0]
    #     predicted = predicted[:, true_column]
    # else:
    #     predicted = np.zeros(test_x.shape[0])
    # test_y = test_set[test_set[:, 1] == addr][:, 2] >= 1
    # test_y = test_set[test_set[:, 1] == addr][:, 2] >= 1
    # print(clf.score(test_x, test_y))
    # print(metrics.confusion_matrix(test_y, predicted))
    # print(test_y)
    return predicted

# predictions = []

# for i, address in enumerate(output_addresses):
#     print("Computed for " + str(i))
#     print("Address: " + str(address))
#     predictions.append(predictLessFeatures(address, relMatrix))


# predictions = np.hstack(predictions)
# np.save("logistic_regression_predictions.npy", predictions)

# actual = np.hstack([test_set[test_set[:, 1] == address][:, 2] >= 1
#                     for address in output_addresses])

# print(classification_report(actual, predictions))

# Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(actual, predictions)
# roc_auc = auc(fpr, tpr)

# nb_fpr, nb_tpr, nb_thresholds = roc_curve(actual, nb_predictions)
# nb_roc_auc = auc(nb_fpr, nb_tpr)

# knn_fpr, knn_tpr, knn_thresholds = roc_curve(actual, knn_predictions)
# knn_roc_auc = auc(knn_fpr, knn_tpr)

# # Pltot ROC curve
# plt.clf()
# # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot(knn_fpr, knn_tpr, label='KNN (area = %0.2f)' % knn_roc_auc)
# plt.plot(nb_fpr, nb_tpr, label='NB (area = %0.2f)' % nb_roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

clf = DummyClassifier()
clf.fit(test_set[:, 0], test_set[:, 2])
prediction_proba = clf.predict_proba(test_set[:, 0])
prediction = clf.predict(test_set[:, 0])

fpr, tpr, thresholds = roc_curve(test_set[:, 2], prediction_proba[:, 1])
roc_auc = auc(fpr, tpr)

print(classification_report(test_set[:, 2], prediction))

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
