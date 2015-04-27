import numpy as np
import scipy
from scipy.sparse import csr_matrix, hstack, vstack
import graph_tool.all as gt
from sklearn import cross_validation
from sklearn import metrics

from sklearn.metrics import classification_report
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
    # clf = KNeighborsClassifier(n_neighbors=10)
    # clf = LinearSVC(class_weight="auto")
    # clf = SGDClassifier(class_weight = "auto", loss = "squared_hinge")
    clf.fit(train_x, train_y)
    test_x = relMatrix[:, columnIndexes][fromAddresses, :]
    test_x = vt_zero_columns.transform(test_x)    
    predicted = clf.predict(test_x)
    # test_y = test_set[test_set[:, 1] == addr][:, 2] >= 1
    # print(clf.score(test_x, test_y))
    # print(metrics.confusion_matrix(test_y, predicted))
    # print(test_y)
    return predicted

# print(predictForOutput(51, relMatrix))
output_addresses = np.unique(test_set[:,1])
predictions = []
for i, address in enumerate(output_addresses):
    print("Computed for " + str(i))
    print("Address: " + str(address))
    predictions.append(predictForOutput(address, relMatrix))


# predictions = np.load("multinomialNB_predictions.npy")
predictions = np.hstack(predictions)

actual = np.hstack([test_set[test_set[:, 1] == address][:, 2] >= 1
                    for address in output_addresses])

# np.save("multinomialNB_predictions.npy", predictions)
print(metrics.confusion_matrix(actual, predictions))
print(classification_report(actual, predictions))

# eliminate column of 'to address'
# eliminate rows of linked 'from addresses'

# clf = LinearSVC(class_weight="auto").fit(relMatrix[:, indexes], relMatrix[:, 22506].toarray().ravel())

# weight = graph.new_edge_property("int32_t")
# for edge in graph.edges():
#   weight[edge] = train_set[(train_set[:,0] == graph.vertex_index[edge.source()]) & (train_set[:,1] == graph.vertex_index[edge.target()]) ][0][2]

# print("Calculated weights")

# distanceList = []
# reverseDistanceList = []
# weightedDistanceList = []
# for x in test_set:
#   distanceList.append(distance(graph, x[0], x[1]))
#   reverseDistanceList.append(reverseDistance(graph, x[0], x[1]))
#   weightedDistanceList.append(weightedDistance(graph, weight, x[0], x[1]))

# print("Calculated distanceData")

# distanceArray = np.array(distanceList)
# reverseArray = np.array(reverseDistanceList)
# weightedArray = np.array(weightedDistanceList)
# allDistance = np.transpose(np.vstack((distanceArray, reverseArray, weightedArray)))


# X_train, X_test, y_train, y_test = cross_validation.train_test_split(allDistance, test_set[:,2], test_size=0.4, random_state=0)
# clf = LinearSVC(class_weight="auto").fit(X_train, y_train)
# clf.score(X_test, y_test)                           
# clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(allDistance, test_set[:,2], test_size=0.4, random_state=2345624)
# clf = SGDClassifier(class_weight="auto").fit(X_train, y_train)
# clf.score(X_test, y_test)
# predicted = clf.predict(X_test)
# print(metrics.confusion_matrix(y_test, predicted))







