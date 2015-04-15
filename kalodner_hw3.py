import numpy as np
from scipy.sparse import csr_matrix
import graph_tool.all as gt
from sklearn import cross_validation
from sklearn import metrics

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

def distance(g, inV, outV):
	return gt.shortest_distance(g, source=g.vertex(inV), target=g.vertex(outV))

def weightedDistance(g, weightsG, inV, outV):
	return gt.shortest_distance(g, weights=weightsG, source=g.vertex(inV), target=g.vertex(outV))

def reverseDistance(g, inV, outV):
	return gt.shortest_distance(gt.GraphView(g, reversed=True, directed=True), source=g.vertex(inV), target=g.vertex(outV))

# train_set = np.genfromtxt('txTripletsCounts.txt')
# np.save("txTripletsCounts", train_set.astype('int64', casting='unsafe'))
# test_set = np.genfromtxt('testTriplets.txt')
# np.save("testTriplets", test_set.astype('int64', casting='unsafe'))

train_set = np.load("txTripletsCounts.npy").astype('int64', casting='unsafe')
test_set = np.load("testTriplets.npy").astype('int64', casting='unsafe')

graph = gt.Graph()
graph.add_edge_list(train_set[:,[0,1]])

data = np.ones((train_set.shape[0]))
columns = train_set[:,0]
rows = train_set[:,1]

relMatrix = csr_matrix((data, (rows, columns)), shape=(444075, 444075))

# relMatrix = lil_matrix((444075,444075), dtype=np.bool)
# for i in range(444075):
	# relMatrix[i, train_set[train_set[:,0] == i][:,1]] = 1


# weight = graph.new_edge_property("int32_t")
# for edge in graph.edges():
# 	weight[edge] = train_set[(train_set[:,0] == graph.vertex_index[edge.source()]) & (train_set[:,1] == graph.vertex_index[edge.target()]) ][0][2]

# print("Calculated weights")

# distanceList = []
# reverseDistanceList = []
# weightedDistanceList = []
# for x in test_set:
# 	distanceList.append(distance(graph, x[0], x[1]))
# 	reverseDistanceList.append(reverseDistance(graph, x[0], x[1]))
# 	weightedDistanceList.append(weightedDistance(graph, weight, x[0], x[1]))

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







