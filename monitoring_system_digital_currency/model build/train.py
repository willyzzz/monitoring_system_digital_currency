from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import random


def assemble_layer(data):
	output = {}
	keys = data[0].keys()
	for key in keys:
		value = 0
		for inner_data in data:
			value += float(inner_data[key])
		output[key] = value/len(data)
	output = sorted(output.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
	return output

def min_max_normalization_layer(features):
	normal_features = []
	_min = min(features)
	_max = max(features)
	for i in range(len(features)):
		normal_features.append((features[i] - _min) / (_max - _min))
	return normal_features

def categorical_layer(features):
	feature_kinds = list(set(features))
	kind_dict = {}
	for i in range(len(feature_kinds)):
		kind_dict[feature_kinds[i]] = i
	multi_features = []
	for i in range(len(features)):
		ss = [0]*len(feature_kinds)
		ss[kind_dict[features[i]]] = 1
		multi_features.append(ss)
	return multi_features

def _LR(x_train, y_train, x_test):
	lr_model = LogisticRegression()
	lr_model.fit(x_train, y_train)
	probability = []
	for binary_scores in lr_model.predict_proba(x_test):
		probability.append(binary_scores[1])
	return probability

def _SVC(x_train, y_train, x_test):
	svc_model = SVC(probability=True)
	svc_model.fit(x_train, y_train)
	probability = []
	for binary_scores in svc_model.predict_proba(x_test):
		probability.append(binary_scores[1])
	return probability

# def _BYS(x_train, y_train, x_test):
# 	bys_model = MultinomialNB()
# 	bys_model.fit(x_train, y_train)
# 	probability = []
# 	for binary_scores in bys_model.predict_proba(x_test):
# 		probability.append(binary_scores[1])
# 	return probability

def _KNN(x_train, y_train, x_test):
	knn_model = KNeighborsClassifier()
	knn_model.fit(x_train, y_train)
	probability = []
	for binary_scores in knn_model.predict_proba(x_test):
		probability.append(binary_scores[1])
	return probability

def _RF(x_train, y_train, x_test):
	rf_model = RandomForestClassifier()
	rf_model.fit(x_train, y_train)
	probability = []
	for binary_scores in rf_model.predict_proba(x_test):
		probability.append(binary_scores[1])
	return probability

def _LGB(x_train, y_train, x_test):
	lgb_model = lgb.LGBMClassifier()
	lgb_model.fit(x_train, y_train)
	probability = []
	for binary_scores in lgb_model.predict_proba(x_test):
		probability.append(binary_scores[1])
	return probability

def _MLP(x_train, y_train, x_test):
	mlp_model = MLPClassifier(solver='sgd', activation='relu', max_iter=100, alpha=1e-5, hidden_layer_sizes=(40, 80, 40), random_state=1, verbose = False)
	mlp_model.fit(x_train, y_train)
	probability = []
	for binary_scores in mlp_model.predict_proba(x_test):
		probability.append(binary_scores[1])
	return probability

def read_file(file):
	name_list = []
	feature_list = []
	count = 0
	for line in open(file):
		count += 1
		if count > 1:
			data = line.split(',')
			data[-1] = data[-1].replace('\n', '')
			name_list.append(data[1])
			# print(data)
			feature_list.append([float(x) for x in data[2:]])
	if file == 'positive_data.csv':
		for i in range(len(feature_list)):
			feature_list[i].append(1.0)
		return name_list, feature_list
	elif file == 'negative_data.csv':
		for i in range(len(feature_list)):
			feature_list[i].append(0.0)
		return name_list, feature_list
	else:
		return name_list, feature_list

def get_feature_and_label(data):
	random.shuffle(data)
	x_train = []
	y_train = []
	for inner_data in data:
		x_train.append(inner_data[:len(inner_data)-1])
		y_train.append(inner_data[-1])
	return np.array(x_train), np.array(y_train)


positive_names, positive_features = read_file('positive_data.csv')
negative_names, negative_features = read_file('negative_data.csv')
to_predict_names, to_predict_features = read_file('etherscan_tag_data.csv')

sampling_ratio = 3
data = positive_features + random.sample(negative_features, min(len(negative_features), len(positive_features)*sampling_ratio))
x_train, y_train = get_feature_and_label(data)
x_test = to_predict_features


probability_1 = _LR(x_train, y_train, x_test)
probability_2 = _SVC(x_train, y_train, x_test)
probability_3 = _KNN(x_train, y_train, x_test)
probability_4 = _RF(x_train, y_train, x_test)
probability_5 = _LGB(x_train, y_train, x_test)
probability_6 = _MLP(x_train, y_train, x_test)


dict_1 = {}
dict_2 = {}
dict_3 = {}
dict_4 = {}
dict_5 = {}
dict_6 = {}
for i in range(len(to_predict_names)):
	dict_1[to_predict_names[i]] = probability_1[i]
	dict_2[to_predict_names[i]] = probability_2[i]
	dict_3[to_predict_names[i]] = probability_3[i]
	dict_4[to_predict_names[i]] = probability_4[i]
	dict_5[to_predict_names[i]] = probability_5[i]
	dict_6[to_predict_names[i]] = probability_6[i]

output = assemble_layer([dict_1, dict_2, dict_3, dict_4, dict_5, dict_6])
for results in output[:10]:
	print(results[0], results[1])
