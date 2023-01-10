from sklearn.datasets import load_boston
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, rand_score
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch



warnings.filterwarnings("ignore")
boston = load_boston()
df = pd.DataFrame(boston.data)
df.columns = [boston.feature_names]

df['PRICE'] = boston.target
df.isnull()

# print(df.describe())
print(df)
print(df.corr())

mask = np.zeros_like(df.corr())
plt.figure(figsize=(16,10))
sns.heatmap(df.corr(), mask=mask, annot=True, annot_kws={'size': 14})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()




prices = np.log(df['PRICE'])
features = df.drop('PRICE', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size = 0.2, random_state=10)


regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Коэффициент детерминации(тренировочные данные): ', regr.score(X_train, Y_train))
print('Коэффициент детерминации(тестовые данные): ', regr.score(X_test, Y_test))
print('Интерцепт: ', regr.intercept_[0])


X_incl_const = sm.add_constant(X_train)
model = sm.OLS(Y_train, X_incl_const)
results = model.fit()

reduced_log_mse = round(results.mse_resid, 3)

print('2 СКО в log:', 2 * np.sqrt(reduced_log_mse))

upper_price = np.log(30) + 2 * np.sqrt(reduced_log_mse)
print('Максимальная цена: $', np.e ** upper_price * 1000)

lower_price = np.log(30) - 2 * np.sqrt(reduced_log_mse)
print('Минимальная цена: $', np.e ** lower_price * 1000)


pred = regr.predict(X_test)
print('mean_squared_error : ', mean_squared_error(Y_test, pred))
print('mean_absolute_error : ', mean_absolute_error(Y_test, pred))



my_dict = {'PRICE': df['PRICE'], 'RM': df['RM']}
X = pd.DataFrame([my_dict])

X = np.array(X, dtype=float)

wcss = []
for i in range(1, 11):
	kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

df2 = df
y_kmeans_list = list(y_kmeans)

for i in range(len(y_kmeans_list)):
	y_kmeans_list[i] += 1

tmp = list(df.iloc[:, 6].values)
k = 0
clusters = []

for i in range(len(tmp)):
	clusters.append(y_kmeans_list[k])
	k += 1

true_clusters = []
pricesT = list(df2['PRICE'])

for i in range(len(pricesT)):
	if float(pricesT[i]) < 21.6:
		true_clusters.append(2)
	else:
		true_clusters.append(1)

df2['Cluster'] = clusters
df2['True cluster'] = true_clusters
# df2.to_excel('cluster.xlsx')
df2.head()




plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')

plt.title('Clusters')
plt.xlabel('Price')
plt.ylabel('RM')
plt.legend()

plt.show()

X_old = df.iloc[:, [6, 6]].values
for i in range(len(X_old)):
	if X_old[i][1] < 7.185:
		X_old[i] = 0
X = []
for i in range(len(X_old)):
	if sum(X_old[i]) == 0:
		pass
	elif sum(X_old[i]) > 0:
		X.append([X_old[i][0], X_old[i][1]])
X = np.array(X, dtype=float)
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('RM')
plt.ylabel('Euclidean distances')
plt.show()