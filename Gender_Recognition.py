import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# read input file
data = pd.read_csv('voices.csv')
print(data.head()) # will give top five rows

print(data.shape)
print(data.describe())
print(data.isnull().values.any()) # check data has any values null/nan or not)

column_names = data.columns.tolist()
print(column_names)   # print columns name


data = data.sample(frac=1)


# Check how many datas are of male and female
print("number of male: ", data[data['gender'] == 'male'].shape[0])
print("number of female: ", data[data['gender'] == 'female'].shape[0])

# for checking difference between mal and female
a = data[data['gender'] == 'male'].mean()
b = data[data['gender'] == 'female'].mean()

# Distribution of male and female it's another way of box plot

sns.FacetGrid(data, hue="gender").map(sns.kdeplot, "mean").add_legend()


# Distribution of male and female
sns.FacetGrid(data, hue="gender").map(sns.kdeplot, "median").add_legend()

sns.FacetGrid(data, hue="gender").map(sns.kdeplot, "iqr").add_legend()



######################################## SVM ##########################################
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# convert srting data into numberic eg. male 1, female 0
df = data
df = df.drop(['gender'],axis = 1)
df = df.drop(['id'],axis = 1)
X = df.values
y = data['gender'].values


# plot correlation matrix
correlation = df.corr()
f, ax = plt.subplots(figsize = (8,8))
# Draw the heatmap using seaborn
sns.heatmap(correlation, square = True)
plt.show()
# only one column has object type so we encode it
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print("labels:", y)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Running SVM 
from sklearn.svm import SVC
from sklearn import metrics
svc = SVC(kernel='rbf', random_state=0, C=5)
svc.fit(X_train, y_train)
y_pred1 = svc.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_test,y_pred1))
print("precision:", metrics.precision_score(y_test,y_pred1))
print("recall:", metrics.recall_score(y_test,y_pred1))
print("f1 score:", metrics.f1_score(y_test,y_pred1))
print("confusion matrix:", metrics.confusion_matrix(y_test,y_pred1))
