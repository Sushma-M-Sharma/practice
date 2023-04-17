import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import pickle

df = pd.read_csv(r"C:\Users\Sushma Sharma\Downloads\sushma\data_science_course\ML\data\Loan.csv")
print(df.head())
print(df.isna().sum())
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
print(df.isna().sum())
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna(method='ffill',limit=1,inplace=True)
df.dropna(inplace=True)

print(df.count())
df.drop('Loan_ID',axis=1,inplace=True)

# scatterplot
plt.figure(figsize=(10,5))
ax = plt.axes()
ax.set_facecolor('#ffeecc')
sns.scatterplot(x='ApplicantIncome',y='LoanAmount',data=df,hue='Loan_Status')
plt.xlabel('Applicant Income',fontweight='bold')
plt.ylabel('Loan Amount',fontweight='bold')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()

df1 = df[['Gender','LoanAmount']]
df2 = df[['Married','LoanAmount']]
df3 = df[['Education','LoanAmount']]
df4 = df[['Self_Employed','LoanAmount']]
df5 = df[['Property_Area','LoanAmount']]
df6 = df[['Loan_Status','LoanAmount']]

df1 = df1.groupby('Gender').sum()
df2 = df2.groupby('Married').sum()
df3 = df3.groupby('Education').sum()
df4 = df4.groupby('Self_Employed').sum()
df5 = df5.groupby('Property_Area').sum()
df6 = df6.groupby('Loan_Status').sum()

print(df1)
print()
print(df2)
print()
print(df3)
print()
print(df4)
print()
print(df5)
print()
print(df6)

#barplot

c = ['#f2f2f2','#595959','lavender']
plt.figure(figsize=(20,20))
sns.set(rc={'axes.facecolor':'#e6f2ff'})
plt.subplot(331)
sns.barplot(x=df1.index,y='LoanAmount',data=df1,palette=c)

plt.subplot(332)
sns.barplot(x=df2.index,y='LoanAmount',data=df2,palette=c)

plt.subplot(333)
sns.barplot(x=df3.index,y='LoanAmount',data=df3,palette=c)

plt.subplot(334)
sns.barplot(x=df4.index,y='LoanAmount',data=df4,palette=c)

plt.subplot(335)
sns.barplot(x=df5.index,y='LoanAmount',data=df5,palette=c)

plt.subplot(336)
sns.barplot(x=df6.index,y='LoanAmount',data=df6,palette=c)

plt.show()

# distplot
sns.distplot(df3['LoanAmount'],rug=True,fit=norm)


#pairplot
sns.pairplot(df,hue='Loan_Status')

#data manipulation
print(df['Gender'].value_counts(),'\n')
print(df['Married'].value_counts(),'\n')
print(df['Education'].value_counts(),'\n')
print(df['Self_Employed'].value_counts(),'\n')
print(df['Property_Area'].value_counts(),'\n')
print(df['Loan_Status'].value_counts())

print(df['Gender'].unique(),'\n')
print(df['Married'].unique(),'\n')
print(df['Education'].unique(),'\n')
print(df['Self_Employed'].unique(),'\n')
print(df['Property_Area'].unique(),'\n')
print(df['Loan_Status'].unique())

df['Gender'].replace('Male',0,inplace=True)
df['Gender'].replace('Female',1,inplace=True)
df['Married'].replace('No',0,inplace=True)
df['Married'].replace('Yes',1,inplace=True)
df['Dependents'].replace('3+',4,inplace=True)

#0 for graduate, not self_employed, no property, urban, Y loan_status
#1 for not graduate, self_employed, property, rural, N loan_status
#2 for semiurban

label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Self_Employed'] = label_encoder.fit_transform(df['Self_Employed'])
df['Property_Area'] = label_encoder.fit_transform(df['Property_Area'])
df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'])

#classification

x = df.drop(['LoanAmount','Loan_Amount_Term','Loan_Status'],axis=1)
y = df[['Loan_Status']]
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

print(type(x_train),type(x_test))

logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)
print(logistic_model.score(x_test,y_test))

kclassifier_model = KNeighborsClassifier()
kclassifier_model.fit(x_train,y_train)
print(kclassifier_model.score(x_test,y_test))

naive_model = GaussianNB()
naive_model.fit(x_train,y_train)
print(naive_model.score(x_test,y_test))

random_classifier_model = RandomForestClassifier(n_estimators=100)
random_classifier_model.fit(x_train,y_train)
print(random_classifier_model.score(x_test,y_test))

svm_model = SVC(kernel = 'linear')
svm_model.fit(x_train,y_train)
print(svm_model.score(x_test,y_test))

svm_poly_model = SVC(kernel='poly',degree=2)
svm_poly_model.fit(x_train,y_train)
print(svm_poly_model.score(x_test,y_test))

svm_rbf_model = SVC(kernel='rbf',random_state=0)
svm_rbf_model.fit(x_train,y_train)
print(svm_rbf_model.score(x_test,y_test))

svm_sigmoid_model = SVC(kernel='sigmoid')
svm_sigmoid_model.fit(x_train,y_train)
print(svm_sigmoid_model.score(x_test,y_test))

decision_classifier_model = DecisionTreeClassifier(criterion='gini')
decision_classifier_model.fit(x_train,y_train)
print(decision_classifier_model.score(x_test,y_test))

decision_classifier_model1 = DecisionTreeClassifier(criterion='entropy')
decision_classifier_model1.fit(x_train,y_train)
print(decision_classifier_model1.score(x_test,y_test))

filename = "C:/Users/Sushma Sharma/Downloads/logistic_model.sav"
pickle.dump(logistic_model,open(filename,'wb'))
print('successful')

load_model = pickle.load(open(filename,'rb'))
y_pred = load_model.predict(x_test)

print('root mean squared error : ', np.sqrt(
    metrics.mean_squared_error(y_test, y_pred)))




#Regression
df.head()
x1 = df.drop(['LoanAmount','Loan_Amount_Term','Loan_Status'],axis=1)
y1 = df[['LoanAmount']]

print(x1)
print(y1)

print(x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3))
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
print(linear_model.score(x_test,y_test))

kregressor_model = KNeighborsRegressor()
kregressor_model.fit(x_train,y_train)
print(kregressor_model.score(x_test,y_test))


