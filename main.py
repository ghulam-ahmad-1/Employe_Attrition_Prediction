import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Reading the CSV File
df = pd.read_csv("Employee-Attrition.csv")

# Preprocessing the Data (Data Cleaning)

df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
df.drop(['EmployeeCount', 'EmployeeNumber', 'DailyRate', 'Gender','StandardHours', 'Over18' ,'MaritalStatus'], axis=1, inplace=True)
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
df['BusinessTravel'] = df['BusinessTravel'].map({'Travel_Rarely': 1, 'Travel_Frequently': 2, 'Non-Travel': 0})
df['Department'] = df["Department"].map({'Sales':0 , 'Research & Development' : 1 ,"Human Resources" : 2})
df["EducationField"]= df['EducationField'].map({'Life Sciences' : 0, 'Other' : 1, 'Medical' : 2, 'Marketing' :3,'Technical Degree' : 4 , 'Human Resources': 5 })
# df["Gender"] = df['Gender'].map({"Female" : 0  , "Male":1})
df["JobRole"] = df['JobRole'].map({'Sales Executive':0, 'Research Scientist':1, 'Laboratory Technician':2,
       'Manufacturing Director':3, 'Healthcare Representative':4, 'Manager':5,
       'Sales Representative':6, 'Research Director':7, 'Human Resources':8})

# Separating the Target column and independent Varibles Column

X = df.drop('Attrition', axis=1)
X=df.values
Y = df['Attrition'].values

# Plotting Some data Insights

df.hist(figsize=(20,20));
plt.show()
# Attrition by DistanceFromHome
plt.bar(df['Attrition'].unique(), df['Attrition'].value_counts(), color=['red', 'green']);
plt.show()

# Importing the RandomForest Regressor to Train the Model :
from sklearn.ensemble import RandomForestRegressor

# Calling the Class
lr = RandomForestRegressor()

# Splitting the Training and Testing Data With the test size Of 0.2
from sklearn.model_selection import train_test_split
X_train ,X_test ,Y_train ,Y_test =train_test_split(X,Y ,test_size=0.2 ,random_state=50)

# Trainig the Model
model = lr.fit(X_train,Y_train)

# Prediction Of Test Data
Y_predict = model.predict(X_test)

# Accuracy Of teh Model
Model_Accuracy = model.score(X_test,Y_test)
print(Model_Accuracy)

