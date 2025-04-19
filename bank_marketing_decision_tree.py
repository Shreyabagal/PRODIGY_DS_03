# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

url = 'bank.csv'  
data = pd.read_csv(url, sep=';')  

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

data = data.fillna(data.median())  

X = data.drop(columns=['y'])  
y = data['y']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 15)) 
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], fontsize=12, rounded=True, precision=3)

plt.savefig('decision_tree_plot.png', format='png')

plt.show()

print("Decision tree plot saved as 'decision_tree_plot.png'")
