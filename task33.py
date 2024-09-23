import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/bhumikahiremath/documents/IdeaProjects/task3/bank-additional/bank-additional-full.csv', sep=';')
data = pd.get_dummies(data, drop_first=True)

X = data.drop(columns=['y_yes'])
y = data['y_yes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = ExtraTreesClassifier()
model.fit(X_train, y_train)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()

selector = SelectFromModel(model, prefit=True, threshold=0.02)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train_selected, y_train)

y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns[selector.get_support()], class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

tree.export_graphviz(clf, out_file="tree.dot", feature_names=X.columns[selector.get_support()], class_names=['No', 'Yes'], filled=True)
