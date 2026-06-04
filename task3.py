import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("bank_fixed.csv")

# Convert text columns into numeric columns
df = pd.get_dummies(df, drop_first=True)

# Target
y = df["y_yes"]

# Features
X = df.drop("y_yes", axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Decision Tree Visualization
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier")

plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")

plt.show()