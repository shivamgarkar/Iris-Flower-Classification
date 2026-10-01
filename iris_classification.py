# ==========================================
# Iris Flower Classification
# AIML Internship - Task 2
# ==========================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the machine learning model
model = DecisionTreeClassifier(random_state=42)


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# Display results
print("=" * 50)
print("       IRIS FLOWER CLASSIFICATION")
print("          AIML INTERNSHIP - TASK 2")
print("=" * 50)

print("Dataset: Iris Flower Dataset")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model: Decision Tree Classifier")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("=" * 50)

print("\nSample Predictions:")

for i in range(5):
    predicted_name = iris.target_names[y_pred[i]]
    actual_name = iris.target_names[y_test[i]]

    print(
        "Predicted:",
        predicted_name,
        "| Actual:",
        actual_name
    )

print("=" * 50)