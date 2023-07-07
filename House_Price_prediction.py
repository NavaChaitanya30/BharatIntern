from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the House Prices dataset
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_regression.predict(X_test)

# Define a threshold to convert regression predictions into binary classes
threshold = 2.5

# Convert predictions into binary classes
y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]

# Calculate accuracy based on the binary classes
accuracy = accuracy_score(y_test >= threshold, y_pred_binary)
print("Accuracy:", accuracy)
