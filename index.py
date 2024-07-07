import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("pima-indians-diabetes.data.csv")

# Split into input (X) and output (y) variables
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]
X = np.array(X)
y = np.array(y)
print(X)
print(y)

# Define the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# Evaluate the model
accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: %.2f %%' % (accuracy[1] * 100))

# Make predictions
prediction = (model.predict(X) > 0.5).astype(int).flatten()

# Print some predictions along with expected values
for i in range(50):
    print('%s \t=> %d \t(expected %d)' % (X[i].tolist(), prediction[i], y[i]))
