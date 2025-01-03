import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
# Read the dataset
df = pd.read_csv("crops.csv")



# Separate features (X) and target (y)
X = df.drop("Recommended_Crop", axis=1)  # Assuming 'Recommended_Crop' is the target column
y = df["Recommended_Crop"]  # Target variable

# Create and train the RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model using pickle
pickle.dump(model, open('crop_model.pkl', 'wb'))
