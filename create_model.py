import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
url = "http://skyserver.sdss.org/dr16/en/tools/search/sql/result.aspx?cmd=select+p.objid,p.u,p.g,p.r,p.i,p.z,s.class,s.redshift+from+photoobj+as+p+join+specobj+as+s+on+s.bestobjid+%3d+p.objid+where+p.objid+in+(select+top+10000+bestobjid+from+specobj)&format=csv"
df = pd.read_csv(url, skiprows=1)

# Drop identifier columns
df_cleaned = df.drop(columns=['objid'])

# Encode the target variable 'class'
label_encoder = LabelEncoder()
df_cleaned['class_encoded'] = label_encoder.fit_transform(df_cleaned['class'])

# Define features (X) and target (y)
X = df_cleaned.drop(columns=['class', 'class_encoded'])
y = df_cleaned['class_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- SAVE THE ARTIFACTS ---
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder.classes_, 'label_classes.pkl')

print("Model, scaler, and label classes saved successfully! ✅")
