import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import sys

# --- 1. Install Hugging Face 'datasets' library ---
# We need to ensure the library is installed in the virtual environment.
try:
    from datasets import load_dataset
except ImportError:
    print("Hugging Face 'datasets' library not found.")
    print("Please run: pip install datasets")
    sys.exit()

print("--- Starting Model Training using Hugging Face Dataset ---")

# --- 2. Load Data Directly from Hugging Face ---
try:
    print("Loading dataset from Hugging Face... (This may take a moment)")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(dataset['train'])
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset from Hugging Face: {e}")
    sys.exit()

# --- 3. Preprocess Data ---
# We only need the user's message ('instruction') and the department ('category')
df = df[['instruction', 'category']]
df.dropna(inplace=True)
print(f"Data preprocessed. Total records: {len(df)}")
print(f"Unique categories found: {df['category'].unique().tolist()}")

# Define our features (X) and target (y)
X = df['instruction']
y = df['category']

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# --- 5. Feature Engineering (Text to Numbers) ---
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_vec = tfidf_vectorizer.transform(X_test)
print("Text data vectorized using TF-IDF.")

# --- 6. Model Training ---
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
print("Training the Logistic Regression model... (This will take a minute or two)")
model.fit(X_train_vec, y_train)
print("Model training complete.")

# --- 7. Model Evaluation ---
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# --- 8. Save the Model and Vectorizer ---
joblib.dump(model, 'ticket_classifier_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("Model and vectorizer saved successfully.")
print("--- Training Script Finished ---")