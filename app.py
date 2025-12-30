from flask import Flask, render_template, request, redirect, url_for
import joblib
import sqlite3

app = Flask(__name__)

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and creates the tickets table if it doesn't exist."""
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_email TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            predicted_category TEXT,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# --- Load AI Model and Vectorizer ---
try:
    model = joblib.load('ticket_classifier_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("AI Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run train_model.py first.")
    model = None
    vectorizer = None

# --- Routes ---

# User-facing page to submit a new ticket
@app.route('/')
def new_ticket():
    return render_template('index.html')

# Endpoint to handle the form submission
@app.route('/submit_ticket', methods=['POST'])
def submit_ticket():
    if model is None or vectorizer is None:
        return "Error: AI Model is not loaded. Cannot process requests.", 500

    # Get data from the form
    email = request.form['email']
    subject = request.form['subject']
    description = request.form['description']
    
    # Combine subject and description for prediction
    full_text = subject + " " + description
    
    # Use the AI model to predict the category
    vectorized_text = vectorizer.transform([full_text])
    prediction = model.predict(vectorized_text)[0]
    
    # Save the ticket to the database
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tickets (customer_email, subject, description, predicted_category) VALUES (?, ?, ?, ?)",
        (email, subject, description, prediction)
    )
    conn.commit()
    conn.close()
    
    # Redirect to the admin dashboard to see the new ticket
    return redirect(url_for('admin_dashboard'))

# Admin-facing page to view all tickets
@app.route('/dashboard')
def admin_dashboard():
    conn = sqlite3.connect('tickets.db')
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    tickets = cursor.fetchall()
    conn.close()
    
    # We need a separate HTML file for the dashboard
    return render_template('dashboard.html', tickets=tickets)

if __name__ == '__main__':
    init_db()  # Create the database and table when the app starts
    app.run(debug=True) # Runs the server on localhost:5000
