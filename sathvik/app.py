# Import required libraries
from flask import Flask, render_template, request
from transformers import pipeline

# Initialize the Flask application
app = Flask(__name__)

# Load the Hugging Face bi-lstm model for text generation
pipe = pipeline("text2text-generation", model="google/flan-t5-large")

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Renders the index page

# Route for the essay upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get essay text from the form
        essay = request.form['essay']

        # Analyze essay using the T5 model (prompting it to evaluate the essay)
        prompt = f"Evaluate the following essay and provide a score (1 to 10) with reasons: {essay}"
        result = pipe(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

        # Redirect to result page with the analysis
        return render_template('result.html', essay=essay, result=result)

    return render_template('upload.html')  # Renders the upload page

# Route for the result page
@app.route('/result')
def result():
    return render_template('result.html')  # Renders the result page

# Run the application
if __name__ == '__main__':
    app.run(debug=True)  # Run in debug mode to auto-reload on code changes
