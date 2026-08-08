# Importing necessary libraries
from flask import Flask, render_template, request, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Set a secret key for session management

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")

# Home route
@app.route('/')
def home():
    # Renders the homepage
    return render_template('home.html')

# Route to display the word and upload its meaning
@app.route('/display_word', methods=['GET', 'POST'])
def display_word():
    if request.method == 'POST':
        # Get the word and meaning entered by the user
        word = request.form['word']
        meaning = request.form['meaning']

        # Store word and meaning in the session
        session['word'] = word
        session['meaning'] = meaning

        # Redirect to the quiz page
        return redirect(url_for('quiz'))
    return render_template('display_word.html')  # Render the word input page

# Route to display the quiz
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    # Retrieve the word from the session
    word = session.get('word', 'No word provided')
    
    # Define basic quiz questions based on the word
    questions = [
        f"What is the meaning of {word}?",
        f"Use {word} in a sentence.",
        f"Is {word} a noun, verb, or adjective?"
    ]

    if request.method == 'POST':
        # Retrieve answers from the form
        answers = [request.form.get(f'q{i+1}') for i in range(3)]
        
        # Simple scoring logic: score 1 point for each answer
        score = sum(1 for answer in answers if answer)
        
        # Save score in the session
        session['score'] = score

        # Redirect to the score page
        return redirect(url_for('score'))
    
    return render_template('quiz.html', questions=questions)  # Render quiz page with questions

# Route to display the score
@app.route('/score')
def score():
    # Retrieve score from the session
    score = session.get('score', 0)
    return render_template('score.html', score=score)  # Render the score page

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
