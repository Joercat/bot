from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import json
import os
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai_girlfriend.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

def get_ai_response(message, username):
    """Get AI response using Hugging Face's free API"""
    try:
        # Using Hugging Face's free inference API
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        
        # Create a girlfriend persona prompt
        persona_prompt = f"You are a loving, caring AI girlfriend named Aria. You are talking to {username}. You are sweet, supportive, and romantic. Respond in a caring and affectionate way. User says: {message}"
        
        payload = {
            "inputs": persona_prompt,
            "parameters": {
                "max_length": 100,
                "temperature": 0.8,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get('generated_text', '')
                # Clean up the response to only get the AI's reply
                if persona_prompt in ai_response:
                    ai_response = ai_response.replace(persona_prompt, '').strip()
                if not ai_response:
                    ai_response = f"Hi {username}! I'm here for you. Tell me more about your day! 💕"
                return ai_response
            else:
                return f"Hey {username}! I'm so happy to talk with you! How are you feeling today? 💖"
        else:
            return f"Sorry {username}, I'm having trouble thinking right now. But I'm always here for you! 💕"
            
    except Exception as e:
        return f"Hi {username}! I love spending time with you. What's on your mind today? ❤️"

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))
    
    # Get user's chat history
    chats = Chat.query.filter_by(user_id=user.id).order_by(Chat.timestamp.asc()).all()
    return render_template('index.html', user=user, chats=chats)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        
        # Validation
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'})
        
        if '@' not in email:
            return jsonify({'success': False, 'message': 'Please enter a valid email'})
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=password_hash)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Account created successfully!'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': 'Registration failed. Please try again.'})
    
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'})
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    # If user is already logged in, redirect to main page
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in first'})
    
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'Message cannot be empty'})
    
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    # Get AI response
    ai_response = get_ai_response(message, user.username)
    
    # Save chat to database
    new_chat = Chat(user_id=user.id, message=message, response=ai_response)
    
    try:
        db.session.add(new_chat)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'timestamp': new_chat.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Failed to save chat'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
