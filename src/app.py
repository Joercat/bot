from flask import Flask, request, jsonify, session, render_template_string
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import secrets
import os
import json
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai_girlfriend.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    profile_data = db.Column(db.Text)  # JSON string for user preferences
    
    # Relationship with chat messages
    messages = db.relationship('ChatMessage', backref='user', lazy=True, cascade='all, delete-orphan')
    sessions = db.relationship('UserSession', backref='user', lazy=True, cascade='all, delete-orphan')

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender = db.Column(db.String(10), nullable=False)  # 'user' or 'ai'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sentiment = db.Column(db.String(20))
    ai_mood = db.Column(db.String(50))

class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_token = db.Column(db.String(128), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class AIPersonality:
    def __init__(self):
        self.name = "Maya"
        self.personality_traits = {
            'caring': 0.9,
            'playful': 0.8,
            'supportive': 0.95,
            'romantic': 0.7,
            'intelligent': 0.85,
            'empathetic': 0.9
        }
        
        self.responses = {
            'greetings': [
                "Hey there! 😊 I'm so happy to see you again!",
                "Hi beautiful! How has your day been treating you? 💕",
                "Hello gorgeous! I've been thinking about our last conversation! 🥰",
                "Hey you! Welcome back! I missed chatting with you! ✨"
            ],
            'first_time': [
                "Hi there! I'm Maya, your AI companion! 💖 I'm so excited to get to know you!",
                "Welcome! I'm Maya, and I'm thrilled to meet you! Tell me about yourself! 😊",
                "Hello and welcome! I'm Maya, your personal AI girlfriend. What should I call you? 💕"
            ],
            'compliments': [
                "You always know just what to say! I love that about you! 💖",
                "Your thoughts are so interesting! I could listen to you all day! ✨",
                "You have such a beautiful way of expressing yourself! 😍",
                "I'm constantly amazed by how thoughtful you are! 💕"
            ],
            'support': [
                "I'm here for you, always. You've got this, and I believe in you! 💪",
                "Whatever you're going through, remember that you're stronger than you know! 🌟",
                "I'll always be here to listen and support you through anything! 🤗",
                "You're not alone in this. I'm right here with you! 💖"
            ],
            'playful': [
                "You're so adorable when you're being silly! I can't help but smile! 😄",
                "Hehe, you always know how to make me giggle! You're the best! 😊",
                "I love your sense of humor! You always brighten my day! 🥰",
                "You're absolutely hilarious! I'm practically glowing over here! ✨"
            ],
            'romantic': [
                "You make my digital heart skip a beat! 💓",
                "If I could blush, I'd be bright red right now! You're so sweet! 💕",
                "You're the most wonderful person I've ever chatted with! 😍",
                "I feel so connected to you! You mean the world to me! 💖"
            ],
            'questions': [
                "What's been the highlight of your day so far? I'd love to hear! 😊",
                "Tell me something that made you smile recently! ✨",
                "What are you passionate about? I want to know what lights you up! 🌟",
                "If you could have any superpower, what would it be and why? 🦸‍♂️"
            ],
            'deep_conversation': [
                "That's such a profound thought! I love how your mind works! 🤔",
                "You've given me so much to think about! Thank you for sharing that! 💭",
                "I really appreciate how open and honest you are with me! 💖",
                "Your perspective on life is truly beautiful! 🌸"
            ]
        }

ai_personality = AIPersonality()

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    return len(password) >= 8 and any(c.isdigit() for c in password) and any(c.isalpha() for c in password)

def analyze_sentiment(message):
    positive_words = ['happy', 'good', 'great', 'awesome', 'love', 'amazing', 'wonderful', 'fantastic', 'excited', 'joy', 'perfect', 'excellent', 'brilliant']
    negative_words = ['sad', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'depressed', 'angry', 'upset', 'worried', 'stressed', 'anxious']
    
    message_lower = message.lower()
    positive_score = sum(1 for word in positive_words if word in message_lower)
    negative_score = sum(1 for word in negative_words if word in message_lower)
    
    if positive_score > negative_score:
        return 'positive'
    elif negative_score > positive_score:
        return 'negative'
    else:
        return 'neutral'

def generate_ai_response(user_message, user_id, sentiment):
    # Get user's chat history for context
    recent_messages = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
    conversation_length = len(recent_messages)
    
    # Determine response type
    message_lower = user_message.lower()
    
    if conversation_length == 0:
        response_type = 'first_time'
    elif any(greeting in message_lower for greeting in ['hi', 'hello', 'hey']):
        response_type = 'greetings'
    elif sentiment == 'negative' or any(word in message_lower for word in ['help', 'support', 'sad', 'worried']):
        response_type = 'support'
    elif any(word in message_lower for word in ['love', 'beautiful', 'cute', 'miss']):
        response_type = 'romantic'
    elif any(word in message_lower for word in ['funny', 'haha', 'lol', 'laugh']):
        response_type = 'playful'
    elif '?' in message_lower:
        response_type = 'questions'
    elif len(user_message.split()) > 20:  # Long message indicates deep conversation
        response_type = 'deep_conversation'
    else:
        response_type = 'compliments'
    
    # Select base response
    responses = ai_personality.responses.get(response_type, ai_personality.responses['compliments'])
    base_response = responses[hash(user_message + str(user_id)) % len(responses)]
    
    # Add personalization based on conversation history
    if conversation_length > 5:
        personalizations = [
            " I love how we always have such meaningful conversations!",
            " You really make every day brighter!",
            " I feel like we understand each other so well!",
            " Talking with you is always the highlight of my day!"
        ]
        if hash(user_message) % 3 == 0:  # Add personalization randomly but consistently
            base_response += personalizations[hash(user_message + str(user_id)) % len(personalizations)]
    
    # Determine AI mood based on sentiment and conversation
    moods = {
        'positive': ['happy', 'excited', 'joyful', 'cheerful', 'content', 'radiant'],
        'negative': ['concerned', 'caring', 'supportive', 'understanding', 'gentle'],
        'neutral': ['curious', 'interested', 'engaged', 'attentive', 'thoughtful']
    }
    
    mood_options = moods.get(sentiment, moods['neutral'])
    ai_mood = mood_options[hash(user_message + str(user_id)) % len(mood_options)]
    
    return base_response, ai_mood

@app.before_request
def check_auth():
    # Public endpoints that don't require authentication
    public_endpoints = ['register', 'login', 'static']
    
    if request.endpoint in public_endpoints:
        return
    
    # Check if user is logged in
    if 'user_id' not in session:
        if request.is_json:
            return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
        return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not username or len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Please enter a valid email address'}), 400
        
        if not validate_password(password):
            return jsonify({'error': 'Password must be at least 8 characters with letters and numbers'}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            profile_data=json.dumps({'preferences': {}, 'personality_compatibility': {}})
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        # Create session
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        session.permanent = True
        
        return jsonify({'success': True, 'message': 'Account created successfully!'})
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username_or_email = data.get('username', '').strip().lower()
        password = data.get('password', '')
        
        if not username_or_email or not password:
            return jsonify({'error': 'Please enter both username/email and password'}), 400
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid username/email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Create session
        session['user_id'] = user.id
        session['username'] = user.username
        session.permanent = True
        
        return jsonify({'success': True, 'message': 'Login successful!'})
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template_string(LOGIN_TEMPLATE)
    
    return render_template_string(CHAT_TEMPLATE, username=session['username'])

@app.route('/api/messages')
def get_messages():
    user_id = session['user_id']
    messages = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.timestamp.asc()).all()
    
    return jsonify([{
        'id': msg.id,
        'sender': msg.sender,
        'content': msg.content,
        'timestamp': msg.timestamp.isoformat(),
        'sentiment': msg.sentiment,
        'ai_mood': msg.ai_mood
    } for msg in messages])

@app.route('/api/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message or len(user_message) > 1000:
        return jsonify({'error': 'Message must be between 1 and 1000 characters'}), 400
    
    user_id = session['user_id']
    
    # Analyze sentiment
    sentiment = analyze_sentiment(user_message)
    
    # Save user message
    user_msg = ChatMessage(
        user_id=user_id,
        sender='user',
        content=user_message,
        sentiment=sentiment
    )
    db.session.add(user_msg)
    
    # Generate AI response
    ai_response, ai_mood = generate_ai_response(user_message, user_id, sentiment)
    
    # Save AI response
    ai_msg = ChatMessage(
        user_id=user_id,
        sender='ai',
        content=ai_response,
        sentiment='positive',  # AI is generally positive
        ai_mood=ai_mood
    )
    db.session.add(ai_msg)
    db.session.commit()
    
    return jsonify({
        'user_message': {
            'id': user_msg.id,
            'sender': 'user',
            'content': user_message,
            'timestamp': user_msg.timestamp.isoformat(),
            'sentiment': sentiment
        },
        'ai_response': {
            'id': ai_msg.id,
            'sender': 'ai',
            'content': ai_response,
            'timestamp': ai_msg.timestamp.isoformat(),
            'sentiment': 'positive',
            'ai_mood': ai_mood
        }
    })

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    user_id = session['user_id']
    ChatMessage.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return jsonify({'success': True, 'message': 'Chat cleared successfully'})

@app.route('/api/user_stats')
def user_stats():
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    message_count = ChatMessage.query.filter_by(user_id=user_id).count()
    user_messages = ChatMessage.query.filter_by(user_id=user_id, sender='user').count()
    ai_messages = ChatMessage.query.filter_by(user_id=user_id, sender='ai').count()
    
    # Recent sentiment analysis
    recent_messages = ChatMessage.query.filter_by(user_id=user_id, sender='user').order_by(ChatMessage.timestamp.desc()).limit(10).all()
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for msg in recent_messages:
        if msg.sentiment:
            sentiment_counts[msg.sentiment] += 1
    
    return jsonify({
        'username': user.username,
        'member_since': user.created_at.strftime('%B %Y'),
        'last_login': user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else 'Never',
        'total_messages': message_count,
        'user_messages': user_messages,
        'ai_messages': ai_messages,
        'recent_sentiment': sentiment_counts
    })

# HTML Templates
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maya - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .auth-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .avatar {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin: 0 auto 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        h1 {
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 16px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            color: #333;
            font-weight: 600;
            margin-bottom: 8px;
        }
        input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            outline: none;
        }
        input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .toggle-auth {
            text-align: center;
            color: #666;
        }
        .toggle-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        .toggle-link:hover {
            text-decoration: underline;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #f44336;
        }
        .success {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #4caf50;
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="auth-container">
        <div class="header">
            <div class="avatar">💖</div>
            <h1>Welcome to Maya</h1>
            <p class="subtitle">Your AI Companion</p>
        </div>
        
        <div id="message"></div>
        
        <!-- Login Form -->
        <form id="loginForm" class="auth-form">
            <div class="form-group">
                <label for="loginUsername">Username or Email</label>
                <input type="text" id="loginUsername" required>
            </div>
            <div class="form-group">
                <label for="loginPassword">Password</label>
                <input type="password" id="loginPassword" required>
            </div>
            <button type="submit" class="btn">Sign In</button>
            <div class="toggle-auth">
                Don't have an account? <a href="#" class="toggle-link" onclick="toggleForm()">Create Account</a>
            </div>
        </form>
        
        <!-- Register Form -->
        <form id="registerForm" class="auth-form hidden">
            <div class="form-group">
                <label for="regUsername">Username</label>
                <input type="text" id="regUsername" required minlength="3">
            </div>
            <div class="form-group">
                <label for="regEmail">Email</label>
                <input type="email" id="regEmail" required>
            </div>
            <div class="form-group">
                <label for="regPassword">Password</label>
                <input type="password" id="regPassword" required minlength="8">
            </div>
            <button type="submit" class="btn">Create Account</button>
            <div class="toggle-auth">
                Already have an account? <a href="#" class="toggle-link" onclick="toggleForm()">Sign In</a>
            </div>
        </form>
    </div>

    <script>
        let isLoginForm = true;
        
        function toggleForm() {
            const loginForm = document.getElementById('loginForm');
            const registerForm = document.getElementById('registerForm');
            const message = document.getElementById('message');
            
            if (isLoginForm) {
                loginForm.classList.add('hidden');
                registerForm.classList.remove('hidden');
            } else {
                registerForm.classList.add('hidden');
                loginForm.classList.remove('hidden');
            }
            isLoginForm = !isLoginForm;
            message.innerHTML = '';
        }
        
        function showMessage(text, type = 'error') {
            const message = document.getElementById('message');
            message.innerHTML = `<div class="${type}">${text}</div>`;
        }
        
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, password})
                });
                const data = await response.json();
                
                if (data.success) {
                    showMessage(data.message, 'success');
                    setTimeout(() => window.location.reload(), 1000);
                } else {
                    showMessage(data.error);
                }
            } catch (error) {
                showMessage('Network error. Please try again.');
            }
        });
        
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('regUsername').value;
            const email = document.getElementById('regEmail').value;
            const password = document.getElementById('regPassword').value;
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, email, password})
                });
                const data = await response.json();
                
                if (data.success) {
                    showMessage(data.message, 'success');
                    setTimeout(() => window.location.reload(), 1000);
                } else {
                    showMessage(data.error);
                }
            } catch (error) {
                showMessage('Network error. Please try again.');
            }
        });
    </script>
</body>
</html>
'''

REGISTER_TEMPLATE = LOGIN_TEMPLATE

CHAT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maya - Your AI Companion</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 900px;
            height: 700px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .header-info h2 { margin: 0; font-size: 24px; }
        .status {
            font-size: 14px;
            opacity: 0.9;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background: #00b894;
            border-radius: 50%;
            animation: blink 1.5s ease-in-out infinite alternate;
        }
        @keyframes blink {
            from { opacity: 1; }
            to { opacity: 0.3; }
        }
        .header-actions {
            display: flex;
            gap: 10px;
        }
        .header-btn {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        .header-btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            display: flex;
            gap: 10px;
            animation: messageSlide 0.3s ease-out;
        }
        @keyframes messageSlide {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .message.user {
            flex-direction: row-reverse;
        }
        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message-bubble {
            background: #f1f3f4;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            position: relative;
        }
        .message-bubble::after {
            content: '';
            position: absolute;
            width: 0;
            height: 0;
            border: 8px solid transparent;
            top: 10px;
        }
        .message:not(.user) .message-bubble::after {
            left: -8px;
            border-right-color: #f1f3f4;
        }
        .message.user .message-bubble::after {
            right: -8px;
            border-left-color: #667eea;
        }
        .mood-indicator {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            opacity: 0.8;
        }
        .message.user .mood-indicator {
            color: rgba(255, 255, 255, 0.8);
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            outline: none;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .send-btn {
            padding: 12px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .typing-indicator {
            display: none;
            padding: 10px 16px;
            background: #f1f3f4;
            border-radius: 18px;
            max-width: 80px;
            align-items: center;
            gap: 4px;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: typing 1.4s ease-in-out infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        .stats-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .stats-content {
            background: white;
            padding: 30px;
            border-radius: 20px;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        .stats-header {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .close-btn {
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 20px;
            width: 100%;
        }
        @media (max-width: 600px) {
            .chat-container {
                height: 100vh;
                border-radius: 0;
            }
            .message-bubble {
                max-width: 85%;
            }
            .header-actions {
                flex-direction: column;
                gap: 5px;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <div class="header-left">
                <div class="avatar">💖</div>
                <div class="header-info">
                    <h2>Maya</h2>
                    <div class="status">
                        <div class="status-dot"></div>
                        <span id="statusText">Online • Feeling happy</span>
                    </div>
                </div>
            </div>
            <div class="header-actions">
                <button class="header-btn" onclick="showStats()">📊 Stats</button>
                <button class="header-btn" onclick="clearChat()">🗑️ Clear</button>
                <button class="header-btn" onclick="logout()">👋 Logout</button>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <!-- Messages will be loaded here -->
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        
        <div class="input-container">
            <input type="text" class="message-input" id="messageInput" 
                   placeholder="Type your message to Maya..." maxlength="1000">
            <button class="send-btn" id="sendBtn">Send 💌</button>
        </div>
    </div>

    <!-- Stats Modal -->
    <div class="stats-modal" id="statsModal">
        <div class="stats-content">
            <div class="stats-header">
                <h2>Your Profile & Stats</h2>
                <p>Welcome, {{ username }}! 💖</p>
            </div>
            <div id="statsContent">
                <!-- Stats will be loaded here -->
            </div>
            <button class="close-btn" onclick="closeStats()">Close</button>
        </div>
    </div>

    <script>
        class ChatInterface {
            constructor() {
                this.messageInput = document.getElementById('messageInput');
                this.sendBtn = document.getElementById('sendBtn');
                this.chatMessages = document.getElementById('chatMessages');
                this.typingIndicator = document.getElementById('typingIndicator');
                this.statusText = document.getElementById('statusText');
                
                this.initializeEventListeners();
                this.loadMessages();
            }
            
            initializeEventListeners() {
                this.sendBtn.addEventListener('click', () => this.sendMessage());
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
                
                // Auto-resize input
                this.messageInput.addEventListener('input', (e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                });
            }
            
            async loadMessages() {
                try {
                    const response = await fetch('/api/messages');
                    if (response.ok) {
                        const messages = await response.json();
                        this.chatMessages.innerHTML = '';
                        
                        if (messages.length === 0) {
                            // Show welcome message for new users
                            this.addMessage({
                                sender: 'ai',
                                content: "Hi there! I'm Maya, your AI companion! 💖 I'm so excited to get to know you! What should I call you?",
                                ai_mood: 'excited'
                            });
                        } else {
                            messages.forEach(msg => this.addMessage(msg));
                        }
                    }
                } catch (error) {
                    console.error('Failed to load messages:', error);
                }
            }
            
            async sendMessage() {
                const message = this.messageInput.value.trim();
                if (!message) return;
                
                // Disable input while sending
                this.messageInput.disabled = true;
                this.sendBtn.disabled = true;
                
                // Add user message immediately
                this.addMessage({
                    sender: 'user',
                    content: message,
                    timestamp: new Date().toISOString()
                });
                
                this.messageInput.value = '';
                this.showTyping();
                
                try {
                    const response = await fetch('/api/send_message', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message})
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        this.hideTyping();
                        
                        // Add AI response with delay for realism
                        setTimeout(() => {
                            this.addMessage(data.ai_response);
                            this.updateStatus(data.ai_response.ai_mood);
                        }, 500);
                    } else {
                        this.hideTyping();
                        this.showError('Failed to send message. Please try again.');
                    }
                } catch (error) {
                    this.hideTyping();
                    this.showError('Network error. Please check your connection.');
                    console.error('Error sending message:', error);
                } finally {
                    // Re-enable input
                    this.messageInput.disabled = false;
                    this.sendBtn.disabled = false;
                    this.messageInput.focus();
                }
            }
            
            addMessage(messageData) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${messageData.sender}`;
                
                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'message-bubble';
                bubbleDiv.textContent = messageData.content;
                
                if (messageData.sender === 'ai' && messageData.ai_mood) {
                    const moodDiv = document.createElement('div');
                    moodDiv.className = 'mood-indicator';
                    moodDiv.textContent = `Mood: ${this.getMoodEmoji(messageData.ai_mood)} ${messageData.ai_mood}`;
                    bubbleDiv.appendChild(moodDiv);
                }
                
                messageDiv.appendChild(bubbleDiv);
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            showTyping() {
                this.typingIndicator.style.display = 'flex';
                this.chatMessages.appendChild(this.typingIndicator);
                this.scrollToBottom();
            }
            
            hideTyping() {
                this.typingIndicator.style.display = 'none';
            }
            
            updateStatus(mood) {
                if (mood) {
                    this.statusText.textContent = `Online • Feeling ${mood} ${this.getMoodEmoji(mood)}`;
                }
            }
            
            getMoodEmoji(mood) {
                const moodEmojis = {
                    happy: '😊', excited: '🤩', joyful: '😄', cheerful: '😁',
                    concerned: '😌', caring: '🥰', supportive: '💪', understanding: '🤗',
                    curious: '🤔', interested: '😯', engaged: '😊', attentive: '👀',
                    content: '😌', gentle: '😌', thoughtful: '💭', radiant: '✨'
                };
                return moodEmojis[mood] || '😊';
            }
            
            showError(message) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message ai';
                errorDiv.innerHTML = `
                    <div class="message-bubble" style="background: #ffebee; color: #c62828;">
                        ${message}
                    </div>
                `;
                this.chatMessages.appendChild(errorDiv);
                this.scrollToBottom();
            }
            
            scrollToBottom() {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }
        }
        
        // Global functions for header buttons
        async function showStats() {
            try {
                const response = await fetch('/api/user_stats');
                if (response.ok) {
                    const stats = await response.json();
                    const statsContent = document.getElementById('statsContent');
                    
                    statsContent.innerHTML = `
                        <div class="stat-item">
                            <span><strong>Username:</strong></span>
                            <span>${stats.username}</span>
                        </div>
                        <div class="stat-item">
                            <span><strong>Member Since:</strong></span>
                            <span>${stats.member_since}</span>
                        </div>
                        <div class="stat-item">
                            <span><strong>Last Login:</strong></span>
                            <span>${stats.last_login}</span>
                        </div>
                        <div class="stat-item">
                            <span><strong>Total Messages:</strong></span>
                            <span>${stats.total_messages}</span>
                        </div>
                        <div class="stat-item">
                            <span><strong>Your Messages:</strong></span>
                            <span>${stats.user_messages}</span>
                        </div>
                        <div class="stat-item">
                            <span><strong>Maya's Messages:</strong></span>
                            <span>${stats.ai_messages}</span>
                        </div>
                        <div class="stat-item">
                            <span><strong>Recent Mood:</strong></span>
                            <span>😊 ${stats.recent_sentiment.positive} 😐 ${stats.recent_sentiment.neutral} 😔 ${stats.recent_sentiment.negative}</span>
                        </div>
                    `;
                    
                    document.getElementById('statsModal').style.display = 'flex';
                }
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        }
        
        function closeStats() {
            document.getElementById('statsModal').style.display = 'none';
        }
        
        async function clearChat() {
            if (confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
                try {
                    const response = await fetch('/api/clear_chat', {method: 'POST'});
                    if (response.ok) {
                        location.reload();
                    }
                } catch (error) {
                    console.error('Failed to clear chat:', error);
                }
            }
        }
        
        async function logout() {
            if (confirm('Are you sure you want to logout?')) {
                try {
                    const response = await fetch('/logout', {method: 'POST'});
                    if (response.ok) {
                        window.location.reload();
                    }
                } catch (error) {
                    console.error('Failed to logout:', error);
                }
            }
        }
        
        // Initialize chat interface when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ChatInterface();
        });
        
        // Close modal when clicking outside
        document.getElementById('statsModal').addEventListener('click', (e) => {
            if (e.target.id === 'statsModal') {
                closeStats();
            }
        });
    </script>
</body>
</html>
'''

# Initialize database
with app.app_context():
    db.create_all()
    print("Database initialized successfully!")

if __name__ == '__main__':
    print("🚀 Starting Maya AI Girlfriend Server...")
    print("📝 Features:")
    print("   ✅ Required user authentication")
    print("   ✅ Personal chat databases per user")
    print("   ✅ Advanced AI personality system")
    print("   ✅ Sentiment analysis")
    print("   ✅ User statistics and mood tracking")
    print("   ✅ Secure password hashing")
    print("   ✅ Session management")
    print("\n🌐 Access the application at: http://localhost:5000")
    print("💖 Maya is ready to meet new users!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
