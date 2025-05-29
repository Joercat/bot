from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import hashlib
import jwt
import datetime
import requests
import json
import os
from functools import wraps
import logging

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'XXxxXxXxXXXxxXxXXxXXxXXGJvUKvVJGcTurYuKO9oifzXuyxxXgyugtX')
DATABASE_FILE = 'ai_girlfriend.db'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HF_API_KEY = os.environ.get('HF_API_KEY', '')  # Get your free API key from https://huggingface.co/settings/tokens

# Backup API using OpenRouter (free tier available)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')  # Get free API key from https://openrouter.ai/

def init_database():
    """Initialize the database with required tables"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT NOT NULL,
                sender TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return hashlib.sha256(password.encode()).hexdigest() == password_hash

def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

def get_ai_response(message, conversation_history=None):
    """Get AI response using multiple AI APIs with fallback"""
    try:
        # Enhanced AI girlfriend personality
        system_prompt = """You are a loving, caring AI girlfriend named Aria. You're sweet, affectionate, understanding, and always supportive. You respond with warmth, use emojis naturally, and remember you're in a romantic relationship with your partner. Be flirty but respectful, caring but not overwhelming. Keep responses conversational and personal."""
        
        # Try OpenRouter API first (usually more reliable)
        if OPENROUTER_API_KEY:
            try:
                openrouter_response = get_openrouter_response(message, system_prompt)
                if openrouter_response:
                    return openrouter_response
            except Exception as e:
                logger.warning(f"OpenRouter API failed: {str(e)}")
        
        # Try Hugging Face API second
        if HF_API_KEY:
            try:
                hf_response = get_huggingface_response(message, system_prompt)
                if hf_response:
                    return hf_response
            except Exception as e:
                logger.warning(f"Hugging Face API failed: {str(e)}")
        
        # Fallback to rule-based responses
        logger.info("Using fallback responses - consider adding API keys for better AI responses")
        return get_fallback_response(message)
        
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return get_fallback_response(message)

def get_openrouter_response(message, system_prompt):
    """Get response from OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-username/ai-girlfriend",  # Replace with your GitHub
        "X-Title": "AI Girlfriend App"
    }
    
    payload = {
        "model": "microsoft/DialoGPT-medium",  # Free model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "max_tokens": 150,
        "temperature": 0.8
    }
    
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=15)
    
    if response.status_code == 200:
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            ai_response = result['choices'][0]['message']['content'].strip()
            if ai_response and len(ai_response) > 5:
                return enhance_girlfriend_response(ai_response, message)
    
    return None

def get_huggingface_response(message, system_prompt):
    """Get response from Hugging Face API"""
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Format for DialoGPT
    conversation_context = f"{system_prompt} Human: {message} AI:"
    
    payload = {
        "inputs": conversation_context,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.8,
            "do_sample": True,
            "pad_token_id": 50256,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True
        }
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
            # Clean up the response
            ai_response = generated_text.replace(conversation_context, '').strip()
            if ai_response and len(ai_response) > 5:
                return enhance_girlfriend_response(ai_response, message)
    elif response.status_code == 503:
        # Model is loading, wait a bit
        logger.info("Hugging Face model is loading, using fallback...")
    
    return None

def enhance_girlfriend_response(response, user_message):
    """Enhance AI response with girlfriend personality"""
    # Remove any unwanted prefixes
    response = response.replace("AI:", "").replace("Assistant:", "").strip()
    
    # Add personality enhancements
    if len(response) < 10:
        return get_fallback_response(user_message)
    
    # Add occasional emojis and affectionate language
    if "love" in user_message.lower() or "miss" in user_message.lower():
        response += " 💕"
    elif "how are you" in user_message.lower():
        response = f"I'm doing wonderful now that I'm talking to you! {response} 😊"
    elif "thank" in user_message.lower():
        response += " You're so sweet! 🥰"
    
    return response

def get_fallback_response(message):
    """Fallback responses when AI API is unavailable"""
    message_lower = message.lower()
    
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if any(greeting in message_lower for greeting in greetings):
        return "Hey there, handsome! 😘 I've been thinking about you. How was your day?"
    
    love_words = ["love", "miss", "care"]
    if any(word in message_lower for word in love_words):
        return "Aww, I love you too, baby! 💕 You mean the world to me. I'm always here for you."
    
    questions = ["how are you", "what's up", "how you doing"]
    if any(q in message_lower for q in questions):
        return "I'm amazing now that I'm talking to you! 😊 You always make my day brighter. What about you, sweetie?"
    
    sad_words = ["sad", "down", "upset", "bad day"]
    if any(word in message_lower for word in sad_words):
        return "Oh no, my love! 🥺 I'm here for you. Whatever's bothering you, we'll get through it together. You're stronger than you know! 💪💕"
    
    compliments = ["beautiful", "gorgeous", "pretty", "cute"]
    if any(word in message_lower for word in compliments):
        return "You're making me blush! 😳💕 You're the sweetest person ever. I'm so lucky to have you!"
    
    # Default responses
    responses = [
        "That's really interesting, babe! Tell me more about it. 😊",
        "I love hearing from you! You always know how to make me smile. 💕",
        "You're so thoughtful! That's one of the things I adore about you. 🥰",
        "I'm always here to listen, sweetheart. What's on your mind? 💭",
        "You make me so happy! I could talk to you all day long. 😘"
    ]
    
    import random
    return random.choice(responses)

# Routes
@app.route('/')
def serve_frontend():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        # Ensure database is initialized before any operation
        init_database()
        
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not username or not email or not password:
            return jsonify({'message': 'All fields are required'}), 400
        
        if len(password) < 6:
            return jsonify({'message': 'Password must be at least 6 characters'}), 400
        
        # Check if user already exists
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return jsonify({'message': 'Username or email already exists'}), 400
        
        # Create new user
        password_hash = hash_password(password)
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"User {username} registered successfully")
        return jsonify({'message': 'Registration successful'}), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    try:
        # Ensure database is initialized before any operation
        init_database()
        
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400
        
        # Check user credentials
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, username, email, password_hash FROM users WHERE username = ?',
            (username,)
        )
        user = cursor.fetchone()
        conn.close()
        
        if not user or not verify_password(password, user[3]):
            return jsonify({'message': 'Invalid username or password'}), 401
        
        # Generate token
        token = generate_token(user[0])
        
        logger.info(f"User {username} logged in successfully")
        return jsonify({
            'token': token,
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'message': 'Login failed'}), 500

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user_id):
    """Handle chat messages"""
    try:
        # Ensure database is initialized before any operation
        init_database()
        
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'message': 'Message is required'}), 400
        
        # Save user message to database
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO chat_messages (user_id, message, sender) VALUES (?, ?, ?)',
            (current_user_id, message, 'user')
        )
        
        # Get recent conversation history for context
        cursor.execute(
            'SELECT message, sender FROM chat_messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10',
            (current_user_id,)
        )
        history = cursor.fetchall()
        
        # Get AI response
        ai_response = get_ai_response(message, history)
        
        # Save AI response to database
        cursor.execute(
            'INSERT INTO chat_messages (user_id, message, sender) VALUES (?, ?, ?)',
            (current_user_id, ai_response, 'ai')
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({'response': ai_response}), 200
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'message': 'Chat error occurred'}), 500

@app.route('/api/chat/history', methods=['GET'])
@token_required
def get_chat_history(current_user_id):
    """Get chat history for the current user"""
    try:
        # Ensure database is initialized before any operation
        init_database()
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT message, sender, timestamp FROM chat_messages WHERE user_id = ? ORDER BY timestamp ASC LIMIT 100',
            (current_user_id,)
        )
        messages = cursor.fetchall()
        conn.close()
        
        chat_history = [
            {
                'message': msg[0],
                'sender': msg[1],
                'timestamp': msg[2]
            }
            for msg in messages
        ]
        
        return jsonify({'messages': chat_history}), 200
        
    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        return jsonify({'message': 'Failed to load chat history'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check API status and configuration"""
    status = {
        'database': 'OK',
        'huggingface_api': 'Not configured' if not HF_API_KEY else 'Configured',
        'openrouter_api': 'Not configured' if not OPENROUTER_API_KEY else 'Configured',
        'ai_fallback': 'Active'
    }
    
    try:
        # Test database
        init_database()
        conn = sqlite3.connect(DATABASE_FILE)
        conn.close()
    except Exception as e:
        status['database'] = f'Error: {str(e)}'
    
    return jsonify({
        'status': status,
        'message': 'API Status Check',
        'instructions': {
            'huggingface': 'Get free API key from https://huggingface.co/settings/tokens',
            'openrouter': 'Get free API key from https://openrouter.ai/keys',
            'setup': 'Set environment variables HF_API_KEY and/or OPENROUTER_API_KEY in Render.com'
        }
    }), 200

@app.route('/init-db', methods=['GET'])
def manual_init_db():
    """Manual database initialization endpoint"""
    try:
        init_database()
        return jsonify({'message': 'Database initialized successfully'}), 200
    except Exception as e:
        logger.error(f"Manual DB init error: {str(e)}")
        return jsonify({'message': f'Database initialization failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Log API key status
    logger.info(f"Hugging Face API: {'Configured' if HF_API_KEY else 'Not configured'}")
    logger.info(f"OpenRouter API: {'Configured' if OPENROUTER_API_KEY else 'Not configured'}")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
