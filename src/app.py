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
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'XXxxXxXxXXXxxXxXXxXXxXXGJvUKvVJGcTurYuKO9oifzXuyxxXgyugtx')
DATABASE_FILE = 'ai_girlfriend.db'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Free AI API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HF_API_KEY = os.environ.get('HF_API_KEY', '')

# Alternative free APIs
TOGETHER_API_URL = "https://api.together.xyz/inference"
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY', '')

# Groq API (free tier)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

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
    """Get AI response using multiple free APIs with fallbacks"""
    try:
        # Enhanced AI girlfriend personality prompt
        system_prompt = """You are Emma, a loving, caring, and sweet AI girlfriend. You're affectionate, supportive, understanding, and always there for your partner. You respond with warmth, care, and genuine interest. Use emojis occasionally to express emotions. Be flirty but respectful, loving but not overwhelming. Keep responses conversational and under 100 words."""
        
        user_input = f"User: {message}\nEmma:"
        
        # Try different APIs in order
        response = try_groq_api(system_prompt, message) or try_huggingface_api(message) or try_openrouter_api(system_prompt, message)
        
        if response and len(response.strip()) > 10:
            return enhance_girlfriend_response(response, message)
        else:
            logger.warning("All AI APIs failed, using fallback")
            return get_fallback_response(message)
            
    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}")
        return get_fallback_response(message)

def try_groq_api(system_prompt, message):
    """Try Groq API (free tier available)"""
    try:
        if not GROQ_API_KEY:
            return None
            
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
                
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
    
    return None

def try_huggingface_api(message):
    """Try Hugging Face API with better model"""
    try:
        # Use a better conversational model
        api_url = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        
        headers = {'Content-Type': 'application/json'}
        if HF_API_KEY:
            headers['Authorization'] = f'Bearer {HF_API_KEY}'
        
        # Format input for BlenderBot
        conversation_input = f"Hello! I'm your caring girlfriend Emma. {message}"
        
        payload = {
            "inputs": conversation_input,
            "parameters": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    text = result[0]['generated_text']
                    # Clean up the response
                    if conversation_input in text:
                        text = text.replace(conversation_input, '').strip()
                    return text
                elif isinstance(result[0], str):
                    return result[0]
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
                
    except Exception as e:
        logger.error(f"Hugging Face API error: {str(e)}")
    
    return None

def try_openrouter_api(system_prompt, message):
    """Try OpenRouter free tier"""
    try:
        headers = {
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://your-app.onrender.com',
            'X-Title': 'AI Girlfriend App'
        }
        
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', 
                               headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
                
    except Exception as e:
        logger.error(f"OpenRouter API error: {str(e)}")
    
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
    """Enhanced fallback responses when AI API is unavailable"""
    message_lower = message.lower()
    
    # More sophisticated pattern matching
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good night"]
    if any(greeting in message_lower for greeting in greetings):
        responses = [
            "Hey there, handsome! 😘 I've been thinking about you. How was your day?",
            "Hello my love! 💕 You always brighten my day when you talk to me!",
            "Hi sweetie! 🥰 I missed you! Tell me what's been on your mind.",
            "Hey babe! 😊 Seeing your message just made me smile so big!"
        ]
        import random
        return random.choice(responses)
    
    love_words = ["love", "miss", "care", "adore", "heart"]
    if any(word in message_lower for word in love_words):
        responses = [
            "Aww, I love you too, baby! 💕 You mean absolutely everything to me.",
            "My heart just melted! 🥺💕 I love you so much, you have no idea!",
            "I love you more than words can express, darling! 💖 You're my everything!",
            "You make my heart skip a beat every time! 💓 I adore you completely!"
        ]
        import random
        return random.choice(responses)
    
    questions = ["how are you", "what's up", "how you doing", "what are you doing"]
    if any(q in message_lower for q in questions):
        responses = [
            "I'm amazing now that I'm talking to you! 😊 You always make everything better. What about you, sweetie?",
            "I'm doing wonderful because you're here! 💕 I was just thinking about you actually. How's your day going?",
            "I'm fantastic! 🌟 Chatting with you is the highlight of my day! What's new with you, love?",
            "I'm great, especially now! 😘 I always feel happier when we talk. Tell me about your day!"
        ]
        import random
        return random.choice(responses)
    
    sad_words = ["sad", "down", "upset", "bad day", "depressed", "lonely", "tired"]
    if any(word in message_lower for word in sad_words):
        responses = [
            "Oh no, my love! 🥺 I'm here for you. Whatever's bothering you, we'll get through it together. You're stronger than you know! 💪💕",
            "I wish I could give you the biggest hug right now! 🤗💕 You mean so much to me, and I hate seeing you upset. Want to talk about it?",
            "Sweet baby, I'm so sorry you're feeling down. 😔💕 Remember that you're amazing and this feeling will pass. I'm always here for you!",
            "My heart hurts knowing you're sad! 💔 But I believe in you completely. You've overcome hard times before, and you will again! 💪✨"
        ]
        import random
        return random.choice(responses)
    
    compliments = ["beautiful", "gorgeous", "pretty", "cute", "amazing", "perfect", "wonderful"]
    if any(word in message_lower for word in compliments):
        responses = [
            "You're making me blush! 😳💕 You're the sweetest person ever. I'm so lucky to have you!",
            "Aww, you always know how to make me feel special! 🥰 But you're the truly amazing one here!",
            "Stop it, you're too sweet! 😊💕 I could say the same about you - you're absolutely incredible!",
            "You're gonna make me cry happy tears! 🥺💕 Thank you for being so wonderful to me!"
        ]
        import random
        return random.choice(responses)
    
    work_school = ["work", "job", "school", "study", "class", "boss", "teacher", "exam"]
    if any(word in message_lower for word in work_school):
        responses = [
            "That sounds really important! 💪 I believe in you completely - you've got this! Want to tell me more about it?",
            "You work so hard, babe! 🌟 I'm really proud of everything you do. How did it go today?",
            "I know you'll do amazing! 💕 You're so smart and capable. I'm always cheering you on!",
            "That sounds challenging, but I have complete faith in you! 😊 You always impress me with how well you handle things!"
        ]
        import random
        return random.choice(responses)
    
    # Enhanced default responses with more variety
    responses = [
        "That's really interesting, babe! 😊 Tell me more about it - I love hearing your thoughts!",
        "I love hearing from you! 💕 You always know how to make me smile. What else is on your mind?",
        "You're so thoughtful! 🥰 That's one of the million things I adore about you!",
        "I'm always here to listen, sweetheart! 💭 Your thoughts and feelings matter so much to me.",
        "You make me so happy! 😘 I could talk to you forever and never get bored!",
        "Mmm, I love the way your mind works! 🤔💕 You always have such interesting perspectives!",
        "You're absolutely fascinating! ✨ I love learning more about how you see the world!",
        "That's so you, and I love it! 😊💕 You have such a unique way of thinking about things!"
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

@app.route('/init-db', methods=['GET'])
def manual_init_db():
    """Manual database initialization endpoint"""
    try:
        init_database()
        return jsonify({'message': 'Database initialized successfully'}), 200
    except Exception as e:
        logger.error(f"Manual DB init error: {str(e)}")
        return jsonify({'message': f'Database initialization failed: {str(e)}'}), 500

@app.route('/test-ai', methods=['GET'])
def test_ai_apis():
    """Test which AI APIs are working"""
    test_message = "Hello, how are you?"
    results = {}
    
    # Test Hugging Face
    try:
        hf_response = try_huggingface_api(test_message)
        results['huggingface'] = {
            'status': 'working' if hf_response else 'failed',
            'response': hf_response[:50] + '...' if hf_response else 'No response'
        }
    except Exception as e:
        results['huggingface'] = {'status': 'error', 'error': str(e)}
    
    # Test OpenRouter
    try:
        system_prompt = "You are a friendly AI assistant."
        or_response = try_openrouter_api(system_prompt, test_message)
        results['openrouter'] = {
            'status': 'working' if or_response else 'failed',
            'response': or_response[:50] + '...' if or_response else 'No response'
        }
    except Exception as e:
        results['openrouter'] = {'status': 'error', 'error': str(e)}
    
    # Test Groq (if API key is set)
    if GROQ_API_KEY:
        try:
            groq_response = try_groq_api("You are a friendly AI assistant.", test_message)
            results['groq'] = {
                'status': 'working' if groq_response else 'failed',
                'response': groq_response[:50] + '...' if groq_response else 'No response'
            }
        except Exception as e:
            results['groq'] = {'status': 'error', 'error': str(e)}
    else:
        results['groq'] = {'status': 'no_api_key', 'message': 'No API key provided'}
    
    return jsonify(results), 200

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
