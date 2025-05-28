from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import re
import random
import math
import numpy as np
from collections import Counter, defaultdict
import string
import pickle
from datetime import datetime
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

app = Flask(__name__)
CORS(app)

class WebDevCodeBot:
    def __init__(self):
        self.knowledge_base = self._load_web_dev_knowledge()
        self.code_templates = self._load_code_templates()
        self.conversation_history = []
        self.user_preferences = {}
        self.context_window = 10
        self.intent_classifier = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self.code_patterns = self._compile_code_patterns()
        self.learning_data = []
        self._initialize_ml_models()
        self._setup_database()
        
    def _setup_database(self):
        """Initialize SQLite database for learning and conversation storage"""
        self.conn = sqlite3.connect('chatbot_memory.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create tables for storing conversations and learning data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                bot_response TEXT,
                intent TEXT,
                confidence REAL,
                timestamp DATETIME,
                feedback INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT,
                description TEXT,
                code TEXT,
                tags TEXT,
                usage_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0
            )
        ''')
        
        self.conn.commit()
    
    def _initialize_ml_models(self):
        """Initialize and train ML models for intent classification"""
        training_data = [
            ("create a button", "html_element"),
            ("make a div", "html_element"),
            ("style this button", "css_styling"),
            ("add hover effect", "css_styling"),
            ("write javascript function", "js_function"),
            ("create event listener", "js_function"),
            ("responsive design", "css_responsive"),
            ("flexbox layout", "css_layout"),
            ("grid system", "css_layout"),
            ("form validation", "js_validation"),
            ("ajax request", "js_async"),
            ("fetch api", "js_async"),
            ("css animation", "css_animation"),
            ("javascript loop", "js_control"),
            ("conditional statement", "js_control"),
            ("html form", "html_form"),
            ("input field", "html_form"),
            ("navbar", "html_component"),
            ("footer", "html_component"),
            ("modal", "html_component"),
            ("carousel", "js_component"),
            ("dropdown menu", "js_component"),
            ("color scheme", "css_design"),
            ("typography", "css_design"),
            ("box model", "css_fundamentals"),
            ("positioning", "css_fundamentals"),
            ("dom manipulation", "js_dom"),
            ("event handling", "js_events"),
            ("local storage", "js_storage"),
            ("cookies", "js_storage"),
            ("media queries", "css_responsive"),
            ("mobile first", "css_responsive")
        ]
        
        texts, labels = zip(*training_data)
        
        # Create and train intent classification pipeline
        self.intent_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        self.intent_classifier.fit(texts, labels)
        
        # Save the model
        joblib.dump(self.intent_classifier, 'intent_classifier.pkl')
    
    def _compile_code_patterns(self):
        """Compile regex patterns for code detection and analysis"""
        return {
            'html_tag': re.compile(r'<[^>]+>'),
            'css_property': re.compile(r'[a-zA-Z-]+\s*:\s*[^;]+;'),
            'js_function': re.compile(r'function\s+\w+\s*\([^)]*\)\s*{'),
            'js_variable': re.compile(r'(let|const|var)\s+\w+'),
            'css_selector': re.compile(r'[.#]?[\w-]+\s*{'),
            'html_attribute': re.compile(r'\w+\s*=\s*["\'][^"\']*["\']')
        }
    
    def _load_web_dev_knowledge(self):
        """Load comprehensive web development knowledge base"""
        return {
            # HTML Knowledge
            "create html button": {
                "response": "Here's how to create an HTML button:",
                "code": '<button type="button" onclick="myFunction()">Click Me!</button>',
                "explanation": "The button element creates a clickable button. You can add onclick events, styling, and different types.",
                "intent": "html_element"
            },
            
            "html form": {
                "response": "Here's a basic HTML form structure:",
                "code": '''<form action="/submit" method="POST">
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" required>
    
    <label for="email">Email:</label>
    <input type="email" id="email" name="email" required>
    
    <button type="submit">Submit</button>
</form>''',
                "explanation": "Forms collect user input. Use proper labels, input types, and validation attributes.",
                "intent": "html_form"
            },
            
            "responsive navbar": {
                "response": "Here's a responsive navigation bar:",
                "code": '''<nav class="navbar">
    <div class="nav-brand">
        <a href="#" class="brand-link">Logo</a>
    </div>
    <div class="nav-toggle" id="mobile-menu">
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
    </div>
    <ul class="nav-menu">
        <li class="nav-item">
            <a href="#" class="nav-link">Home</a>
        </li>
        <li class="nav-item">
            <a href="#" class="nav-link">About</a>
        </li>
        <li class="nav-item">
            <a href="#" class="nav-link">Contact</a>
        </li>
    </ul>
</nav>''',
                "explanation": "A responsive navbar that adapts to different screen sizes with a mobile hamburger menu.",
                "intent": "html_component"
            },
            
            # CSS Knowledge
            "flexbox layout": {
                "response": "Here's how to create a flexbox layout:",
                "code": '''.container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}

.flex-item {
    flex: 1;
    min-width: 200px;
    padding: 20px;
    background: #f0f0f0;
    border-radius: 8px;
}''',
                "explanation": "Flexbox provides a flexible way to arrange elements. Use justify-content for horizontal alignment and align-items for vertical alignment.",
                "intent": "css_layout"
            },
            
            "css grid": {
                "response": "Here's a CSS Grid layout example:",
                "code": '''.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    grid-gap: 20px;
    padding: 20px;
}

.grid-item {
    background: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Responsive grid */
@media (max-width: 768px) {
    .grid-container {
        grid-template-columns: 1fr;
    }
}''',
                "explanation": "CSS Grid is perfect for 2D layouts. Use auto-fit and minmax for responsive grids.",
                "intent": "css_layout"
            },
            
            "css animations": {
                "response": "Here are some CSS animation examples:",
                "code": '''/* Fade in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}

/* Button hover effect */
.animated-button {
    background: #007bff;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.animated-button:hover {
    background: #0056b3;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,123,255,0.3);
}''',
                "explanation": "CSS animations can enhance user experience. Use @keyframes for complex animations and transitions for simple effects.",
                "intent": "css_animation"
            },
            
            # JavaScript Knowledge
            "javascript function": {
                "response": "Here are different ways to create JavaScript functions:",
                "code": '''// Function declaration
function greetUser(name) {
    return `Hello, ${name}! Welcome to our site.`;
}

// Arrow function
const calculateTotal = (price, tax) => {
    return price + (price * tax);
};

// Function with default parameters
function createUser(name, role = 'user', active = true) {
    return {
        name,
        role,
        active,
        createdAt: new Date()
    };
}

// Usage examples
console.log(greetUser('John'));
console.log(calculateTotal(100, 0.08));
console.log(createUser('Alice', 'admin'));''',
                "explanation": "Functions are reusable blocks of code. Use arrow functions for shorter syntax and regular functions for hoisting.",
                "intent": "js_function"
            },
            
            "dom manipulation": {
                "response": "Here's how to manipulate the DOM with JavaScript:",
                "code": '''// Select elements
const button = document.getElementById('myButton');
const container = document.querySelector('.container');
const items = document.querySelectorAll('.item');

// Create and add elements
function addNewItem(text) {
    const newItem = document.createElement('div');
    newItem.className = 'item';
    newItem.textContent = text;
    
    // Add event listener
    newItem.addEventListener('click', function() {
        this.classList.toggle('active');
    });
    
    container.appendChild(newItem);
}

// Modify existing elements
function updateContent() {
    const title = document.querySelector('h1');
    title.textContent = 'Updated Title';
    title.style.color = '#007bff';
}

// Remove elements
function removeItem(element) {
    element.remove();
}''',
                "explanation": "DOM manipulation allows you to dynamically change webpage content. Always check if elements exist before manipulating them.",
                "intent": "js_dom"
            },
            
            "fetch api": {
                "response": "Here's how to use the Fetch API for HTTP requests:",
                "code": '''// GET request
async function fetchData() {
    try {
        const response = await fetch('https://api.example.com/data');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log(data);
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

// POST request
async function submitData(userData) {
    try {
        const response = await fetch('/api/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData)
        });
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Submit error:', error);
    }
}

// Usage
fetchData().then(data => {
    // Handle the data
    displayData(data);
});''',
                "explanation": "Fetch API provides a modern way to make HTTP requests. Always handle errors and check response status.",
                "intent": "js_async"
            },
            
            "form validation": {
                "response": "Here's comprehensive form validation in JavaScript:",
                "code": '''class FormValidator {
    constructor(formId) {
        this.form = document.getElementById(formId);
        this.errors = {};
        this.init();
    }
    
    init() {
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.validateForm();
        });
        
        // Real-time validation
        const inputs = this.form.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
        });
    }
    
    validateField(field) {
        const value = field.value.trim();
        const name = field.name;
        
        // Clear previous errors
        this.clearError(field);
        
        // Required field validation
        if (field.hasAttribute('required') && !value) {
            this.setError(field, `${name} is required`);
            return false;
        }
        
        // Email validation
        if (field.type === 'email' && value) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                this.setError(field, 'Please enter a valid email');
                return false;
            }
        }
        
        // Password validation
        if (field.type === 'password' && value) {
            if (value.length < 8) {
                this.setError(field, 'Password must be at least 8 characters');
                return false;
            }
        }
        
        return true;
    }
    
    setError(field, message) {
        field.classList.add('error');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
    }
    
    clearError(field) {
        field.classList.remove('error');
        const errorMsg = field.parentNode.querySelector('.error-message');
        if (errorMsg) {
            errorMsg.remove();
        }
    }
    
    validateForm() {
        const inputs = this.form.querySelectorAll('input, textarea');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });
        
        if (isValid) {
            this.submitForm();
        }
    }
    
    submitForm() {
        const formData = new FormData(this.form);
        const data = Object.fromEntries(formData);
        
        // Submit the form data
        fetch(this.form.action, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            console.log('Success:', result);
            this.form.reset();
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}

// Initialize validator
const validator = new FormValidator('myForm');''',
                "explanation": "This validation class provides real-time validation, error handling, and form submission. It's reusable and extensible.",
                "intent": "js_validation"
            }
        }
    
    def _load_code_templates(self):
        """Load code templates for quick generation"""
        return {
            "html_boilerplate": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <!-- Navigation content -->
        </nav>
    </header>
    
    <main>
        <!-- Main content -->
    </main>
    
    <footer>
        <!-- Footer content -->
    </footer>
    
    <script src="script.js"></script>
</body>
</html>''',
            
            "css_reset": '''/* CSS Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
}

img {
    max-width: 100%;
    height: auto;
}

a {
    text-decoration: none;
    color: inherit;
}

ul, ol {
    list-style: none;
}''',
            
            "js_module": '''// ES6 Module Template
class ComponentName {
    constructor(element, options = {}) {
        this.element = element;
        this.options = {
            // default options
            ...options
        };
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.render();
    }
    
    bindEvents() {
        // Event listeners
    }
    
    render() {
        // Render logic
    }
    
    destroy() {
        // Cleanup
    }
}

export default ComponentName;'''
        }
    
    def _extract_intent(self, user_input):
        """Extract intent using trained ML model"""
        try:
            if self.intent_classifier:
                intent = self.intent_classifier.predict([user_input])[0]
                confidence = max(self.intent_classifier.predict_proba([user_input])[0])
                return intent, confidence
        except:
            pass
        
        # Fallback intent detection
        user_lower = user_input.lower()
        
        intent_keywords = {
            'html_element': ['button', 'div', 'span', 'paragraph', 'heading', 'link', 'image'],
            'html_form': ['form', 'input', 'textarea', 'select', 'checkbox', 'radio'],
            'html_component': ['navbar', 'header', 'footer', 'sidebar', 'modal', 'card'],
            'css_styling': ['style', 'color', 'background', 'font', 'border', 'margin', 'padding'],
            'css_layout': ['flexbox', 'grid', 'layout', 'position', 'float', 'display'],
            'css_responsive': ['responsive', 'mobile', 'tablet', 'media query', 'breakpoint'],
            'css_animation': ['animation', 'transition', 'hover', 'keyframes', 'transform'],
            'js_function': ['function', 'method', 'arrow function', 'callback'],
            'js_dom': ['dom', 'element', 'selector', 'manipulation', 'query'],
            'js_events': ['event', 'click', 'listener', 'handler', 'trigger'],
            'js_async': ['fetch', 'ajax', 'promise', 'async', 'await', 'api'],
            'js_validation': ['validation', 'validate', 'check', 'verify', 'form validation']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return intent, 0.7
        
        return 'general', 0.3
    
    def _semantic_search(self, user_input, intent):
        """Advanced semantic search using TF-IDF and cosine similarity"""
        user_tokens = self._advanced_tokenize(user_input)
        best_match = None
        best_score = 0.0
        
        # Filter knowledge base by intent
        relevant_items = []
        for key, value in self.knowledge_base.items():
            if value.get('intent') == intent or intent == 'general':
                relevant_items.append((key, value))
        
        if not relevant_items:
            relevant_items = list(self.knowledge_base.items())
        
        # Create corpus for TF-IDF
        corpus = [item[0] + " " + item[1].get('response', '') for item in relevant_items]
        corpus.append(user_input)
        
        # Calculate TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity
        user_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(user_vector, tfidf_matrix[:-1]).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > 0.1:
            best_match = relevant_items[best_idx][1]
        
        return best_match, best_score
    
    def _advanced_tokenize(self, text):
        """Advanced tokenization with stemming and lemmatization"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words but keep important technical terms
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'up', 'into'
        }
        
        tokens = [word for word in text.split() if word not in stop_words and len(word) > 1]
        return tokens
    
    def _generate_code_response(self, intent, user_input):
        """Generate code-specific responses based on intent"""
        code_generators = {
            'html_element': self._generate_html_element,
            'html_form': self._generate_html_form,
            'css_styling': self._generate_css_styling,
            'css_layout': self._generate_css_layout,
            'js_function': self._generate_js_function,
            'js_dom': self._generate_js_dom
        }
        
        generator = code_generators.get(intent)
        if generator:
            return generator(user_input)
        
        return None
    
    def _generate_html_element(self, user_input):
        """Generate HTML elements based on user request"""
        user_lower = user_input.lower()
        
        if 'button' in user_lower:
            return {
                'response': "Here's a customizable HTML button:",
                'code': '''<button type="button" class="btn btn-primary" onclick="handleClick()">
    Click Me
</button>

<style>
.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
}

.btn-primary {
    background-color: #007bff;
    color: white;
}

.btn-primary:hover {
    background-color: #0056b3;
    transform: translateY(-1px);
}
</style>

<script>
function handleClick() {
    alert('Button clicked!');
}
</script>''',
                'explanation': "A complete button with styling and JavaScript functionality."
            }
        
        elif 'card' in user_lower:
            return {
                'response': "Here's a responsive card component:",
                'code': '''<div class="card">
    <img src="image.jpg" alt="Card image" class="card-image">
    <div class="card-content">
        <h3 class="card-title">Card Title</h3>
        <p class="card-description">
            This is a description of the card content.
        </p>
        <button class="card-button">Learn More</button>
    </div>
</div>

<style>
.card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    max-width: 300px;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.card-image {
    width: 100%;
    height: 200px;
    object-fit: cover;
}

.card-content {
    padding: 20px;
}

.card-title {
    margin: 0 0 10px 0;
    color: #333;
}

.card-description {
    color: #666;
    line-height: 1.5;
    margin-bottom: 15px;
}

.card-button {
    background: #007bff;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 5px;
    cursor: pointer;
}
</style>''',
                'explanation': "A modern card component with hover effects and responsive design."
            }
        
        return None
    
    def _generate_css_styling(self, user_input):
        """Generate CSS styling based on user request"""
        user_lower = user_input.lower()
        
        if 'gradient' in user_lower:
            return {
                'response': "Here are some beautiful gradient examples:",
                'code': '''/* Linear gradients */
.gradient-1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.gradient-2 {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.gradient-3 {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

/* Radial gradient */
.radial-gradient {
    background: radial-gradient(circle, #ff6b6b, #4ecdc4);
}

/* Animated gradient */
.animated-gradient {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}''',
                'explanation': "Various gradient techniques including animated gradients for modern web design."
            }
        
        return None

        def _generate_html_form(self, user_input):
    """Generate HTML forms based on user request"""
    user_lower = user_input.lower()
    
    if 'contact' in user_lower or 'contact form' in user_lower:
        return {
            'response': "Here's a complete contact form with validation:",
            'code': '''<form class="contact-form" id="contactForm">
    <div class="form-group">
        <label for="name">Full Name *</label>
        <input type="text" id="name" name="name" required>
        <span class="error-message" id="nameError"></span>
    </div>
    
    <div class="form-group">
        <label for="email">Email Address *</label>
        <input type="email" id="email" name="email" required>
        <span class="error-message" id="emailError"></span>
    </div>
    
    <div class="form-group">
        <label for="phone">Phone Number</label>
        <input type="tel" id="phone" name="phone">
    </div>
    
    <div class="form-group">
        <label for="subject">Subject *</label>
        <select id="subject" name="subject" required>
            <option value="">Select a subject</option>
            <option value="general">General Inquiry</option>
            <option value="support">Support</option>
            <option value="business">Business</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="message">Message *</label>
        <textarea id="message" name="message" rows="5" required></textarea>
        <span class="error-message" id="messageError"></span>
    </div>
    
    <button type="submit" class="submit-btn">Send Message</button>
</form>

<style>
.contact-form {
    max-width: 600px;
    margin: 0 auto;
    padding: 30px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.form-group {
    margin-bottom: 20px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: 600;
    color: #333;
}

.form-group input,
.form-group select,
.form-group textarea {
    width: 100%;
    padding: 12px;
    border: 2px solid #e1e5e9;
    border-radius: 5px;
    font-size: 16px;
    transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
    outline: none;
    border-color: #007bff;
}

.error-message {
    color: #dc3545;
    font-size: 14px;
    margin-top: 5px;
    display: block;
}

.submit-btn {
    background: linear-gradient(135deg, #007bff, #0056b3);
    color: white;
    padding: 12px 30px;
    border: none;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;
    transition: transform 0.2s ease;
}

.submit-btn:hover {
    transform: translateY(-2px);
}
</style>

<script>
document.getElementById('contactForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Basic validation
    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const message = document.getElementById('message').value.trim();
    
    let isValid = true;
    
    if (name.length < 2) {
        document.getElementById('nameError').textContent = 'Name must be at least 2 characters';
        isValid = false;
    } else {
        document.getElementById('nameError').textContent = '';
    }
    
    if (!email.includes('@')) {
        document.getElementById('emailError').textContent = 'Please enter a valid email';
        isValid = false;
    } else {
        document.getElementById('emailError').textContent = '';
    }
    
    if (message.length < 10) {
        document.getElementById('messageError').textContent = 'Message must be at least 10 characters';
        isValid = false;
    } else {
        document.getElementById('messageError').textContent = '';
    }
    
    if (isValid) {
        alert('Form submitted successfully!');
        this.reset();
    }
});
</script>''',
            'explanation': "A complete contact form with client-side validation, modern styling, and responsive design."
        }
    
    elif 'login' in user_lower:
        return {
            'response': "Here's a modern login form:",
            'code': '''<div class="login-container">
    <form class="login-form" id="loginForm">
        <h2>Login</h2>
        
        <div class="form-group">
            <input type="email" id="email" name="email" placeholder="Email Address" required>
        </div>
        
        <div class="form-group">
            <input type="password" id="password" name="password" placeholder="Password" required>
        </div>
        
        <div class="form-options">
            <label class="checkbox-container">
                <input type="checkbox" id="remember">
                <span class="checkmark"></span>
                Remember me
            </label>
            <a href="#" class="forgot-password">Forgot Password?</a>
        </div>
        
        <button type="submit" class="login-btn">Login</button>
        
        <div class="signup-link">
            Don't have an account? <a href="#">Sign up</a>
        </div>
    </form>
</div>

<style>
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-form {
    background: white;
    padding: 40px;
    border-radius: 10px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    width: 100%;
    max-width: 400px;
}

.login-form h2 {
    text-align: center;
    margin-bottom: 30px;
    color: #333;
}

.form-group {
    margin-bottom: 20px;
}

.form-group input {
    width: 100%;
    padding: 15px;
    border: 2px solid #e1e5e9;
    border-radius: 5px;
    font-size: 16px;
    transition: border-color 0.3s ease;
}

.form-group input:focus {
    outline: none;
    border-color: #667eea;
}

.form-options {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.checkbox-container {
    display: flex;
    align-items: center;
    font-size: 14px;
    cursor: pointer;
}

.forgot-password {
    color: #667eea;
    text-decoration: none;
    font-size: 14px;
}

.login-btn {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border: none;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;
    transition: transform 0.2s ease;
}

.login-btn:hover {
    transform: translateY(-2px);
}

.signup-link {
    text-align: center;
    margin-top: 20px;
    font-size: 14px;
}

.signup-link a {
    color: #667eea;
    text-decoration: none;
}
</style>''',
            'explanation': "A modern login form with gradient styling and smooth animations."
        }
    
    return None

def _generate_css_layout(self, user_input):
    """Generate CSS layout based on user request"""
    user_lower = user_input.lower()
    
    if 'flexbox' in user_lower or 'flex' in user_lower:
        return {
            'response': "Here's a comprehensive flexbox layout system:",
            'code': '''<!-- Flexbox Layout Examples -->
<div class="flex-container">
    <div class="flex-item">Item 1</div>
    <div class="flex-item">Item 2</div>
    <div class="flex-item">Item 3</div>
</div>

<div class="flex-grid">
    <div class="flex-card">Card 1</div>
    <div class="flex-card">Card 2</div>
    <div class="flex-card">Card 3</div>
    <div class="flex-card">Card 4</div>
</div>

<style>
/* Basic Flexbox Container */
.flex-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 10px;
    margin-bottom: 30px;
}

.flex-item {
    flex: 1;
    padding: 20px;
    background: white;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Responsive Flex Grid */
.flex-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    padding: 20px;
}

.flex-card {
    flex: 1 1 calc(50% - 10px);
    min-width: 250px;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    text-align: center;
    transition: transform 0.3s ease;
}

.flex-card:hover {
    transform: translateY(-5px);
}

/* Responsive Design */
@media (max-width: 768px) {
    .flex-container {
        flex-direction: column;
    }
    
    .flex-card {
        flex: 1 1 100%;
    }
}

/* Utility Classes */
.flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.flex-between {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.flex-column {
    display: flex;
    flex-direction: column;
}

.flex-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
}
</style>''',
            'explanation': "A complete flexbox system with responsive design and utility classes for modern layouts."
        }
    
    elif 'grid' in user_lower:
        return {
            'response': "Here's a powerful CSS Grid layout system:",
            'code': '''<!-- CSS Grid Layout Examples -->
<div class="grid-container">
    <header class="grid-header">Header</header>
    <nav class="grid-nav">Navigation</nav>
    <main class="grid-main">Main Content</main>
    <aside class="grid-sidebar">Sidebar</aside>
    <footer class="grid-footer">Footer</footer>
</div>

<div class="card-grid">
    <div class="grid-card">Card 1</div>
    <div class="grid-card">Card 2</div>
    <div class="grid-card">Card 3</div>
    <div class="grid-card">Card 4</div>
    <div class="grid-card">Card 5</div>
    <div class="grid-card">Card 6</div>
</div>

<style>
/* Website Layout Grid */
.grid-container {
    display: grid;
    grid-template-areas:
        "header header header"
        "nav main sidebar"
        "footer footer footer";
    grid-template-rows: auto 1fr auto;
    grid-template-columns: 200px 1fr 250px;
    gap: 20px;
    min-height: 100vh;
    padding: 20px;
}

.grid-header {
    grid-area: header;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.grid-nav {
    grid-area: nav;
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}

.grid-main {
    grid-area: main;
    background: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.grid-sidebar {
    grid-area: sidebar;
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}

.grid-footer {
    grid-area: footer;
    background: #343a40;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

/* Responsive Card Grid */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    padding: 20px;
    margin-top: 30px;
}

.grid-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    text-align: center;
    transition: transform 0.3s ease;
}

.grid-card:hover {
    transform: scale(1.05);
}

/* Responsive Design */
@media (max-width: 768px) {
    .grid-container {
        grid-template-areas:
            "header"
            "nav"
            "main"
            "sidebar"
            "footer";
        grid-template-columns: 1fr;
    }
}
</style>''',
            'explanation': "A complete CSS Grid system with responsive website layout and auto-fitting card grid."
        }
    
    return None

def _generate_js_dom(self, user_input):
    """Generate JavaScript DOM manipulation based on user request"""
    user_lower = user_input.lower()
    
    if 'dom' in user_lower or 'element' in user_lower:
        return {
            'response': "Here's a comprehensive DOM manipulation guide:",
            'code': '''// DOM Selection Methods
const element = document.getElementById('myId');
const elements = document.getElementsByClassName('myClass');
const queryElement = document.querySelector('.my-selector');
const queryElements = document.querySelectorAll('.my-selector');

// Creating and Modifying Elements
function createDynamicContent() {
    // Create new element
    const newDiv = document.createElement('div');
    newDiv.className = 'dynamic-content';
    newDiv.innerHTML = '<h3>Dynamic Content</h3><p>Created with JavaScript!</p>';
    
    // Add styles
    newDiv.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    newDiv.style.color = 'white';
    newDiv.style.padding = '20px';
    newDiv.style.borderRadius = '10px';
    newDiv.style.margin = '10px 0';
    
    // Append to container
    const container = document.getElementById('container');
    container.appendChild(newDiv);
}

// DOM Manipulation Class
class DOMManager {
    constructor() {
        this.elements = new Map();
    }
    
    // Register element for easy access
    register(name, selector) {
        this.elements.set(name, document.querySelector(selector));
        return this;
    }
    
    // Get registered element
    get(name) {
        return this.elements.get(name);
    }
    
    // Add class with animation
    addClass(name, className) {
        const element = this.get(name);
        if (element) {
            element.classList.add(className);
        }
        return this;
    }
    
    // Toggle visibility with fade effect
    toggleVisibility(name) {
        const element = this.get(name);
        if (element) {
            if (element.style.opacity === '0') {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            } else {
                element.style.opacity = '0';
                element.style.transform = 'translateY(-20px)';
            }
        }
        return this;
    }
    
    // Update content with typing effect
    typeText(name, text, speed = 50) {
        const element = this.get(name);
        if (element) {
            element.innerHTML = '';
            let i = 0;
            const typeInterval = setInterval(() => {
                element.innerHTML += text.charAt(i);
                i++;
                if (i > text.length) {
                    clearInterval(typeInterval);
                }
            }, speed);
        }
        return this;
    }
}

// Usage Example
const dom = new DOMManager();
dom.register('header', '#main-header')
   .register('content', '.content-area')
   .register('button', '#action-button');

// Event Delegation Example
document.addEventListener('click', function(e) {
    // Handle button clicks
    if (e.target.matches('.dynamic-btn')) {
        e.target.style.transform = 'scale(0.95)';
        setTimeout(() => {
            e.target.style.transform = 'scale(1)';
        }, 150);
    }
    
    // Handle card clicks
    if (e.target.matches('.card')) {
        e.target.classList.toggle('expanded');
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
        }
    });
}, observerOptions);

// Observe all elements with animation class
document.querySelectorAll('.animate-on-scroll').forEach(el => {
    observer.observe(el);
});''',
            'explanation': "Complete DOM manipulation toolkit with modern JavaScript patterns, event delegation, and intersection observer for animations."
        }
    
    elif 'event' in user_lower:
        return {
            'response': "Here's a comprehensive event handling system:",
            'code': '''// Modern Event Handling Examples

// Event Listener Class
class EventManager {
    constructor() {
        this.listeners = new Map();
    }
    
    // Add event listener with automatic cleanup
    on(element, event, handler, options = {}) {
        const key = `${element.id || 'element'}-${event}`;
        
        // Store for cleanup
        this.listeners.set(key, { element, event, handler });
        
        element.addEventListener(event, handler, options);
        return this;
    }
    
    // Remove specific event listener
    off(element, event) {
        const key = `${element.id || 'element'}-${event}`;
        const listener = this.listeners.get(key);
        
        if (listener) {
            element.removeEventListener(event, listener.handler);
            this.listeners.delete(key);
        }
        return this;
    }
    
    // Remove all listeners
    cleanup() {
        this.listeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });
        this.listeners.clear();
    }
}

// Custom Event System
class CustomEventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }
    
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
}

// Usage Examples
const eventManager = new EventManager();
const emitter = new CustomEventEmitter();

// Button click with debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

const debouncedClick = debounce((e) => {
    console.log('Button clicked!', e.target);
    emitter.emit('buttonClicked', { target: e.target });
}, 300);

// Form handling with validation
function setupFormEvents() {
    const form = document.querySelector('#myForm');
    const inputs = form.querySelectorAll('input, textarea');
    
    // Real-time validation
    inputs.forEach(input => {
        eventManager.on(input, 'input', (e) => {
            validateField(e.target);
        });
        
        eventManager.on(input, 'blur', (e) => {
            validateField(e.target, true);
        });
    });
    
    // Form submission
    eventManager.on(form, 'submit', (e) => {
        e.preventDefault();
        if (validateForm(form)) {
            submitForm(form);
        }
    });
}

function validateField(field, showError = false) {
    const value = field.value.trim();
    let isValid = true;
    let message = '';
    
    if (field.required && !value) {
        isValid = false;
        message = 'This field is required';
    } else if (field.type === 'email' && value && !isValidEmail(value)) {
        isValid = false;
        message = 'Please enter a valid email';
    }
    
    // Update UI
    field.classList.toggle('invalid', !isValid);
    const errorElement = field.nextElementSibling;
    if (errorElement && errorElement.classList.contains('error-message')) {
        errorElement.textContent = showError ? message : '';
    }
    
    return isValid;
}

function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + S to save
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        emitter.emit('save', { timestamp: Date.now() });
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        emitter.emit('closeModal');
    }
});

// Touch events for mobile
let touchStartY = 0;
document.addEventListener('touchstart', (e) => {
    touchStartY = e.touches[0].clientY;
});

document.addEventListener('touchend', (e) => {
    const touchEndY = e.changedTouches[0].clientY;
    const diff = touchStartY - touchEndY;
    
    if (Math.abs(diff) > 50) {
        emitter.emit('swipe', { 
            direction: diff > 0 ? 'up' : 'down',
            distance: Math.abs(diff)
        });
    }
});

// Initialize everything
document.addEventListener('DOMContentLoaded', () => {
    setupFormEvents();
    
    // Listen to custom events
    emitter.on('buttonClicked', (data) => {
        console.log('Custom event received:', data);
    });
    
    emitter.on('save', (data) => {
        console.log('Save triggered at:', new Date(data.timestamp));
    });
});''',
            'explanation': "Advanced event handling system with custom events, debouncing, form validation, keyboard shortcuts, and touch support."
        }
    
    return None

# Add this method to the WebDevCodeBot class as well
def _generate_css_responsive(self, user_input):
    """Generate responsive CSS based on user request"""
    user_lower = user_input.lower()
    
    if 'responsive' in user_lower or 'mobile' in user_lower:
        return {
            'response': "Here's a comprehensive responsive design system:",
            'code': '''/* Mobile-First Responsive Design */

/* Base styles (mobile first) */
.container {
    width: 100%;
    padding: 15px;
    margin: 0 auto;
}

.grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Typography scaling */
.heading {
    font-size: clamp(1.5rem, 4vw, 3rem);
    line-height: 1.2;
}

.text {
    font-size: clamp(0.875rem, 2.5vw, 1.125rem);
    line-height: 1.6;
}

/* Responsive breakpoints */
@media (min-width: 480px) {
    .container {
        padding: 20px;
    }
    
    .grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (min-width: 768px) {
    .container {
        max-width: 750px;
        padding: 30px;
    }
    
    .grid {
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
    }
    
    .card {
        padding: 30px;
    }
}

@media (min-width: 1024px) {
    .container {
        max-width: 1200px;
    }
    
    .grid {
        grid-template-columns: repeat(4, 1fr);
    }
}

@media (min-width: 1200px) {
    .container {
        max-width: 1400px;
    }
}

/* Responsive navigation */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
}

.nav-menu {
    display: none;
    flex-direction: column;
    position: absolute;
    top: 100%;
    left: 0;
    width: 100%;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.nav-menu.active {
    display: flex;
}

.nav-toggle {
    display: block;
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
}

@media (min-width: 768px) {
    .nav-menu {
        display: flex;
        flex-direction: row;
        position: static;
        width: auto;
        background: none;
        box-shadow: none;
    }
    
    .nav-toggle {
        display: none;
    }
}

/* Responsive images */
.responsive-img {
    width: 100%;
    height: auto;
    object-fit: cover;
}

.img-container {
    position: relative;
    overflow: hidden;
    border-radius: 10px;
}

/* Aspect ratio containers */
.aspect-16-9 {
    aspect-ratio: 16 / 9;
}

.aspect-4-3 {
    aspect-ratio: 4 / 3;
}

.aspect-square {
    aspect-ratio: 1 / 1;
}

/* Responsive utilities */
.hide-mobile {
    display: none;
}

.show-mobile {
    display: block;
}

@media (min-width: 768px) {
    .hide-mobile {
        display: block;
    }
    
    .show-mobile {
        display: none;
    }
    
    .hide-desktop {
        display: none;
    }
    
    .show-desktop {
        display: block;
    }
}

/* Container queries (modern browsers) */
@container (min-width: 400px) {
    .card {
        padding: 25px;
    }
}

/* Print styles */
@media print {
    .no-print {
        display: none;
    }
    
    .container {
        max-width: none;
        padding: 0;
    }
    
    .card {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #ddd;
    }
}''',
            'explanation': "Complete responsive design system with mobile-first approach, flexible grid, responsive typography, and modern CSS features."
        }
    
    return None


    
    def _generate_js_function(self, user_input):
        """Generate JavaScript functions based on user request"""
        user_lower = user_input.lower()
        
        if 'debounce' in user_lower:
            return {
                'response': "Here's a debounce function for performance optimization:",
                'code': '''// Debounce function
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

// Usage example
const searchInput = document.getElementById('search');
const debouncedSearch = debounce(function(event) {
    console.log('Searching for:', event.target.value);
    // Perform search API call here
}, 300);

searchInput.addEventListener('input', debouncedSearch);

// Alternative: Modern debounce with AbortController
class SearchManager {
    constructor() {
        this.controller = null;
    }
    
    async search(query) {
        // Cancel previous request
        if (this.controller) {
            this.controller.abort();
        }
        
        this.controller = new AbortController();
        
        try {
            const response = await fetch(`/api/search?q=${query}`, {
                signal: this.controller.signal
            });
            const results = await response.json();
            this.displayResults(results);
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Search error:', error);
            }
        }
    }
    
    displayResults(results) {
        // Display search results
        console.log(results);
    }
}''',
                'explanation': "Debouncing prevents excessive function calls, especially useful for search inputs and API calls."
            }
        
        return None
    
    def _learn_from_feedback(self, user_input, response, feedback):
        """Learn from user feedback to improve responses"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_input, bot_response, feedback, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_input, response, feedback, datetime.now()))
        self.conn.commit()
        
        # Update learning data
        self.learning_data.append({
            'input': user_input,
            'response': response,
            'feedback': feedback,
            'timestamp': datetime.now()
        })
        
        # Retrain model if we have enough feedback data
        if len(self.learning_data) > 50:
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain ML models with new data"""
        try:
            # Get positive feedback examples
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT user_input, intent FROM conversations
                WHERE feedback > 0
            ''')
            
            positive_examples = cursor.fetchall()
            
            if len(positive_examples) > 10:
                # Prepare training data
                texts, labels = zip(*positive_examples)
                
                # Retrain intent classifier
                self.intent_classifier.fit(texts, labels)
                
                # Save updated model
                joblib.dump(self.intent_classifier, 'intent_classifier.pkl')
                
                print(f"Model retrained with {len(positive_examples)} examples")
                
        except Exception as e:
            print(f"Retraining error: {e}")
    
    def _generate_contextual_response(self, user_input):
        """Generate intelligent contextual responses"""
        # Extract intent
        intent, intent_confidence = self._extract_intent(user_input)
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in user_input.lower() for greeting in greetings):
            responses = [
                "Hello! I'm your web development coding assistant. I can help you with HTML, CSS, and JavaScript. What would you like to build today?",
                "Hi there! Ready to create some awesome web code? Ask me about HTML elements, CSS styling, or JavaScript functions!",
                "Hey! I'm here to help you with web development. Whether you need HTML structure, CSS styling, or JavaScript functionality, just ask!",
                "Good day! I specialize in HTML, CSS, and JavaScript. What coding challenge can I help you solve?"
            ]
            return random.choice(responses), 0.9, intent
        
        # Check for goodbyes
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell', 'take care']
        if any(goodbye in user_input.lower() for goodbye in goodbyes):
            responses = [
                "Goodbye! Happy coding! Feel free to come back anytime you need help with web development.",
                "Take care! I hope the code examples were helpful. Keep building amazing things!",
                "See you later! Remember, I'm always here to help with your HTML, CSS, and JavaScript needs.",
                "Farewell! May your code be bug-free and your websites beautiful!"
            ]
            return random.choice(responses), 0.9, intent
        
        # Check for thanks
        thanks = ['thank', 'thanks', 'appreciate', 'grateful']
        if any(thank in user_input.lower() for thank in thanks):
            responses = [
                "You're very welcome! I'm glad I could help with your web development needs. Need anything else?",
                "Happy to help! That's what I'm here for. Got any other coding questions?",
                "My pleasure! I love helping developers create amazing web experiences. What's next?",
                "You're welcome! Keep experimenting with the code and don't hesitate to ask for more help!"
            ]
            return random.choice(responses), 0.8, intent
        
        # Try to generate code-specific response
        code_response = self._generate_code_response(intent, user_input)
        if code_response:
            formatted_response = f"{code_response['response']}\n\n```html\n{code_response['code']}\n```\n\n💡 **Explanation:** {code_response['explanation']}"
            return formatted_response, 0.9, intent
        
        # Use semantic search
        best_match, confidence = self._semantic_search(user_input, intent)
        
        if best_match and confidence > 0.2:
            response = best_match['response']
            if 'code' in best_match:
                # Determine the primary language for syntax highlighting
                code = best_match['code']
                if '<' in code and '>' in code:
                    lang = 'html'
                elif '{' in code and (':' in code or 'function' in code):
                    if 'function' in code or 'const' in code or 'let' in code:
                        lang = 'javascript'
                    else:
                        lang = 'css'
                else:
                    lang = 'html'
                
                response += f"\n\n```{lang}\n{code}\n```"
            
            if 'explanation' in best_match:
                response += f"\n\n💡 **Explanation:** {best_match['explanation']}"
            
            return response, confidence, intent
        
        # Fallback responses for low confidence
        fallback_responses = [
            f"I'd love to help you with that! As a web development specialist, I can assist with HTML structure, CSS styling, and JavaScript functionality. Could you be more specific about what you'd like to create?",
            f"That's an interesting request! I specialize in HTML, CSS, and JavaScript. Could you provide more details about what kind of web component or functionality you need?",
            f"I want to give you the best code example possible! Could you clarify if you need help with HTML elements, CSS styling, JavaScript functions, or something else?",
            f"I'm here to help with web development! Whether you need responsive layouts, interactive components, or modern JavaScript features, just let me know more details about your project."
        ]
        
        return random.choice(fallback_responses), 0.3, intent
    
    def _store_conversation(self, user_input, response, intent, confidence):
        """Store conversation in database for learning"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_input, bot_response, intent, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_input, response, intent, confidence, datetime.now()))
        self.conn.commit()
    
    def get_response(self, user_input):
        """Main method to get chatbot response with advanced NLP"""
        try:
            # Generate response using advanced NLP
            response, confidence, intent = self._generate_contextual_response(user_input)
            
            # Store conversation for learning
            self._store_conversation(user_input, response, intent, confidence)
            
            # Update conversation history
            self.conversation_history.append({
                'user': user_input,
                'bot': response,
                'intent': intent,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent conversations
            if len(self.conversation_history) > self.context_window:
                self.conversation_history = self.conversation_history[-self.context_window:]
            
            return {
                'response': response,
                'confidence': confidence,
                'intent': intent,
                'timestamp': datetime.now().isoformat(),
                'suggestions': self._get_suggestions(intent)
            }
            
        except Exception as e:
            return {
                'response': "I apologize, but I encountered an error processing your request. Please try asking about HTML, CSS, or JavaScript topics!",
                'confidence': 0.0,
                'intent': 'error',
                'error': str(e)
            }
    
    def _get_suggestions(self, intent):
        """Get relevant suggestions based on current intent"""
        suggestions = {
            'html_element': [
                "Create a responsive navigation bar",
                "Build a contact form",
                "Make a card component",
                "Design a hero section"
            ],
            'css_styling': [
                "Add hover animations",
                "Create gradient backgrounds",
                "Style form inputs",
                "Make responsive typography"
            ],
            'css_layout': [
                "Build a flexbox layout",
                "Create CSS grid system",
                "Make responsive columns",
                "Center content perfectly"
            ],
            'js_function': [
                "Create form validation",
                "Build a carousel component",
                "Add smooth scrolling",
                "Make an API call with fetch"
            ],
            'js_dom': [
                "Manipulate DOM elements",
                "Add event listeners",
                "Create dynamic content",
                "Build interactive features"
            ],
            'general': [
                "Show me HTML boilerplate",
                "Create a responsive website",
                "Build a modern button",
                "Make a loading animation"
            ]
        }
        
        return suggestions.get(intent, suggestions['general'])

# Initialize the enhanced chatbot
chatbot = WebDevCodeBot()

@app.route('/')
def index():
    """Serve the enhanced HTML interface"""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with advanced NLP"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get response from enhanced chatbot
        result = chatbot.get_response(user_message)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback for learning"""
    try:
        data = request.get_json()
        user_input = data.get('user_input')
        response = data.get('response')
        feedback_score = data.get('feedback', 0)  # 1 for positive, -1 for negative
        
        if user_input and response:
            chatbot._learn_from_feedback(user_input, response, feedback_score)
            return jsonify({'status': 'feedback_recorded'})
        
        return jsonify({'error': 'Invalid feedback data'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Feedback error: {str(e)}'}), 500

@app.route('/suggestions')
def get_suggestions():
    """Get coding suggestions"""
    suggestions = {
        'popular_requests': [
            "Create a responsive navigation bar",
            "Build a modern card component",
            "Make a contact form with validation",
            "Design a hero section with animation",
            "Create a CSS grid layout",
            "Build a JavaScript carousel",
            "Make a loading spinner",
            "Design a modal popup"
        ],
        'categories': {
            'HTML': ['Forms', 'Navigation', 'Cards', 'Tables', 'Semantic Elements'],
            'CSS': ['Flexbox', 'Grid', 'Animations', 'Responsive Design', 'Modern Styling'],
            'JavaScript': ['DOM Manipulation', 'Event Handling', 'Async/Await', 'Form Validation', 'API Calls']
        }
    }
    return jsonify(suggestions)

@app.route('/templates')
def get_templates():
    """Get code templates"""
    return jsonify(chatbot.code_templates)

@app.route('/health')
def health():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'features': ['NLP', 'ML', 'Code Generation', 'Learning'],
        'knowledge_base_size': len(chatbot.knowledge_base),
        'conversation_count': len(chatbot.conversation_history),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Enhanced Web Development AI Bot Starting...")
    print("Knowledge base loaded with", len(chatbot.knowledge_base), "web dev topics")
    print("ML models initialized for intent classification")
    print("Database setup complete for learning and memory")
    print("Server will be available at http://localhost:5000")
    print("Features: Advanced NLP, Code Generation, Machine Learning, Continuous Learning")
    print("Specialized for: HTML, CSS, JavaScript development")
    print("Ready to help you build amazing web experiences!")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
