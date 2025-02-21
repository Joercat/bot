from flask import Flask, render_template, request, jsonify
import spacy
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from collections import deque
import json
from pathlib import Path
from datetime import datetime
import queue
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class WritingAssistantBot:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.context_memory = deque(maxlen=10)
        self.response_queue = queue.Queue()
        
        self.writing_styles = {
            'formal': {
                'tone': 'professional',
                'complexity': 'high',
                'rules': ['Use sophisticated vocabulary', 'Maintain professional tone']
            },
            'creative': {
                'tone': 'imaginative',
                'complexity': 'moderate',
                'rules': ['Use vivid descriptions', 'Employ metaphors and similes']
            },
            'technical': {
                'tone': 'precise',
                'complexity': 'high',
                'rules': ['Use domain-specific terminology', 'Maintain clarity']
            },
            'casual': {
                'tone': 'friendly',
                'complexity': 'low',
                'rules': ['Use conversational tone', 'Keep sentences simple']
            }
        }

        self.knowledge_base = {
            "writing_patterns": {
                "improve": ["enhance", "better", "upgrade", "polish", "refine"],
                "style": ["tone", "voice", "formal", "casual", "style"],
                "grammar": ["structure", "syntax", "correct", "grammar"],
                "creative": ["imaginative", "original", "unique", "creative"]
            },
            "learned_responses": {},
            "style_examples": {
                "formal": {
                    "templates": [
                        "It is our pleasure to inform you that...",
                        "We would like to bring to your attention..."
                    ],
                    "vocabulary": ["furthermore", "subsequently", "nevertheless"]
                },
                "creative": {
                    "templates": [
                        "The sun painted the sky with...",
                        "Whispers of wind danced through..."
                    ],
                    "vocabulary": ["vibrant", "mesmerizing", "enchanting"]
                }
            },
            "user_preferences": {},
            "improvement_history": []
        }

    def analyze_text(self, text):
        doc = self.nlp(text)
        analysis = {
            'complexity_score': self._calculate_complexity(doc),
            'formality_score': self._calculate_formality(doc),
            'key_phrases': [chunk.text for chunk in doc.noun_chunks],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'sentiment': self._analyze_sentiment(doc),
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents))
        }
        return analysis

    def _calculate_complexity(self, doc):
        long_words = len([token for token in doc if len(token.text) > 6])
        return (long_words / len(doc)) if len(doc) > 0 else 0

    def _calculate_formality(self, doc):
        formal_pos = ['NOUN', 'ADJ', 'NUM', 'PROPN']
        informal_pos = ['INTJ', 'PART', 'PRON']
        formal_count = len([token for token in doc if token.pos_ in formal_pos])
        informal_count = len([token for token in doc if token.pos_ in informal_pos])
        return formal_count / (informal_count + 1)

    def _analyze_sentiment(self, doc):
        positive_words = set(['good', 'great', 'excellent', 'amazing'])
        negative_words = set(['bad', 'poor', 'terrible', 'awful'])
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        return (positive_count - negative_count) / (len(doc) + 1)

    def improve_writing(self, text, style='formal'):
        prompt = f"Transform this text into a {style} style while maintaining its meaning: {text}"
        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
                do_sample=True
            )
        
        improved_text = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return improved_text.replace(prompt, '').strip()

    def get_response(self, user_input, style='formal'):
        analysis = self.analyze_text(user_input)
        improved_version = self.improve_writing(user_input, style)
        
        response = {
            'original_text': user_input,
            'improved_text': improved_version,
            'analysis': analysis,
            'suggestions': self.generate_suggestions(analysis, style),
            'timestamp': datetime.now().isoformat()
        }
        
        self.context_memory.append(response)
        return response

    def generate_suggestions(self, analysis, style):
        suggestions = []
        style_rules = self.writing_styles[style]['rules']
        
        if analysis['complexity_score'] < 0.3 and style in ['formal', 'technical']:
            suggestions.append("Consider using more sophisticated vocabulary")
        
        if analysis['formality_score'] < 1.5 and style == 'formal':
            suggestions.append("Increase formality by using more professional language")
        
        suggestions.extend(style_rules)
        return suggestions

bot = WritingAssistantBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/improve', methods=['POST'])
def improve_text():
    data = request.get_json()
    text = data.get('text', '')
    style = data.get('style', 'formal')
    
    response = bot.get_response(text, style)
    return jsonify(response)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
