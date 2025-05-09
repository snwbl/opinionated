"""
Sentiment analysis module with multiple analysis methods.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from typing import Dict, List, Union
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize the sentiment analyzer with multiple methods.
        """
        self.vader = SentimentIntensityAnalyzer()
        self.transformer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        return self.vader.polarity_scores(text)
    
    def analyze_transformer(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment using transformer model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment label and score
        """
        result = self.transformer(text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }
    
    def analyze_aspect(self, text: str, aspect: str) -> Dict[str, float]:
        """
        Analyze sentiment for a specific aspect in the text.
        
        Args:
            text: Input text
            aspect: Aspect to analyze
            
        Returns:
            Dictionary containing aspect-specific sentiment scores
        """
        # Simple window-based approach
        words = text.split()
        aspect_idx = words.index(aspect) if aspect in words else -1
        
        if aspect_idx == -1:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
            
        # Get context window around the aspect
        window_size = 3
        start = max(0, aspect_idx - window_size)
        end = min(len(words), aspect_idx + window_size + 1)
        context = " ".join(words[start:end])
        
        return self.analyze_vader(context)
    
    def get_confidence_score(self, text: str) -> float:
        """
        Calculate confidence score for the sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score between 0 and 1
        """
        vader_scores = self.analyze_vader(text)
        transformer_result = self.analyze_transformer(text)
        
        # Combine confidence from both methods
        vader_confidence = abs(vader_scores["compound"])
        transformer_confidence = transformer_result["score"]
        
        return (vader_confidence + transformer_confidence) / 2 