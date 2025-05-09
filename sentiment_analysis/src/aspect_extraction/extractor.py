"""
Aspect extraction module for identifying and extracting aspects from text.
"""

import spacy
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import networkx as nx

class AspectExtractor:
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the aspect extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp = spacy.load(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        
    def extract_aspects(self, text: str) -> List[Dict]:
        """
        Extract aspects from the given text using dependency parsing.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing aspect information
        """
        doc = self.nlp(text)
        aspects = []
        
        for token in doc:
            # Look for noun phrases and named entities
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                aspect = {
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "pos": token.pos_,
                    "deps": [child.text for child in token.children]
                }
                aspects.append(aspect)
                
        return aspects
    
    def extract_aspect_opinion_pairs(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract aspect-opinion pairs from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of tuples containing (aspect, opinion, sentiment_score)
        """
        doc = self.nlp(text)
        pairs = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                # Find adjectives connected to the aspect
                opinions = [child for child in token.children if child.pos_ == "ADJ"]
                
                for opinion in opinions:
                    # Calculate sentiment score (placeholder for now)
                    sentiment_score = 0.5  # This will be replaced with actual sentiment analysis
                    pairs.append((token.text, opinion.text, sentiment_score))
                    
        return pairs
    
    def get_bert_embeddings(self, text: str) -> torch.Tensor:
        """
        Get BERT embeddings for the given text.
        
        Args:
            text: Input text
            
        Returns:
            BERT embeddings tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1) 