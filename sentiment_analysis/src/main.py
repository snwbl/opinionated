"""
Main analyzer class that combines all components of the sentiment analysis system.
"""

from typing import Dict, List, Union
import pandas as pd
from .aspect_extraction.extractor import AspectExtractor
from .sentiment_analysis.analyzer import SentimentAnalyzer
from .visualization.visualizer import Visualizer

class AspectSentimentAnalyzer:
    def __init__(self):
        """Initialize the main analyzer with all components."""
        self.aspect_extractor = AspectExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.visualizer = Visualizer()
        
    def analyze(self, text: str) -> Dict:
        """
        Perform complete analysis of the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract aspects
        aspects = self.aspect_extractor.extract_aspects(text)
        
        # Extract aspect-opinion pairs
        pairs = self.aspect_extractor.extract_aspect_opinion_pairs(text)
        
        # Analyze sentiment for each aspect
        aspect_sentiments = {}
        for aspect in aspects:
            sentiment = self.sentiment_analyzer.analyze_aspect(text, aspect["text"])
            aspect_sentiments[aspect["text"]] = sentiment
            
        # Get overall sentiment
        overall_sentiment = self.sentiment_analyzer.analyze_vader(text)
        
        return {
            "aspects": aspects,
            "aspect_opinion_pairs": pairs,
            "aspect_sentiments": aspect_sentiments,
            "overall_sentiment": overall_sentiment,
            "confidence": self.sentiment_analyzer.get_confidence_score(text)
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        return [self.analyze(text) for text in texts]
    
    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a markdown report from analysis results.
        
        Args:
            analysis_results: Analysis results dictionary
            
        Returns:
            Markdown formatted report
        """
        report = ["# Sentiment Analysis Report\n"]
        
        # Overall sentiment
        report.append("## Overall Sentiment")
        report.append(f"Compound Score: {analysis_results['overall_sentiment']['compound']:.2f}")
        report.append(f"Confidence: {analysis_results['confidence']:.2f}\n")
        
        # Aspect analysis
        report.append("## Aspect Analysis")
        for aspect, sentiment in analysis_results["aspect_sentiments"].items():
            report.append(f"### {aspect}")
            report.append(f"- Compound Score: {sentiment['compound']:.2f}")
            report.append(f"- Positive: {sentiment['pos']:.2f}")
            report.append(f"- Negative: {sentiment['neg']:.2f}")
            report.append(f"- Neutral: {sentiment['neu']:.2f}\n")
            
        return "\n".join(report)
    
    def visualize_results(self, analysis_results: Dict) -> Dict[str, object]:
        """
        Generate visualizations for the analysis results.
        
        Args:
            analysis_results: Analysis results dictionary
            
        Returns:
            Dictionary of visualization figures
        """
        visualizations = {}
        
        # Sentiment distribution
        sentiments = [analysis_results["overall_sentiment"]]
        visualizations["sentiment_distribution"] = self.visualizer.plot_sentiment_distribution(sentiments)
        
        # Aspect network
        visualizations["aspect_network"] = self.visualizer.plot_aspect_network(
            analysis_results["aspect_opinion_pairs"]
        )
        
        return visualizations 