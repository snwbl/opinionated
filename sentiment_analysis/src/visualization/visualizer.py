"""
Visualization module for sentiment analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        """Initialize the visualizer."""
        pass
        
    def plot_sentiment_distribution(self, sentiments: List[Dict[str, float]], title: str = "Sentiment Distribution") -> go.Figure:
        """
        Create a sentiment distribution plot.
        
        Args:
            sentiments: List of sentiment scores
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        df = pd.DataFrame(sentiments)
        fig = px.histogram(df, x="compound", nbins=20, title=title)
        return fig
    
    def plot_aspect_network(self, aspect_pairs: List[tuple], title: str = "Aspect-Opinion Network") -> go.Figure:
        """
        Create a network visualization of aspects and opinions.
        
        Args:
            aspect_pairs: List of (aspect, opinion, sentiment) tuples
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        G = nx.Graph()
        
        # Add nodes and edges
        for aspect, opinion, sentiment in aspect_pairs:
            G.add_node(aspect, type="aspect")
            G.add_node(opinion, type="opinion")
            G.add_edge(aspect, opinion, sentiment=sentiment)
            
        # Create layout
        pos = nx.spring_layout(G)
        
        # Create plot
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
        )
        
        node_trace = go.Scatter(
            x=[], y=[], mode="markers", hoverinfo="text", marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=10
            )
        )
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)
            
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_trace["x"] += (x,)
            node_trace["y"] += (y,)
            
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode="closest",
                           margin=dict(b=20, l=5, r=5, t=40)
                       ))
        return fig
    
    def plot_temporal_trends(self, data: pd.DataFrame, aspect: str, title: str = "Temporal Sentiment Trends") -> go.Figure:
        """
        Create a temporal trend visualization for sentiment over time.
        
        Args:
            data: DataFrame with timestamp and sentiment columns
            aspect: Aspect to analyze
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = px.line(data, x="timestamp", y="sentiment", title=title)
        return fig
    
    def plot_keyword_associations(self, keywords: Dict[str, float], title: str = "Keyword Associations") -> go.Figure:
        """
        Create a keyword association plot.
        
        Args:
            keywords: Dictionary of keywords and their scores
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        df = pd.DataFrame(list(keywords.items()), columns=["keyword", "score"])
        fig = px.bar(df, x="keyword", y="score", title=title)
        return fig 