from setuptools import setup, find_packages

setup(
    name="sentiment_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "spacy>=3.1.0",
        "transformers>=4.11.0",
        "torch>=1.9.0",
        "vaderSentiment>=3.3.2",
        "networkx>=2.6.3",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "plotly>=5.3.1",
        "nltk>=3.6.3",
        "gensim>=4.1.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced aspect-based sentiment analysis system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sentiment_analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 