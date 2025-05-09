"""
Basic example of using the sentiment analysis system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sentiment_analysis.src.main import AspectSentimentAnalyzer

def main():
    # Initialize the analyzer
    analyzer = AspectSentimentAnalyzer()
    
    # Example text
    text = """
        Vladimir Putin has led Russia's Victory Day commemorations with a parade in Red Square and heightened security after days of Ukrainian strikes targeting the capital.
        China's Xi Jinping joined Putin as he told thousands of soldiers and more than 20 international leaders that Russia remembered the lessons of World War Two.
        Putin used his speech to tie the war to today's full-scale invasion of Ukraine, and said all of Russia was behind what he called the "special military operation" - now well into its fourth year.
        For the first time, a column of trucks carrying various combat drones took part in the Victory Day parade, apparently because of their widescale use in Ukraine.
        A unilateral, three-day ceasefire was announced by Russia to coincide with the lavish 80th anniversary event, which Ukraine rejected as a "theatrical show".
        Kyiv has labelled the truce as a farce, accusing Russia of launching thousands of attacks since it came into force at midnight on Wednesday. Russia says it has observed the ceasefire and accuses Ukraine of hundreds of violations.
        In the hours before the ceasefire, Ukrainian drone strikes prompted airport closures and disruption for thousands of air passengers in Russia.
        Heavy security and restrictions were in place in the centre of Moscow on Friday as Russia marked the Soviet Union's victory over Nazi Germany.
    """
    
    # Perform analysis
    results = analyzer.analyze(text)
    
    # Generate report
    report = analyzer.generate_report(results)
    print(report)
    
    # Generate visualizations
    visualizations = analyzer.visualize_results(results)
    
    # Save visualizations
    for name, fig in visualizations.items():
        fig.write_html(f"{name}.html")
        
if __name__ == "__main__":
    main() 