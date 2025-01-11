import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def calculate_tds(row):
    # Get the actual feature columns by name instead of position
    asymmetry = row['0']  # Asymmetry score (0-2)
    border = row['1']     # Border score (0-8)
    color = row['2']      # Color score
    diameter = row['3']   # Diameter score
    
    # Apply ABCD rule weights
    asymmetry_score = asymmetry * 1.3
    border_score = border * 0.1
    color_score = min(color, 6) * 0.5    # Cap color at 6 colors
    diameter_score = min(diameter, 5) * 0.5  # Cap diameter at 5mm
    
    tds = asymmetry_score + border_score + color_score + diameter_score
    return tds

def main():
    # Read the CSV file
    df = pd.read_csv('asymmetry_module/lesion_features.csv')
    
    # Calculate TDS for each row
    tds_scores = df.apply(calculate_tds, axis=1)
    
    # Add TDS scores to the dataframe
    df['tds_score'] = tds_scores
    
    # Create binary classification based on TDS threshold
    # TDS < 4.75: benign
    # TDS >= 4.75: suspicious
    # TDS >= 5.45: highly suspicious
    df['predicted_class'] = pd.cut(df['tds_score'], 
                                 bins=[-np.inf, 4.75, 5.45, np.inf],
                                 labels=['benign', 'suspicious', 'highly_suspicious'])
    
    # Print summary statistics
    print("\nTDS Score Statistics:")
    print(df['tds_score'].describe())
    
    # Plot TDS score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['tds_score'], bins=30)
    plt.title('Distribution of TDS Scores')
    plt.xlabel('TDS Score')
    plt.ylabel('Frequency')
    plt.savefig('tds_distribution.png')
    plt.close()
    
    # Print sample results
    print("\nSample Results (first 5 rows):")
    print(df[['label', 'tds_score', 'predicted_class']].head())
    
    # Save results to CSV
    df[['image_path', 'label', 'tds_score', 'predicted_class']].to_csv('tds_results.csv', index=False)
    print("\nResults have been saved to 'tds_results.csv'")

if __name__ == "__main__":
    main()
