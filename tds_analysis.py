import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from calculate_tds import calculate_tds
from sklearn.metrics import confusion_matrix, classification_report
from remove_hair import remove_hair
from segmentation_v2 import ImageProcessing


def preprocess_image(image):
    # Step 1: Remove hair
    clean_image = remove_hair(image)

    # Step 2: Create and apply mask
    img_processor = ImageProcessing(clean_image)
    mask = img_processor.global_thresholding()
    masked_image = img_processor.apply_mask(mask)

    return masked_image


def load_and_analyze_images(base_path):
    all_results = []  # Store all results including suspicious cases
    stats_results = []  # Store only non-suspicious cases for statistics

    # Define categories
    malignant_categories = [
        'basal cell carcinoma',
        'squamous cell carcinoma',
        "melanoma"
    ]

    benign_categories = [
        'Dermatofibroma',
        'Nevus',
        'Pigmented benign keratosis',
        'Seborrheic keratosis',
        'Vascular lesion',
        'Actinic keratosis'
    ]

    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            print(f"Processing {category}...")
            for img_name in os.listdir(category_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_path, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Calculate TDS
                            tds_result = calculate_tds(img)
                            true_label = 'Malignant' if category in malignant_categories else 'Benign'

                            result = {
                                'image': img_name,
                                'category': category,
                                'tds_score': tds_result['tds'],
                                'predicted_category': tds_result['category'],
                                'true_category': true_label,
                                'asymmetry_score': tds_result['scores']['asymmetry'],
                                'border_score': tds_result['scores']['border'],
                                'color_score': tds_result['scores']['color'],
                                'differential_structures_score': tds_result['scores']['differential_structures']
                            }

                            # Add to all results
                            all_results.append(result)

                            # Only add to stats if not suspicious
                            if tds_result['category'] != 'Suspicious':
                                stats_results.append(result)

                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

    # Create two dataframes
    all_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(stats_results)

    return all_df, stats_df


def create_visualizations(df):
    # Only create visualizations for non-suspicious cases
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='category', y='tds_score', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of TDS Scores by Category (Excluding Suspicious Cases)')
    plt.tight_layout()
    plt.savefig('tds_scores_by_category.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='true_category', y='tds_score', data=df)
    plt.title('TDS Score Distribution: Malignant vs Benign (Excluding Suspicious Cases)')
    plt.savefig('tds_scores_malignant_vs_benign.png')
    plt.close()

    score_columns = ['asymmetry_score', 'border_score', 'color_score',
                     'differential_structures_score', 'tds_score']
    correlation = df[score_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Different Scores (Excluding Suspicious Cases)')
    plt.tight_layout()
    plt.savefig('score_correlations.png')
    plt.close()


def calculate_performance_metrics(df):
    # Calculate metrics only for non-suspicious cases
    y_true = (df['true_category'] == 'Malignant').astype(int)
    y_pred = (df['predicted_category'] == 'Malignant').astype(int)

    report = classification_report(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Excluding Suspicious Cases)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return report


def main():
    base_path = 'images/ISIC_2017/masked_lesions'
    print("Loading and analyzing images...")
    all_df, stats_df = load_and_analyze_images(base_path)

    # Save all results to CSV
    all_df.to_csv('all_tds_analysis_results.csv', index=False)

    print("\nCreating visualizations (excluding suspicious cases)...")
    create_visualizations(stats_df)

    print("\nCalculating performance metrics (excluding suspicious cases)...")
    performance_report = calculate_performance_metrics(stats_df)

    # Save statistics results to CSV
    stats_df.to_csv('stats_tds_analysis_results.csv', index=False)

    # Save performance report
    with open('performance_report.txt', 'w') as f:
        f.write(performance_report)

    # Count suspicious cases
    suspicious_count = len(all_df[all_df['predicted_category'] == 'Suspicious'])
    total_count = len(all_df)

    print("\nAnalysis complete!")
    print(f"\nProcessed {total_count} images total")
    print(f"Found {suspicious_count} suspicious cases ({(suspicious_count / total_count) * 100:.1f}%)")
    print("\nResults have been saved to:")
    print("- all_tds_analysis_results.csv (includes suspicious cases)")
    print("- stats_tds_analysis_results.csv (excludes suspicious cases)")
    print("- performance_report.txt")
    print("- tds_scores_by_category.png")
    print("- tds_scores_malignant_vs_benign.png")
    print("- score_correlations.png")
    print("- confusion_matrix.png")


if __name__ == "__main__":
    main()