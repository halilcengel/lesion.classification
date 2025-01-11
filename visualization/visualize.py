import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gorsellestime_olustur(veri):
    # Sadece şüpheli olmayan vakalar için görseller oluştur
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='category', y='tds_score', data=veri)
    plt.xticks(rotation=45, ha='right')
    plt.title('Kategorilere Göre TDS Skorlarının Dağılımı (Şüpheli Vakalar Hariç)')
    plt.tight_layout()
    plt.savefig('kategori_bazli_tds_skorlari.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='true_category', y='tds_score', data=veri)
    plt.title('TDS Skor Dağılımı: Malign ve Benign (Şüpheli Vakalar Hariç)')
    plt.savefig('malign_benign_tds_skorlari.png')
    plt.close()

    skor_sutunlari = ['asymmetry_score', 'border_score', 'color_score',
                      'differential_structures_score', 'tds_score']
    korelasyon = veri[skor_sutunlari].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(korelasyon, annot=True, cmap='coolwarm', center=0)
    plt.title('Farklı Skorlar Arasındaki Korelasyon (Şüpheli Vakalar Hariç)')
    plt.tight_layout()
    plt.savefig('skor_korelasyonlari.png')
    plt.close()


def performans_metrikleri_hesapla(veri):
    # Sadece şüpheli olmayan vakalar için metrikleri hesapla
    gercek_deger = (veri['true_category'] == 'Malignant').astype(int)
    tahmin_deger = (veri['predicted_category'] == 'Malignant').astype(int)

    rapor = classification_report(gercek_deger, tahmin_deger)

    karmasiklik_matrisi = confusion_matrix(gercek_deger, tahmin_deger)

    plt.figure(figsize=(8, 6))
    sns.heatmap(karmasiklik_matrisi, annot=True, fmt='d', cmap='Blues')
    plt.title('Karmaşıklık Matrisi (Şüpheli Vakalar Hariç)')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig('karmasiklik_matrisi.png')
    plt.close()

    return rapor


# CSV dosyasını oku
veri = pd.read_csv('../all_tds_analysis_results.csv')

print("\nGörseller oluşturuluyor (şüpheli vakalar hariç)...")
gorsellestime_olustur(veri)

print("\nPerformans metrikleri hesaplanıyor (şüpheli vakalar hariç)...")
performans_raporu = performans_metrikleri_hesapla(veri)
print("\nPerformans Raporu:")
print(performans_raporu)