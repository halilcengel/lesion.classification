from model import MelanomaDatasetProcessor, MelanomaTrainer
import json
# Kullanım örneği
async def main():
    # Dataset processor oluştur
    processor = MelanomaDatasetProcessor(
        benign_path="data/benign",
        malignant_path="data/malignant"
    )

    # Dataset'i hazırla
    X, y = await processor.prepare_dataset()

    # Model eğitici oluştur ve eğit
    trainer = MelanomaTrainer()
    metrics = trainer.train(X, y)

    # Modeli kaydet
    trainer.save_model("models/melanoma_classifier.joblib")

    # Metrikleri kaydet
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())