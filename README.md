# 🩺 Skin Cancer Detection System using Vision Transformer (ViT)


This project is an AI-powered Skin Cancer Detector built with Hugging Face Transformers and the HAM10000 dataset.
The model classifies dermatoscopic skin images into 7 skin disease categories, assisting in early detection and diagnosis.
The entire project was developed and trained in Google Colab, ensuring easy replication without local GPU requirements.

## ▶️ Want To Test

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArafathJ/skincancervit/skin_cancer_detector_system.ipynb)


## 🚀 Features
- Runs seamlessly in Google Colab (no setup hassle).

- Uses the HAM10000 dermatology dataset (Kaggle).

- Fine-tuned Vision Transformer (ViT) model: google/vit-base-patch16-224-in21k.

- Automatic preprocessing with Hugging Face’s AutoImageProcessor.

- Custom PyTorch Dataset for efficient image loading.

- Integrated with Hugging Face Trainer for streamlined training.

- Early stopping to prevent overfitting.

- Tracks training metrics with Weights & Biases (wandb).

- Saves trained models to Google Drive for persistence.

- Outputs predicted label + confidence score for test images.


## 📊 Dataset

Name: ```HAM10000``` — Human Against Machine with 10000 Training Images.

#### Classes (7 skin conditions):

- ```akiec``` : Actinic keratoses / intraepithelial carcinoma

- ```bcc``` : Basal cell carcinoma

- ```bkl``` : Benign keratosis-like lesions

- ```df``` : Dermatofibroma

- ```mel``` : Melanoma

- ```nv``` : Melanocytic nevi

- ```vasc``` : Vascular lesions

⚠️ Dataset not included — download from Kaggle.
 Place it in your own Google Drive when running the notebook.
## ⚙️ Tech Stack

- Google Colab (Training environment)

- Python 3

- PyTorch

- Hugging Face Transformers

- Pandas / NumPy / Matplotlib

- Scikit-learn

- Weights & Biases (wandb)



## 🏗️ How It Works

1. Open the notebook in Google Colab using the badge above.

2. Upload your own kaggle.json (from Kaggle > Account > Create API Token).

3. Download the HAM10000 dataset into Google Drive.

4. Preprocess images → Resize, normalize, and map disease labels to integers.

5. Split dataset → 80% training, 20% testing (stratified).

6. Train model → Vision Transformer fine-tuned for 3 epochs on GPU.

7. Evaluate → Accuracy and prediction confidence on validation set.

8. Predict → Load a random test image → output disease type & confidence score.


## 📈 Sample Output

```
Image path: /content/drive/MyDrive/ham10000/HAM10000_images_part_1/ISIC_0024306.jpg
The actual skin cancer type for the image is: mel
The predicted skin cancer type from the model is: mel
Confidence score (probability): 0.9453
```


## 🔮 Future Improvements
- Add a Flask/Django web app for real-time predictions.

- Extend training with more dermatology datasets for better generalization.

- Deploy as a REST API or mobile app for dermatology assistance.

- Integrate explainability tools (Grad-CAM, SHAP) to visualize model decisions.


## 🧏‍♀️ Acknowledgements

 - Developed entirely in Google Colab

 - Dataset: [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
 - Model: [Hugging Face Vision Transformer](https://huggingface.co/google/vit-base-patch16-224-in21k)

## 💡Author

-  [Arafath J](https://github.com/ArafathJ)
