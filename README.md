# CAPTCHA Recognition and Snapp Authentication Automation

## Introduction

While exploring the Snapp website, I set out to determine whether it was possible to log in without using a web browser or the official mobile app. My approach involved inspecting the site's API calls during the login process to identify usable endpoints.

## API Discovery and Analysis

By analyzing network requests, I identified key endpoints used for authentication:

- **Base API URL:** `https://app.snapp.taxi/api`
- **Login Endpoint:** `https://app.snapp.taxi/login`
- **OTP Request Endpoint:** `https://app.snapp.taxi/api-passenger-oauth/v3/mutotp`
- **Captcha Generation Endpoint:** `https://app.snapp.taxi/captcha/api/v1/generate/text/numeric/71C84A80-395B-448E-A240-B7DC939186D3`

Each endpoint serves a unique purpose, such as handling authentication, issuing login tokens, or generating CAPTCHAs.

### Security Observations

I found a minor security issue: calling the CAPTCHA endpoint directly always returns a fresh CAPTCHA. Additionally, switching `numeric` to `alphabetic` in the URL generates a different CAPTCHA:

```
https://app.snapp.taxi/api/captcha/api/v1/generate/text/alphabetic/71C84A80-395B-448E-A240-B7DC939186D3
```

Upon requesting a CAPTCHA, the response includes a `ref_id`, which must be provided in subsequent login attempts:

```json
{
  "captcha": {
    "client_id": "71C84A80-395B-448E-A240-B7DC939186D3",
    "solution": pred_text,
    "ref_id": captcha_data["ref_id"],
    "type": "numeric"
  }
}
```

Interestingly, Snapp uses `svg-captcha`, a free, open-source CAPTCHA generator, without modifying its font or complexity. This is not a major issue because the critical security measure is the SMS token, which has rate limitations.

## CAPTCHA Recognition Model

To automate CAPTCHA solving, I generated a dataset of 10 synthetic CAPTCHA images similar to Snapp's and trained a **Convolutional Recurrent Neural Network (CRNN)** using PyTorch.

### Model Performance

- **Accuracy:** 0.998
- **Confidence after 5 epochs:** 0.9776
- **Confidence after 10 epochs:** 0.9981
- **Model sizes:**
  - Full model (`model.pth`): 175MB
  - Compressed model (`model_half.pth`): 30MB (still achieving 0.9711 confidence)

### Confusion Matrices

<table>
  <tr>
    <td><img src="./checkpoints/epoch_cm_01.png" alt="Confusion Matrix 1" width="150"></td>
    <td><img src="./checkpoints/epoch_cm_01.png" alt="Confusion Matrix 2" width="150"></td>
    <td><img src="./checkpoints/epoch_cm_01.png" alt="Confusion Matrix 10" width="150"></td>
  </tr>
  <tr>
    <td>Epoch: 1</td>
    <td>Epoch: 2</td>
    <td>Epoch: 10</td>
  </tr>
</table>




The improvement in confusion matrices across epochs demonstrates the model's learning progression and reduced misclassification.

## Automating Snapp Authentication

### Fetching CAPTCHA

```bash
python snapp_captcha_getter.py
```

- Calls Snapp's CAPTCHA API
- Saves the CAPTCHA image
- Predicts the CAPTCHA solution using the trained model

### Completing Login

```bash
python snapp_going_inside.py
```

- Submits a login request
- Uses the CAPTCHA prediction for authentication
- Automates the entire login process

## Educational Value

This project highlights the importance of understanding web security and automation techniques. By analyzing API requests and leveraging deep learning for CAPTCHA recognition, it demonstrates real-world applications of:

- **Reverse Engineering Web Authentication**: How web applications authenticate users and where vulnerabilities may exist.
- **Deep Learning for OCR**: Using CRNNs to solve image-to-text problems effectively.
- **Automation with Ethical Considerations**: Exploring automation responsibly without violating terms of service.

## Disclaimer

This project is for educational and research purposes only. Unauthorized automation of protected services may violate terms of use and legal policies. Use responsibly.


## Dataset Generation

To ensure high-quality training data, the dataset was generated through a customized version of `svg-captcha`. The dataset creation process included:

1. **Customizing `svg-captcha`**: The core library was cloned and modified to generate darker-colored CAPTCHAs for improved contrast and legibility.
2. **Ensuring Numeric Balance**:
   - A script was developed to generate CAPTCHAs while ensuring each digit (0-9) appears at least **2000 times**.
   - Captchas were stored as **JPEG images** with their labels saved in corresponding JSON files.
3. **Final Dataset Balancing**:
   - An additional Python script was created to verify and balance the dataset.
   - This script adjusted the dataset by **removing excess occurrences** of overrepresented digits, ensuring a uniform distribution.

### Image Specifications

- **Characters**: 5-digit numeric CAPTCHAs
- **Height**: 50px
- **Width**: 150px
- **Color Mode**: Grayscale with darkened characters
- **Noise & Distortion**: Controlled noise addition for realism

## Model Architecture

The model follows a **CRNN** (Convolutional Recurrent Neural Network) architecture that integrates feature extraction and sequence modeling, making it highly effective for CAPTCHA recognition. It consists of the following key components:

### 1. Convolutional Feature Extraction

- **5 Convolutional Layers**:
  - Uses **3x3 kernels** to capture fine-grained spatial features.
  - **ReLU activations** for non-linearity and efficient training.
- **Batch Normalization**:
  - Stabilizes training and prevents internal covariate shift.
- **MaxPooling Layers**:
  - Reduces spatial dimensions while retaining essential patterns.
  - Two of the pooling layers use asymmetric pooling `(2,1)` to preserve width dimensions.
- **Dropout Layers**:
  - Applied after each convolutional block to reduce overfitting.

### 2. Recurrent Sequence Processing

- **Two-layer BiLSTM (Bidirectional Long Short-Term Memory)**:
  - Captures contextual dependencies in both forward and backward directions.
  - Improves accuracy in sequential data processing.
- **Linear Layer Projection**:
  - Maps the BiLSTM output to character logits.
- **Log Softmax Activation**:
  - Converts raw predictions into probability distributions for decoding.

### 3. Loss Function: Connectionist Temporal Classification (CTC)

- **Why CTC?**
  - CAPTCHAs have variable-length character sequences.
  - No explicit alignment between input images and labels.
  - CTC loss enables sequence learning without predefined character segmentation.
- **CTC Greedy Decoding**
  - Eliminates repeated characters and blank labels to extract the final prediction.

## Training Details

- **Optimizer**: AdamW with weight decay of `1e-5`.
- **Learning Rate Scheduler**: Reduces LR when validation loss plateaus.
- **Batch Size**: 16 (adjusted for optimal GPU usage).
- **Epochs**: 10 (achieves near-optimal accuracy by epoch 5).
- **Gradient Scaling**: Uses mixed-precision training to prevent gradient underflow.
- **Efficient Checkpointing**: Saves model weights and optimizer states for resuming training.
- **Dynamic Input Lengths**: CTC loss allows prediction of varying-length sequences without explicit labels.

## Inference Pipeline

A trained model is used for inference with the following steps:

1. Load the latest trained checkpoint.
2. Preprocess input image (grayscale, resizing, tensor conversion).
3. Forward pass through the CRNN.
4. Decode predictions using CTC Greedy Decoding.
5. Output predicted CAPTCHA text.

## Testing and Real-World Evaluation

To validate the model's performance on **unseen** CAPTCHAs from Snapp, an automated evaluation pipeline was implemented:

1. **Real CAPTCHA Collection**:
   - `snapp_captcha_getter.py` was used to request CAPTCHAs directly from Snapp's API.
   - Captchas were retrieved in base64 format and converted into images.
   - Each CAPTCHA and its API response were logged for further analysis.
2. **Inference on Unseen CAPTCHAs**:
   - The model was tested on these fresh CAPTCHAs to verify generalization.
   - Predictions were compared against ground truth values.
   - Confidence scores and failure cases were analyzed to refine model robustness.
3. **Automated Login Testing**:
   - `snapp_going_inside.py` was used to test the full authentication pipeline.
   - The script successfully extracted CAPTCHA images, predicted text, and submitted the solution.
   - Validation confirmed that the model maintained high accuracy when integrated into real-world API interactions.

## Ethical Considerations

This research aims to enhance OCR and machine learning knowledge. CAPTCHA bypassing should be used ethically, ensuring it does not violate service agreements or legal guidelines.

