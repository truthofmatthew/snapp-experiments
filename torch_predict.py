import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import logging

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
CAPTCHAS = os.path.join(SCRIPT_DIR, "captcha")
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "final_model.pth")
METADATA_FILE = os.path.join(CHECKPOINT_DIR, "checkpoints_metadata.json")

def ctc_decode(predicted_indices, probs, target_length=5, blank_idx=10):
    decoded = []
    confidences = []
    prev = None

    for t, idx in enumerate(predicted_indices):
        if idx != prev and idx != blank_idx:
            decoded.append(idx)
            confidences.append(probs[t, idx])
        prev = idx


    if len(decoded) < target_length:
        best = np.argmax(probs.mean(axis=0)[:blank_idx])
        while len(decoded) < target_length:
            decoded.append(best)
            confidences.append(probs[:, best].max())

    return decoded[:target_length], confidences[:target_length]


def load_metadata():

    if not os.path.exists(METADATA_FILE):
        logger.warning(f"Metadata file not found at {METADATA_FILE}. Using default values.")
        return {}
    with open(METADATA_FILE, 'r') as f:
        try:
            metadata = json.load(f)
            if not isinstance(metadata, dict):
                logger.error("Metadata file is not a dictionary. Using default values.")
                return {}
            return metadata
        except json.JSONDecodeError:
            logger.error("Failed to decode metadata file. Using default values.")
            return {}

def create_default_mapping():

    all_chars = "0123456789"
    mapping = {char: i for i, char in enumerate(all_chars)}
    mapping_inv = {v: char for char, v in mapping.items()}
    num_class = len(mapping) + 1
    return mapping, mapping_inv, num_class


def ctc_greedy_decode(seq, blank_idx):
    decoded = []
    prev = None
    for token in seq:
        if token != blank_idx and token != prev:
            decoded.append(token)
            prev = token
    decoded = decoded[:5] + [blank_idx] * (5 - len(decoded))
    return decoded


class Bidirectional(torch.nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = torch.nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = torch.nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = torch.nn.Linear(hidden * 2, out)

    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return torch.nn.functional.log_softmax(out, dim=2)

class CRNN(torch.nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d((2, 2), (2, 1)),
            torch.nn.Conv2d(256, 512, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d((2, 2), (2, 1)),
            torch.nn.Conv2d(512, 512, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Dropout(0.25)
        )
        dummy_input = torch.randn(1, in_channels, 50, 150)
        out = self.cnn(dummy_input)
        _, C, H, W = out.size()
        self.calculated_size = C * H
        self.linear = torch.nn.Linear(self.calculated_size, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output)

    def forward(self, X):
        out = self.cnn(X)
        N, C, H, W = out.size()
        out = out.permute(0, 3, 1, 2)
        out = out.reshape(N, W, -1)
        out = self.linear(out)
        out = self.bn1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out.permute(1, 0, 2)
        out = self.rnn(out)
        return out


def predict_image(image_path, model, device, mapping_inv, num_class):
    model.eval()
    transform = T.Compose([T.Grayscale(), T.Resize((50, 150)), T.ToTensor()])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        decoded = ctc_greedy_decode(output.permute(1, 0, 2).argmax(2).cpu().numpy()[0], blank_idx=num_class - 1)
        predicted_text = ''.join([mapping_inv[i] for i in decoded if i != (num_class - 1)])
    return predicted_text


def generate_html_report(predictions, output_html="captcha_report.html"):
    html_content = """
    <html>
        <head>
            <title>Captcha Predictions Report</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .card { border: 1px solid #ccc; padding: 10px; margin: 10px; }
                img { max-width: 150px; }
            </style>
        </head>
        <body>
            <h1>Captcha Predictions Report</h1>
    """

    for image_path, predicted_text, confidences in predictions:
        avg_conf = f"{np.mean(confidences):.2f}" if confidences else "N/A"
        conf_str = " ".join([f"{predicted_text[i]}:{confidences[i]:.2f}" for i in range(len(predicted_text))]) if confidences else "N/A"

        abs_path = os.path.abspath(image_path)
        html_content += f"""
        <div class="card">
            <img src="{abs_path}" alt="Captcha" />
            <p><strong>Prediction:</strong> {predicted_text}</p>
            <p><strong>Avg Confidence:</strong> {avg_conf}</p>
            <p><strong>Confidence Details:</strong> {conf_str}</p>
        </div>
        """

    html_content += """
        </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report saved to {output_html}")

def predict_captcha_with_confidence(image_path, model, mapping_inv, num_class, device):
    model.eval()
    try:
        image = Image.open(image_path).convert('L')
        transform = T.Compose([T.ToTensor()])
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)

        probs = torch.exp(output).permute(1, 0, 2).cpu().numpy()[0]
        predicted_indices = probs.argmax(axis=1)

        decoded_indices, confidences = ctc_decode(predicted_indices, probs, target_length=5, blank_idx=num_class - 1)
        predicted_text = ''.join([mapping_inv.get(i, '') for i in decoded_indices])

        if not predicted_text:
            predicted_text = "UNKNOWN"

        return predicted_text, confidences, np.mean(confidences) if confidences else 0.0

    except Exception as e:
        print(f"Prediction error for {image_path}: {e}")
        return "ERROR", [], 0.0

def generate_captcha_predictions_report(directory, model, mapping_inv, num_class, device, output_html="captcha_report.html"):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    if not image_paths:
        print(f"No .jpg images in {directory}.")
        return

    predictions = []
    for image_path in image_paths:
        pred_text, confs, _ = predict_captcha_with_confidence(image_path, model, mapping_inv, num_class, device)
        predictions.append((image_path, pred_text, confs))

    generate_html_report(predictions, output_html)

def show_single_prediction(image_path, predicted_text):
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {predicted_text}")
    plt.axis('off')
    plt.show()

def show_predictions_in_grid(directory, model, mapping_inv, num_class, device, cols=4):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    if not image_paths:
        print(f"No .jpg images in {directory}.")
        return
    predictions = []
    for image_path in image_paths:
        pred_text = predict_image(image_path, model, device, mapping_inv, num_class)
        predictions.append((image_path, pred_text))

    rows = (len(predictions) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, (img_path, pred_text) in enumerate(predictions):
        ax = axes[i]
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {pred_text}", fontsize=10)
        ax.axis('off')
    for j in range(len(predictions), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def get_prediction_function(image_path):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    metadata = load_metadata()
    if 'mapping' not in metadata or 'mapping_inv' not in metadata:
        logger.warning("Metadata file is missing required keys ('mapping' or 'mapping_inv'). Using default values.")
        mapping, mapping_inv, num_class = create_default_mapping()
    else:
        mapping = metadata['mapping']
        mapping_inv = metadata['mapping_inv']
        num_class = len(mapping) + 1

    logger.info(f"Loading model from {MODEL_CHECKPOINT}")
    model = CRNN(in_channels=1, output=num_class).to(DEVICE)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    pred_text, confs, _ = predict_captcha_with_confidence(image_path, model, mapping_inv, num_class, DEVICE)

    return pred_text, confs, _



if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    metadata = load_metadata()
    if 'mapping' not in metadata or 'mapping_inv' not in metadata:
        logger.warning("Metadata file is missing required keys ('mapping' or 'mapping_inv'). Using default values.")
        mapping, mapping_inv, num_class = create_default_mapping()
    else:
        mapping = metadata['mapping']
        mapping_inv = metadata['mapping_inv']
        num_class = len(mapping) + 1


    logger.info(f"Loading model from {MODEL_CHECKPOINT}")
    model = CRNN(in_channels=1, output=num_class).to(DEVICE)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])




    # prediction = predict_image("./captcha/captcha_1739620858.jpg", model, DEVICE, mapping_inv, num_class)
    # logger.info(f"Predicted CAPTCHA1: {prediction}")
    # show_single_prediction(image_path, prediction)

    # pred_text, confs, _ = get_prediction_function("./captcha/captcha_1739620858.jpg")

    # print(f"Predicted CAPTCHA2: {pred_text, confs, _}")


    IMAGE_DIR = CAPTCHAS
    show_predictions_in_grid(IMAGE_DIR, model, mapping_inv, num_class, DEVICE, cols=4)

    # generate_captcha_predictions_report(IMAGE_DIR, model, mapping_inv, num_class, DEVICE, "final.html")