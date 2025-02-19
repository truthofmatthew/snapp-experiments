
import collections
import glob
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import string
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
                logging.FileHandler("training.log"),
                logging.StreamHandler()
                ]
        )
logger = logging.getLogger()


CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
METADATA_FILE = os.path.join(CHECKPOINT_DIR, "checkpoints_metadata.json")
IMG_HEIGHT = 50
IMG_WIDTH = 150
CONFUSION_METADATA_FILE = os.path.join(CHECKPOINT_DIR, "confusion_metadata.json")

def load_confusion_metadata():
    if not os.path.exists(CONFUSION_METADATA_FILE):
        return []
    with open(CONFUSION_METADATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_confusion_metadata(metadata):
    with open(CONFUSION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4, default=lambda x: int(x) if isinstance(x, np.int64) else x)

def save_confusion_info(epoch, sample_preds, sample_targs, unique_preds, unique_targs, cm_array):

    confusion_data = load_confusion_metadata()


    cm_list = cm_array.tolist() if hasattr(cm_array, "tolist") else cm_array

    new_entry = {
        "epoch": epoch,
        "sample_predictions": sample_preds,
        "sample_targets": sample_targs,
        "unique_predictions": list(unique_preds),
        "unique_targets": list(unique_targs),
        "confusion_matrix": cm_list
    }

    confusion_data.append(new_entry)
    save_confusion_metadata(confusion_data)


def load_data(data_dir):
    return glob.glob(os.path.join(data_dir, '*.jpg'))


def create_mapping():
    all_chars = "0123456789"
    mapping = {char: i for i, char in enumerate(all_chars)}
    mapping_inv = {v: char for char, v in mapping.items()}
    return mapping, mapping_inv, len(mapping) + 1


def preprocess_data(data, path, mapping):
    datas = collections.defaultdict(list)
    for d in data:
        x = d.split('/')[-1]
        filename_without_extension = x.split('.')[0]

        if len(filename_without_extension) != 5 or not all(char in mapping for char in filename_without_extension):
            logger.warning(f"Skipping invalid filename: {x}")
            continue

        try:
            label = [mapping[char] for char in filename_without_extension]
        except KeyError as e:
            logger.error(f"Unexpected character in filename '{x}': {e}")
            continue

        datas['image'].append(x)
        datas['label'].append(label)

    return pd.DataFrame(datas)


def split_data(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, shuffle=True)


def summarize_data(df_train, df_test):
    total_images = len(df_train) + len(df_test)
    train_images = len(df_train)
    test_images = len(df_test)

    unique_labels = set(
            ''.join(df_train['label'].apply(lambda x: ''.join(map(str, x)))) +
            ''.join(df_test['label'].apply(lambda x: ''.join(map(str, x))))
            )

    logger.info(f"Total Images: {total_images}")
    logger.info(f"Training Images: {train_images}")
    logger.info(f"Testing Images: {test_images}")
    logger.info(f"Unique Characters in Labels: {len(unique_labels)}")
    logger.info(f"Characters Present: {''.join(sorted(unique_labels))}")



def ctc_greedy_decode(seq, blank_idx):
    decoded = []
    prev = None
    for token in seq:
        if token != blank_idx and token != prev:
            decoded.append(token)
            prev = token

    return decoded[:5] + [blank_idx] * (5 - len(decoded))



class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, path='./allcapchas'):
        self.df = df
        self.transform = transform if transform else T.Compose([T.ToTensor()])
        self.path = path
        self.images = []

        for idx in range(len(self.df)):
            image_path = os.path.join(self.path, self.df.iloc[idx]['image'])
            image = Image.open(image_path).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)
        return image, label



class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden * 2, out)

    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return F.log_softmax(out, dim=2)


class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25),
            nn.MaxPool2d((2, 2), (2, 1)),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25),
            nn.MaxPool2d((2, 2), (2, 1)),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25)
        )

        dummy_input = torch.randn(1, in_channels, 50, 150)
        out = self.cnn(dummy_input)
        _, C, H, W = out.size()
        self.calculated_size = C * H

        self.linear = nn.Linear(self.calculated_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output)

    def forward(self, X, y=None, criterion=None):
        assert X.shape[2:] == (50, 150), f"Input size must be (50, 150), got {X.shape[2:]}"

        out = self.cnn(X)
        N, C, H, W = out.size()
        out = out.permute(0, 3, 1, 2)
        out = out.reshape(N, W, -1)
        out = self.linear(out)
        out = self.bn1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out.permute(1, 0, 2)
        out = self.rnn(out)

        if not self.training or y is None or criterion is None:
            return out

        N = out.size(1)
        T = out.size(0)
        input_lengths = torch.full((N,), T, dtype=torch.int32, device=X.device)
        target_lengths = torch.full((N,), fill_value=5, dtype=torch.int32, device=X.device)
        loss = criterion(out, y.view(-1), input_lengths, target_lengths)
        return out, loss




def save_checkpoint(state, epoch, loss, filename="checkpoint.pth"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    logger.info(f"Saving checkpoint to {filepath}")
    torch.save(state, filepath)

    metadata = load_metadata()
    metadata.append(
            {
                    "epoch": epoch + 1,
                    "train_loss": state['train_loss'],
                    "val_loss": state['val_loss'],
                    "path": filepath
                    }
            )
    save_metadata(metadata)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        logger.info("Checkpoint not found. Starting training from scratch.")
        return model, optimizer, 0

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint.get('epoch', 0)


def get_latest_checkpoint():
    metadata = load_metadata()
    if not metadata:
        logger.warning("No checkpoint files found.")
        return None

    metadata.sort(key=lambda x: x['epoch'], reverse=True)
    latest_checkpoint = metadata[0]['path']
    logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return []
    with open(METADATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)



class Engine:
    def __init__(
            self, model, optimizer, criterion, epochs=50, early_stop=False, patience=5, lr_patience=5, device='cpu',
            checkpoint_path=None
            ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.lr_patience = lr_patience
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.writer = SummaryWriter("runs/captcha_training")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=patience // 2
                )
        self.prev_lr = None
        self.lr_no_change_count = 0

    def fit(self, train_dataloader, val_dataloader, save_every=10, start_epoch=0):
        hist_train_loss = []
        hist_val_loss = []
        best_val_loss = float('inf')
        no_improvement_count = 0
        start_time = time.time()

        scaler = torch.amp.GradScaler(enabled=False)

        for epoch in range(start_epoch, self.epochs):

            self.model.train()
            tk = tqdm(train_dataloader, total=len(train_dataloader))
            epoch_loss = 0
            epoch_start_time = time.time()

            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    out, loss = self.model(data, target, criterion=self.criterion)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()
                tk.set_postfix(
                        {
                                'Epoch': epoch + 1,
                                'Loss': f"{loss.item():.4f}"
                                }
                        )

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            hist_train_loss.append(avg_epoch_loss)


            elapsed_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.prev_lr == current_lr:
                self.lr_no_change_count += 1
            else:
                self.lr_no_change_count = 0
            self.prev_lr = current_lr

            remaining_epochs = self.epochs - epoch - 1
            remaining_time = remaining_epochs * elapsed_time
            remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

            if self.lr_no_change_count >= self.lr_patience:
                logger.warning(f"Stopping early due to no change in learning rate for {self.lr_patience} epochs.")
                break


            logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_epoch_loss:.4f} - Time per epoch: {elapsed_time:.2f}s - Estimated remaining time: {remaining_time_str}"
                    )
            self.writer.add_scalar("Loss/train", avg_epoch_loss, epoch)


            val_loss, val_accuracy = self.evaluate(val_dataloader, show_cm=True, epoch=epoch+1)
            hist_val_loss.append(val_loss)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                    )

            self.scheduler.step(val_loss)


            if (epoch + 1) % save_every == 0:
                checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
                save_checkpoint(
                        state={
                                'epoch': epoch + 1,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'train_loss': avg_epoch_loss,
                                'val_loss': val_loss
                                },
                        epoch=epoch + 1,
                        loss=val_loss,
                        filename=checkpoint_filename
                        )


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                save_checkpoint(
                        state={
                                'epoch': epoch + 1,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'train_loss': avg_epoch_loss,
                                'val_loss': val_loss
                                },
                        epoch=epoch + 1,
                        loss=val_loss,
                        filename="best_model.pth"
                        )
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    logger.warning(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        self.writer.close()
        return hist_train_loss, hist_val_loss


    def evaluate(self, dataloader, show_cm=False, epoch=None):
        self.model.eval()
        total_loss = 0
        sample_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(self.device)
                out = self.model(data)
                N = out.size(1)
                T = out.size(0)
                input_lengths = torch.full((N,), T, dtype=torch.int64, device=self.device)
                target_lengths = torch.full((N,), 5, dtype=torch.int64, device=self.device)
                loss = self.criterion(out, target.view(-1), input_lengths, target_lengths)
                total_loss += loss.item()
                predictions = out.permute(1, 0, 2).argmax(2).cpu().numpy()
                targets = target.cpu().numpy()
                for i in range(N):
                    decoded = ctc_greedy_decode(predictions[i], blank_idx=num_class - 1)
                    target_seq = targets[i].tolist()
                    total_samples += 1
                    if decoded == target_seq:
                        sample_correct += 1
                    if len(decoded) < 5:
                        decoded = decoded + [num_class - 1] * (5 - len(decoded))
                    elif len(decoded) > 5:
                        decoded = decoded[:5]
                    all_predictions.extend(decoded)
                    all_targets.extend(target_seq)
                accuracy = sample_correct / total_samples if total_samples > 0 else 0
                tk.set_postfix(
                        {
                                'Val Loss': total_loss / len(tk),
                                'Accuracy': f"{accuracy:.4f}"
                                }
                        )
        avg_val_loss = total_loss / len(dataloader)
        print("Sample Predictions:", all_predictions[:10])
        print("Sample Targets:", all_targets[:10])
        unique_predictions = set(all_predictions)
        unique_targets = set(all_targets)
        print("Unique Predictions:", unique_predictions)
        print("Unique Targets:", unique_targets)

        if show_cm and all_predictions and all_targets:
            cm = confusion_matrix(
                    all_targets,
                    all_predictions,
                    labels=list(range(num_class - 1))
                    )
            disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=[mapping_inv[i] for i in range(num_class - 1)]
                    )
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')


            if epoch is not None:
                cm_filename = os.path.join(CHECKPOINT_DIR, f"epoch_cm_{epoch:02d}.png")
                plt.savefig(cm_filename)
                logger.info(f"Confusion matrix saved to {cm_filename}")
            plt.show()

            save_confusion_info(
                    epoch=epoch,
                    sample_preds=all_predictions[:10],
                    sample_targs=all_targets[:10],
                    unique_preds=set(all_predictions),
                    unique_targs=set(all_targets),
                    cm_array=cm
                    )

        return avg_val_loss, accuracy

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('L')
        transform = T.Compose([T.ToTensor()])
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            decoded = ctc_greedy_decode(output.permute(1, 0, 2).argmax(2).cpu().numpy()[0], blank_idx=num_class - 1)
            print("Decoded Prediction:", decoded)
        return decoded



if __name__ == "__main__":
    DATA_DIR = './allcapchas'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data = load_data(DATA_DIR)
    mapping, mapping_inv, num_class = create_mapping()
    print("mapping", mapping)

    df = preprocess_data(data, DATA_DIR, mapping)
    df_train, df_test = split_data(df)
    label_counts = Counter([item for sublist in df_train['label'] for item in sublist])
    print("Label Distribution:", label_counts)
    summarize_data(df_train, df_test)

    print(Counter(str(item) for sublist in df_train['label'] for item in sublist))

    train_transform = T.Compose(
            [
                    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    T.RandomRotation(10),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    T.ToTensor()
                    ]
            )

    test_transform = T.Compose([T.ToTensor()])
    train_data = CaptchaDataset(df_train, transform=train_transform)
    test_data = CaptchaDataset(df_test, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=8)



    model = CRNN(in_channels=1, output=num_class).to(DEVICE)
    dummy_input = torch.randn(16, 1, 50, 150).to(DEVICE)
    output = model(dummy_input)

    logger.info(f"Model Output Shape: {output.shape}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CTCLoss(blank=num_class - 1, reduction='mean', zero_infinity=True)


    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        model, optimizer, start_epoch = load_checkpoint(latest_checkpoint, model, optimizer)
    else:
        start_epoch = 0
        logger.info("No checkpoint found. Starting training from scratch.")

    engine = Engine(
            model, optimizer, criterion, device=DEVICE, checkpoint_path=latest_checkpoint, early_stop=True, patience=5
            )


    # try:
    #     val_loss, val_accuracy = engine.evaluate(test_loader)
    #     logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    # except Exception as e:
    #     logger.error(f"Error during evaluation: {e}")
    #     raise


    try:
        hist_train_loss, hist_val_loss = engine.fit(train_loader, test_loader, save_every=5, start_epoch=start_epoch)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


    test_loss, test_accuracy = engine.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")



    save_checkpoint(
            state={
                    'state_dict': engine.model.state_dict(),
                    'optimizer': engine.optimizer.state_dict(),
                    'mapping': mapping,
                    'mapping_inv': mapping_inv,
                    'train_loss': hist_train_loss[-1] if hist_train_loss else None,
                    'val_loss': test_loss
                    },
            epoch=engine.epochs,
            loss=test_loss,
            filename="final_model.pth"
            )


    ids = np.random.randint(len(data))
    image_path = data[ids]
    prediction = engine.predict(image_path)


    def show_prediction(image_path, prediction, mapping_inv):
        image = Image.open(image_path)
        plt.imshow(image, cmap='gray')
        predicted_text = ''.join([mapping_inv[i] for i in prediction if i != (num_class - 1)])
        plt.title(f"Prediction: {predicted_text}")
        plt.axis('off')
        plt.show()


    show_prediction(image_path, prediction, mapping_inv)
