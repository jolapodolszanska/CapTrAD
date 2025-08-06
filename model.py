import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import (
    DataLoader, 
    random_split, 
    WeightedRandomSampler
    )
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score
)
from gym import spaces
import gym
import torch.optim as optim
import seaborn as sns
from collections import Counter
from torchsummary import summary
from matplotlib.patches import FancyBboxPatch, FancyArrow
import cv2
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary
import pandas as pd
import time

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  

    def forward(self, inputs, targets):
        if self.weight is not None:
            self.weight = self.weight.to(inputs.device)
        
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_routes=3):
        super(CapsuleLayer, self).__init__()
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = x[:, :, None, :, None] 
        W = self.W.expand(batch_size, -1, -1, -1, -1)  

        u_hat = torch.matmul(W, x)  
        u_hat = u_hat.squeeze(-1)   

        b_ij = torch.zeros(batch_size, self.in_capsules, self.out_capsules).to(x.device)

        for _ in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=2)  

            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)  

            v_j = self.squash(s_j) 

            v_j_expanded = v_j.unsqueeze(1)  
            agreement = (u_hat * v_j_expanded).sum(dim=-1)  

            b_ij = b_ij + agreement

        return v_j  

    def squash(self, s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        scale = (s_j_norm ** 2) / (1 + s_j_norm ** 2)
        v_j = scale * (s_j / (s_j_norm + 1e-8))  
        return v_j

class CapsuleRoutingEnv(gym.Env):
    def __init__(self, in_capsules, out_capsules, num_routes):
        super(CapsuleRoutingEnv, self).__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.num_routes = num_routes

        self.action_space = spaces.Discrete(self.out_capsules)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(in_capsules,))

    def reset(self):
        self.state = np.random.randn(self.in_capsules)  
        return self.state

    def step(self, action):
        reward = self.compute_reward(action)
        self.state = np.random.randn(self.in_capsules)  
        done = True  
        return self.state, reward, done, {}

    def compute_reward(self, action):
        return np.random.random()  
    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HybridCNN(pl.LightningModule):
    def __init__(self):
        super(HybridCNN, self).__init__()

        # ResNet jako ekstraktor cech
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Parametry warstwy kapsułkowej
        self.in_capsules = 64
        self.in_dim = 32
        self.out_capsules = 10
        self.out_dim = 16

        self.fc_transform = nn.Linear(self.resnet_out_features, self.in_capsules * self.in_dim)
        self.capsule_layer = CapsuleLayer(
            in_capsules=self.in_capsules,
            in_dim=self.in_dim,
            out_capsules=self.out_capsules,
            out_dim=self.out_dim,
            num_routes=3
        )

        # CLS token – uczony parametr
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_dim))

        # Transformer (batch_first = True dla zgodności z [B, seq_len, dim])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Klasyfikator
        self.fc1 = nn.Linear(self.resnet_out_features + self.out_dim, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

        self.criterion = FocalLoss(alpha=1, gamma=2, weight=torch.tensor([1.0, 7.0, 1.0, 2.0]))

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = nn.ModuleDict({
            'val_precision': MulticlassPrecision(num_classes=4, average='macro').to(self.device_type),
            'val_recall': MulticlassRecall(num_classes=4, average='macro').to(self.device_type),
            'val_f1': MulticlassF1Score(num_classes=4, average='macro').to(self.device_type),
        })

        self.validation_preds = []
        self.validation_labels = []

    def forward(self, x):
        resnet_output = self.resnet(x)  # [B, 2048]
        transformed_output = self.fc_transform(resnet_output)  # [B, 64*32]
        capsule_input = transformed_output.view(x.size(0), self.in_capsules, self.in_dim)  # [B, 64, 32]

        capsule_output = self.capsule_layer(capsule_input)  # [B, 10, 16]

        # Dodajemy CLS token
        B = capsule_output.size(0)
        cls_tokens = self.cls_token.expand(B, 1, self.out_dim)  # [B, 1, 16]
        capsule_with_cls = torch.cat((cls_tokens, capsule_output), dim=1)  # [B, 11, 16]

        # Transformer
        transformer_output = self.transformer_encoder(capsule_with_cls)  # [B, 11, 16]
        cls_output = transformer_output[:, 0, :]  # [B, 16]

        combined_features = torch.cat((resnet_output, cls_output), dim=1)  # [B, 2048 + 16]
        combined_features = self.dropout(combined_features)
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)

        self.validation_preds.append(preds.cpu())
        self.validation_labels.append(labels.cpu())

        f1_score = self.metrics['val_f1'](preds, labels)
        precision = self.metrics['val_precision'](preds, labels)
        self.log('val_f1_step', f1_score, prog_bar=True, on_step=True)
        self.log('val_precision_step', precision, prog_bar=True, on_step=True)
        return {"preds": preds, "labels": labels}

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_preds, dim=0).to(self.device_type)
        all_labels = torch.cat(self.validation_labels, dim=0).to(self.device_type)

        f1_score = self.metrics['val_f1'](all_preds, all_labels)
        precision = self.metrics['val_precision'](all_preds, all_labels)

        self.log('val_f1', f1_score, prog_bar=True, on_epoch=True)
        self.log('val_precision', precision, prog_bar=True, on_epoch=True)

        self.validation_preds = []
        self.validation_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        self.validation_preds = []
        self.validation_labels = []


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=64):
        super(CustomDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if not self.train_dataset or not self.val_dataset:
            dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
            print(f"Train dataset length: {len(self.train_dataset)}")
            print(f"Validation dataset length: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

def generate_classification_images(val_loader, model):
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 299, 299)
    
    torch.onnx.export(
        model,
        dummy_input,
        "cnn_hybrid_model.onnx",
        export_params=True,
        opset_version=17,  # ← teraz minimum 14, ale najlepiej 17
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    print("ONNX file saved as cnn_hybrid_model.onnx")
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    num_images = min(len(images), 10)  
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  
    axes = axes.flatten() 
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(f'Predicted: {preds[i].item()}, True: {labels[i].item()}')
        ax.axis('off')
    
    for ax in axes[num_images:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_individual_classification_images(val_loader, model):
    class_labels = ['Healthy', 'Mild', 'Moderate', 'Severe']

    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    for i in range(len(images)):
        plt.figure(figsize=(4, 4))  
        plt.imshow(images[i])
        plt.title(f'Predicted: {preds[i].item()}, True: {labels[i].item()}')
        plt.axis('off')

        plt.savefig(f"individual_image_{i+1}.png")
        plt.close()

    print(f"Images saved as individual_image_1.png, individual_image_2.png, ..., up to {len(images)} images.")

    """
    Generate individual classification images and save them as separate files.
    
    Parameters:
        val_loader: DataLoader
            Validation data loader.
        model: nn.Module
            Trained model for classification.
        class_labels: list
            List of class labels for predictions, e.g., ['Healthy', 'Mild', 'Moderate', 'Severe'].
    """
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    num_images = min(len(images), 4) 
    
    for i in range(num_images):
        plt.figure(figsize=(4, 4)) 
        plt.imshow(images[i])
        plt.title(f'Predicted: {class_labels[preds[i].item()]}, True: {class_labels[labels[i].item()]}')
        plt.axis('off')
        
        plt.savefig(f'image_{i+1}.png', bbox_inches='tight')
        plt.close() 

    print(f"Saved {num_images} individual images as PNG files.")

    """
    Generate individual classification images and save them as separate files.
    
    Parameters:
        val_loader: DataLoader
            Validation data loader.
        model: nn.Module
            Trained model for classification.
        class_labels: list
            List of class labels for predictions, e.g., ['Healthy', 'Mild', 'Moderate', 'Severe'].
    """
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    num_images = min(len(images), 4) 
    
    for i in range(num_images):
        plt.figure(figsize=(4, 4)) 
        plt.imshow(images[i])
        plt.title(f'Predicted: {class_labels[preds[i].item()]}, True: {class_labels[labels[i].item()]}')
        plt.axis('off')
        
        # Save image as PNG file
        plt.savefig(f'image_{i+1}.png', bbox_inches='tight')
        plt.close() 

    print(f"Saved {num_images} individual images as PNG files.")
    
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().detach().numpy()
        return cam

    
def collect_predictions(model, dataloader, device):
    model.to(device) 
    model.eval()
    all_preds = []
    all_labels = []
    all_proba = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)  
            labels = labels.to(device)  
            outputs = model(inputs)  
            preds = torch.argmax(outputs, dim=1)
            probas = F.softmax(outputs, dim=1)  # Dodane

            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(probas.cpu().numpy())  # Dodane

    return np.array(all_preds), np.array(all_labels), np.array(all_proba)


def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
def show_cam_on_image(img, mask, alpha=0.5):
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    cam_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    brain_mask = create_brain_mask(img)
    if brain_mask.max() > 1:
        brain_mask = brain_mask / 255.0
    brain_mask_3ch = np.repeat(brain_mask[..., np.newaxis], 3, axis=2)

    # Maskujemy obraz i mapę cieplną, żeby były tylko na mózgu
    img_masked = img_np * brain_mask_3ch
    heatmap_masked = heatmap * brain_mask_3ch

    # Łączenie
    overlay = heatmap_masked + img_masked
    overlay = overlay / np.max(overlay + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Grad-CAM (masked)")
    plt.show()

    
def generate_gradcam_comparison(val_loader, model, class_labels, target_layer):
    grad_cam = GradCAM(model, target_layer)
    images_per_class = {}
    
    # Znajdź po 1 obrazie dla każdej klasy
    for batch in val_loader:
        inputs, labels = batch
        for i in range(inputs.size(0)):
            label = labels[i].item()
            if label not in images_per_class:
                images_per_class[label] = inputs[i].unsqueeze(0)
            if len(images_per_class) == len(class_labels):
                break
        if len(images_per_class) == len(class_labels):
            break

    # Tworzenie figure z mapami Grad-CAM
    fig, axes = plt.subplots(1, len(class_labels), figsize=(4 * len(class_labels), 4))

    for idx, class_idx in enumerate(sorted(images_per_class.keys())):
        input_tensor = images_per_class[class_idx].to(model.device)
        original_img = input_tensor.squeeze(0)

        cam = grad_cam.generate(input_tensor, target_class=class_idx)
        mask_resized = cv2.resize(cam, (original_img.shape[2], original_img.shape[1]))

        # Stwórz maskę mózgu
        brain_mask = create_brain_mask(original_img)
        img_np = original_img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        overlay = (heatmap * brain_mask[..., np.newaxis]) + (img_np * brain_mask[..., np.newaxis])
        overlay = overlay / np.max(overlay)

        axes[idx].imshow(overlay)
        axes[idx].set_title(class_labels[class_idx])
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
    
def generate_gradcam_grid(val_loader, model, target_layer, num_images=10):
    grad_cam = GradCAM(model, target_layer)
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    num_images = min(num_images, images.size(0))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(num_images):
        input_tensor = images[i].unsqueeze(0)
        cam = grad_cam.generate(input_tensor, target_class=preds[i].item())

        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        brain_mask = create_brain_mask(images[i])
        heatmap_masked = heatmap * brain_mask[..., np.newaxis]
        img_masked = img_np * brain_mask[..., np.newaxis]
        overlay = heatmap_masked + img_masked
        overlay = overlay / overlay.max()

        axes[i].imshow(overlay)
        axes[i].set_title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
        axes[i].axis('off')

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("gradcam_grid.png", dpi=300)
    plt.show()


def create_brain_mask(img_tensor):
    """
    Zwraca binarną maskę mózgu (0-tło, 1-mózg) w formacie H×W (uint8).
    """
    # --- denormalizacja -----------------------------------------------------
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    # --- konwersja do 8-bit, skala szarości --------------------------------
    gray = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    # --- automatyczny próg Otsu + ewentualna inwersja -----------------------
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # jeżeli po prógowaniu tłem jest biel, odwróć
    if np.mean(gray[thresh == 255]) < np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # --- morfologia (usuwa szumy, domyka dziury) ----------------------------
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask,  cv2.MORPH_CLOSE, k, iterations=2)

    # --- wybór największej składowej (sam mózg) -----------------------------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:                       # 0 = tło
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    return mask      # uint8 (0/255)


def generate_masked_gradcam_grid(val_loader, model, target_layer, num_images=10):
    model.eval()
    grad_cam = GradCAM(model, target_layer)
    class_labels = ['Healthy', 'Mild', 'Moderate', 'Severe']

    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(num_images):
        img = images_denorm[i].detach().cpu()
        cam_mask = grad_cam.generate(images[i].unsqueeze(0), target_class=preds[i].item())

        img_np = img.permute(1, 2, 0).numpy()
        cam_resized = cv2.resize(cam_mask, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        brain_mask = create_brain_mask(img)
        if brain_mask.max() > 1:
            brain_mask = brain_mask / 255.0
        brain_mask_3ch = np.repeat(brain_mask[..., np.newaxis], 3, axis=2)

        img_masked = img_np * brain_mask_3ch
        heatmap_masked = heatmap * brain_mask_3ch
        overlay = heatmap_masked + img_masked
        overlay = overlay / np.max(overlay + 1e-8)

        axes[i].imshow(overlay)
        axes[i].set_title(f'Pred: {preds[i].item()}, True: {labels[i].item()}')
        axes[i].axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("fig1_gradcam_masked.png", dpi=300)
    plt.show()
    
def compute_class_metrics(y_true, y_pred, y_proba=None, class_labels=None):
    """
    Zwraca DataFrame z metrykami: Precision, Recall, F1-score i AUC dla każdej klasy.
    
    Parameters:
        y_true: np.array - rzeczywiste etykiety
        y_pred: np.array - predykcje modelu
        y_proba: np.array or None - prawdopodobieństwa (jeśli dostępne)
        class_labels: list or None - opcjonalne etykiety klas (np. ['HC', 'Mild', 'Mod.', 'Sev.'])

    Returns:
        DataFrame z metrykami per klasa.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()

    # Jeśli mamy prawdopodobieństwa – licz AUC per klasa
    if y_proba is not None:
        n_classes = len(np.unique(y_true))
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        auc_scores = {}
        for i in range(n_classes):
            try:
                auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            except ValueError:
                auc = np.nan
            auc_scores[str(i)] = auc
        for label in auc_scores:
            if label in df.index:
                df.loc[label, "AUC"] = auc_scores[label]

    # Zachowaj tylko interesujące kolumny
    valid_labels = [str(i) for i in range(len(class_labels)) if str(i) in df.index]
    df = df.loc[valid_labels, ["precision", "recall", "f1-score", "AUC"] if "AUC" in df.columns else ["precision", "recall", "f1-score"]]

    if class_labels:
        df.index = class_labels

    return df.round(3)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    class_labels = ['Class1', 'Class2', 'Class3', 'Class4']
    torch.set_float32_matmul_precision('high')
    
    dataset_path = 'D:/Badania/embedded/Dataset'
    data_module = CustomDataModule(dataset_path)

    env = CapsuleRoutingEnv(in_capsules=64, out_capsules=10, num_routes=3)
    state_dim = env.observation_space.shape[0]
    
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    model = HybridCNN().to("cuda")
    summary(model, input_size=(1, 3, 299, 299), device="cuda")
    
    
    logger = TensorBoardLogger('tb_logs', name='transformer-caps')
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, verbose=True)
    progress_bar = ProgressBar()
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar]
    )
     
    trainer.fit(model, datamodule=data_module)

    generate_classification_images(data_module.val_dataloader(), model)
    
    generate_individual_classification_images(data_module.val_dataloader(), model)
    
    val_loader = data_module.val_dataloader() 
    preds, labels, probas = collect_predictions(model, val_loader, device='cuda')  # ✅

    
    plot_confusion_matrix(labels, preds, class_labels)
    
    # Grad-CAM demo
    target_layer = model.resnet.layer4
    generate_gradcam_comparison(data_module.val_dataloader(), model, class_labels=['Healthy', 'Mild', 'Moderate', 'Severe'], target_layer=target_layer)
    grad_cam = GradCAM(model, target_layer)
    
    # Pobierz jedną próbkę walidacyjną
    batch = next(iter(data_module.val_dataloader()))
    images, labels = batch
    images = images.to('cuda')
    input_tensor = images[0].unsqueeze(0)  # [1, 3, 299, 299]
    
    # Generuj mapę cieplną
    cam = grad_cam.generate(input_tensor)
    
    # Pokaż ją
    show_cam_on_image(images[0], cam)
    
    generate_gradcam_grid(
        val_loader=data_module.val_dataloader(),
        model=model,
        target_layer=model.resnet.layer4,  # lub inna warstwa
        num_images=10
    )
    
    target_layer = model.resnet.layer4
    generate_masked_gradcam_grid(data_module.val_dataloader(),
                                 model,
                                 target_layer,
                                 num_images=10)
    
    y_pred, y_true, y_proba = collect_predictions(model, val_loader, device='cuda')

    metrics_df = compute_class_metrics(
        y_true,
        y_pred,
        y_proba=y_proba,
        class_labels=["HC", "Mild", "Mod.", "Sev."]
    )
    print(metrics_df)
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    start = time.time()
    trainer.fit(model, datamodule=data_module)
    end = time.time()
    elapsed_time = end - start
    print(f"Training time: {elapsed_time / 3600:.2f} hours")
