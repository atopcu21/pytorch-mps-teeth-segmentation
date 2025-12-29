import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
# We resize to 512x512 to ensure consistency and fit in memory.
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 8         # Selected to fit in MPS memory
LEARNING_RATE = 1e-4   # Standard starting rate for Adam
NUM_EPOCHS = 50        # Max epochs, controlled by Early Stopping
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. DATASET CLASS ---
class TeethDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List all PNG files
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Load as Grayscale (1 channel)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize to fixed size
        resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        image = resize(image)
        mask = resize(mask)

        # Convert to Tensor and Normalize
        # Images: [0, 255] -> [0.0, 1.0]
        # Masks: [0, 255] -> [0.0, 1.0] (Thresholded later)
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        return image, mask, self.images[index] # Return name for tracking

# --- 3. U-NET ARCHITECTURE (Modified for 16 channels) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Modified: Start with 16 features instead of 64
        features = [16, 32, 64, 128] 
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net (Contracting Path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck (The bottom of the U)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # 128 -> 256

        # Up part of U-Net (Expansive Path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1] # Reverse list

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # If input size was not perfectly divisible by 2, sizes might slightly differ
            # We resize x to match skip_connection if needed (though 512x512 avoids this)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- 4. TRAINING & METRIC FUNCTIONS ---

def calculate_metrics(pred, target, threshold=0.5):
    # Flatten tensors
    pred = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.view(-1)

    # True Positives, False Positives, False Negatives
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()

def train_model():
    # Paths (Modify these to your actual absolute paths if needed)
    base_dir = "X-ray-Teeth"
    
    train_ds = TeethDataset(f"{base_dir}/images/train", f"{base_dir}/masks/train")
    val_ds = TeethDataset(f"{base_dir}/images/val", f"{base_dir}/masks/val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss() # Combines Sigmoid + BCE

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    print("Starting Training on:", DEVICE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_prec, train_rec, train_f1 = 0, 0, 0
        
        # --- TRAINING LOOP ---
        for images, masks, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Calculate Training Metrics
            p, r, f1 = calculate_metrics(predictions, masks)
            train_prec += p
            train_rec += r
            train_f1 += f1

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        val_prec, val_rec, val_f1 = 0, 0, 0
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                preds = model(images)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()
                
                p, r, f1 = calculate_metrics(preds, masks)
                val_prec += p
                val_rec += r
                val_f1 += f1

        # Average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_prec = val_prec / len(val_loader)
        avg_val_rec = val_rec / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)

        # Print all metrics for the report
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train -> Loss: {avg_train_loss:.4f} | F1: {avg_train_f1:.4f}")
        print(f"  Val   -> Loss: {avg_val_loss:.4f} | Prec: {avg_val_prec:.4f} | Rec: {avg_val_rec:.4f} | F1: {avg_val_f1:.4f}")
        # Stopping Condition & Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "unet_teeth_best.pth")
            patience_counter = 0
            print("  -> Model Saved (Validation Loss Improved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early Stopping Triggered")
                break

# --- 5. EVALUATION FUNCTION (Run this after training) ---
def evaluate_test_set():
    print("\n--- Evaluating Test Set ---")
    base_dir = "X-ray-Teeth"
    test_ds = TeethDataset(f"{base_dir}/images/test", f"{base_dir}/masks/test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Batch 1 for individual metrics

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("unet_teeth_best.pth"))
    model.eval()

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    # For saving visual results
    save_dir = "visual_results"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (image, mask, fname) in enumerate(test_loader):
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            pred_logits = model(image)
            precision, recall, f1 = calculate_metrics(pred_logits, mask)

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Save some examples (e.g., first 6)
            if idx < 6: 
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float().cpu()
                # Create a composite image: Input | True Mask | Predicted Mask
                # (Need to denormalize input for visualization if we normalized, 
                # but here we just have [0,1] tensors so simply * 255)
                
                img_vis = (image.cpu().squeeze() * 255).byte().numpy()
                true_vis = (mask.cpu().squeeze() * 255).byte().numpy()
                pred_vis = (pred_mask.squeeze() * 255).byte().numpy()
                
                combined = Image.fromarray(np.hstack((img_vis, true_vis, pred_vis)))
                combined.save(f"{save_dir}/{fname[0]}_result.png")

    num_images = len(test_loader)
    print(f"Average Precision: {total_precision/num_images:.4f}")
    print(f"Average Recall:    {total_recall/num_images:.4f}")
    print(f"Average F-Score:   {total_f1/num_images:.4f}")

if __name__ == "__main__":
    train_model()
    evaluate_test_set()