
# Standard
import argparse
import csv
import os
# External
import numpy as np
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module, MSELoss
from torch.utils.data import DataLoader, Subset
# Internal
from data import EmbryoAnnotations, LabelType
from models import ViTBinaryClassifier

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_TYPE_MAP = {
    'bin': LabelType.Binary,
    'dsg': LabelType.DoubleSigmoid
}

parser = argparse.ArgumentParser(description='ubar-pytorch-implementation')
parser.add_argument('--job', type = str, choices = ['train', 'test'], help = 'Indicates whether to train or test', default = 'train')
parser.add_argument('--label-type', type = str, choices = LABEL_TYPE_MAP, help = 'Select label type', default = 'dsg')
parser.add_argument('--epochs', type = int, help = 'Number of epochs for training', default = 50)
parser.add_argument('--patience', type = int, help = 'Early stopping patience', default = 10)
parser.add_argument('--folds', type = int, help = 'Number of folds for cross-validation', default = 8)
parser.add_argument('--seed', type = int, help = 'Random seed for reproducibility', default = 1985)
parser.add_argument('--workers', type = int, help = 'Number of threads', default = 4)
parser.add_argument('--batch-train', type = int, help = 'Batch size for training', default = 32)
parser.add_argument('--batch-test', type = int, help = 'Batch size for testing', default = 16)
opt = parser.parse_args()

# constants
EPOCHS = opt.epochs
PATIENCE = opt.patience
FOLDS = opt.folds
SEED = opt.seed
WORKERS = opt.workers
BATCH_TRAIN = opt.batch_train
BATCH_TEST = opt.batch_test

LABEL_TYPE = LABEL_TYPE_MAP[opt.label_type]

CP_DIR = f'checkpoints/{"bin" if LABEL_TYPE == LabelType.Binary else "dsg"}/'
os.makedirs(CP_DIR, exist_ok = True)

def validate(device, model: Module, dataloader: DataLoader, criterion: Module):
    model.eval()
    criterion.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels, __indexes__ in dataloader:
            images: Tensor = images.to(device)
            labels: Tensor = labels.to(device).unsqueeze(1)  # Shape: [batch_size, 1]
            preds: Tensor = model(images)
            batch_loss: Tensor = criterion(preds, labels)
            epoch_loss += batch_loss.item() * images.size(0)
            all_labels.extend(labels.squeeze().tolist())
            all_preds.extend(preds.squeeze().tolist())
    epoch_loss = epoch_loss / len(all_labels)
    all_labels = [ round(label) for label in all_labels ]
    all_preds = [ round(pred) for pred in all_preds ]
    
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division = 0)
    precision = precision_score(all_labels, all_preds, zero_division = 0)
    recall = recall_score(all_labels, all_preds, zero_division = 0)
    cf = confusion_matrix(all_labels, all_preds, labels = [0, 1])
    tn, fp, fn, tp = (cf / len(all_labels)).ravel()
    
    # Done
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    return metrics

def train():
    # Define the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize the Dataset
    full_dataset = EmbryoAnnotations(label_type = LABEL_TYPE)
    print('Initiating training with label type:', LABEL_TYPE.name)
    
    # Extract labels for StratifiedKFold
    X = [0] * len(full_dataset)
    y = np.zeros(len(full_dataset))
    skf = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)

    # Initialize Metrics Storage
    best_metrics = [ None ] * FOLDS
    last_metrics = [ None ] * FOLDS
    for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1} / {FOLDS}")

        # Create Subsets for Training and Validation
        sub_train = Subset(full_dataset, idx_train)
        sub_valid = Subset(full_dataset, idx_valid)

        # Create DataLoaders
        load_train = DataLoader(sub_train, batch_size = BATCH_TRAIN, shuffle = True, num_workers = WORKERS)
        load_valid = DataLoader(sub_valid, batch_size = BATCH_TEST, shuffle = False, num_workers = WORKERS)

        # Initialize the Model, Loss Function and Optimizer
        model = ViTBinaryClassifier().to(device)
        if LABEL_TYPE == LabelType.DoubleSigmoid:
            criterion = MSELoss().to(device)
        elif LABEL_TYPE == LabelType.Binary:
            criterion = BCEWithLogitsLoss().to(device)
        else:
            raise NotImplementedError("Unsupported label type. Use LabelType.DoubleSigmoid or LabelType.Binary.")
        optimizer = torch.optim.Adam(
            [ { 'params': model.classifier.parameters() } ],
            lr = 1e-4)

        # Training Loop
        stop_counter = 0
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1} / {EPOCHS}")
            model.train()
            criterion.train()
            epoch_loss = 0.0
            for batch_index, (images, labels, __indexes__) in enumerate(load_train):
                print(f"  Batch {batch_index + 1} / {len(load_train)}")
                images: Tensor = images.to(device)
                labels: Tensor = labels.to(device).unsqueeze(1)
                # Forward Pass
                predictions: Tensor = model(images)
                batch_loss: Tensor = criterion(predictions, labels)
                # Backward Pass and Optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item() * images.size(0)

            # Calculate Average Loss for the Epoch
            epoch_loss = epoch_loss / len(sub_train)
            print(f"  Training Loss: {epoch_loss:.3f}")

            # Evaluation on Validation Set
            print("  Validating...")
            metrics = validate(device, model, load_valid, criterion)
            for key, value in metrics.items():
                print(f"   - {key.replace('_', ' ').title()}: {value:.3f}")

            # Check for Improvement
            if best_metrics[fold] is None or metrics['f1_score'] > best_metrics[fold]['f1_score']:
                stop_counter = 0
                best_metrics[fold] = metrics
                torch.save(model.classifier, f'{CP_DIR}/cls.{fold}.pth')
                print("  ✓✓✓")
            else:
                stop_counter += 1
                print(f"  ✗✗✗     [ {stop_counter} / {PATIENCE} ]")
                if stop_counter >= PATIENCE:
                    break
        
        # Store Final Metrics
        last_metrics.append(metrics)

    # Aggregate Metrics Across Folds
    avg_metrics = { key: sum(best_metrics[fold][key] for fold in range(FOLDS)) / FOLDS 
                   for key in best_metrics[0] }

    # Print Average Metrics
    print("Cross-Validation Results:")
    for key in avg_metrics:
        print(f"  Average {key.replace('_', ' ').title()}: {avg_metrics[key]:.3f}")

def test():
    # Define the device for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Initialize the Dataset and DataLoader
    print('Initiating testing with label type:', LABEL_TYPE.name)
    full_dataset = EmbryoAnnotations(label_type = LABEL_TYPE)
    load_test = DataLoader(full_dataset, batch_size = BATCH_TRAIN, shuffle = False, num_workers = WORKERS)
    # Initialize the Model and Predictions Storage
    model = ViTBinaryClassifier().to(device)
    id2data = { id: { 'label': label, 'preds': [ None ] * FOLDS } 
               for id, label in zip(full_dataset.ids, full_dataset.labels) }
    load_args = { 'map_location': device, 'weights_only': False }
    for fold in range(FOLDS):
        # Load the Fold Classifier
        print(f"Loading classifier for fold {fold + 1} / {FOLDS}")
        model.classifier = torch.load(f'{CP_DIR}/cls.{fold}.pth', **load_args)
        # Generate and store predictions
        print(f"Generating predictions for fold {fold + 1} / {FOLDS}")
        model.eval()
        for images, __labels__, indexes in load_test:
            images: Tensor = images.to(device)
            indexes = indexes.tolist()
            with torch.no_grad():
                preds: Tensor = model(images)
            preds = preds.squeeze().tolist()
            for i, pred in zip(indexes, preds):
                id2data[full_dataset.ids[i]]['preds'][fold] = pred
    # Write Predictions to CSV
    with open(f'logs/results-{opt.label_type}.csv', 'w') as fw:
        cw = csv.writer(fw)
        cw.writerow(['id', 'label', 'pred' ] + [f'fold {i + 1}' for i in range(FOLDS)])
        for id, data in id2data.items():
            label = data['label']
            preds = data['preds']
            mean_pred = np.mean(preds)
            cw.writerow([id, label, mean_pred] + preds)

if __name__ == '__main__':
    job = opt.job
    if job == 'train':
        train()
    if job == 'test':
        test()
    print("-" * 40)
