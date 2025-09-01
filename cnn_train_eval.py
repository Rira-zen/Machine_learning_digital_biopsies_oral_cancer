import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import copy



save_dir = 'C:\\Users\\path_to_dir\\saved_models'
os.makedirs(save_dir, exist_ok=True)


class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

        # Map class names to integer labels
        self.class_to_label = {cls: i for i, cls in enumerate(self.classes)}

        # List of (image_path, label) tuples
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_label = self.class_to_label[class_name]

            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                data.append((image_path, class_label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# Desired transforms
data_transform = transforms.Compose([
    transforms.Resize((299, 299), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# Add transforms to dataset
root_dir = "C:\\Users\\path_to_dir\\Labelled training Tiffs"
qmr_dataset = Dataset(root_dir, transform=data_transform)

#print(qmr_dataset[2])

# Get the image paths and corresponding labels from qmr_dataset
all_files = [item[0] for item in qmr_dataset.data]
all_labels = [item[1] for item in qmr_dataset.data]

# Convert labels to numerical indices
label_to_index = {label: i for i, label in enumerate(set(all_labels))}
all_labels = [label_to_index[label] for label in all_labels]

# Convert to numpy arrays for easy indexing
all_files = np.array(all_files)
all_labels = np.array(all_labels)

#Define ground_truth_labels_list
ground_truth_labels_list = []

# Upto this point we have simply loaded and preprocessed the data


# Cross entropy loss function
criterion = nn.CrossEntropyLoss()

# Function to train and evaluate the model
def train_and_evaluate(QMRmodel, train_loader, val_loader, criterion, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QMRmodel.to(device)
    
    train_losses = []
    val_losses = []
    best_val_accuracy = 0.0
    gt_labels_list = []

    for epoch in range(epochs):
        
        # Training phase
        QMRmodel.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            main_output,aux_output = QMRmodel(inputs)  
            loss = criterion(main_output, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = main_output.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train
        train_losses.append(train_loss)


        # Validation phase
        QMRmodel.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        predictions_list = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                gt_labels_list.extend(labels.cpu().numpy())
                
                main_output = QMRmodel(inputs)
               
                main_output_probs = F.softmax(main_output, dim=1)
                
                loss = criterion(main_output_probs, labels)

                running_val_loss += loss.item()
                _, predicted = main_output.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
                #Add to predictions list
                predictions_list.extend(main_output_probs.cpu().numpy())
                
        gt_labels = np.array(gt_labels_list)
        predictions = np.array(predictions_list)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        #empty the gt_list
        gt_labels_list.clear()

        # Save the model if it has the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = copy.deepcopy(QMRmodel.state_dict())

    # Load the best model weights
    QMRmodel.load_state_dict(best_model_weights)

    return gt_labels, predictions, best_val_accuracy




# Range of hyperparameters
epochs_range = [20, 25, 30]
learning_rates = [0.001, 0.01, 0.1]

# DataLoader
loader = DataLoader(qmr_dataset, batch_size=32, shuffle=True)

# Initialize dictionary to store results
results_dict = {'fold':[], 
                'epochs': [],
                'lr':[],
                'accuracy':[],
                'true positives':[],
                'false positives':[],
                'true negatives':[],
                'false negatives':[],
                'sensitivity':[],
                'specificity':[],
                'precision':[],
                'recall':[],
                'f1_score':[]
                }

# Perform 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(all_files, all_labels)):
    train_dataset = torch.utils.data.Subset(qmr_dataset, train_index)
    val_dataset = torch.utils.data.Subset(qmr_dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for epochs in epochs_range:
        for lr in learning_rates:
            
            
            # Load Inception V3 with pre-trained weights
            QMRmodel = models.inception_v3(pretrained=True)

            # Modify the final fully connected layer for binary classification
            num_ftrs = QMRmodel.fc.in_features
            QMRmodel.fc = nn.Linear(num_ftrs, 2)  # 2 output nodes for binary classification
            
            
            # Instantiate model and optimizer with the specified learning rate
            optimizer = optim.SGD(QMRmodel.parameters(), lr=lr)

            # Train and evaluate the model, get accuracy
            gt_labels, predictions, accuracy = train_and_evaluate(QMRmodel, train_loader, val_loader, criterion, optimizer, epochs)

            # Save model
            model_filename = f'{save_dir}/model_fold{fold}_epochs{epochs}_lr{lr}.pth'
            torch.save(QMRmodel,model_filename)
            
            print(f'Accuracy: {accuracy}')
            
            # Convert raw model outputs to class labels
            
            gt_labels_t = [torch.tensor(gt).item() for gt in gt_labels]
            #print(f'gt_labels tensor: {gt_labels_t}')
            #print(len(gt_labels_t))
            
            predicted_labels = [torch.argmax(torch.tensor(pred)).item() for pred in predictions]
            #print(f'Prediction Labels:{predicted_labels}')
            #print(len(predicted_labels))
            
            
            # Compute metrics
            
            tn,fp,fn,tp = confusion_matrix(gt_labels_t, predicted_labels).ravel()
            print(f'TN:{tn} ,  FP:{fp} ,  FN:{fn} ,  TP:{tp}')
            
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = precision_score(gt_labels_t, predicted_labels)
            recall = recall_score(gt_labels_t, predicted_labels)
            f1 = f1_score(gt_labels_t, predicted_labels)
            
            #Store results is dict
            results_dict['fold'].append(fold)
            results_dict['epochs'].append(epochs)
            results_dict['lr'].append(lr)
            results_dict['accuracy'].append(accuracy)
            results_dict['true positives'].append(tp) 
            results_dict['false positives'].append(fp)
            results_dict['true negatives'].append(tn)
            results_dict['false negatives'].append(fn)
            results_dict['sensitivity'].append(sensitivity)
            results_dict['specificity'].append(specificity)
            results_dict['precision'].append(precision)
            results_dict['recall'].append(recall)
            results_dict['f1_score'].append(f1)
            
                  
            
            for key, value in results_dict.items():
                print(f"Length of {key} : {len(value)}")
          
            
            results_df = pd.DataFrame(results_dict)
            
            results_df.to_csv('cnn_model_results.csv', index=False)

# Identify the best hyperparameter combination across all folds
best_index = results_dict['accuracy'].index(max(results_dict['accuracy']))
best_epochs = results_dict['epochs'][best_index]
best_lr = results_dict['lr'][best_index]

print(f"Best Hyperparameter Combination: Epochs = {best_epochs}, Learning Rate = {best_lr}")
