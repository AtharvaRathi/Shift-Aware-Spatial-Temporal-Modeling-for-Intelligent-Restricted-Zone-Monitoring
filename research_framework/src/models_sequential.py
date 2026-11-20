import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from .dataset import SEQUENCE_LENGTH

class SequenceNet(nn.Module):
    def __init__(self, rnn_type='lstm', input_size=7, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3):
        super(SequenceNet, self).__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :] # Take the output of the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out

class SequentialModelPipeline:
    def __init__(self, model_type='lstm', models_dir='research_framework/models/sequential'):
        self.model_type = model_type.lower()
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, f"{self.model_type}_best.pt")
        
        # Apple Silicon support (MPS), else CUDA, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.model = SequenceNet(rnn_type=self.model_type).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
    def train(self, train_dataset, val_dataset, epochs=20, batch_size=128, patience=5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_f1 = 0.0
        patience_counter = 0
        
        print(f"Training {self.model_type.upper()} on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            val_loss, val_f1, _ = self.evaluate(val_loader)
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Early Stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Load best model for future predictions
        self.load()
        
    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                
        val_loss /= len(dataloader)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return val_loss, val_f1, all_preds
        
    def predict_proba(self, dataloader):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
        return all_probs
        
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        
    def load(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def get_feature_importance(self, dataloader, feature_names=None):
        if feature_names is None:
            feature_names = ['speed_variance', 'distance_to_restricted_zone', 'time_near_boundary', 
                             'time_inside_zone', 'sudden_stops', 'carrying_baggage', 'time_of_day_multiplier']
        
        _, baseline_f1, _ = self.evaluate(dataloader)
        importances = []
        
        for i, feature in enumerate(feature_names):
            all_preds = []
            all_labels = []
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_shuffled = X_batch.clone().to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Permute the feature column across the batch
                    indices = torch.randperm(X_shuffled.size(0))
                    X_shuffled[:, :, i] = X_shuffled[indices, :, i]
                    
                    outputs = self.model(X_shuffled)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    
            shuffled_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            drop = baseline_f1 - shuffled_f1
            importances.append((feature, max(0, drop)))
            
        importances_array = [x[1] for x in importances]
        total = sum(importances_array)
        if total > 0:
            importances_array = [x / total for x in importances_array]
        return importances_array
