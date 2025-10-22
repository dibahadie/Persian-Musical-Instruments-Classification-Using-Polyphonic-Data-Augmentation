from huggingface_hub import login
from datasets import load_dataset, load_from_disk, Audio
import soundfile as sf
import re


import pandas as pd
from datasets import Dataset, Audio

ds_main = load_dataset("dibahadie/PCMIR"))



ds_val = ds_main['VAL']
ds = load_dataset("dibahadie/BPM_DASTGAH")


def remove_parentheses(example):
    example['labels'] = [item.split('-')[0] for item in example['labels'].split()]   
    return example

ds = ds.map(remove_parentheses)
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
Rank = 16
Alpha = 8
class LoRALayer(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        rank: int = Rank,
        alpha: float = Alpha
        ):

        super().__init__()
        self.rank = Rank
        self.alpha = Alpha
        self.scaling = self.alpha / self.rank # scaling factor
        self.in_dim = module.in_features
        self.out_dim = module.out_features
        self.pretrained = module

 
        self.MLL_A = nn.Linear(self.in_dim, self.rank, bias=False)
        nn.init.kaiming_uniform_ (self.MLL_A.weight, mode='fan_in')
        self.MLL_B = nn.Linear(self.rank, self.out_dim, bias=False)
        nn.init.zeros_(self.MLL_B.weight)


    def forward(self, x: torch.Tensor):

        pretrained_out = self.pretrained(x) # get the pretrained weights -> x.W

        lora_out = self.MLL_A(x) # x@A
        lora_out = self.MLL_B(lora_out) # x@A@B
        lora_out = lora_out * self.scaling # Scale by the scaling factor

        return pretrained_out + lora_out # x@W + x@A@B*(scaling_factor)
def mutate_model(model: nn.Module, rank: int = Rank, alpha: float = Alpha):
    """
    Replaces all linear layers in the model with LoRALinear layers.
    Freeze all params except LoRA params.
    """
    # make sure there are no LoRALayer is in the model; return if there are any
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            print("Model already contains LoRALinear layers! \n Try reloading the model.")
            return

    # we want to replace all query and value Linear modules with LoRALayer

    # iterate over the children of the model
    for name, module in model.named_children():
        # if the module is linear and the name is for query or value
        if isinstance(module, nn.Linear) and (name == 'q_proj' or name == 'v_proj'):

            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(model, name, lora_layer)
            print(f"Replaced {name} with LoRALinear layer.")
        else:
            mutate_model(module, rank, alpha) # recursively call the function on the module

def freeze_non_LoRA(model, peft_key):
    print('Non freezed weights:')
    for param_name, weights in model.named_parameters():
        weights.requires_grad = peft_key in param_name
        if weights.requires_grad:
            print(param_name)


processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
class ModelMert(nn.Module,
    PyTorchModelHubMixin):
    def __init__(self, n_class):
        super(ModelMert, self).__init__()

        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #mutate_model(self.model)
        #freeze_non_LoRA(self.model, peft_key='MLL')
        
        self.aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(1024, n_class))  # Remove bias=True

    def forward(self, input_values, attention_mask):
        outputs = self.model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2).permute((1, 0, 2))
        weighted_avg_hidden_states = self.aggregator(time_reduced_hidden_states).squeeze()
        return self.classifier(weighted_avg_hidden_states)

def multi_label_acc(y_pred, y_test):
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to get probabilities
    y_pred_tags = (y_pred > 0.5).float()  # Convert probabilities to binary predictions
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / y_test.numel()  # Average across all labels
    acc = torch.round(acc * 100)
    return acc


from transformers import (
    #AdamW,
    get_linear_schedule_with_warmup, get_wsd_schedule, get_cosine_with_hard_restarts_schedule_with_warmup
)


import os
import torch
from tqdm import tqdm

def train_model(model, train_dataloader, test_dataloader, n_epochs, optimizer, criterion, device, scheduler=None, output_dir="outputs_dastgahb"):
    os.makedirs(output_dir, exist_ok=True)  # ensure output directory exists
    
    train_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_dataloader, unit="batch") as batches:
            for data, target in batches:
                batches.set_description(f"Epoch {epoch + 1}")

                input_values = data['input_values'].to(device)
                attention_mask = data['attention_mask'].to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(input_values, attention_mask)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                acc = multi_label_acc(output, target)
                epoch_loss += loss.item()
                batches.set_postfix(train_loss=loss.item(), train_accuracy=acc.item())
        
        valid_loss = 0
        correct = 0
        total = 0
        
        with tqdm(test_dataloader, unit="batch") as batches:
            model.eval()
            with torch.no_grad():
                for data, target in batches:
                    batches.set_description(f"Epoch {epoch + 1}")

                    input_values = data['input_values'].to(device)
                    attention_mask = data['attention_mask'].to(device)
                    target = target.to(device)

                    output = model(input_values, attention_mask)
                    loss = criterion(output, target)
                    valid_loss += loss.item()

                    y_pred_tags = (torch.sigmoid(output) > 0.5).float()
                    correct += (y_pred_tags == target).sum().item()
                    total += target.numel()
        
        valid_loss /= len(test_dataloader)
        accuracy = correct / total
        train_losses.append(epoch_loss)
        
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss = {valid_loss:.4f}, Validation Accuracy = {accuracy:.4f}")
        
        # Save model checkpoint for this epoch
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
        # Track the best model
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model_state = model.state_dict()
    
    print("Training complete. Loading best model...")
    model.load_state_dict(best_model_state)
    return model, train_losses




import random
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


S = 42
set_seed(S)


#ds = ds['train'].train_test_split(test_size=0.2, seed=42)


ds = ds.cast_column('audio', Audio(sampling_rate=24000))
class InstrumentsMERT(Dataset):
    def __init__(self, ds):
        super(InstrumentsMERT, self).__init__()
        self.df = ds
        self.label_map = {'kamancheh': 0, 'ney': 1, 'santur': 2, 'sitar': 3, 'setar': 3, 'tar': 4, 'tonbak': 5, 'daaf': 6, 'avaz': 7, 'piano': 8, 'violin': 9}
        self.num_classes = len(self.label_map) - 1
        self.max_length = 5 * 24000  # Example fixed length for padding/truncation

    def __getitem__(self, index):
        
        labels = self.df[index]['labels'] # This should be a list of labels
        if type(labels) == str:
            labels = labels.split()
        array = self.df[index]['audio']['array']

        # Process audio with padding/truncation
        INPUTS = processor(array, sampling_rate=24000, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = INPUTS['input_values'].squeeze()
        attention_mask = INPUTS['attention_mask'].squeeze()

        # Create multi-label target vector
        label_vector = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.label_map:
                label_vector[self.label_map[label]] = 1.0

        return {'input_values': input_ids, 'attention_mask': attention_mask}, label_vector

    def __len__(self):
        return len(self.df)  # Return the number of samples


Mert_train_dataloader = DataLoader(InstrumentsMERT(ds['train']), batch_size=16, shuffle=True, drop_last=True)
Mert_test_dataloader = DataLoader(InstrumentsMERT(ds_val), batch_size=12, shuffle=False, drop_last=False)



n_classes =10
EPOCHS = 10
#LR = 1e-3
deviceMert = 'cuda' if torch.cuda.is_available() else 'cpu'
temp_modelMert = ModelMert(n_class=n_classes).to(deviceMert)
pos_w = torch.full((n_classes,), 0.8, device=deviceMert)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = torch.optim.AdamW(temp_modelMert.parameters(), lr=1e-4)

trained_modelMert, train_lossesmert = train_model(temp_modelMert, Mert_train_dataloader, Mert_test_dataloader, EPOCHS, optimizer, criterion, deviceMert)
