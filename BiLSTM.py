"""
@author ys3276
"""
import torch 
import torch.nn as nn
from torch.utils import data
import pickle
import numpy as np
import time
import kaldi_io
from arrangeShape import arrangeShape
from accuracy import getAccuracy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters
sequence_length = 30
input_size = 89
hidden_size = 128
num_layers = 2
num_classes = 89
batch_size = 100
num_epochs = 1
learning_rate = 0.03


class Dataset(data.Dataset):
    def __init__(self, features, notes):
        self.features = features
        self.notes = notes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.notes[idx]
#Load features from Kaldi ark file
feats = np.array([], dtype=np.float64).reshape(0,16)
for key,mat in kaldi_io.read_mat_ark("/home/yogesh/Downloads/kaldi-trunk/egs/mycorpus/mfcc/raw_mfcc_pitch_train.1.ark"):
    feats = np.vstack([feats, mat])

target = pickle.load(open('data/target_train.pkl', 'rb'), encoding="latin1")
#target = pickle.load(open('kaldi_target.pkl', 'rb'))#encoding="latin1"

#X_train = []
#for set_inputs_i in range(len(feats) - sequence_length):
#    set_feats = feats[set_inputs_i:set_inputs_i+sequence_length, :]
#    X_train.append(set_feats)
#X_train = np.array(X_train, dtype=np.float32)
#Y_train = target[sequence_length/2:len(feats)-sequence_length/2, :]
#Y_train = np.multiply(Y_train, 1)
X_train, Y_train = arrangeShape(feats, target, 11)
X_train = np.multiply(X_train, 1)
Y_train = np.multiply(Y_train, 1)
train_dataset = Dataset(X_train, Y_train)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=batch_size, 
#                                          shuffle=False)

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.sigmoid(self.fc(out[:, -1, :]))
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    start = time.time()
    for i, (features, notes) in enumerate(train_loader):
        features = features.to(device)
        notes = notes.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, notes.float())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            t = (time.time() - start) / 60
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f} min' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), t))
            start = time.time()

## Test the model
#with torch.no_grad():
#    correct = 0
#    total = 0
#    for images, labels in test_loader:
#        images = images.reshape(-1, sequence_length, input_size).to(device)
#        labels = labels.to(device)
#        outputs = model(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 
#

train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1000, 
                                           shuffle=False)
Y_train_pred = np.array([], dtype=np.float32).reshape(0,89)
with torch.no_grad():
    for features, notes in train_loader1:
        features = features.to(device)
        outputs = model(features)
        Y_train_pred = np.vstack([Y_train_pred, outputs.cpu().numpy()])

            
pickle.dump(Y_train_pred , open('bi2.pkl', 'wb'), protocol = 2)
print(getAccuracy(Y_train_pred, Y_train, 0.2))
print(getAccuracy(Y_train_pred, Y_train, 0.3))
print(getAccuracy(Y_train_pred, Y_train, 0.4))
print(getAccuracy(Y_train_pred, Y_train, 0.5))
print(getAccuracy(Y_train_pred, Y_train, 0.6))
#Y_act = np.array([], dtype=np.int64).reshape(0,89)
#for features, notes in train_loader:
#    notes = notes.to(device)
#    Y_act = np.vstack([Y_act, notes.numpy()])
#    
#Y_pred = np.array([], dtype=np.int64).reshape(0,89)
#for outputs in Y_train_pred:
#    Y_pred = np.vstack([Y_pred, outputs])

# Save the model checkpoint
torch.save(model, 'biLSTM.mdl')
