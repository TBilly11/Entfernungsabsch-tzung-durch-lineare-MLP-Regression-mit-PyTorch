import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
import time


# Datei Pfad
file_path = r'C:\Users\1234\Desktop\Pytorch\data.txt'
file_path_0 = r'C:\Users\1234\Desktop\Pytorch\Neu_Daten'

# Funktion zum Lesen und Filtern der Daten aus der Datei mit 5 Spalten
def read_and_filter_data_5(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip().startswith('0'):
                parts = line.strip().split()
                width = float(parts[1])
                height = float(parts[2])
                zoom = float(parts[3])
                distance = float(parts[4])
                data.append([width, height, zoom, distance])
    return data

# Funktion zum Lesen und Filtern der Daten aus den Dateien mit 6 Spalten
def read_and_filter_data_6(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip().startswith('0'):
                parts = line.strip().split()
                if len(parts) == 6:
                    width = float(parts[1])
                    height = float(parts[2])
                    zoom = float(parts[3])
                    focus=float(parts[4])
                    distance = float(parts[5])  # 5. Spalte (Fokus) wird übersprungen
                    data.append([width, height, zoom, focus, distance])
    return data

# Daten aus allen Dateien im Verzeichnis sammeln
#all_data = read_and_filter_data_5(file_path)
all_data=[]
for root, dirs, files in os.walk(file_path_0):
    for file in files:
        if file.endswith('.txt'):
            all_data.extend(read_and_filter_data_6(os.path.join(root, file)))

# In NumPy-Array umwandeln
data_array = np.array(all_data)
X = data_array[:, :-1]  # Weite, Höhe, Zoom,focus
y = data_array[:, -1]   # Distanz

# Fester Seed für Reproduzierbarkeit
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Daten in Trainings-, Validierungs- und Testdaten aufteilen
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, shuffle=False, random_state=seed)

# In PyTorch-Tensoren umwandeln
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Modell definieren
'''model = nn.Sequential(
    nn.Linear(4, 10),  # Eingänge: Weite, Höhe, Zoom
    nn.ReLU(),         # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(10, 5),  # Transformiert in einen 5-dimensionalen Vektor
    nn.ReLU(),         # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(5, 1)    # Ausgang: Distanz
)'''
model = nn.Sequential(
    nn.Linear(4, 20),    # Erhöhung der Anzahl der Neuronen in der ersten Schicht
    nn.ReLU(),
    #nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(20, 15),   # Neue Schicht hinzugefügt
    nn.ReLU(),
    #nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(15, 10),   # Neue Schicht hinzugefügt
    nn.ReLU(),
   # nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(10, 5),    # Reduzierung der Dimensionen schrittweise
    nn.ReLU(),
    #nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(5, 1)
)

'''model = nn.Sequential(
    nn.Linear(4, 30),    # Erste Schicht: Eingang mit 3 Merkmalen zu 64 Neuronen
    nn.ReLU(),           # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(30, 25),  # Zweite Schicht: 64 Neuronen zu 128 Neuronen
    nn.ReLU(),           # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(25, 20), # Dritte Schicht: 128 Neuronen zu 256 Neuronen
    nn.ReLU(),           # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(20, 15), # Vierte Schicht: 256 Neuronen zu 128 Neuronen
    nn.ReLU(),           # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(15, 10),  # Fünfte Schicht: 128 Neuronen zu 64 Neuronen
    nn.ReLU(),           # Aktivierungsfunktion
    nn.Dropout(0.2),   # Dropout zwischen den Schichten
    nn.Linear(10, 5),     # Ausgangsschicht: 64 Neuronen zu 1 Neuron (Ausgabe)
    nn.ReLU(),
    nn.Linear(5, 1)
)'''

# Verlustfunktion und Optimierer definieren
loss_fn = nn.MSELoss()    # Verlustfunktion
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimierer

# Trainingsschleife
n_epochs = 100
batch_size = 20
best_mse = np.inf
best_weights = None
train_history = []
val_history = []
test_history = []

# Startzeit für die Schätzung der gesamten Arbeitszeit
start_time = time.time()

for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])  # Zufällige Permutation der Indizes
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        # Validation Loss
        val_outputs = model(X_val)
        val_loss = loss_fn(val_outputs, y_val)
        val_mse = val_loss.item()
        val_history.append(val_mse)
        
        # Train Loss
        train_outputs = model(X_train)
        train_loss = loss_fn(train_outputs, y_train)
        train_mse = train_loss.item()
        train_history.append(train_mse)

        # Test Loss
        test_outputs = model(X_test)
        test_loss = loss_fn(test_outputs, y_test)
        test_mse = test_loss.item()
        test_history.append(test_mse)
        
        if val_mse < best_mse:
            best_mse = val_mse
            best_weights = copy.deepcopy(model.state_dict())
    
    # Endezeit für die gesamte Arbeitszeit
    end_time = time.time()
    total_time = end_time - start_time

    
    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_mse}, Validation Loss: {val_mse}, Test Loss: {test_mse}')
print(f'Geschätzte Gesamtzeit: {total_time:.2f} Sekunden')

# Beste Modellgewichte wiederherstellen
model.load_state_dict(best_weights)

# Verlustverlauf plotten
plt.plot(train_history, label='Training Loss')
plt.plot(val_history, label='Validation Loss')
plt.plot(test_history, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Model Loss over Epochs')
plt.legend()
plt.show()

# Modell evaluieren und Ergebnisse plotten
model.eval()
predicted_distances = []
actual_distances = []

with torch.no_grad():
    for i in range(len(X_test)):
        X_sample = X_test[i:i+1]
        y_pred = model(X_sample).item()
        actual_distance = y_test[i].item()
        
        predicted_distances.append(y_pred)
        actual_distances.append(actual_distance)

# Plotten der tatsächlichen gegenüber den vorhergesagten Entfernungen
plt.figure(figsize=(8, 6))
plt.scatter(actual_distances, predicted_distances, color='blue', label='Actual vs Predicted')
plt.plot([min(actual_distances), max(actual_distances)], [min(actual_distances), max(actual_distances)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Distance')
plt.ylabel('Predicted Distance')
plt.title('Actual vs Predicted Distance')
plt.legend()
plt.grid(True)
plt.show()

