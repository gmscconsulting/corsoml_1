# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# checking PyTorch version installed,
print(torch.__version__, torch.cuda.is_available(), torch.cuda.current_device())


# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print("Number of available GPUs:", num_gpus)

# Set the CUDA device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define custom transformations for data preprocessing
custom_transform = transforms.Compose([
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))    # Normalize images with mean and standard deviation for better convergence
])

# Load training data and test data
train_data = datasets.MNIST(root='data', train=True, download=True, transform=custom_transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=custom_transform)

sample_data, sample_target = train_data[0]
print("The number of samples of the training data:", len(train_data))   # 60_000
print("The number of samples of the testing data:", len(test_data))     # 10_000
print("Shape of sample data:", sample_data.shape)                       # Shape of sample data: torch.Size([1, 28, 28])
print("Number of dimensions of sample data:", sample_data.dim())        # Number of dimensions of sample data: 3


def show_images(dataset, num_images=10):
    # Set up a grid of subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    # Loop through the specified number of images
    for i in range(num_images):
        # Get the data and target label for the current image
        image, label = dataset[i]

        # Display the image on the corresponding subplot
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.show()

# Display up to 10 images from the training dataset
# show_images(train_data, num_images=2)


# the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiation of the CNN model
model = CNN().to(device)
# If multiple GPUs are available, parallelize the model across GPUs
# if num_gpus > 1:
#     print("Parallelize model across multiple GPUs...")
#     model = nn.DataParallel(model)


# batch size for training
batch_size = 64
# learning rate
learning_rate = 1e-3


# Create data loaders for training and test data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)


# Lists to store training and testing metrics
train_losses = []
test_losses = []
test_accuracies = []


# the number of epochs for training
num_epochs = 5

# Training loop
for epoch in range(1, num_epochs+1):
    running_loss = 0.0

    # Set the model in training mode
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_function(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset)}")
    train_losses.append(running_loss/len(train_loader.dataset))

# Set the model in evaluation mode
model.eval()

# Evaluation on test data
correct = 0
total = 0
test_loss = 0
test_len = len(test_loader)

# Dictionary to store precision, recall, and fpr/tpr for each class
class_metrics = {}

with torch.no_grad():
    # Testing loop
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted_proba, predicted = torch.max(outputs, 1)

        # Calculate precision and recall for each class
        for class_idx in range(10):  # 10 classes in MNIST
            class_true = (labels == class_idx).float()
            class_pred = (predicted == class_idx).float()

            if class_idx not in class_metrics:
                class_metrics[class_idx] = {'true': [], 'predicted_proba': [], 'precision': [], 'recall': []}

            # Set zero_division parameter to 1 to avoid UndefinedMetricWarning
            precision = precision_score(class_true.cpu().numpy(), class_pred.cpu().numpy(), zero_division=1)
            recall = recall_score(class_true.cpu().numpy(), class_pred.cpu().numpy(), zero_division=1)

            # Append precision and recall
            class_metrics[class_idx]['precision'].append(precision)
            class_metrics[class_idx]['recall'].append(recall)

            class_metrics[class_idx]['true'].append(class_true.cpu().numpy())
            class_metrics[class_idx]['predicted_proba'].append(predicted_proba.cpu().numpy())

# Calculate ROC curve for each class
for class_idx in class_metrics.keys():
    class_true_all = np.concatenate(class_metrics[class_idx]['true'])
    predicted_proba_all = np.concatenate(class_metrics[class_idx]['predicted_proba'])
    fpr, tpr, _ = roc_curve(class_true_all, predicted_proba_all)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {class_idx}')
    plt.legend(loc="lower right")
    plt.show()


# Calculate average precision and recall for each class
avg_precision_recall = {}
for class_idx, metrics in class_metrics.items():
    precision = np.mean(class_metrics[class_idx]['precision'])
    recall = np.mean(class_metrics[class_idx]['recall'])
    avg_precision_recall[class_idx] = {'precision': precision, 'recall': recall}

# # Print average precision and recall for each class
for class_idx, metrics in avg_precision_recall.items():
    print(f"Class {class_idx}: Precision = {metrics['precision']}, Recall = {metrics['recall']}")




# Convert lists to NumPy arrays for plotting
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
test_accuracies = np.array(test_accuracies)
# If GPU was used. Transfer results to CPU before converting to NumPy arrays
# train_losses = train_losses.cpu().numpy()
# test_losses = test_losses.cpu().numpy()
# test_accuracies = test_accuracies.cpu().numpy()

# Visualization of results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')

# plt.subplot(1, 2, 2)
# plt.plot(test_accuracies, label='Test Accuracy', color='green')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.title('Testing Accuracy')

plt.tight_layout()
plt.show()