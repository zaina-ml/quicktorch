import torch
from torch import nn

import matplotlib.pyplot as plt
import random

import torchmetrics

from torchmetrics import ConfusionMatrix
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC    


import mlxtend
from mlxtend.plotting import plot_confusion_matrix


"""
A simple PyTorch library with functions made to make the PyTorch workflow easier
"""

# --- Performs Tests Such as Accuracy, Precision, Recall, F1Core and AUROC --- #

def run_diagnostic(model: torch.nn.Module, dataset: torch.utils.data.Dataset, task: str, device: str, BATCH_SIZE=32) -> tuple:
    """
    Performs Tests Such as Accuracy, Precision, Recall, F1Score and AUROC Over a Given Dataset
    
    Args: 
    model (torch.nn.Module): PyTorch Model to be tested
    dataset (torch.utils.data.Dataset): PyTorch dataset where samples will be extracted
    task (str): either "multiclass" or "binary" (parameter used by torchmetrics)
    device (str): current device that model is on
    BATCH_SIZE (int): batch sizes of dataset when fitted to model
  
    Returns:
    Tuple containing Accuracy, Precision, Recall, F1Scorea and AUROC
    """
    if task != "multiclass" and task != "binary":
        raise Exception("Task parameter must either be (multiclass, binary)")

    accuracy = Accuracy(task=task, num_classes=len(dataset.classes)).to(device)
    precision = Precision(task=task, num_classes=len(dataset.classes)).to(device)
    recall = Recall(task=task, num_classes=len(dataset.classes)).to(device)
    f1_score = F1Score(task=task, num_classes=len(dataset.classes)).to(device)
    auroc = AUROC(task=task, num_classes=len(dataset.classes)).to(device)

    model.eval()
  
    with torch.inference_mode():  
      for X, y in torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE):
        X, y = X.to(device), y.to(device)

        y_hat = model(X)
        y_labels = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)

        accuracy.update(y_labels, y)
        precision.update(y_labels, y)
        recall.update(y_labels, y)
        f1_score.update(y_labels, y)
        auroc.update(y_hat, y)
        
    print(f"Accuracy:  {accuracy.compute().item():.4f}")
    print(f"Precision: {precision.compute().item():.4f}")
    print(f"Recall:    {recall.compute().item():.4f}")
    print(f"F1 Score:  {f1_score.compute().item():.4f}")
    print(f"AUROC:     {auroc.compute().item():.4f}")

    return accuracy.compute().item(), precision.compute().item(), recall.compute().item(), f1_score.compute().item(), auroc.compute().item()

# --- Plots Confusion Matrix --- #

def confusion_matrix(model: torch.nn.Module, dataset: torch.utils.data.Dataset, task: str, device: str):
    """
    Plots Confusion Matrix on a Model with a given Dataset
    
    Args: 
    model (torch.nn.Module): PyTorch Model to be tested
    dataset (torch.utils.data.Dataset): PyTorch dataset where samples will be extracted
    task (str): either "multiclass" or "binary" (parameter used by torchmetrics)
    device (str): current device that model is on
  
    Returns:
    Confusion Matrix in Matplotlib
    """

    if task != "multiclass" and task != "binary":
        raise Exception("Task parameter must either be (multiclass, binary)")

    
    y_true = dataset.targets
    y_hat = make_predictions(model=model, dataset=dataset, device=device)
    y_probs = torch.softmax(y_hat, dim=1)

    confusion_matrix = ConfusionMatrix(task=task, num_classes=len(dataset.classes)).to(device)
    confusion_matrix_tensor = confusion_matrix(preds=y_probs,
                                               target=torch.tensor(dataset.targets, device=device))

    figure, axis = plot_confusion_matrix(
        conf_mat=confusion_matrix_tensor.cpu().numpy(),
        class_names=dataset.classes,
        figsize=(10, 7)
    )

# --- Converts Logits to Probabilities --- #

def convert_to_probabilities(predictions: torch.Tensor, task: str):
    """
    Converts Given Logits to Probabilities

    Args:
    predictions(torch.Tensor): logits given to be converted to prediction probabilities
    task (str): either multiclass or binary (used to determine usage of sigmond or softmax)

    Returns:
    A Torch Tensor Of Probabilities

    """
    if task != "multiclass" and task != "binary":
        raise Exception("Task parameter must either be (multiclass, binary)")

    if task == "multiclass":
        return torch.softmax(predictions, dim=1)
    elif task == "binary":
        return torch.sigmoid(predictions, dim=1)


# --- Generates Random Samples --- #

def generate_samples(dataset: torch.utils.data.Dataset, n_samples: int) -> torch.utils.data.Dataset:
    """
    Generates a list of samples from the given dataset randomly
    
    Args: 
    dataset (torch.utils.data.Dataset): PyTorch dataset where samples will be extracted
    n_samples (int): amount of samples to be extracted from dataset
  
    Returns:
    A new PyTorch dataset of the random samples
    """
    
    random_samples = random.sample(list(dataset), k=n_samples)

    class RandomSampleDataset(torch.utils.data.Dataset):
        def __init__(self, random_samples=random_samples, classes=dataset.classes, class_to_idx=dataset.class_to_idx):
            self.random_samples = random_samples

            self.classes, self.class_to_idx = classes, class_to_idx
            self.targets = self.get_targets()
        
        def get_targets(self):
            targets = []

            for X, y in random_samples:
                targets.append(y)

            return torch.tensor(targets)
        
        def __getitem__(self, idx: int):
            return (random_samples[idx][0], random_samples[idx][1])

        def __len__(self):
            return len(self.random_samples)
    
    random_sample_dataset = RandomSampleDataset()

    return random_sample_dataset


# --- Plots A Set Of Images --- #

def plot_images(dataset: torch.utils.data.Dataset, rows: int, columns: int, figsize: tuple, cmap=None, title=True, fontsize=10) -> None:
    """
    Plots a certain amount of images from the given dataset
  
    Args:
    dataset (torch.utils.data.Dataset): the PyTorch dataset in which images are stored (must be unpackable, if length does not equal rows * columns, the first rows * columns images of the dataset will be plotted)
    rows (int): how many rows of images the figure will consist of 
    columns (int): how many columns of images the figure will consist of
    figsize (tuple): what the resulting figure size of the figure will be
    cmap (str, optional): what the color map of the plotted images will be
    title (bool, optional): whether or not a title will be displayed for each image (defaults to True)
    fontsize (int, optional): font size of the title (defaults to 10)
  
    Returns:
    A matplotlib plot consisting of rows * columns images
    """
    
    plt.figure(figsize=figsize)
  
    for i in range(rows * columns):
        image, label = dataset[i]
      
        plt.subplot(rows, columns, i+1)
  
        if title:
            plt.title(dataset.classes[label], fontsize=fontsize)

        plt.imshow(image.squeeze(), cmap=cmap)  
        plt.axis(False)
        
    plt.show();

# --- Plotting Images And Their Predictions --- #

def plot_image_predictions(predictions: torch.Tensor, dataset: torch.utils.data.Dataset, rows: int, columns: int, figsize: tuple, fontsize=10, cmap=None) -> None:
    """ 
    Plots the image being predicted on and the models prediction on it 
  
    Args:
    predictions (torch.Tensor): the model's predictions on the dataset (must be in probability form)
    dataset (torch.utils.data.Dataset): The PyTorch dataset the model predicted on
    rows (int): how many rows of images the figure will consist of 
    columns (int): how many columns of images the figure will consist of 
    figsize (tuple): the resulting figure size of the plot
    fontsize (int, optional): the fontsize of the resulting plot (default is 10)
    cmap (str, optional): the colormap of the resulting images
  
    Returns:
    A matplotlib plot of all images and their respective predictions
    """
    
    plt.figure(figsize=figsize)
    
    for i in range(rows * columns):
        plt.subplot(rows, columns, i+1)
  
        image, label = dataset[i]
        prediction = predictions[i]
  
        title = f"Predicted: {dataset.classes[prediction.argmax()]}: {prediction.max(): .2f} | Truth: {dataset.classes[label]}"
        color = "green" if dataset.classes[prediction.argmax()] == dataset.classes[label] else "red"
  
        plt.title(title, fontsize=fontsize, c=color)

        if len(torch.squeeze(image).shape) == 2:
            plt.imshow(torch.squeeze(image), cmap=cmap)
        elif len(torch.squeeze(image).shape) == 3:
            plt.imshow(torch.squeeze(image).permute(1, 2, 0))
        
        plt.axis(False)
    plt.show();
  
# --- Plotting Results Function --- #

def plot_results(results: dict, figsize=(15, 7)) -> None:
    """
    Plots a loss curve and an accuracy curve given a results dictionary in the format below

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"epoch": [...],
            "train_loss": [...],
            "test_loss": [...],
            "train_acc": [...],
            "test_acc:" [...]}
        figsize (tuple, optional): what the resulting figure size of the plot will be (defaults to (15, 7))

    Returns:
    A matplotlib plot of the loss curves
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)

    plt.title("Loss")
    plt.plot(results["epoch"], results["train_loss"], label="Train Loss")
    plt.plot(results["epoch"], results["test_loss"], label="Test Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.title("Accuracy")
    plt.plot(results["epoch"], results["train_acc"], label="Train Accuracy")
    plt.plot(results["epoch"], results["test_acc"], label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show();


# --- Function For Making Predictions --- #

def make_predictions(model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: str, BATCH_SIZE=32) -> torch.Tensor:
    """
    Makes predictions using the given model on a given dataset
  
    Args: 
    model (torch.nn.Module): the model predicting on the given dataset
    dataset (torch.utils.data.Dataset): the dataset in which the model will be predicting on 
    device (str): specifies what device the data should be on
    BATCH_SIZE (int, optional): specifies the batch size of data when predicted on (def. 32)
  
    Returns:
    A tensor of predictions from the model (in raw logit form)
    """
    model.eval()
  
    with torch.inference_mode():
      predictions = torch.tensor([], device=device)
      
      for X, y in torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE):
        X = X.to(device)
        
        prediction = model(X)

        predictions = torch.cat((predictions, prediction), dim=0)

    return predictions


# --- Train Function --- #

def train_model(epochs: int, 
                model: torch.nn.Module, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                train_dataloader: torch.utils.data.DataLoader, 
                test_dataloader: torch.utils.data.DataLoader, 
                device:str) -> dict:
    """
    Performs a training &  test loop over an iterable torch dataloader with the given model

    Args:
    epochs (int): how many iterations the model will be trained for
    model (torch.nn.Module): the model being trained
    loss_fn (torch.nn.Module): the loss function being used for training
    optimizer (torch.optim.Optimizer): the optimizer for training the model
    train_dataloader (torch.utils.data.DataLoader): the dataloader used when training the model for the training loop
    test_dataloader (torch.utils.data.DataLoader): dataloader being used for the training loop
    device (str): the desired device for the data that the model will be trained on (must be the same as the device of the model)

    Returns:

    A results dictionary containing the train and testing losses of the model
        eg. {"epoch": [...],
            "train_loss": [...],
            "test_loss": [...],
            "train_acc": [...],
            "test_acc:" [...]}
    """
    
    results = {"epoch": [],
               "train_loss": [],
               "test_loss": [],
               "train_acc": [],
               "test_acc": []}
    
    for epoch in range(epochs):
        results["epoch"].append(epoch)
        
        print(f"\nEpoch: {epoch} \n----------")

        train_loss, test_loss = 0, 0
        train_acc, test_acc = 0,0 

        for batch, (X, y) in enumerate(train_dataloader):
            model.train()

            X, y = X.to(device), y.to(device)

            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()
 
            train_acc += ((torch.eq(y, y_hat.argmax(dim=1)).sum().item()) / len(y)) * 100
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        
        print(f"Train Loss: {train_loss: .5f} | Accuracy: {train_acc :.3f}")
        
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)

                model.eval()

                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                
                test_acc += ((torch.eq(y, y_hat.argmax(dim=1)).sum().item()) / len(y)) * 100

                test_loss += loss.item()

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            print(f"Test Loss: {test_loss: .5f} | Accuracy: {test_acc :.3f}")

    return results
