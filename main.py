import torch
from torch import nn

import matplotlib.pyplot as plt
import random



"""
A simple PyTorch library with 10 functions made to make the PyTorch workflow easier
"""

# --- Generates Random Samples --- #

def generate_samples(dataset, n_samples: int) -> list:
    """
    Generates a list of samples from the given dataset randomly
    
    Args: 
    dataset (Any): dataset where samples will be extracted (must be unpackable)
    n_samples (int): amount of samples to be extracted from dataset
  
    Returns:
    A list consisting of random samples extracted from the dataset with a length of n_samples along with their corresponding label (can be unpacked)
    """
    
    samples = []
  
    for sample, label in random.sample(list(dataset), k=n_samples):
      samples.append((sample, label))
  
    return samples

# --- Function For Unsqueezing a Dataset --- #

def unsqueeze_dataset(dataset, dim) -> list:
    
    """
    Unsqueezes all samples on a given dataset on a given dimension (you should never be overwriting the original dataset)

    Args:
    dataset (Any): dataset being unsqueezed (only samples will be unsqeezed and must be unpackable)
    dim (int): the target dimension of a certain dataset sample being unsqueezed

    Returns:
    An unsqueezed dataset
    """
    
    unsqueezed_dataset = []
    
    for sample, label in dataset:
        unsqueezed_sample = torch.unsqueeze(sample, dim=dim)

        unsqueezed_dataset.append((unsqueezed_sample, label))

    return unsqueezed_dataset

# --- Function For Permuting A Dataset --- #

def permute_dataset(dataset, *argv) -> list:
    """
    Permutes all samples on a given dataset given dimension indexes (you should never be overwriting the original dataset)

    Args:
    dataset (Any): the dataset being permuted (only samples will be permuted)
    *argv (ints): the dimension indexes that the samples will be permuted on

    Returns:
    A permuted dataset
    """
    permuted_dataset = []

    for sample, label in dataset:
        permuted_dataset.append((sample.permute(argv), label))
    
    return permuted_dataset

# --- Function For Plotting Images --- #

def plot_images(dataset, rows: int, columns: int, figsize: tuple, classes=None, cmap=None, title=True, fontsize=10) -> None:
    """
    Plots a certain amount of images from the given dataset
  
    Args:
    dataset (Any): dataset in which images are stored (must be unpackable, if length does not equal rows * columns, the first rows * columns images of the dataset will be plotted)
    rows (int): how many rows of images the figure will consist of 
    columns (int): how many columns of images the figure will consist of
    figsize (tuple): what the resulting figure size of the figure will be
    classes (dict): what the function will use when plotting image labels
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
            plt.title(classes[label], fontsize=fontsize)

        plt.imshow(image.squeeze(), cmap=cmap)  
        plt.axis(False)
        
    plt.show();

# --- Plotting Images And Their Predictions --- #

def plot_image_predictions(predictions, dataset, rows: int, columns: int, figsize: tuple, classes: dict, fontsize=10, cmap=None) -> None:
    """ Plots the image being predicted on and the models prediction on it 
  
    Args:
    predictions (Any): the model's predictions on the dataset (must be in probability form)
    dataset (Any): The dataset the model predicted on
    rows (int): how many rows of images the figure will consist of 
    columns (int): how many columns of images the figure will consist of 
    figsize (tuple): the resulting figure size of the plot
    classes (dict): the classes of the dataset
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
  
        title = f"Predicted: {classes[prediction.argmax()]}: {prediction.max(): .2f} | Truth: {classes[label]}"
        color = "green" if classes[prediction.argmax()] == classes[label] else "red"
  
        plt.title(title, fontsize=fontsize, c=color)
        plt.imshow(image.squeeze(), cmap=cmap)
        
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


# --- Accuracy Evaluation Function --- #

def eval_accuracy(model: torch.nn.Module, dataset, device: str) -> float:
    """
    Evaluates a model's accuracy on a given dataset

    Args: 
    model (torch.nn.Module): the model being evaluated
    dataset (Any): an unpackable iterable (ex. [(sample, label), (sample, label)]) that the model will be evaluated on
    device (str): the target device for the model

    Returns:
    accuracy (float): the accuracy of the model on the dataset
    """

    correct  = 0
    
    with torch.inference_mode():
        model.eval()

        for sample, label in dataset:
            y_hat = model(sample.to(device)) 

            if y_hat.argmax() == label:
                correct += 1

    return correct / len(dataset)

# --- Plots Linear Regression Predictions --- #

def plot_linear_predictions(X: torch.Tensor, y: torch.Tensor, predictions=None, colors=['r','g'], figsize=(10, 7)) -> None:
    """
    Plots the linear predictions of a certain model
  
    Args:
    X (torch.Tensor): the data inputed through the model
    y (torch.Tensor): the labels of the data given
    predictions (torch.Tensor, optional): The predictions of the model
    colors (list, optional): The colors of each of scatter plots (defaults to [r, g])
    figsize (tuple, optional): The figure size of the returned plot
  
    Returns:
    A Matplotlib scatter plot consisting of the data, labels and the models predictions
    """
    plt.figure(figsize=figsize)
  
    plt.scatter(X, y, c=colors[0], s=4, label="Data")
    
    if predictions is not None:
        plt.scatter(X, predictions, c=colors[1], s=4, label="Predictions")
  
    plt.legend(prop={"size": 14})

# --- Function For Making Predictions --- #

def make_predictions(model: torch.nn.Module, dataset, device: str):
    """
    Makes predictions using the given model on a given dataset
  
    Args: 
    model (torch.nn.Module): the model predicting on the given dataset
    dataset (Any): the dataset in which the model will be predicting on
    device (str): specifies what device the data should be on
  
    Returns:
    A tensor of predictions from the model (in raw logit form)
    """
    model.eval()
  
    with torch.inference_mode():
      predictions = []
      
      for X, y in dataset:
        X = X.to(device)
        
        prediction = model(X)
  
        predictions.append(prediction)

    return torch.cat(predictions)


# --- Train Function --- #

def train(epochs: int, 
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
