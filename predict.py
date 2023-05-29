import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, num_classes, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,
                      out_features=num_classes, bias=False)
            # nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.model(x)


def plot_losses(train_losses, train_metrics, val_losses, val_metrics):
    '''
    Plot losses and metrics while training
      - train_losses: sequence of train losses
      - train_metrics: sequence of train MSE values
      - val_losses: sequence of validation losses
      - val_metrics: sequence of validation MSE values
    '''
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[1].plot(range(1, len(train_metrics) + 1), train_metrics, label='train')
    axs[1].plot(range(1, len(val_metrics) + 1), val_metrics, label='val')

    # if max(train_losses) / min(train_losses) > 10:
    #    axs[0].set_yscale('log')
    #
    # if max(train_metrics) / min(train_metrics) > 10:
    #    axs[0].set_yscale('log')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    axs[0].set_ylabel('loss')
    axs[1].set_ylabel('MSE')
    plt.show()


import torch
from tqdm.auto import tqdm


def train_and_validate(device, model, optimizer, criterion, metric, train_loader, val_loader,
                       num_epochs, verbose=True):
    '''
    Train and validate neural network
      - model: neural network to train
      - optimizer: optimizer chained to a model
      - criterion: loss function class
      - metric: function to measure MSE taking neural networks predictions
                 and ground truth labels
      - train_loader: DataLoader with train set
      - val_loader: DataLoader with validation set
      - num_epochs: number of epochs to train
      - verbose: whether to plot metrics during training
      - device: device to use for training and inference, e.g. 'cpu', 'cuda'
    Returns:
      - train_mse: training MSE over the last epoch
      - val_mse: validation MSE after the last epoch
    '''

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_metric = 0, 0
        pbar = tqdm(train_loader, desc=f'Training {epoch}/{num_epochs}') \
            if verbose else train_loader

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch.reshape(
                1, -1)[0].long().to(device))

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric_value = metric(predictions.to(device), y_batch)
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]

            if verbose and i % 100 == 0:
                pbar.set_postfix({'loss': loss.item(), 'MSE': metric_value})

        train_losses += [running_loss / len(train_loader.dataset)]
        train_metrics += [running_metric / len(train_loader.dataset)]

        model.eval()
        running_loss, running_metric = 0, 0
        pbar = tqdm(val_loader, desc=f'Validating {epoch}/{num_epochs}') \
            if verbose else val_loader

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.no_grad():
                predictions = model(torch.flatten(X_batch, start_dim=1))
                loss = criterion(predictions, y_batch.reshape(
                    1, -1)[0].long().to(device))

                metric_value = metric(predictions.to(device), y_batch)
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]

            if verbose and i % 100 == 0:
                pbar.set_postfix({'loss': loss.item(), 'MSE': metric_value})

        val_losses += [running_loss / len(val_loader.dataset)]
        val_metrics += [running_metric / len(val_loader.dataset)]

        if verbose:
            plot_losses(train_losses, train_metrics, val_losses, val_metrics)

    if verbose:
        print(f'Validation MSE: {val_metrics[-1]:.3f}')

    return train_metrics[-1], val_metrics[-1]


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_loader = DataLoader(list(zip(
    X_train, y_train / y_train.max() * (num_classes - 1))), batch_size=16, shuffle=True)
val_loader = DataLoader(list(zip(
    X_test, y_test / y_test.max() * (num_classes - 1))), batch_size=16, shuffle=True)

IN_SIZE = train_loader.dataset[0][0].shape[0]
NUM_EPOCHS = 10
num_classes = np.unique(y).shape[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


my_model = MLP(in_features=IN_SIZE, num_classes=num_classes,
               hidden_size=128).to(device)

#optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
metric = nn.MSELoss()

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
