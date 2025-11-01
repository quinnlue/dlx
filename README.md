# DLX - Deep Learning X

A deep learning framework with custom tensor operations, neural network modules, and transformer implementations.

## Features

- **Custom Tensor Operations**: Implemented from scratch with automatic differentiation
- **Neural Network Modules**: Linear layers, Layer Normalization, and more
- **Transformer Implementation**: Complete transformer architecture with attention mechanisms
- **Optimizers**: AdamW, SGD, and other optimization algorithms
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy with logits
- **GPU Support**: Optional CUDA support via CuPy

## Installation

### From Source

```bash
git clone https://github.com/quinnlue/dlx.git
cd dlx
pip install -e .
```

## Quick Start

```python
from dlx import Module, Tensor
from dlx.nn.losses import BinaryCrossEntropyWithLogits
from dlx.nn.optim import AdamW
from dlx.utils.backend import xp
import pandas as pd


def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    cols = ["variance", "skewness", "curtosis", "entropy", "target"]
    df = pd.read_csv(url, header=None, names=cols)
    df.sample(frac=1).reset_index(drop=True)

    X = df[["variance", "skewness", "curtosis", "entropy"]].to_numpy(dtype=xp.float32)
    y = df["target"].to_numpy(dtype=xp.int64).reshape((-1,1))
    return Tensor(X[:128]), Tensor(y[:128]), Tensor(X[128:]), Tensor(y[128:])

X_test, y_test, X_train, y_train = load_data()

class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self.linear(4, 28, name="fc1")
        self.fc2 = self.linear(28, 2, name="fc2")

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x) # inherited from Module
        x = self.fc2(x)
        return x
    
    def train(self, x, y, optimizer, num_epochs=250):
        for epoch in range(num_epochs):
            y_hat = self.forward(x)
            loss = BinaryCrossEntropyWithLogits(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.data}")
    
    def eval(self, x, y, loss_fn):
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)
        print(f"Loss: {loss.data}")
        return loss.data
        
if __name__ == "__main__":
    net = Net()
    print(net)
    print(X_train.shape, y_train.shape)
    optimizer = AdamW(net.parameters(), lr=0.001)
    net.train(X_train, y_train, optimizer)
    _ = net.eval(X_test, y_test, BinaryCrossEntropyWithLogits)
```

## Package Structure

```
dlx/
├── nn/                 # Core neural network components
│   ├── tensor.py      # Custom tensor implementation
│   ├── module.py      # Base module class
│   ├── losses.py      # Loss functions
│   └── optim.py       # Optimizers
├── modules/           # Neural network layers
│   ├── linear.py      # Linear/fully connected layers
│   ├── linear.py      # Embedding Module
│   ├── layernorm.py   # Layer normalization
│   └── transformer.py # Transformer components
├── utils/             # Utility functions
│   ├── backend.py     # Backend management (numpy/cupy)
│   ├── logger.py      # Logging utilities
│   └── lr_scheduler.py # Learning rate scheduling
├── experiments/       # Experimental code and notebooks
└── tests/            # Test suite
```

