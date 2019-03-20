# Conv1dsparse

## A PyTorch module that perform a convolution operation on sparse binary tensors

### Instalation

To install our package, you need to install it as:

`git clone git@github.com:leosouliotis/conv1dsparse.git`

### Implementation

First we need to import the *Conv1sparse* module

```
from conv1dsparse import Conv1dsparse
```

To use the *Conv1sparse* module, we firstly need to initialize a regular *Conv1d* module to generate the weights

```
conv1 = nn.Conv1d(channels, out_channels, kernel_size)
```

To maximize the efficiency of the module, we give an option to have partially filled inputs to the right. A list with the size of each input is needed.

The module is initialized as

```
model = Conv1dsparse(conv1, input_lengths)
```

and implements the forward pass of a convolutional layer as

```
output_sparse = model(sparse_tensor)
```

### Toy example

We create a simple tensor with 1 input with 7 input channels and input size equal to 13 as

```
i = torch.LongTensor([[0, 0, 0],
                      [0, 2, 1],
                      [0, 6, 2],
                      [0, 2, 3],
                      [0, 1, 4],
                      [0, 4, 5],
                      [0, 3, 6],
                      [0, 0, 7],
                      [0, 5, 8],
                      [0, 2, 9],
                      [0, 2, 10]])
v = torch.FloatTensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
sparse_tensor = torch.sparse.FloatTensor(i.t(), v, torch.Size([1, 7, 13]))
```

and implement a sparse convolutional layer with 5 output channels and kernel size equal to 10. We take advantage that the last value with a non-zero entry is the 11th, we denote the length of the input as a list with the single element 11.
We initialize the dense and the sparse convolutional layer as:

```
in_channels, out_channels, kernel_size = 7, 5, 10
lengths = [11]
conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
model = Conv1dsparse(conv1, genome_lengths)
```
