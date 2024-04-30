---
date: "2024-04-12T14:46:31+05:30"
title: Paper Implementation
---

{{< cards >}}
{{< card link="https://arxiv.org/pdf/1606.02147.pdf" title="ENet Academic Paper" icon="academic-cap" subtitle="arXiv PDF Link" >}}
{{< card link="https://github.com/mora-bprs/segnet/blob/main/enet_paper.ipynb" title="ENet Notebook" icon="book-open" subtitle="Jupyter Notebook Link" >}}
{{< /cards >}}
{{< callout type="info" >}}
Please visit GitHub to see the latest releases.
{{< /callout >}}

## CamVid Implementation

Paper implementation of ENet on CamVid dataset [4]. This guide assumes basic familiarity with notebooks and will include a brief setup process to get started with Google Colab.

{{% steps %}}

### Environment Setup

This is the first step.

### Notebook Environment Setup: Google Colab

- Go to \colaburl -> File -> Open Notebook -> Search for the notebook from the Github Repo \segnetrepourl and open it.
- Alternatively you can open the colab notebook from this link \enetcolaburl.
- Connect to GPU Runtime: In the menubar, go to Runtime -> Change runtime type. In the pop-up window, Runtime type as Python -> Select T4 GPU as the hardware accelerator -> Click Save.
- A Google Account is required. Colab interface is constantly changing, and it will autodetect recommended configurations for the notebook at launch. User is expected to do the best in either cases as GPU will improve the training time dramatically.
- Importing dependencies: Execute the first cell in the notebook to prepare the python environment by importing required dependencies.
- You can ignore the warning about the notebook being not authored by Google after you've reviewed the source code

### Initialize Datasets

### Training the model: `optional`

### Inference and Results

{{% /steps %}}

## Network Architecture

{{< tabs items="InitialBlock,UBNeck,RDDNeck,ASNeck,ENet" >}}

{{< tab >}}

```python {filename="enet_blocks.py"}
class InitialBlock(nn.Module):
    # Initial block of the model:
    #         Input
    #        /     \
    #       /       \
    # maxpool2d    conv2d-3x3
    #       \       /
    #        \     /
    #      concatenate

    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

        self.prelu = nn.PReLU(16)

        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.batchnorm(main)

        side = self.maxpool(x)

        # concatenating on the channels axis
        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)

        return x
```

{{< /tab >}}

{{< tab >}}

```python {filename="enet_model.py"}
class UBNeck(nn.Module):
    # Upsampling bottleneck:
    #     Bottleneck Input
    #        /        \
    #       /          \
    # conv2d-1x1     convTrans2d-1x1
    #      |             | PReLU
    #      |         convTrans2d-3x3
    #      |             | PReLU
    #      |         convTrans2d-1x1
    #      |             |
    # maxunpool2d    Regularizer
    #       \           /
    #        \         /
    #      Summing + PReLU
    #
    #  Params:
    #  projection_ratio - ratio between input and output channels
    #  relu - if True: relu used as the activation function else: Prelu us used

    def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):
        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.main_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout2d(p=0.1)

        self.convt1 = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.prelu1 = activation

        # This layer used for Upsampling
        self.convt2 = nn.ConvTranspose2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

        self.prelu2 = activation

        self.convt3 = nn.ConvTranspose2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.prelu3 = activation

        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, indices):
        x_copy = x

        # Side Branch
        x = self.convt1(x)
        x = self.batchnorm(x)
        x = self.prelu1(x)

        x = self.convt2(x)
        x = self.batchnorm(x)
        x = self.prelu2(x)

        x = self.convt3(x)
        x = self.batchnorm2(x)

        x = self.dropout(x)

        # Main Branch

        x_copy = self.main_conv(x_copy)
        x_copy = self.unpool(x_copy, indices, output_size=x.size())

        # summing the main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        return x
```

{{< /tab >}}

{{< tab >}}

```python {filename="enet_model.py"}
class RDDNeck(nn.Module):
    def __init__(
        self,
        dilation,
        in_channels,
        out_channels,
        down_flag,
        relu=False,
        projection_ratio=4,
        p=0.1,
    ):
        # Regular|Dilated|Downsampling bottlenecks:
        #
        #     Bottleneck Input
        #        /        \
        #       /          \
        # maxpooling2d   conv2d-1x1
        #      |             | PReLU
        #      |         conv2d-3x3
        #      |             | PReLU
        #      |         conv2d-1x1
        #      |             |
        #  Padding2d     Regularizer
        #       \           /
        #        \         /
        #      Summing + PReLU
        #
        # Params:
        #  dilation (bool) - if True: creating dilation bottleneck
        #  down_flag (bool) - if True: creating downsampling bottleneck
        #  projection_ratio - ratio between input and output channels
        #  relu - if True: relu used as the activation function else: Prelu us used
        #  p - dropout ratio

        super().__init__()

        # Define class variables
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag

        # calculating the number of reduced channels
        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=True
        )

        self.dropout = nn.Dropout2d(p=p)

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.prelu1 = activation

        self.conv2 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=3,
            stride=self.stride,
            padding=self.dilation,
            bias=True,
            dilation=self.dilation,
        )

        self.prelu2 = activation

        self.conv3 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.prelu3 = activation

        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.prelu1(x)

        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.batchnorm2(x)

        x = self.dropout(x)

        # Main Branch
        if self.down_flag:
            x_copy, indices = self.maxpool(x_copy)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels

            # padding and concatenating in order to match the channels axis of the side and main branches
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Summing main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        if self.down_flag:
            return x, indices
        else:
            return x
```

{{< /tab >}}

{{< tab >}}

```python {filename="enet_model.py"}
class ASNeck(nn.Module):
    def __init__(self, in_channels, out_channels, projection_ratio=4):
        # Asymetric bottleneck:
        #
        #     Bottleneck Input
        #        /        \
        #       /          \
        #      |         conv2d-1x1
        #      |             | PReLU
        #      |         conv2d-1x5
        #      |             |
        #      |         conv2d-5x1
        #      |             | PReLU
        #      |         conv2d-1x1
        #      |             |
        #  Padding2d     Regularizer
        #       \           /
        #        \         /
        #      Summing + PReLU
        #
        # Params:
        #  projection_ratio - ratio between input and output channels

        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        self.dropout = nn.Dropout2d(p=0.1)

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.reduced_depth,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.prelu1 = nn.PReLU()

        self.conv21 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2),
            bias=False,
        )

        self.conv22 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.reduced_depth,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
            bias=False,
        )

        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(
            in_channels=self.reduced_depth,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.prelu3 = nn.PReLU()

        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.prelu1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.batchnorm(x)
        x = self.prelu2(x)

        x = self.conv3(x)

        x = self.dropout(x)
        x = self.batchnorm2(x)

        # Main Branch

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels

            # padding and concatenating in order to match the channels axis of the side and main branches
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Summing main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        return x
```

{{< /tab >}}
{{< tab >}}

```python {filename="enet_model.py"}
class ENet(nn.Module):
    # Creating Enet model!

    def __init__(self, C):
        super().__init__()

        # Define class variables
        # C - number of classes
        self.C = C

        # The initial block
        self.init = InitialBlock()

        # The first bottleneck
        self.b10 = RDDNeck(
            dilation=1, in_channels=16, out_channels=64, down_flag=True, p=0.01
        )

        self.b11 = RDDNeck(
            dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01
        )

        self.b12 = RDDNeck(
            dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01
        )

        self.b13 = RDDNeck(
            dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01
        )

        self.b14 = RDDNeck(
            dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01
        )

        # The second bottleneck
        self.b20 = RDDNeck(dilation=1, in_channels=64, out_channels=128, down_flag=True)

        self.b21 = RDDNeck(
            dilation=1, in_channels=128, out_channels=128, down_flag=False
        )

        self.b22 = RDDNeck(
            dilation=2, in_channels=128, out_channels=128, down_flag=False
        )

        self.b23 = ASNeck(in_channels=128, out_channels=128)

        self.b24 = RDDNeck(
            dilation=4, in_channels=128, out_channels=128, down_flag=False
        )

        self.b25 = RDDNeck(
            dilation=1, in_channels=128, out_channels=128, down_flag=False
        )

        self.b26 = RDDNeck(
            dilation=8, in_channels=128, out_channels=128, down_flag=False
        )

        self.b27 = ASNeck(in_channels=128, out_channels=128)

        self.b28 = RDDNeck(
            dilation=16, in_channels=128, out_channels=128, down_flag=False
        )

        # The third bottleneck
        self.b31 = RDDNeck(
            dilation=1, in_channels=128, out_channels=128, down_flag=False
        )

        self.b32 = RDDNeck(
            dilation=2, in_channels=128, out_channels=128, down_flag=False
        )

        self.b33 = ASNeck(in_channels=128, out_channels=128)

        self.b34 = RDDNeck(
            dilation=4, in_channels=128, out_channels=128, down_flag=False
        )

        self.b35 = RDDNeck(
            dilation=1, in_channels=128, out_channels=128, down_flag=False
        )

        self.b36 = RDDNeck(
            dilation=8, in_channels=128, out_channels=128, down_flag=False
        )

        self.b37 = ASNeck(in_channels=128, out_channels=128)

        self.b38 = RDDNeck(
            dilation=16, in_channels=128, out_channels=128, down_flag=False
        )

        # The fourth bottleneck
        self.b40 = UBNeck(in_channels=128, out_channels=64, relu=True)

        self.b41 = RDDNeck(
            dilation=1, in_channels=64, out_channels=64, down_flag=False, relu=True
        )

        self.b42 = RDDNeck(
            dilation=1, in_channels=64, out_channels=64, down_flag=False, relu=True
        )

        # The fifth bottleneck
        self.b50 = UBNeck(in_channels=64, out_channels=16, relu=True)

        self.b51 = RDDNeck(
            dilation=1, in_channels=16, out_channels=16, down_flag=False, relu=True
        )

        # Final ConvTranspose Layer
        self.fullconv = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=self.C,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x):
        # The initial block
        x = self.init(x)

        # The first bottleneck
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)

        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)

        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)

        # Final ConvTranspose Layer
        x = self.fullconv(x)

        return x
```

{{< /tab >}}
{{< /tabs >}}

{{% steps %}}

### Model Instantiation

```python {filename="enet.py"}
enet = ENet(12)  # instantiate a 12 class ENet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enet = enet.to(device)
```

### Loading Dataset

```python {filename="loader.py"}
def loader(training_path, segmented_path, batch_size, h=320, w=1000):
    filenames_t = os.listdir(training_path)
    total_files_t = len(filenames_t)

    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)

    assert total_files_t == total_files_s

    if str(batch_size).lower() == "all":
        batch_size = total_files_s

    idx = 0
    while 1:
        # Choosing random indexes of images and labels
        batch_idxs = np.random.randint(0, total_files_s, batch_size)

        inputs = []
        labels = []

        for jj in batch_idxs:
            # Reading normalized photo
            img = plt.imread(training_path + filenames_t[jj])
            # Resizing using nearest neighbor method
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            inputs.append(img)

            # Reading semantic image
            img = Image.open(segmented_path + filenames_s[jj])
            img = np.array(img)
            # Resizing using nearest neighbor method
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            labels.append(img)

        inputs = np.stack(inputs, axis=2)
        # Changing image format to C x H x W
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)

        labels = torch.tensor(labels)

        yield inputs, labels
```

{{% /steps %}}
