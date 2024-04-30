---
date: "2024-04-12T14:46:31+05:30"
title: Paper Implementation
---

## CamVid Implementation

Paper implementation of ENet on CamVid dataset. This guide assumes basic familiarity with notebooks and will include a brief setup process to get started with Google Colab.

{{% steps %}}

### Get the Notebook

Open the notebook in Google Colab and connect to a GPU runtime.

- Go to the [ENet Notebook link](/docs/models/enet/paper/#important-links) below & click on the `Open in Colab` button.
- Connect to GPU Runtime: In the menubar, go to Runtime Change runtime type. In the pop-up window, Runtime type as Python Select T4 GPU as the hardware accelerator Click Save.
- A Google Account is required. Colab interface is constantly changing, and it will autodetect recommended configurations for the notebook at launch. User is expected to do the best in either cases as GPU will improve the training time dramatically.

### Initial Setup

Importing dependencies: Execute the first cell in the notebook to prepare the python environment by importing required dependencies.
{{< callout type="warning" >}}
You can safelt ignore the warning about the notebook being `not authored by Google` or you can opt to reviewed the source code and run it after.
{{< /callout >}}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import cv2
import os
from tqdm import tqdm
from PIL import Image

root_path = "/content" # use this for google colab
```

### Initialize Datasets

Uncomment the next cell to download the CamVid dataset and extract it.

```python
!wget "https://www.dropbox.com/s/pxcz2wdz04zxocq/CamVid.zip?dl=1" -O CamVid.zip
!unzip CamVid.zip
```

### ENet Architecture

{{< callout type="warning" >}}
Include the ENet class after the other 4 class blocks
{{< /callout >}}

Refer to [ENet Architecture](/docs/models/enet/#enet-architecture) in the documentation for the architecture code.

### Model Instantiation

```python
enet = ENet(12)  # instantiate a 12 class ENet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enet = enet.to(device)
```

### Loading Dataset

```python
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

### Defining Class Weights

```python
def get_class_weights(num_classes, c=1.02):
    pipe = loader(f"{root_path}/train/", f"{root_path}/trainannot/", batch_size="all")
    _, labels = next(pipe)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights


class_weights = get_class_weights(12)
```

### Defining Hyper Parameters

```python
lr = 5e-4
batch_size = 10

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
optimizer = torch.optim.Adam(enet.parameters(), lr=lr, weight_decay=2e-4)

print_every = 5
eval_every = 5
```

### Training the model: `optional`

```python
train_losses = []
eval_losses = []

bc_train = 367 // batch_size  # mini_batch train
bc_eval = 101 // batch_size  # mini_batch validation

# Define pipeline objects
pipe = loader(f"{root_path}/train/", f"{root_path}/trainannot/", batch_size)
eval_pipe = loader(f"{root_path}/val/", f"{root_path}/valannot/", batch_size)

epochs = 100

# Train loop

for e in range(1, epochs + 1):
    train_loss = 0
    print("-" * 15, "Epoch %d" % e, "-" * 15)

    enet.train()

    for _ in tqdm(range(bc_train)):
        X_batch, mask_batch = next(pipe)

        # assign data to cpu/gpu
        X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()

        out = enet(X_batch.float())

        # loss calculation
        loss = criterion(out, mask_batch.long())
        # update weights
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print()
    train_losses.append(train_loss)

    if (e + 1) % print_every == 0:
        print("Epoch {}/{}...".format(e, epochs), "Loss {:6f}".format(train_loss))

    if e % eval_every == 0:
        with torch.no_grad():
            enet.eval()

            eval_loss = 0

            # Validation loop
            for _ in tqdm(range(bc_eval)):
                inputs, labels = next(eval_pipe)

                inputs, labels = inputs.to(device), labels.to(device)

                out = enet(inputs)

                out = out.data.max(1)[1]

                eval_loss += (labels.long() - out.long()).sum()

            print()
            print("Loss {:6f}".format(eval_loss))

            eval_losses.append(eval_loss)

    if e % print_every == 0:
        checkpoint = {"epochs": e, "state_dict": enet.state_dict()}
        torch.save(
            checkpoint, "{}/ckpt-enet-{}-{}.pth".format(root_path, e, train_loss)
        )
        print("Model saved!")

print(
    "Epoch {}/{}...".format(e, epochs),
    "Total Mean Loss: {:6f}".format(sum(train_losses) / epochs),
)
```

### Inference and Results

```python
state_dict = torch.load(f'{root_path}/ckpt-enet.pth')['state_dict']
enet.load_state_dict(state_dict)

fname = "Seq05VD_f05100.png"
tmg_ = plt.imread(f"{root_path}/test/" + fname)
tmg_ = cv2.resize(tmg_, (512, 512), cv2.INTER_NEAREST)
tmg = torch.tensor(tmg_).unsqueeze(0).float()
tmg = tmg.transpose(2, 3).transpose(1, 2).to(device)

enet.to(device)
with torch.no_grad():
    out1 = enet(tmg.float()).squeeze(0)

# load the labeled (inferred) image
smg_ = Image.open(f'{root_path}/testannot/' + fname)
smg_ = cv2.resize(np.array(smg_), (512, 512), cv2.INTER_NEAREST)

# move the output to cpu TODO: why?
out2 = out1.cpu().detach().numpy()

mno = 8  # Should be between 0 - n-1 | where n is the number of classes

figure = plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.axis("off")
plt.imshow(tmg_)
plt.subplot(1, 3, 2)
plt.title("Output Image")
plt.axis("off")
plt.imshow(out2[mno, :, :])
plt.show()

b_ = out1.data.max(0)[1].cpu().numpy()
# Define the function that maps a 2D image with all the class labels to a segmented image with the specified colored maps

def decode_segmap(image):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colours = np.array(
        [
            Sky,
            Building,
            Pole,
            Road_marking,
            Road,
            Pavement,
            Tree,
            SignSymbol,
            Fence,
            Car,
            Pedestrian,
            Bicyclist,
        ]
    ).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, 12):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb

# decode the images
true_seg = decode_segmap(smg_)
pred_seg = decode_segmap(b_)

# plot the decoded segments
figure = plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title('Input Image')
plt.axis('off')
plt.imshow(tmg_)
plt.subplot(1, 3, 2)
plt.title('Predicted Segmentation')
plt.axis('off')
plt.imshow(pred_seg)
plt.subplot(1, 3, 3)
plt.title('Ground Truth')
plt.axis('off')
plt.imshow(true_seg)
plt.show()
```

{{% /steps %}}

## Important Links

{{< cards >}}
{{< card link="https://arxiv.org/pdf/1606.02147.pdf" title="ENet Academic Paper" icon="academic-cap" subtitle="arXiv PDF Link" >}}
{{< card link="https://github.com/mora-bprs/segnet/blob/main/enet_paper.ipynb" title="ENet Notebook" icon="book-open" subtitle="Jupyter Notebook Link" >}}
{{< /cards >}}
{{< callout type="info" >}}
Latest fixes and updates to the code can be obtained from the above GitHub link
{{< /callout >}}
