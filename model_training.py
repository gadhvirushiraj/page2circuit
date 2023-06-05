
import os
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,random_split
import datetime


# generate labels set
data_path = os.path.join(os.getcwd(), 'dataset_final/')
label_list = [l.lower().strip() for l in os.listdir(data_path) if not l.startswith('.')]
labels = {ele:i for i, ele in enumerate(label_list)}
print('------ Labels ------', labels, sep='\n')

# calculate mean and std of dataset
# mean = 0.
# std = 0.

# for images, _ in dataset:
#     mean += images.view(1, -1).mean(dim=1)
#     std += images.view(1, -1).std(dim=1)

# mean /= len(dataset)
# std /= len(dataset)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.00683), std=(0.2440)), # mean and std are calculated above
    transforms.Resize((64,64)),
    transforms.Lambda(lambda x: x[0,:,:].T),
])


dataset = datasets.ImageFolder(data_path, transform=transform)
train_data, val_data = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

# model class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(1, 32, 3, padding=1)
        self.acv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.con2 = nn.Conv2d(32, 64, 3, padding=1)
        self.acv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.con3 = nn.Conv2d(64, 32, 3, padding=1)
        self.acv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.con4 = nn.Conv2d(32, 16, 3, padding=1)
        self.acv4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(16*4*4, 1024)
        self.act5 = nn.ReLU()
        self.lin2 = nn.Linear(1024, 37)

    def forward(self, img):
        out = self.pool1(self.acv1(self.con1(img)))
        out = self.pool2(self.acv2(self.con2(out)))
        out = self.pool3(self.acv3(self.con3(out)))
        out = self.pool4(self.acv4(self.con4(out)))
        out = out.view(-1, 16*4*4)
        out = self.act5(self.lin1(out))
        out = self.lin2(out)

        return out



model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)


epochs = 100
for epoch in range(epochs + 1):
    loss_train = 0.0
    for imgs, labels in train_loader:
        outputs = model(imgs.unsqueeze(1))
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    if epoch % 10 == 0:
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch,loss_train / len(train_loader)))


val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)


y_pred = []
y_true = []

for name, loader in [('Train', train_loader), ('Validation', val_loader)]:
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_pred.extend(predicted)
            y_true.extend(labels)

    print('Accuracy of the network on the {} images: {} %'.format(name, 100 * correct / total))


model.eval()
torch.save(model, 'model.pkl')