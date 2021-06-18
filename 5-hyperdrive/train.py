"""Script to train our PyTorch CIFAR10 model.

Based on standard PyTorch example:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import argparse
import os

from azureml.core import Run
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Net

# connect to Azure ML Run
run = Run.get_context()

# download CIFAR 10 data
trainset = torchvision.datasets.CIFAR10(
   root="../data",
   train=True,
   download=True,
   transform=torchvision.transforms.ToTensor(),
)
trainloader = torch.utils.data.DataLoader(
   trainset, batch_size=4, shuffle=True, num_workers=2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="../outputs", help="Directory to save model state_dict")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()

    # define convolutional network
    net = Net()
    net.to(device)

    # set up pytorch loss /  optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # train the network
    for epoch in range(args.num_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # unpack the data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                loss = running_loss / 2000
                print(f"epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}")
                run.log("loss", loss)
                running_loss = 0.0

    print("Finished Training")

    print(f"Saving model state_dict to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "net.pt")
    torch.save(net.state_dict(), path)
