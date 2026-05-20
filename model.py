import torch


class Tiny(torch.nn.Module):
    def __init__(self, input_dim=30):
        super(Tiny, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Small(torch.nn.Module):
    def __init__(self, input_dim=30):
        super(Small, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Medium(torch.nn.Module):
    def __init__(self, input_dim=30):
        super(Medium, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 32)
        self.linear4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Large(torch.nn.Module):
    def __init__(self, input_dim=30):
        super(Large, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 32)
        self.linear5 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.linear5(x)
        return x
