import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FAUST
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj
from filelock import FileLock
from pathlib import Path

from sacred import Experiment
from sacred.observers import FileStorageObserver
from models.faust import Net

ex = Experiment('faust_pdo_layer')

# home path set to allow calling from ray/tune
home = Path(__file__).parent
ex.observers.append(FileStorageObserver.create(
    home / 'output' / 'file_observer')
)

# add_config didn't accept a file path object so str conversion
ex.add_config(str(home / 'configs' / 'faust.yml'))


@ex.capture
def train(loader, model, dev, example_count, target, batch_size,
          edge_dropout, _run):
    model.train()
    loss_list = []
    acc_list = np.zeros(example_count * int(target.shape[0]/batch_size))
    for batch_idx, data in enumerate(loader):
        model.optimizer.zero_grad()
        data.x = data.pos
        data.to(dev)

        data.edge_index, data.edge_attr = dropout_adj(
            data.edge_index, edge_attr=data.edge_attr, p=edge_dropout, training=True,
            num_nodes=data.num_nodes
        )

        out = model(data)
        loss = model.loss(out, target)
        loss.backward()
        model.optimizer.step()
        acc = target.cpu().numpy() == out.detach().cpu().numpy().argmax(axis=1)
        acc_list[batch_idx * acc.shape[0]:batch_idx * acc.shape[0] + acc.shape[0]] = acc
        loss_list.append(loss.item())

    acc, loss = np.array(acc_list).mean(), np.array(loss_list).mean()
    _run.log_scalar('train_loss', loss)
    _run.log_scalar('train_accuracy', acc)

    return acc, loss


@ex.capture
def test(loader, model, dev, example_count, target, batch_size, _run):
    model.eval()
    loss_list = []
    acc_list = np.zeros(example_count * int(target.shape[0]/batch_size))
    for batch_idx, data in enumerate(loader):
        data.x = data.pos
        data.to(dev)
        out = model(data)
        loss = model.loss(out, target)
        acc = target.cpu().numpy() == out.detach().cpu().numpy().argmax(axis=1)
        acc_list[batch_idx * acc.shape[0]:batch_idx * acc.shape[0] + acc.shape[0]] = acc
        loss_list.append(loss.item())

    acc, loss = np.array(acc_list).mean(), np.array(loss_list).mean()
    _run.log_scalar('test_loss', loss)
    _run.log_scalar('test_accuracy', acc)
    return acc, loss


@ex.automain
def main(epochs, batch_size, learning_rate, weight_decay, device,
         model_options, _log):

    pre_transform = T.Compose([T.FaceToEdge(), T.Constant(value=1.0)])

    with FileLock(home / 'train_data.lock'):
        train_dataset = FAUST(
            home / 'data/FAUST', train=True,
            pre_transform=pre_transform, transform=T.Cartesian()
        )
    with FileLock(home / 'test_data.lock'):
        test_dataset = FAUST(
            home / 'data/FAUST', train=False,
            pre_transform=pre_transform, transform=T.Cartesian()
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = Net(3, train_dataset[0].num_nodes, **model_options)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                       weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 'min', patience=5, min_lr=learning_rate / 1000
    )
    model.loss = torch.nn.CrossEntropyLoss()
    model.to(dev)

    target = torch.arange(train_dataset[0].num_nodes, dtype=torch.long).repeat(batch_size).to(dev)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    _log.info(f"MODEL PARAMS: {params}")

    best_train_acc = 0
    best_train_test_acc = 0

    for epoch in range(epochs):
        train_acc, train_loss = train(train_loader, model, dev,
                                      len(train_dataset), target)
        test_acc, test_loss = test(test_loader, model, dev, len(test_dataset), target)
        scheduler.step(test_loss)  # apply lr reduction

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_test_acc = test_acc
            _log.info("New best training acc {}\t test {}".format(train_acc,
                                                                  test_acc))

        _log.info(
            'Epoch: %d\tTrain acc: %.4f loss: %.3f\t Test acc: %.4f loss: %.3f'
            % (epoch + 1, train_acc, train_loss, test_acc, test_loss)
        )
    return best_train_test_acc
