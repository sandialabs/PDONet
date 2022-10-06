import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import dropout_adj
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from models.superpixel_classifier import Net
from datasets.superpixel import get_datasets

ex = Experiment('superpixel_classification_w_pdo_layer')

# home path set to allow calling from ray/tune
home = Path(__file__).parent

ex.observers.append(FileStorageObserver.create(
    home / 'output' / 'file_observer')
)

# add_config didn't accept a file path object so str conversions
ex.add_config(str(home / 'configs' / 'default.yml'))


@ex.capture
def train(loader, model, dev, example_count, batch_size,
          edge_dropout, _run):
    model.train()
    loss_list = []
    acc_list = np.zeros(example_count)
    for batch_idx, data in enumerate(loader):
        model.optimizer.zero_grad()
        data.to(dev)

        data.edge_index, data.edge_attr = dropout_adj(
            data.edge_index, edge_attr=data.edge_attr, p=edge_dropout, training=True,
            num_nodes=data.num_nodes
        )

        out = model(data)
        loss = model.loss(out, data.y)
        loss.backward()
        model.optimizer.step()
        acc = data.y.cpu().numpy() == out.detach().cpu().numpy().argmax(axis=1)
        acc_list[batch_idx * batch_size:batch_idx * batch_size + acc.shape[0]] = acc
        loss_list.append(loss.item())

    acc, loss = np.array(acc_list).mean(), np.array(loss_list).mean()
    _run.log_scalar('train_loss', loss)
    _run.log_scalar('train_accuracy', acc)

    return acc, loss


@ex.capture
def test(loader, model, dev, example_count, batch_size, _run):
    model.eval()
    loss_list = np.zeros(example_count)
    acc_list = np.zeros(example_count)
    for batch_idx, data in enumerate(loader):
        data.to(dev)
        out = model(data)
        loss = model.loss(out, data.y)
        acc = data.y.cpu().numpy() == out.detach().cpu().numpy().argmax(axis=1)
        acc_list[batch_idx * batch_size:batch_idx * batch_size + acc.shape[0]] = acc
        loss_list[batch_idx * batch_size:batch_idx * batch_size + acc.shape[0]] = loss.item()

    acc, loss = acc_list.mean(), loss_list.mean()
    _run.log_scalar('test_loss', loss)
    _run.log_scalar('test_accuracy', acc)
    return acc, loss


@ex.automain
def main(dataset, hierarchical, prune_edges, epochs, batch_size, learning_rate,
         weight_decay, model_options, device, _log, _run):
    train_dataset, test_dataset = get_datasets(dataset, hierarchical, prune_edges)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_node_features, train_dataset.num_classes,
                **model_options)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                       weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, 'min', patience=5, min_lr=learning_rate / 1000
    )
    model.loss = torch.nn.CrossEntropyLoss()
    model.to(dev)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    _log.info(f"MODEL PARAMS: {params}")

    best_train_acc = 0
    best_train_test_acc = 0

    for epoch in range(epochs):
        train_acc, train_loss = train(train_loader, model, dev,
                                      len(train_dataset))
        test_acc, test_loss = test(test_loader, model, dev, len(test_dataset))
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
