import torch
import numpy as np
import random

def fit(train_loader, val_loader, batch_sampler, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=0, metrics=[]):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = pass_epoch(train_loader, batch_sampler, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = pass_epoch(val_loader, batch_sampler, model, loss_fn, optimizer, cuda, log_interval, metrics, train=False)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

def pass_epoch(loader, batch_sampler, model, loss_fn, optimizer, cuda, log_interval, metrics, train=True):
    # the dataset provides its batch sampling function
    for metric in metrics:
        metric.reset()

    if train:
        model.train()
        losses = []
    else:
        model.eval()

    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        intermod_triplet_data = batch_sampler(batch, cuda)

        optimizer.zero_grad()
        outputs = model(*intermod_triplet_data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        if type(batch[1]) not in (tuple, list):
            batch[1] = (batch[1],)

        loss_inputs = outputs

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        total_loss += loss.item()

        if train:
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0 and train:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(batch[1][0]), len(loader.dataset),
                100. * batch_idx / len(loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

