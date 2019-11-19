import torch
import numpy as np
import random

IMAGE_LABEL = 1
TEXT_LABEL = 0

def fit(train_loader, val_loader, batch_sampler, proj_model, d_model, proj_loss_fn, d_loss_fn,
        proj_optimizer, d_optimizer, n_epochs, cuda, log_interval, start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    def pass_epoch(loader, train=True):

        def get_classifier_loss(text_projections, image_projections, detach=False):
            if detach:
                image_projections = image_projections.detach()
                text_projections = text_projections.detach()

            pred_images = d_model(image_projections)
            pred_text = d_model(text_projections)

            image_label = torch.full((len(image_projections), 1), IMAGE_LABEL).cuda()
            text_label =  torch.full((len(image_projections), 1), TEXT_LABEL).cuda()

            i_loss = d_loss_fn(pred_images, image_label)
            t_loss = d_loss_fn(pred_text, text_label)

            loss = i_loss + t_loss

            return loss

        # the dataset provides its batch sampling function
        if train:
            proj_model.train()
            d_model.train()
            proj_losses = []
            d_losses = []
        else:
            proj_model.eval()
            d_model.eval()

        total_proj_loss = 0
        total_disc_loss = 0

        k = 6
        for batch_idx, batch in enumerate(loader):
            train_discriminator = (batch_idx % k) == 0
            intermod_triplet_data = batch_sampler(batch, cuda)
            outputs = proj_model(*intermod_triplet_data)
            proj_loss = 0.5 * proj_loss_fn(*outputs)

            if train:
                if train_discriminator:
                    d_optimizer.zero_grad()
                    d_loss = get_classifier_loss(outputs[3], outputs[0], detach=True)
                    (d_loss - proj_loss).backward()
                    d_optimizer.step()
                    d_losses.append(d_loss.item())
                else:
                    proj_optimizer.zero_grad()
                    d_optimizer.zero_grad()
                    d_loss = get_classifier_loss(outputs[3], outputs[0], detach=False)
                    (proj_loss - d_loss).backward()
                    proj_optimizer.step()
                    proj_losses.append(proj_loss.item())
            else:
                d_loss = get_classifier_loss(outputs[3], outputs[0], detach=False)

            total_proj_loss += proj_loss.item()
            total_disc_loss += d_loss.item()

            if batch_idx % log_interval == 0 and train:
                message = 'Train: [{}/{} ({:.0f}%)]\tTriplet Loss: {:.6f}\t Discriminator Loss: {:.6f}'.format(
                    batch_idx * len(batch[0]), len(loader.dataset),
                    100. * batch_idx / len(loader), np.mean(proj_losses), np.mean(d_losses))

                print(message)
                proj_losses = []
                d_losses = []

        total_proj_loss /= (batch_idx + 1)
        total_disc_loss /= (batch_idx + 1)
        return total_proj_loss, total_disc_loss

    for epoch in range(start_epoch, n_epochs):
        #scheduler.step()

        # Train stage
        train_loss = pass_epoch(train_loader)

        message = 'Epoch: {}/{}. Train set: Average Triplet loss: {:.4f} Average Discriminator loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss[0], train_loss[1])

        val_loss = pass_epoch(val_loader, train=False)

        message += '\nEpoch: {}/{}. Validation set: Average Triplet loss: {:.4f} Average Discriminator loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss[0], val_loss[1])

        print(message)
