import numpy as np
import torch


def train(model, trainloader, testloader, optimizer, NO_EPOCHS, BATCH_SIZE):
    for epoch in range(NO_EPOCHS):
        mean_epoch_loss = []
        mean_epoch_loss_val = []

        for batch, label in trainloader:
            t = torch.randint(0, model.timesteps, (BATCH_SIZE,)).long()
            batch_noisy, noise = model.forward(batch, t) 
            predicted_noise = model(batch_noisy, t, labels = label.reshape(-1,1).float())

            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(noise, predicted_noise) 
            mean_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        for batch, label in testloader:

            t = torch.randint(0, model.timesteps, (BATCH_SIZE,)).long()

            batch_noisy, noise = model.forward(batch, t) 
            predicted_noise = model(batch_noisy, t, labels = label.reshape(-1,1).float())

            loss = torch.nn.functional.mse_loss(noise, predicted_noise) 
            mean_epoch_loss_val.append(loss.item())

        if epoch % 10 == 0:
            print('---')
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")