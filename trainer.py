import torch
from tqdm import tqdm
import numpy as np

class Trainer(object):
    def __init__(self, model, device, criterion, optimizer, scheduler = None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for index, batch in enumerate(tqdm(dataloader, desc = "Iteration")):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            y_true = batch.y
            y_pred = self.model(batch)
            y_pred = y_pred.view(y_true.size())

            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()

        total_loss /= (index + 1)

        return total_loss

    def eval_step(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        y_trues, y_preds = [], []

        with torch.inference_mode():
            for index, batch in enumerate(tqdm(dataloader, desc = "Iteration")):
            #for index, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                
                y_true = batch.y
                y_pred = self.model(batch)
                y_pred = y_pred.view(y_true.size())

                loss = self.criterion(y_pred, y_true).item()
                total_loss += loss

                y_trues.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())

        total_loss /= (index + 1)
        y_trues = np.concatenate(y_trues, axis = 0)
        y_preds = np.concatenate(y_preds, axis = 0)

        return total_loss, y_trues, y_preds

    def predict_step(self, dataloader):
        self.model.eval()

        y_preds = []

        with torch.inference_mode():
            for index, batch in enumerate(tqdm(dataloader, desc = "Iteration")):
            #for index, batch in enumerate(dataloader):
                batch = batch.to(self.device)

                y_pred = self.model(batch)
                y_preds.append(y_pred.cpu().numpy())

        y_preds = np.concatenate(y_preds, axis = 0)

        return y_preds

    def train(self, num_epochs, train_dataloader, val_dataloader, early_stopping = False, patience = None):
        if early_stopping:
            assert(patience is not None)

        best_val_loss = np.inf
        best_model = None

        for epoch in range(num_epochs):
            print("---- Epoch {} ----".format(epoch))
            
            print("Training...")
            train_loss = self.train_step(train_dataloader)
            
            print("Evaluating...")
            val_loss, _, _ = self.eval_step(val_dataloader)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model

                current_patience = patience

            else:
                current_patience -= 1

            #Logging
            print(
                "Epoch {} || ".format(epoch), 
                "Train Loss: {} || ".format(train_loss),
                "Val Loss: {} || ".format(val_loss),
                "Learning rate: {0:.{1}f} || ".format(self.optimizer.param_groups[0]["lr"], 6),
                "Patience: {} \n".format(current_patience) 
            )

            # wandb.log({
            #     "epochs": epoch,
            #     "train_loss": train_loss,
            #     "val_loss": val_loss,
            #     })

            if early_stopping and current_patience == 0:
                print("Stop Early")
                break

        return self.model, best_model