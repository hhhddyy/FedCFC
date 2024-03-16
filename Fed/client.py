from torch.optim import optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class Client:
    def __init__(self, client_id, model, data_loader, mainloss,CUD,CUM,name='CFC'):
        self.client_id = client_id
        self.model = model
        self.name = name
        self.data_loader = data_loader
        self.optimizer = optimizer.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.optimizerF = optimizer.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # the update direction
        self.d = CUD
        # control the mag
        self.m = CUM

        if mainloss == 'MSE':
            self.criterion = nn.MSELoss()
        elif mainloss == 'NLL':
            self.criterion = nn.NLLLoss()
        elif mainloss == 'KL':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(" The main loss is not supported!")

    def train(self, lock, global_gradients,lambda_d,lambda_m,lambda_a):
        for data, target in self.data_loader:

            self.optimizer.zero_grad()
            self.optimizerF.zero_grad()

            output = self.model(data)
            mainloss = self.criterion(output, target)
            mainloss.backward()
            # Compute gradients for all parameters
            self.optimizer.step()

            # Disable gradient update for f temporarily
            if self.name == 'CFC':

                for param in self.model.rnn_cell.ff1.parameters():
                    param.requires_grad = False

                # Update g and h
                optimizer.step()

                # Re-enable gradient for f
                for param in self.model.rnn_cell.ff1.parameters():
                    param.requires_grad = True

                # Extract gradient for the last layer of f as needed
                grad_f = self.model.rnn_cell.ff1[-1].weight.grad.clone()

                flat_d = self.d.view(-1)
                assert grad_f.size(0) == flat_d.size(0)
                cosine_loss = 1 - F.cosine_similarity(grad_f, flat_d, 0)
                mag_loss = max(0,torch.norm(grad_f, p=2)-self.m)

                # Combine losses for f update #todo add lambda
                total_loss_f = mainloss + lambda_d*cosine_loss+lambda_m*mag_loss
                total_loss_f.backward()  # This should only update f due to the grad requirement settings

                # Update f using its optimizer
                self.optimizerF.step()




            # Update the global gradients list
            with lock:
                global_gradients[self.client_id] = self.model.rnn_cell.ff1[-1].weight.grad.clone()
