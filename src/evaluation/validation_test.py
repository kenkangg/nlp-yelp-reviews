import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch

device = 'cuda'

def test_model(model, dloader, num_data, desc=None):
    """
    Test model on validation set


    Returns:
        average loss
        accuracy

    """
    loss_arr = np.array([])
    preds = []
    for x,y in tqdm(dloader, total=num_data//BATCH_SIZE, desc=desc):
        x_cuda = x.to(device)
        y_cuda = y.to(device)

        output = model(x_cuda)
        loss = lossfn(output,y_cuda)
        with out:
                clear_output(wait=True)
                print(output.data.cpu().numpy())
                print(y_cuda.data.cpu().numpy())
                print(loss)

        preds.append((torch.round(output).data.cpu().numpy().T, y.data.cpu().numpy().T))
        loss_arr = np.concatenate((loss_arr, np.array([loss.data.cpu().numpy()])))
        del x, y, output, loss, x_cuda, y_cuda

    pred = np.concatenate([x[0][0] for x in preds])
    actual = np.concatenate([x[1][0] for x in preds])
    cm = confusion_matrix(actual, pred, labels=[1,2,3,4,5])
    acc = accuracy_score(actual, pred)
    f1sc = f1_score(actual,pred, average='macro')
    return np.mean(loss_arr), acc, cm, f1sc
