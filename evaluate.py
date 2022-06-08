import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader


def evaluate(model, loss_fn, dataloader, metrics, params):
    model.eval()

    summ = []

    for x, y in dataloader:
        # move to GPU if available
        if params.cuda:
            x, y = x.cuda(
                non_blocking=True), y.cuda(non_blocking=True)

        # compute model output
        ypred = model(x)
        loss = loss_fn(ypred, y)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        ypred = ypred.data.cpu().numpy()
        y = y.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](ypred, y)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """

    data_dir = "data"
    model_dir = "experiments/base_model"
    restore_file = "best"

    # Load the parameters
    json_path = os.path.join(model_dir, 'params.json')

    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(77)
    if params.cuda:
        torch.cuda.manual_seed(77)

    # Get the logger
    utils.set_logger(os.path.join(model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.TwoLayerNet(2,2,1).cuda() if params.cuda else net.TwoLayerNet(2,4,1)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        model_dir, restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        model_dir, "metrics_test_{}.json".format(restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
