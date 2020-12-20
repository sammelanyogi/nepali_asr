from model import HancyModel
from dataloader import NlpDataset, data_processing
import torch.optim as optim
from torch.utils.data import Dataloader
from torch.nn import functional as F
import torch.nn as nn
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
import torch
import pickle

from .utils import TextProcess, processLabels
import pytorch_lightning as pl


def checkpoint_callback(args):
    return ModelCheckpoint(
        filepath=args.save_model_path,
        save_top_k=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )


class SpeechRecog(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.50, patience=6
        )
        return [self.optimizer], [self.scheduler]

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.step(train_batch)
        logs = {"loss": loss, "lr": self.optimizer.param_groups[0]["lr"]}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        loss = self.step(val_batch)
        return {"val_loss": loss}


def main(args, train_loader, val_loader):
    model = HancyModel()
    speechmodule = SpeechRecog(model)
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=2,
        num_nodes=8,
        distributed_backend=None,
        gradient_clip_val=1.0,
        val_check_interval=0.25,
        checkpoint_callback=checkpoint_callback(args),
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    trainer.fit(speechmodule, train_loader, val_loader)


if __name__ == "__main__":

    parser = ArgumentParser()

    # dir and path for models and logs
    parser.add_argument(
        "--save_model_path",
        default=None,
        required=True,
        type=str,
        help="path to save model",
    )
    parser.add_argument(
        "--load_model_from",
        default=None,
        required=False,
        type=str,
        help="path to load a pretrain model to continue training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        required=False,
        type=str,
        help="check path to resume from",
    )

    # training file path
    parser.add_argument(
        "--load_x",
        default=None,
        required=False,
        type=str,
        help="path to load a tensor x file",
    )
    parser.add_argument(
        "--load_y",
        default=None,
        required=False,
        type=str,
        help="path to load a tensor label file",
    )

    parser.add_argument(
        "--logdir",
        default="tb_logs",
        required=False,
        type=str,
        help="path to save logs",
    )
    args = parser.parse_args()
    tprocess = TextProcess()

    x = torch.load("/dest")
    y = processLabels(pickle.load("/dest"))

    dataset = NlpDataset(x, y)
    train, val = random_split(dataset, len(dataset) * 0.8, len(dataset) * 0.2)
    train_loader = Dataloader(
        dataset=train, collate_fn=data_processing(dataset, tprocess)
    )
    val_loader = Dataloader(dataset=val, collate_fn=data_processing(dataset, tprocess))
    main(args, train_loader, val_loader)
