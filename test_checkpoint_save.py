"""Quick test: can Lightning save a checkpoint to scratch?"""
import os
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

SAVE_DIR = os.path.join(os.path.dirname(__file__), "logs", "ckpt_test")


class TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def training_step(self, batch, batch_idx):
        loss = self.linear(batch[0]).sum()
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


dataset = torch.utils.data.TensorDataset(torch.randn(64, 10))
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

checkpoint_callback = ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="test_ckpt",
    monitor="train/loss",
    save_top_k=1,
    every_n_train_steps=1,
    save_weights_only=False,
)

trainer = L.Trainer(
    max_steps=2,
    accelerator="cpu",
    callbacks=[checkpoint_callback],
    enable_progress_bar=True,
    logger=False,
)

print(f"Saving to: {SAVE_DIR}")
model = TinyModel()
trainer.fit(model, loader)
print(f"Done. Files: {os.listdir(SAVE_DIR)}")
