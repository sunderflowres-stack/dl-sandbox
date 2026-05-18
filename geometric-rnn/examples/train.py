import sys
import os
import random
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from grnn import GeometricRNN

COMPILE_MODEL = False
FAST_DEV = False

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LitCharRNN(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=64, num_layers=2, lr=3e-3):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = GeometricRNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=0,
            use_gate=True,
            return_sequences=True,
            dropout=0.2,
        )
        self.W_out = nn.Linear(hidden_size, vocab_size, bias=False)
        self.W_jepa = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h0=None):
        emb = self.embedding(x)
        h_seq, h_last = self.rnn(emb, h0)
        logits = self.W_out(h_seq)
        e_hat = self.W_jepa(h_seq)
        return logits, e_hat, h_last

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, e_hat, _ = self(x)

        loss_ce = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))

        with torch.no_grad():
            e_target = F.normalize(self.embedding(y).detach(), dim=-1)

        e_hat_norm = F.normalize(e_hat, dim=-1)
        loss_jepa = 1.0 - (e_hat_norm * e_target).sum(dim=-1).mean()

        loss = loss_ce + loss_jepa

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_ce", loss_ce)
        self.log("loss_jepa", loss_jepa)

        a_norms = [
            cell.rotor.last_A_norm
            for cell in self.rnn.cells
            if cell.rotor.last_A_norm is not None
        ]
        if a_norms:
            avg = torch.stack(a_norms).mean() if isinstance(a_norms[0], torch.Tensor) else sum(a_norms) / len(a_norms)
            self.log("rotor_A_norm", avg)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _, _ = self(x)
        loss_ce = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        self.log("val_loss", loss_ce, prog_bar=True)
        return loss_ce

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    @torch.no_grad()
    def generate(self, start_str, itos, stoi, max_len=100, temperature=0.8):
        assert len(start_str) > 0, "start_str cannot be empty"
        self.eval()
        device = next(self.parameters()).device

        chars = list(start_str)
        x = torch.tensor([[stoi[c] for c in chars]], dtype=torch.long, device=device)

        _, _, h = self(x)
        curr_x = x[:, -1:]

        for _ in range(max_len):
            logits, _, h = self(curr_x, h)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1)
            chars.append(itos[next_char_idx.item()])
            curr_x = next_char_idx

        return "".join(chars)


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = os.path.join(current_dir, "shakespeare.txt")

    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = [stoi[c] for c in text]

    SEQ_LEN = 64
    BATCH_SIZE = 128
    HIDDEN_SIZE = 64

    chunk_size = SEQ_LEN + 1
    chunks = [
        data[i : i + chunk_size]
        for i in range(0, len(data) - chunk_size, chunk_size)
    ]
    random.shuffle(chunks)

    if FAST_DEV:
        chunks = chunks[:4000]

    split = int(0.9 * len(chunks))
    train_chunks = chunks[:split]
    val_chunks = chunks[split:]

    train_data = [tok for chunk in train_chunks for tok in chunk]
    val_data = [tok for chunk in val_chunks for tok in chunk]

    train_dataset = CharDataset(train_data, SEQ_LEN)
    val_dataset = CharDataset(val_data, SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    model = LitCharRNN(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        lr=3e-3,
    )

    if COMPILE_MODEL:
        try:
            model = torch.compile(model, backend="eager")
            print("torch.compile: enabled (eager)")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n--- Generation ---")
    print(model.generate("ROMEO:\n", itos, stoi, max_len=200))
