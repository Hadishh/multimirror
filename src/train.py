import torch
import os
import argparse
from tqdm import tqdm

from src.data.dataset import AlignmentDataset
from src.model.multimirror_model import MultiMirror
from torch.utils.data import DataLoader
from src.utils import generate_loss_mask
from src.evaluation import evaluate

def train(model, loss_fn, optimizer, epochs, train_loader, output_dir, val_loader=None, batch_size=32, device="cpu"):
    
    for epoch in range(epochs):
        running_loss = 0
        iters = 0
        pbar = tqdm(train_loader)
        for bert_input, labels, indices in pbar:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            predictions = model(bert_input, indices)
            total_loss = 0
            for pred, label in zip(predictions, labels):
                pred = torch.flatten(pred).to(device)
                label = torch.flatten(label).to(device)
                
                loss = loss_fn(pred, label)
                loss = torch.sum(loss) / loss.shape[0]
                
                total_loss += loss / len(predictions)
            
            total_loss.backward()

            optimizer.step()

            running_loss += total_loss.item()
            iters += 1
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {running_loss / iters:05}")
            
        if val_loader:
            precision, recall, f_score = evaluate(model, val_loader, threshold=0.5)
            torch.save(model, os.path.join(args.output_dir, f"chackpoint.{epoch + 1}_{f_score:.5f}.pt"))
        else:
            torch.save(model, os.path.join(args.output_dir, f"chackpoint.{epoch + 1}.pt"))
        running_loss = running_loss / iters
        

        


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_fn = torch.nn.BCELoss(reduction="none")
    train_ds = AlignmentDataset(args.train_data)
    model = MultiMirror(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_loader = DataLoader(train_ds, 
                    batch_size=args.batch_size, 
                    collate_fn=AlignmentDataset.collate_fn
                )
    
    val_loader = None
    if args.val_data:
        val_ds = AlignmentDataset(args.val_data)
        val_loader = DataLoader(val_ds, 
                    batch_size=args.batch_size, 
                    collate_fn=AlignmentDataset.collate_fn
                )
    os.makedirs(args.output_dir, exist_ok=True)
    train(model, loss_fn, optimizer, args.epoch, training_loader, batch_size=args.batch_size, device=device, output_dir=args.output_dir, val_loader=val_loader)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args)
