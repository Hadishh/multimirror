import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch 

from src.data.dataset import AlignmentDataset
from src.model.multimirror_model import MultiMirror
from src.utils import calculate_prf

def evaluate(model:MultiMirror, dataloader, threshold):
    total_scores = []
    model.eval()
    for bert_input, labels, indices in tqdm(dataloader):
        preds = model(bert_input, indices)
        preds = [(pred > threshold).float().cpu() for pred in preds]
        scores = calculate_prf(preds, labels)
        total_scores.append(scores)
    
    return sum([p for p, r, f in total_scores]) / len(total_scores), \
           sum([r for p, r, f in total_scores]) / len(total_scores), \
           sum([f for p, r, f in total_scores]) / len(total_scores)
        

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = AlignmentDataset(args.input)
    dataloader = DataLoader(ds, 
                            batch_size=32, 
                            collate_fn=AlignmentDataset.collate_fn
                            )

    model = MultiMirror(device=device)
    model= torch.load(args.model_path)
    model.device = device
    p, r, f = evaluate(model, dataloader, args.threshold)

    print(f"Precision: {p:.4f}, Recall: {r:.4f} F1-Score: {f:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()
    main(args)