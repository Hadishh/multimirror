import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch 
import os

from src.data.dataset import AlignmentDataset
from src.model.multimirror_model import MultiMirror
from src.utils import calculate_f1

def __parse_prediction(prediction) -> str:
    l, k = prediction.shape[0], prediction.shape[1]
    alignments = []
    for i in range(l):
        for j in range(k):
            if prediction[i, j].item() == 1:
                alignments.append(f"{i}-{j}")
            
    return " ".join(alignments)


def predict(model:MultiMirror, dataloader, threshold):
    model.eval()
    alignments = []
    for bert_input, labels, indices in tqdm(dataloader):
        preds = model(bert_input, indices)
        preds = [(pred > threshold).float().cpu() for pred in preds]
        alignments.extend([__parse_prediction(pred) for pred in preds])
    
    return alignments
        

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
    alignments = predict(model, dataloader, args.threshold)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "en-fa.MM.align")
    f = open(output_path, "w")
    for line in alignments:
        f.write(line + "\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()
    main(args)