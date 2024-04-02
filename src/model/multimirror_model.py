import torch
from transformers import BertModel, BertTokenizer
from torch import nn

class MultiMirror(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # Bert
        self.hidden_size = 768
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, 
                                                   nhead=12, 
                                                   dim_feedforward=3072, 
                                                   dropout=0.1,
                                                   activation=nn.functional.gelu)
        self.alignment_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)

        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)

    def __mean_tokens(self, representations, l, indices):
        s_representations = []
        mean_tensors = [representations[1, :]]
        for idx in range(2, l):
            if indices[idx - 1] == indices[idx]:
                mean_tensors.append(representations[idx, :])
                if idx == l - 1:
                    word_representation = torch.mean(torch.stack(mean_tensors, dim=0), dim=0)
                    s_representations.append(word_representation)
            else:
                word_representation = torch.mean(torch.stack(mean_tensors, dim=0), dim=0)
                s_representations.append(word_representation)
                if idx == l - 1:
                    s_representations.append(representations[idx, :])
                else:
                    mean_tensors = [representations[idx, :]]
        return torch.stack(s_representations, dim=0).to(self.device)
    
    def __step_forward_alignment(self, repr, indices):

        l = len(indices[0])
        k = len(indices[1])
        s1_repr = self.__mean_tokens(repr[:l, :], l, indices[0])
        s2_repr = self.__mean_tokens(repr[l:l+k], k, indices[1])

        l = s1_repr.shape[0]
        k = s2_repr.shape[0]

        repr = torch.concat([s1_repr, s2_repr], dim = 0).to(self.device)

        repr = self.alignment_encoder(repr) # (l + k) x 768

        s1_repr = repr[:l, :] # l x 768
        s2_repr = repr[l:l + k, :] # k x 768
        # s1_repr = s1_repr.unsqueeze(1).repeat(1, k, 1) # l x k x 768
        # s2_repr = s2_repr.unsqueeze(0).repeat(l, 1, 1) # l x k x 768
        s1_repr = s1_repr.unsqueeze(1) # l x 1 x h
        s2_repr = s2_repr.unsqueeze(0) # 1 x k x h

        repr = s1_repr * s2_repr # l x k x 768

        repr = self.linear_1(repr)
        repr = self.relu_1(repr) # l x k x 768

        repr = self.linear_2(repr)
        repr = self.relu_2(repr) # l x k x 768

        repr = self.linear_3(repr)
        repr = self.sigmoid(repr) # l x k x 1

        return repr
    

    def forward(self, bert_input, indices):
        bert_input.to(self.device)
        bert_out = self.bert(**bert_input).last_hidden_state # batch_size x seq_len x 768
        output = []
        for b in range(bert_out.shape[0]):
            out = self.__step_forward_alignment(bert_out[b], indices[b])
            output.append(out)
        return output