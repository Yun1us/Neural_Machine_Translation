import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from evaluate import load 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

MAX_LEN    = 30
BATCH_SIZE = 256
LR         = 1e-4
EMBED_DIM  = 256
HIDDEN_DIM = 512
ATTN_DIM   = 128
EPOCHS     = 25

# 1) Daten laden
dataset = pd.read_csv("data.csv")
en_sentences = dataset["EN"].tolist()
fr_sentences = dataset["FR"].tolist()

# 2) Split into train + validation (here 90% train / 10% val)
en_train, en_val, fr_train, fr_val = train_test_split(
    en_sentences,
    fr_sentences,
    test_size=0.10,
    random_state=42,
    shuffle=True,
)
# 2) Vocab-Klasse (unverändert variablennamen)
class Vocab:
    def __init__(self):
        self.max_len = 30
        self.token_to_ids = {"<UNK>": 0}
        self.id_to_token = {0: "<UNK>"}
        self.special_tokens = ["<pad>", "<sos>", "<eos>"]
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_ids:
            idx = len(self.token_to_ids)
            self.token_to_ids[token] = idx
            self.id_to_token[idx] = token

    def sentence_to_id(self, sentence, maxlen=MAX_LEN):
        attention_id = []
        id_list = []
        words = ["<sos>"] + sentence.split() + ["<eos>"]
        for word in words:
            if word in self.token_to_ids:
                id_list.append(self.token_to_ids[word])
            else:
                id_list.append(self.token_to_ids["<UNK>"])
            attention_id.append(1)
        if len(id_list) > maxlen:
            id_list = id_list[:maxlen]
        while len(id_list) < maxlen:
            id_list.append(self.token_to_ids["<pad>"])
            attention_id.append(0)
        return id_list, attention_id

    def id_to_sentence(self, ids):
        words = []
        for i in ids:
            token = self.id_to_token.get(i, "<UNK>")
            if token == "<eos>":
                break
            if token in ["<sos>", "<pad>"]:
                continue
            words.append(token)
        return " ".join(words)

    def build_Vocab(self, sentences):
        for sentence in sentences:
            for token in sentence.lower().split():
                self.add_token(token)

# Vokabulare instanziieren
en_vocab = Vocab() 
fr_vocab = Vocab()
en_vocab.build_Vocab(en_train)
fr_vocab.build_Vocab(fr_train)

pad_idx = fr_vocab.token_to_ids["<pad>"]  # changed to token_to_ids

# 3) Dataset
class EN_FR_Dataset(torch.utils.data.Dataset):
    def __init__(self, en_sentences, fr_sentences, en_vocab, fr_vocab):
        self.en_sentences = en_sentences
        self.fr_sentences = fr_sentences
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, index):
        en_ids, en_attention = self.en_vocab.sentence_to_id(self.en_sentences[index], MAX_LEN)
        fr_ids, fr_attention = self.fr_vocab.sentence_to_id(self.fr_sentences[index], MAX_LEN)
        return (
            torch.tensor(en_ids, dtype=torch.long),
            torch.tensor(en_attention, dtype=torch.bool),
            torch.tensor(fr_ids, dtype=torch.long),
            torch.tensor(fr_attention, dtype=torch.bool),
        )

train_dataset = EN_FR_Dataset(en_train, fr_train, en_vocab, fr_vocab)
val_dataset = EN_FR_Dataset(en_val, fr_val, en_vocab, fr_vocab)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader= DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
dataset = EN_FR_Dataset(en_sentences, fr_sentences, en_vocab, fr_vocab)
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

# 4) BahdanauAttention 
class BahdanauAttention(nn.Module):
    def __init__(self, attention_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(encoder_hidden_dim, attention_dim)
        self.w2 = nn.Linear(decoder_hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
    def forward(self, decoder_hidden, encoder_outputs, att_mask):
        h = self.w1(encoder_outputs)                        # [batch,seq,attn]
        s = self.w2(decoder_hidden).unsqueeze(1)            # [batch,1,attn]
        score = torch.tanh(h + s)                          # [batch,seq,attn]
        score_full = self.v(score).squeeze(-1)              # [batch,seq]

        #attention mask
        score_full = score_full.masked_fill(att_mask == 0, -1e10)


        weights = torch.softmax(score_full, dim=1)          # [batch,seq]

        context_vector = torch.sum(weights.unsqueeze(2) * encoder_outputs, dim=1)
        return context_vector, weights

# 5) Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_dim, dropout=0.2):
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.input_embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True, bidirectional=False)
    def forward(self, batch):
        embedded_vector = self.input_embed(batch)
        embedded_vector = self.dropout(embedded_vector)
        encoder_output, last_hidden = self.gru(embedded_vector)
        return encoder_output, last_hidden

# 6) Decoder
class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim, hidden_dim, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.attention = BahdanauAttention(128, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + hidden_dim, target_vocab_size)


    def forward(self, input_token, last_hidden, encoder_outputs, att_mask):

        token_embed = self.embedding(input_token) 

        context_vec, attn_weights = self.attention(last_hidden[0], encoder_outputs, att_mask)  
        concat_vec = torch.cat([token_embed, context_vec], dim=1)
        gru_input = concat_vec.unsqueeze(1)

        decoder_output, decoder_hidden = self.gru(gru_input, last_hidden)
        decoder_output = decoder_output.squeeze(1)
        dec_concat = torch.cat([decoder_hidden[0], context_vec], dim=1)
        logits, hidden = self.fc(dec_concat), decoder_hidden  

        return logits, hidden, attn_weights

# 7) Training-Setup
source_vocab_size = len(en_vocab.token_to_ids)
target_vocab_size = len(fr_vocab.token_to_ids)
attention_dim = 128

encoder = Encoder(source_vocab_size, EMBED_DIM, HIDDEN_DIM)
decoder = Decoder(target_vocab_size, EMBED_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
writer = SummaryWriter(log_dir="runs/mt_experiment")
all_hyps = [] 
all_refs = [] 
val_loss = 0.0
bleu = load("sacrebleu")
eos_idx = fr_vocab.token_to_ids["<eos>"]

for epoch in range(EPOCHS):
    # --- TRAINING ---
    encoder.train(); decoder.train()
    epoch_loss = 0.0

    for en_ids, en_att, fr_ids, fr_att in train_loader:
        encoder_outputs, encoder_last_hidden = encoder(en_ids)
        decoder_hidden = encoder_last_hidden.clone()
        decoder_input  = fr_ids[:, 0] #sos?

        # 2) Loss für das Batch berechnen (mit Masking)
        batch_loss  = 0.0
        valid_steps = 0
        for t in range(1, fr_ids.size(1)):
            logits, decoder_hidden, _ = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                en_att
            )
            true_token = fr_ids[:, t]
            mask       = fr_att[:, t]   # nur gültige Tokens
            if mask.any():
                step_loss   = criterion(logits[mask], true_token[mask])
                batch_loss += step_loss
                valid_steps += 1
            decoder_input = true_token

        # 3) Mittelung und aufs Epoch-Loss aufsummieren
        batch_loss   = batch_loss / valid_steps
        epoch_loss  += batch_loss.item()

        # 4) Backprop + Optimizer‐Step
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()

    avg_epoch_loss = epoch_loss / len(train_loader)
    perplexity     = torch.exp(torch.tensor(avg_epoch_loss))

    # --- VALIDATION ---
    encoder.eval(); decoder.eval()
    val_loss = 0.0                    
    all_hyps = []                   
    all_refs = []                   

    with torch.no_grad():
        for en_ids, en_att, fr_ids, fr_att in val_loader:
            weights_per_t = []     #one [batch,src_len] tensor per step

            # encode
            enc_out, enc_h = encoder(en_ids)
            dec_hidden     = enc_h.clone()
            dec_input      = fr_ids[:, 0]
            batch_size     = en_ids.size(0)
            max_len        = fr_ids.size(1)

            # compute batch loss
            batch_loss  = 0.0            
            valid_steps = 0             
            batch_pred_ids = torch.zeros(batch_size, max_len, dtype=torch.long) 
            for t in range(1, max_len):
                logits, dec_hidden, attn_weights = decoder(dec_input, dec_hidden, enc_out, en_att)
                weights_per_t.append(attn_weights[0])
                
                true_token = fr_ids[:, t]
                mask       = fr_att[:, t]
                if mask.any():
                    step_loss   = criterion(logits[mask], true_token[mask])
                    batch_loss += step_loss
                    valid_steps += 1

                
                pred_token = logits.argmax(dim=1)
                if (pred_token == eos_idx).all():
                    break
                batch_pred_ids[:, t] = pred_token
                dec_input = pred_token

            #build NumPy Array
            attn_mat = torch.stack(weights_per_t).cpu().numpy()

            # actual token labels
            src_tokens = en_vocab.id_to_sentence(en_ids[0].tolist()).split()
            tgt_tokens = fr_vocab.id_to_sentence(batch_pred_ids[0].tolist()).split()

            plt.figure(figsize=(8,6))
            plt.imshow(attn_mat, aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha='right')
            plt.yticks(range(len(tgt_tokens)), tgt_tokens)
            plt.xlabel("Source Tokens")
            plt.ylabel("Target Tokens")
            plt.title(f"Epoch {epoch} — Example 0 Attention")
            plt.tight_layout()
            plt.show()

            # batch loss into val_loss
            val_loss += (batch_loss / valid_steps).item()  

            # collect for BLEU
            for i in range(batch_size):
                hyp_tokens = fr_vocab.id_to_sentence(batch_pred_ids[i].tolist()).split()
                ref_tokens = fr_vocab.id_to_sentence(fr_ids[i].tolist()).split()
                # join back into strings:
                hyp_str = " ".join(hyp_tokens)
                ref_str = " ".join(ref_tokens)
                all_hyps.append(hyp_str)         # List[str]
                all_refs.append([ref_str])       # List[List[str]]         

    # validation metrics
    avg_val_loss = val_loss / len(val_loader)         
    bleu_res     = bleu.compute(predictions=all_hyps, references=all_refs) 

    # --- EPOCH SUMMARY ---
    print(
        f"[Epoch {epoch:02d}]  "
        f"Train Loss={avg_epoch_loss:.4f}  "
        f"Val Loss={avg_val_loss:.4f}  "  
        f"PPL={perplexity:.2f}  "
        f"Val BLEU={bleu_res['bleu']*100:.2f}"
    )
        # --- TENSORBOARD LOGGING ---
    writer.add_scalar("train/loss", avg_epoch_loss, epoch)
    writer.add_scalar("train/ppl",  perplexity,     epoch)
    writer.add_scalar("val/bleu",  bleu_res["bleu"]*100, epoch)
    writer.add_scalar("val/loss", avg_val_loss, epoch)
