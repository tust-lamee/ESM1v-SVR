import os
import torch
import esm
import pandas as pd
import numpy as np
import joblib
from Bio import SeqIO
from tqdm import tqdm
import math

esm_model_path = "esm1v-svr/checkpoints/esm1v_t33_650M_UR90S_1.pt"
model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model_path)
batch_converter = alphabet.get_batch_converter()
torch.set_grad_enabled(False)
model.eval()


def extract_features_from_fasta(fasta_file, model, alphabet, layer=33, chunk_size=32):

    sequences = []
    skipped = []   

    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        seq_raw = str(record.seq).strip()
        seq = seq_raw.upper()

        seq = seq[:1020]
        sequences.append((record.id, seq))

    total_seqs = len(sequences)

    if total_seqs == 0:
        return np.empty((0, model.args.embed_dim)), [], skipped

    all_features = []
    seq_ids = []

    def split_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    total_batches = math.ceil(total_seqs / chunk_size)

    for chunk in tqdm(list(split_list(sequences, chunk_size)), total=total_batches):
        batch_labels, batch_strs, batch_tokens = batch_converter(chunk)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        token_reps = results["representations"][layer]

        for i, length in enumerate(batch_lens):
            rep = token_reps[i, 1:length - 1]
            feature = rep.mean(0).float().cpu().numpy()
            all_features.append(feature)
            seq_ids.append(chunk[i][0])

    return np.array(all_features), seq_ids, skipped


def predict_from_fasta(fasta_file, model_file, scaler_file, output_csv):

    features, seq_ids, skipped = extract_features_from_fasta(
        fasta_file,
        model=model,
        alphabet=alphabet,
        layer=33,
        chunk_size=32
    )

    svr = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    features_scaled = scaler.transform(features)
    preds = svr.predict(features_scaled)

    df = pd.DataFrame({"Sequence_ID": seq_ids, "Predicted Value": preds})
    df.to_csv(output_csv, index=False)




if __name__ == "__main__":
    fasta_input = "esm1v-svr/0.fasta"
    model_file = "9579_svr.pkl"
    scaler_file = "9579_svr_scaler.pkl"
    output_file = "esm1v-svr/0.csv"

    predict_from_fasta(fasta_input, model_file, scaler_file, output_file)

