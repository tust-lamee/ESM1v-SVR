import esm
import torch.hub
torch.hub.set_dir('esm1v-svr')

def split_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


all_data = []
my_list = list(range(0, 30000))

line_number = 0

with open("9579", 'r', encoding='utf-8') as file:
    for j in file:
        line_number += 1  
        try:
            i = my_list[line_number - 1] 
            data, label = j.strip().split('\t')
            if len(data) > 1020: 
                data = data[:1020]
        except ValueError as e:
            print(f"Warning: Line {line_number} caused an error due to incorrect format - {j.strip()} with error: {e}")
            continue  

        data_mutation = data
        data_ = (str(label) + '_' + str(i) + '_' + str(data), data_mutation)
        all_data.append(data_)

small_lists = list(split_list(all_data, 4))
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()   
batch_converter = alphabet.get_batch_converter()
model.eval()  

with open('9579.csv', 'w', encoding='utf-8') as svm_file:
    for one in small_lists:
        batch_labels, batch_strs, batch_tokens = batch_converter(one)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

        for i, tokens_len in enumerate(batch_lens):
            sequence_tensor = token_representations[i, 1:tokens_len - 1]
            sequence_feature_vector = sequence_tensor.mean(dim=0).cpu().numpy()
            label = batch_labels[i].split('_')[0]
            feature_string = ','.join(map(str, sequence_feature_vector))
            svm_file.write(f"{label},{feature_string}\n")

