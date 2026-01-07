import tiktoken
import safetensors
import torch
from torch.nn import functional as F




f = open("tiny-shakespeare.txt")
text = f.read()

enc = tiktoken.encoding_for_model("gpt2")
tokens = enc.encode(text)

x = torch.tensor(tokens[:10])
y = torch.tensor(tokens[1:11])

print(x, y)

#text_rev = enc.decode(tokens)
#print(text_rev)


# Loading Model Weights
f = open("model.safetensors", "rb")
fts = safetensors.deserialize(f.read())

params = {}
for ft in fts:
    name = ft[0]
    shape = ft[1]['shape']
    data = ft[1]['data']
    tensor_1d = torch.frombuffer(data, dtype=torch.float32) # this creates a 1 dimensional tensor
    tensor_2d = tensor_1d.reshape(shape) # reshaping the 1d tensor to 2d tensor
    params[name] = tensor_2d

#print(params['wte.weight'].shape)

wte_out = F.embedding(x, params['wte.weight'])
wpe_out = F.embedding(torch.arange(0, 10), params['wpe.weight'])
embedding_out = wte_out + wpe_out

print(f"wte_out.shape:  {wte_out.shape}")
print(f"wpe_out.shape:  {wpe_out.shape}")
print(f"embedding_out.shape: {embedding_out.shape}")

print(wpe_out[0])
print(params['wpe.weight'][0])

print()

#print(params['wpe.weight'].shape)
#print(wte_out.shape)
#print(wte_out)


