

layer_name_map = {
    "wte.weight": "wte",
    "wpe.weight": "wpe",
}

block_map = {
    "h.{i}.ln_1.weight": "h.{i}.ln1.weight",
    "h.{i}.ln_1.bias": "h.{i}.ln1.bias",
    "h.{i}.attn.bias": "h.{i}.attn.causal.mask",
    "h.{i}.attn.c_attn.weight": "h.{i}.qkv.weight",
    "h.{i}.attn.c_attn.bias": "h.{i}.qkv.bias",
    "h.{i}.attn.c_proj.weight": "h.{i}.attn.proj.weight",
    "h.{i}.attn.c_proj.bias": "h.{i}.attn.proj.bias",
    "h.{i}.ln_2.weight": "h.{i}.ln2.weight",
    "h.{i}.ln_2.bias": "h.{i}.ln2.bias",
    "h.{i}.mlp.c_fc.weight": "h.{i}.mlp.up.weight",
    "h.{i}.mlp.c_fc.bias": "h.{i}.mlp.up.bias",
    "h.{i}.mlp.c_proj.weight": "h.{i}.mlp.down.weight",
    "h.{i}.mlp.c_proj.bias": "h.{i}.mlp.down.bias",
}

for i in range(12):
    for src, dst in block_map.items():
        layer_name_map[src.format(i=i)] = dst.format(i=i)

layer_name_map["ln_f.weight"] = "ln.weight"
layer_name_map["ln_f.bias"] = "ln.bias"



from safetensors.torch import load_file

hf_path = "/Users/uonliaquat/Downloads/gpt2.safetensors"
zg_path = "/Users/uonliaquat/workspace/zerograd/my_model.safetensors"

tensors_hf = load_file(hf_path)
tensors_zg = load_file(zg_path)

max_diff_overall = -10
for key_hf, key_zg in layer_name_map.items():
    # key_hf = "wte.weight"
    # key_zg = layer_name_map[key_hf]

    # print(f"key_hf: {key_hf}, key_zg: {key_zg}\n\n")

    if key_hf not in tensors_hf:
        print("key_hf not found in tensors_hf\n")
        continue
    if key_zg not in tensors_zg:
        print("key_zg not found in tensors_zg\n")
        continue

    t_hf = tensors_hf[key_hf]
    t_zg = tensors_zg[key_zg]

    print(f"Name: t_hf:         {key_hf}    | t_zg: {key_zg}")
    print(f"hf.shape:           {list(t_hf.shape)}  | zg.shape: {list(t_zg.shape)}")
    print(f"Are shapes equal:   {list(t_hf.shape) == list(t_zg.shape)}")
    diff = abs((t_hf.flatten() - t_zg.flatten())).max().item()
    print(f"Diff of data:       {diff}")
    if diff > max_diff_overall:
        max_diff_overall = diff

print(f"\n\nmax_diff_overall: {max_diff_overall}")