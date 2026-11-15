#### This is the complete rewrite of the zerograd engine, the previous 2 branches had implemented GPT2 only and the architecture had a lot of flaws so that needs to be improved here. 
##### These are the steps at very high level that we'll be building

1. **Model definition:** We'll have the .json or .txt template file for every different family of LLM. It'll be a hand written file per model family
2. **Graph Construction:** Once we've the template file per model family, we'll take that along with the .config of the model it self and create 2 virtual graph (1 is for prefill and 1 is for decode) in memory using Tensor and Ops
3. **Weight Binding:** We'll take the graph from previous step and also take .gguf file and update all the weight's Tensor metadata.
4. **Shape Inference:** Once we've the graph poulated with Weight Tensors and also ops, we'll have to traverse the graph to populate the tensor metadata for all the activation tensors.
5. **Graph Optimization:**
    1. **Operator Fusion:** At this step, we'll have to fuse the operations to reduce memory round trips
6. **Device Placement:** At this step we'll be assigning the device ids to all the ops and we'll also be adding SEND/REC Ops inbetween if tensor/activation needs to travel from one device or memory to another
7. **Scheduling**: Needs topoloical sorting
8. **Kernel Selection (lowering/dispatch):** At this step we'll be using op type, dtype, quantization, device etc and assign the kernels to ops
9. **Memory Planning:** At this step, we'll have to plan memory needed for each device, weights memory + acitivations memory + KV Cache 
10. **Memory Allocation:** Create one arena per device at the planned size and set each Tensor's data pointer to arena + offset
11. **Weights Loading**: Use .gguf file and mmap the weights
12. **Execution:**
    1. Tokenize
    2. Prefill: Run the prefill graph on all T tokens, This also populated the KV Cache
    3. Sample: Pick the next token from logits (it depends on the sampling technique being used)
    4. Decode: Run the decode graph on 1 token, sample again and repeat until the stop token
    5. Detokenize: Token ids back to text
