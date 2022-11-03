from mem_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from mem_pytorch import TritonTransformer

import torch
import numpy as np

import bittensor as bt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(dim: int, depth: int, heads: int, seq_len: int) -> torch.nn.Module:
    model = TritonTransformer(
        num_tokens=50257,
        dim=dim,
        max_seq_len=seq_len,
        depth=depth,
        heads=heads,
        causal=True,
        use_triton=False,
        # q_bucket_size=1024,
        # k_bucket_size=2048,
        # ff_chunks=5,
        # use_flash_attn=True,
    )

    model = AutoregressiveWrapper(model)

    pre_model = torch.load("./mem-1.3b/1.3b_20000.pt")

    model.load_state_dict(pre_model)
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Number of parameters: {params:,}")

    return model


# load in model

model = create_model(2048, 24, 24, 16_384)

x = input("Enter a sentence: ")

seq_len = 1024
tokenizer = bt.tokenizer()

inputs = tokenizer.encode(x)
inputs = torch.tensor(inputs).unsqueeze(0).to(device)

outputs = model.generate(inputs, seq_len=seq_len)
print(outputs)

decoded = tokenizer.decode(outputs.tolist()[0])
print(decoded)