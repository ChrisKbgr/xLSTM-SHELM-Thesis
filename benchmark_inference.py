import time
import argparse
import numpy as np
import torch
import os
import random
from tqdm import tqdm

# Need a dummy env action space for the policy constructor
class DummyActionSpace:
    def __init__(self, n):
        self.n = n

def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Overhead")
    parser.add_argument("--model", type=str, default="SHELM", choices=["SHELM", "HELM"], 
                        help="Model class to instantiate (usually SHELM)")
    parser.add_argument("--arch", type=str, choices=["xLSTM", "TrXL"], required=True, 
                        help="The underlying architecture of the model in this folder (xLSTM or TrXL)")
    parser.add_argument("--n_steps", type=int, default=2048, 
                        help="Number of rollout steps to simulate")
    parser.add_argument("--n_envs", type=int, default=1, 
                        help="Number of parallel environments (batch size)")
    args = parser.parse_args()

    # Model and environment parameters exactly like psychlab
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    action_space = DummyActionSpace(11) # Psychlab discrete 11
    input_dim = (3, 80, 80) # Psychlab obs shape

    optimizer = "Adam" # Doesn't matter for inference
    lr = 1e-4

    print(f"Initializing {args.model} model...")
    if args.model == "SHELM":
        from model import SHELM
        import warnings
        warnings.filterwarnings("ignore") # Ignore huggingface warnings
        
        policy = SHELM(
            action_space=action_space, 
            input_dim=input_dim, 
            optimizer=optimizer, 
            learning_rate=lr, 
            env_id='psychlab_continuous_recognition', 
            topk=1, 
            device=device, 
            mem_len=511, 
            clip_encoder='ViT-B/16'
        ).to(device)
    else:
        from model import HELM
        policy = HELM(
            action_space=action_space, 
            input_dim=input_dim, 
            optimizer=optimizer, 
            learning_rate=lr, 
            beta=1.0, 
            device=device
        ).to(device)
    
    policy.eval() # Set to evaluation mode

    print(f"Model {args.model} initialized. Starting Benchmark for {args.n_steps} steps (Seq Length = 1 per step)")

    # Prepare dummy observations (scaled 0-1 like in the trainer)
    dummy_obs_np = np.random.rand(args.n_envs, *input_dim).astype(np.float32)
    observations = torch.tensor(dummy_obs_np).to(device)

    # Setup Memory
    if args.arch == "xLSTM":
        last_mems = policy.init_memory(args.n_envs)
    else:
        # TrXL memory init logic (works for both HELM and original SHELM)
        # Check if policy.model has d_embed and n_layer (HuggingFace object)
        d_embed = getattr(policy.model, "d_embed", policy.model.config.d_embed)
        n_layer = getattr(policy.model, "n_layer", policy.model.config.n_layer)
        last_mems = [torch.zeros((policy.mem_len, args.n_envs, d_embed)).to(device)
                               for _ in range(n_layer)]

    # Dummy dones
    dones = np.zeros(args.n_envs, dtype=bool)

    # WARMUP RUN
    print("Warming up (10 steps)...")
    with torch.no_grad():
        for _ in range(10):
             policy.memory = last_mems
             action, value, log_prob, hidden = policy(observations)
             last_mems = policy.memory
             
    # PRE-BENCHMARK MEMORY RESET 
    # (To emulate the TransfoXL starting with empty cache and accumulating up to mem_len)
    if args.arch == "xLSTM":
        last_mems = policy.init_memory(args.n_envs)
    else:
        last_mems = [torch.zeros((policy.mem_len, args.n_envs, d_embed)).to(device)
                               for _ in range(n_layer)]

    torch.cuda.synchronize() if device == "cuda" else None
    
    # ACTUAL BENCHMARK
    print(f"Starting actual benchmark loop for {args.n_steps} steps...")
    start_time = time.time()
    
    for step in tqdm(range(args.n_steps), desc="Benchmarking Inference", unit="step"):
        with torch.no_grad():
            policy.memory = last_mems
            action, value, log_prob, hidden = policy(observations)
            last_mems = policy.memory
            
            # Emulate the continuous accumulation (we do NOT reset memory here)
            # because we want TransfoXL to hit mem_len and do the sliding window!
            
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = (args.n_steps * args.n_envs) / total_time
    
    print("="*40)
    print(f"BENCHMARK RESULTS: {args.model}")
    print(f"Total Steps: {args.n_steps} (Batch Size/Envs: {args.n_envs})")
    print(f"Total Wall-Clock Time: {total_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} steps per second")
    print("="*40)

if __name__ == '__main__':
    main()
