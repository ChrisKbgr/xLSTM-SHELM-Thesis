import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional

class XLSTMAdapter(nn.Module):
    def __init__(self, model_path="NX-AI/xLSTM-7b", device="cuda"):
        super().__init__()
        
        # 1. Configuration for 8-bit Quantization (Optimized for 1080 Ti)
        # 4-bit NF4 is slow on Pascal (1080 Ti) due to unoptimized kernels.
        # 8-bit (Int8) is faster and fits in VRAM (7B model ~7.5GB).
        
        
        print(f"Loading xLSTM from {model_path} in 8-bit...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False, # Enable Int8
            trust_remote_code=True,
            device_map=device, 
            low_cpu_mem_usage=True
        )

        # 2. Expose the backbone and config
        # xLSTM structure usually puts the transformer/recurrence in .model
        self.backbone = self.model.base_model 
        self.config = self.model.config
        
        # 3. Expose dimensions for SHELM
        # xLSTM-7b hidden size is 4096
        self.d_embed = self.config.hidden_size 
        
        # 4. Expose the embedding layer (SHELM needs this for 'word_embs')
        self.word_emb = self.model.get_input_embeddings()

        self.n_layer = self.model.config.num_hidden_layers

    def forward(self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.LongTensor] = None, cache_params=None, **kwargs):
        """
        Adapts SHELM's call signature to xLSTM.
        SHELM passes: inputs_embeds (Batch, Seq, Dim), mems (State)
        """
        outputs = self.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        cache_params=cache_params,
        output_hidden_states=True,
        **kwargs
        ) 

        
        # outputs.last_hidden_state: (B, L, H)
        # outputs.past_key_values: The new state (Cache object)
        return outputs.hidden_states[-1], outputs.cache_params

    def reset_cache(self, cache_params, dones):
        """
        Resets the cache for environments that are done.
        
        Args:
            cache_params: The current cache params object (xLSTMCache or similar)
            dones: Boolean array of shape (Batch_Size,) indicating which envs are done.
        """
        if cache_params is None:
            return
            
        # 1. Attempt to resolve 'rnn_state' if this is an xLSTMCache object
        # Based on inspection, we know 'rnn_state' exists.
        target_cache = cache_params
        if hasattr(cache_params, 'rnn_state'):
            target_cache = cache_params.rnn_state

        # 2. Heuristic to determine device from first tensor found
        def find_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.device
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    d = find_device(item)
                    if d: return d
            return None
        
        device = find_device(target_cache)
        if device is None:
            device = "cuda" # Fallback

        # 3. Prepare Dones Tensor
        if not isinstance(dones, torch.Tensor):
            dones_tensor = torch.tensor(dones, device=device, dtype=torch.bool)
        else:
            dones_tensor = dones.to(device).bool()

        # 4. Recursive Reset Function
        def zero_out(obj):
            if isinstance(obj, torch.Tensor):
                # Check compatibility - usually (Batch, ...) so dimension 0 should match dones
                if obj.shape[0] == dones_tensor.shape[0]:
                    obj[dones_tensor] = 0.0
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    zero_out(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    zero_out(item)
            elif hasattr(obj, 'state'): 
                 # Some variants might wrap state
                 zero_out(obj.state)
            elif hasattr(obj, 'rnn_state'):
                 # Handle nested xLSTMCache layers if structured that way
                 zero_out(obj.rnn_state)
        
        # 5. Apply Reset
        # print(f"[DEBUG] reset_cache called. Dones sum: {dones_tensor.sum().item()}/{dones_tensor.shape[0]}")
        zero_out(target_cache)