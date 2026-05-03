import torch
from xlstm_adapter import XLSTMAdapter
import sys

def inspect_output():
    print("Loading adapter...")
    try:
        adapter = XLSTMAdapter(model_path="NX-AI/xLSTM-7b", device="cuda")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("Running forward pass...")
    input_ids = torch.tensor([[1, 2, 3]]).to("cuda") # Dummy input
    
    with torch.no_grad():
        # Call the underlying HF model directly to inspect raw output
        raw_output = adapter.model(input_ids=input_ids, output_hidden_states=True)
        
    print(f"Output type: {type(raw_output)}")
    print(f"Output keys: {raw_output.keys() if hasattr(raw_output, 'keys') else 'No keys'}")
    
    if hasattr(raw_output, 'cache_params'):
        print("✅ 'cache_params' found in output.")
        print(f"cache_params type: {type(raw_output.cache_params)}")
        print(f"Attributes: {dir(raw_output.cache_params)}")
        
        # Try to find what holds the data
        for attr in ['key_values', 'past_key_values', 'states', 'memory', 'param_dict']:
            if hasattr(raw_output.cache_params, attr):
                val = getattr(raw_output.cache_params, attr)
                print(f"Found attribute '{attr}': type {type(val)}")
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    print(f"  Content sample type: {type(val[0])}")
    else:
        print("❌ 'cache_params' NOT found in output.")
        if hasattr(raw_output, 'past_key_values'):
             print("⚠️ Found 'past_key_values' instead.")
    
    # Also check our adapter's return
    print("\nChecking Adapter.forward return...")
    try:
        hidden, state = adapter(input_ids=input_ids)
        print(f"Adapter returned state type: {type(state)}")
        if state is None:
             print("❌ Adapter returned None for state!")
        else:
             print("✅ Adapter state is not None.")
    except Exception as e:
        print(f"Adapter forward failed: {e}")

if __name__ == "__main__":
    inspect_output()
