import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Configuration specific for GTX 1080 Ti
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # Pascal GPUs (1080 Ti) prefer float32 compute. 
    # float16 is very slow on Pascal!
    bnb_4bit_compute_dtype=torch.float32, 
    bnb_4bit_use_double_quant=True,
)

print("Attempting to load xLSTM-7b on GTX 1080 Ti...")

try:
    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        "NX-AI/xLSTM-7b",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Model loaded successfully!")
    
    # 3. Check VRAM usage
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ VRAM Used: {mem_used:.2f} GB")
    
    # 4. Quick Inference Test
    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b", trust_remote_code=True)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
    
    # Run a tiny forward pass to ensure the custom kernels work
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)
    print("✅ Forward pass successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Tip: If this failed with a CUDA error, update your bitsandbytes: pip install -U bitsandbytes")