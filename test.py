import wandb

# Your account or team name
entity = "chr-kuehberger-johannes-kepler-universit-t-linz"  # e.g., "janedoe" or "myteam"

# Your project name
project = "shelm-minigrid-memory-xlstm-final"

# Runs that logged per-env timesteps incorrectly
broken_runs = [
    "SHELM-18LayerTrXL-MiniGrid-MemoryS11-v0",
    "SHELM-2LayerTrXL-MiniGrid-MemoryS11-v0"
]

api = wandb.Api()

for run in api.runs(f"{entity}/{project}"):
    if run.name in broken_runs:
        print(f"Fixing {run.name} ...")
        # Fetch run history
        history = run.history(samples=1000000)
        # Log a new corrected metric
        for _, row in history.iterrows():
            if "_step" in row:
                run.log({
                    "total_timesteps": row["_step"] * 16
                })
print("Done! Use 'total_timesteps' as X-axis in W&B.")
