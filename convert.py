import os
import sox
from tqdm import tqdm
source_dir = "/home/users/nus/e1342275/TSVAD_pytorch/DIHARD\ 3/third_dihard_challenge_dev/data/flac"
target_dir = "/home/users/nus/e1342275/TSVAD_pytorch/DIHARD\ 3/third_dihard_challenge_dev/data/wav"

    # Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Create a transformer object
tfm = sox.Transformer()

# Loop through all FLAC files in the source directory
for filename in tqdm(os.listdir(source_dir)):
    if filename.endswith(".flac"):
        # Construct full file paths
        flac_path = os.path.join(source_dir, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(target_dir, wav_filename)

        # Convert FLAC to WAV
        tfm.build(flac_path, wav_path)

        print(f"Converted {filename} to {wav_filename}")

print("Conversion Complete")
