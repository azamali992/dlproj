import os
import shutil
import time
import kagglehub

# set paths
current_dir = os.path.dirname(__file__)
raw_data_dir = os.path.join(current_dir, "raw")
os.makedirs(raw_data_dir, exist_ok=True)

print("Downloading via kagglehub...")
# Retry for large downloads that may fail on unstable connections.
max_attempts = 5
base_wait_seconds = 5
download_path = None

for attempt in range(1, max_attempts + 1):
    try:
        # dl latest version
        download_path = kagglehub.competition_download('aptos2019-blindness-detection')
        break
    except Exception as exc:
        if attempt == max_attempts:
            raise RuntimeError(
                f"Failed to download after {max_attempts} attempts. Last error: {exc}"
            ) from exc

        wait_seconds = base_wait_seconds * (2 ** (attempt - 1))
        print(
            f"Download failed on attempt {attempt}/{max_attempts}: {exc}. "
            f"Retrying in {wait_seconds}s..."
        )
        time.sleep(wait_seconds)

if download_path is None:
    raise RuntimeError("Download did not complete and no path was returned.")

print(f"Downloaded to: {download_path}")
print("Copying to raw folder...")

# copy to raw dir
for item in os.listdir(download_path):
    s = os.path.join(download_path, item)
    d = os.path.join(raw_data_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Done. Data ready in {raw_data_dir}")