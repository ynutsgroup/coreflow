import os
import re
import shutil

src_dir = "/opt/coreflow/data/dukascopy/dukascopy_ticks"
dst_root = "/opt/coreflow/data/dukascopy/EURUSD"

pattern = re.compile(r"EURUSD_(\d{4})(\d{2})(\d{2})_(\d{2})\.bi5")

for filename in os.listdir(src_dir):
    match = pattern.match(filename)
    if match:
        year, month, day, hour = match.groups()
        dst_dir = os.path.join(dst_root, year, month, day)
        os.makedirs(dst_dir, exist_ok=True)
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, f"{hour}h_ticks.bi5")
        shutil.move(src_path, dst_path)
        print(f"✅ Verschoben: {filename} -> {dst_path}")
