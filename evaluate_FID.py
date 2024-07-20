
import os
import shutil
from pytorch_fid import fid_score

# 文件夹路径
path1 = 'picture-gene'
path2 = "picture-gene-onlygt"
temp_path1 = 'temppath'

# 创建临时文件夹
os.makedirs(temp_path1, exist_ok=True)

for filename in os.listdir(path1):
    if 'gt' not in filename:
        shutil.copy(os.path.join(path1, filename), os.path.join(temp_path1, filename))

fid_value = fid_score.calculate_fid_given_paths([temp_path1, path2], batch_size=50, device='cuda', dims=2048)

print('FID:', fid_value)

shutil.rmtree(temp_path1)

