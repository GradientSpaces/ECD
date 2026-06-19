import ogbench

envs = [
    'antmaze-giant-stitch-v0',
    'antmaze-large-stitch-v0',
    'antmaze-medium-stitch-v0',
    'humanoidmaze-giant-stitch-v0',
    'humanoidmaze-large-stitch-v0',
    'humanoidmaze-medium-stitch-v0',
    'pointmaze-giant-stitch-v0',
    'pointmaze-large-stitch-v0',
    'pointmaze-medium-stitch-v0',
]

print("Downloading OGBench datasets to ~/.ogbench/data (if not already present)...")
ogbench.download_datasets(envs, dataset_dir='~/.ogbench/data')
print("Done")