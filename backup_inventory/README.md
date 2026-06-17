# Backup Inventory

This directory records small, git-safe metadata for local state that will not survive if the machine is released.

Files:

- `model_cache/cache_summary.tsv`: Hugging Face cache directories, model ids, snapshot ids, sizes, and refs.
- `model_cache/blob_inventory.tsv`: blob filenames and sizes from `/scratch/hf_hub_cache`.
- `model_cache/snapshot_inventory.tsv`: snapshot entries and symlink targets.
- `docker_images.tsv`: local Docker tags, image IDs, sizes, and repo digest fields.
- `docker_image_inspect/*.json`: `docker image inspect` output for local benchmark images.
- `repo_file_index.tsv`: file index for this git repo excluding `.git`.
- `repo_dir_sizes.txt`: top-level repo directory sizes.
- `root_scripts/repro_qwen35_fp8_1k1k_tp8.sh`: legacy scratch runner that was not part of the final result bundles.

These files are not a backup of model weights or Docker layers. They are a restore checklist for the next machine or agent.
