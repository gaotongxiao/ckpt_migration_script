# Introduction

This is a light toolbox that copy the weights across checkpoints in a rather intuitive way.

## Usage

1. Grab all available keys from source and target models and export them into key files.

```bash
python export_ckpt_keys.py from.pth from.txt
python export_ckpt_keys.py to.pth to.txt
```

2. Edit both key files and make sure a pair of the source key and target key in their original files share the same line number. The weights from source keys of the source model will be migrated to their corresponding target keys of the target model. Below is an example:

| line | from.txt | to.txt |
| ---  | --- | --- |
| 1 | proj.weight | decoder.proj.weight|
| 2 | token_encoder.pe | decoder.token_encoder.position_table |
| 3 | pos_encoder.pe | decoder.pos_encoder.position_table |

**Note: The final line counts of both key files have to be the same.**

3. Overwrite the weights in `to.pth` with the weights in `from.pth` according to the key files.

```bash
python export_ckpt_keys.py from.pth from.txt to.pth to.txt out.pth
```

Note that other keys that are missing from the migration remain the same as in `to.pth`, unless `--zero-missing-keys` is specified which resets all missing keys to zeros.
