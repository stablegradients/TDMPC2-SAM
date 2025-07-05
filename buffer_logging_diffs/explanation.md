# Buffer Saving and Resuming Functionality

This document explains the new functionality added to the TD-MPC2 codebase for saving and resuming training from replay buffers and model checkpoints.

## Changes Made

The following changes were implemented to support this functionality:

1.  **`tdmpc2/common/buffer.py`**:
    *   Added `save(path)` and `load(path)` methods to the `Buffer` class. These methods allow the replay buffer's state to be saved to and loaded from a file.
    *   The existing `load` method was renamed to `load_from_disk` to avoid confusion.

2.  **`tdmpc2/config.yaml`**:
    *   Added two new configuration parameters:
        *   `checkpoint_dir`: Specifies the directory where model and buffer checkpoints will be saved. Default is `null`.
        *   `resume_from_checkpoint`: Specifies the path to a model checkpoint file to resume training from. Default is `null`.

3.  **`tdmpc2/trainer/online_trainer.py`**:
    *   The `OnlineTrainer` now supports saving and loading checkpoints.
    *   **Saving**: During evaluation (`eval` method), if `checkpoint_dir` is set, the trainer will save the model state, and the `buffer` and `test_buffer` contents. The saved files are organized in the following directory structure:
        ```
        <checkpoint_dir>/<exp_name>/<seed>/
        ```
        The files are named with the current training step, e.g., `model_10000.pt`, `buffer_10000.pt`.
    *   **Loading**: If `resume_from_checkpoint` is specified, the trainer will load the model state and the corresponding replay buffers at the beginning of training. It infers the buffer paths from the model checkpoint path.

## Usage Examples

Here are some examples of how to use the new functionality:

### Saving Checkpoints

To save model and buffer checkpoints during training, specify the `checkpoint_dir` argument.

```bash
python train.py task=dog-run checkpoint_dir=/path/to/checkpoints
```

This will save checkpoints in `/path/to/checkpoints/default/1/` (assuming `exp_name` is `default` and `seed` is `1`).

### Resuming from a Checkpoint

To resume training from a specific checkpoint, use the `resume_from_checkpoint` argument.

```bash
python train.py task=dog-run resume_from_checkpoint=/path/to/checkpoints/default/1/model_50000.pt
```

This will load the model state from `model_50000.pt` and attempt to load the corresponding buffers `buffer_50000.pt` and `test_buffer_50000.pt` from the same directory.

You can also combine both arguments to resume from a checkpoint and continue saving new checkpoints.

```bash
python train.py task=dog-run checkpoint_dir=/path/to/checkpoints resume_from_checkpoint=/path/to/checkpoints/default/1/model_50000.pt
``` 