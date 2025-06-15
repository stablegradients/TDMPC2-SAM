# SAM (Sharpness-Aware Minimization) Optimization for TDMPC2

This implementation adds support for SAM-based optimization for the world model in TDMPC2. SAM helps find flatter minima which can lead to better generalization.

## Usage

### Command Line Usage

To enable SAM optimization for the world model, use the `optimizer=SAM` argument:

```bash
# Use SAM optimizer with default rho=0.05
python train.py task=dog-run optimizer=SAM

# Use SAM optimizer with custom rho value
python train.py task=dog-run optimizer=SAM sam_rho=0.1

# Standard Adam optimizer (default behavior)
python train.py task=dog-run optimizer=Adam
```

### Configuration Parameters

Add these parameters to your configuration:

```yaml
# SAM optimization
optimizer: SAM      # Options: Adam, SAM
sam_rho: 0.05      # Sharpness-aware minimization rho parameter (only used when optimizer=SAM)
```

### Multi-task Training Example

```bash
# Use SAM for multi-task training
python train.py task=mt30 model_size=317 optimizer=SAM sam_rho=0.02
python train.py task=mt80 model_size=48 optimizer=SAM sam_rho=0.05
```

## How It Works

1. **SAM Wrapper**: When `optimizer=SAM` is specified, the world model optimizer is wrapped with the SAM optimizer
2. **Two-Step Process**: SAM uses a two-step optimization process:
   - **First step**: Compute gradients and move to a "sharpness-aware" point  
   - **Second step**: Recompute forward pass and gradients at the new point, then update parameters
3. **World Model Only**: SAM is applied only to the world model optimizer, not the policy optimizer
4. **Configurable Rho**: The `sam_rho` parameter controls the neighborhood size for sharpness-aware updates

## Technical Details

- **Gradient Computation**: SAM requires two forward-backward passes per update
- **Memory Overhead**: Minimal additional memory overhead (stores parameter perturbations)
- **Computation Cost**: Approximately 2x forward pass computation per update
- **Policy Update**: Policy updates continue to use standard Adam optimization

## Benefits

- **Better Generalization**: SAM finds flatter minima which often generalize better
- **Improved Robustness**: Can lead to more robust world models
- **Drop-in Replacement**: Easy to enable/disable via configuration

## Implementation Notes

The implementation ensures:
- Proper gradient flow and computation graph management
- Policy updates use original latent states (not SAM perturbed states)
- Gradient clipping is applied correctly in both SAM steps
- No interference with other optimizers or training components

## Performance Considerations

- Training time increases by ~50-100% due to additional forward pass
- Memory usage remains approximately the same
- Consider using smaller batch sizes if memory becomes a constraint 