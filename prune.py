import os
import torch
import torch_pruning as tp

from model import LightDepthNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading model for structural pruning...")

    model = LightDepthNet()
    model.load_state_dict(torch.load("checkpoints/best_model_relu.pth", map_location=device))
    model.to(device)
    model.eval()

    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=1)

    # Ignore the final conv AND the entire Encoder ---
    ignored_layers = [model.final_conv]
    
    # Iterate through the encoder and protect all its convolutions
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.Conv2d):
            ignored_layers.append(m)

    # Initialize Pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=0.20, 
        ignored_layers=ignored_layers, # Pass the expanded ignore list
    )

    base_macs, base_params = tp.utils.count_macs_and_params(model, example_inputs)

    print("Tracing graph and slicing decoder tensors...")
    pruner.step()

    pruned_macs, pruned_params = tp.utils.count_macs_and_params(model, example_inputs)

    print("\n" + "="*40)
    print(" PRUNING RESULTS (Decoder Only)")
    print("="*40)
    print(f"Parameters: {base_params/1e6:.2f}M  ->  {pruned_params/1e6:.2f}M ({(pruned_params/base_params)*100:.1f}%)")
    print(f"MACs/FLOPs: {base_macs/1e9:.2f}G   ->  {pruned_macs/1e9:.2f}G ({(pruned_macs/base_macs)*100:.1f}%)")
    print("="*40)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/pruned_lightdepthnet.pth"
    torch.save(model, save_path)
    print(f"\nPhysically pruned model saved to {save_path}")

if __name__ == "__main__":
    main()
