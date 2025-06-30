import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm


def analyze_differences(original_dir, new_dir, dataset_name, img_path):
    """
    Detailed analysis of differences between vector fields with visualization using log scale
    """
    # Load vector fields
    original = np.load(os.path.join(original_dir, f"{dataset_name}_vector_field.npy"))
    new = np.load(os.path.join(new_dir, f"{dataset_name}_vector_field.npy"))

    # Load original image
    img = Image.open(os.path.join(img_path, f"{dataset_name}_map.png"))
    img = np.array(img.convert('RGB'))[:, :, 0]
    img = img > 0.5
    img_padded = np.pad(img, ((img.shape[0] // 2,) * 2, (img.shape[1] // 2,) * 2),
                        'constant', constant_values=0)

    # Compute differences
    diff = np.sqrt(np.sum((original - new) ** 2, axis=-1))

    # Find points with large differences
    threshold = 10  # Consider differences larger than 10 pixels as significant
    large_diff_mask = diff > threshold

    # Get coordinates of points with large differences
    y_coords, x_coords = np.where(large_diff_mask)

    # Create visualization
    plt.figure(figsize=(20, 15))

    # Plot 1: Difference heatmap (log scale)
    plt.subplot(221)
    eps = 1e-10
    log_diff = np.log10(diff + eps)
    im = plt.imshow(log_diff)
    plt.colorbar(im, label='Log10 Distance difference (pixels)')
    plt.title(f'Difference Heatmap (Log Scale)\nMax diff: {np.max(diff):.2f}, Mean diff: {np.mean(diff):.4f}')

    # Plot 2: Difference heatmap (linear scale)
    plt.subplot(222)
    im = plt.imshow(diff)
    plt.colorbar(im, label='Distance difference (pixels)')
    plt.title('Difference Heatmap (Linear Scale)')

    # Plot 3: Histogram of log differences
    plt.subplot(223)
    non_zero_diff = diff[diff > eps]
    if len(non_zero_diff) > 0:
        plt.hist(np.log10(non_zero_diff), bins=50)
        plt.xlabel('Log10 Distance difference')
        plt.ylabel('Count')
        plt.title('Histogram of Differences (Log Scale)')

    # Plot 4: Original image with vectors
    plt.subplot(224)
    plt.imshow(img_padded, cmap='gray')

    # Plot vectors for points with large differences
    if len(y_coords) > 0:
        # Sample up to 100 points to avoid cluttering
        sample_size = min(100, len(y_coords))
        sample_indices = np.random.choice(len(y_coords), sample_size, replace=False)

        # Get the sampled coordinates
        sample_y = y_coords[sample_indices]
        sample_x = x_coords[sample_indices]

        # Original vectors
        orig_dest_y = original[sample_y, sample_x, 0]
        orig_dest_x = original[sample_y, sample_x, 1]

        # New vectors
        new_dest_y = new[sample_y, sample_x, 0]
        new_dest_x = new[sample_y, sample_x, 1]

        # Plot origin points
        plt.scatter(sample_x, sample_y, c='green', s=20, label='Origin points')

        # Plot original vectors (red)
        plt.quiver(sample_x, sample_y,
                  orig_dest_x - sample_x,
                  orig_dest_y - sample_y,
                  color='red', alpha=0.5, label='Original vectors',
                  angles='xy', scale_units='xy', scale=1)

        # Plot new vectors (blue)
        plt.quiver(sample_x, sample_y,
                  new_dest_x - sample_x,
                  new_dest_y - sample_y,
                  color='blue', alpha=0.5, label='New vectors',
                  angles='xy', scale_units='xy', scale=1)

        # Plot destination points
        plt.scatter(orig_dest_x, orig_dest_y, c='red', s=10, alpha=0.5, label='Original destinations')
        plt.scatter(new_dest_x, new_dest_y, c='blue', s=10, alpha=0.5, label='New destinations')

    plt.legend()
    plt.title('Vector Field Differences\nGreen: Origin, Red: Original, Blue: New')

    # Save the visualization
    plt.tight_layout()
    plt.savefig(f'analysis_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics about large differences
    if len(y_coords) > 0:
        print(f"\nDetailed analysis for {dataset_name}:")
        print(f"Number of points with difference > {threshold} pixels: {len(y_coords)}")

        # Get statistics for different ranges of differences
        ranges = [0, 1, 10, 50, 100, np.inf]
        for i in range(len(ranges)-1):
            mask = (diff >= ranges[i]) & (diff < ranges[i+1])
            count = np.sum(mask)
            if count > 0:
                percentage = (count / diff.size) * 100
                print(f"Differences {ranges[i]}-{ranges[i+1]}: {count} points ({percentage:.4f}%)")


def main():
    original_dir = "./datasets/vectorfield-orig"
    new_dir = "./datasets/vectorfield"
    img_path = "./datasets/image"

    datasets = [
        'seq_eth',
        'seq_hotel',
        'students003',
        'crowds_zara01',
        'crowds_zara02'
    ]

    for dataset in datasets:
        try:
            analyze_differences(original_dir, new_dir, dataset, img_path)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")


if __name__ == "__main__":
    main()
