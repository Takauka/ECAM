import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_vector_field_sample(vector_field_path, image_path, sample_size=100):
    """
    Visualizes a sample of vectors from the vector field on top of the corresponding image.

    Args:
    vector_field_path (str): Path to the .npy file containing the vector field.
    image_path (str): Path to the image file.
    sample_size (int): Number of vectors to sample for visualization.
    """

    # Load vector field.
    vector_field = np.load(vector_field_path)

    # Load image.
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, 0]
    img = img > 0.5
    img = img.astype(np.int32)

    # Pad the image (as done in your original code).
    img_padded = np.pad(img, ((img.shape[0] // 2,) * 2, (img.shape[1] // 2,) * 2),
                        'constant', constant_values=0)

    # Randomly select sample points.
    h, w, _ = vector_field.shape
    indices = np.random.choice(h * w, sample_size, replace=False)
    y_coords, x_coords = np.unravel_index(indices, (h, w))

    # Get destination points.
    dest_y = vector_field[y_coords, x_coords, 0].astype(int)
    dest_x = vector_field[y_coords, x_coords, 1].astype(int)

    # Create visualization.
    plt.figure(figsize=(10, 10))
    plt.imshow(img_padded, cmap='gray')

    # Plot origin points.
    plt.scatter(x_coords, y_coords, c='green', s=20, label='Origin points')

    # Plot vectors.
    plt.quiver(x_coords, y_coords,
               dest_x - x_coords,
               dest_y - y_coords,
               color='red', alpha=0.7, angles='xy', scale_units='xy', scale=1)

    # Plot destination points.
    plt.scatter(dest_x, dest_y, c='red', s=10, alpha=0.7, label='Destination points')

    plt.legend()
    plt.title('Sample of Vectors from Vector Field')

    # Extract dataset name from file path for saving.
    dataset_name = os.path.basename(vector_field_path).split('_vector_field')[0]
    plt.savefig(f'vector_field_sample_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    DATASETS = [
        'seq_eth',
        'seq_hotel',
        'students003',
        'crowds_zara01',
        'crowds_zara02',

        'bookstore_0',
        'bookstore_1',
        'bookstore_2',
        'bookstore_3',
        'coupa_0',
        'coupa_1',
        'coupa_3',
        'deathCircle_0',
        'deathCircle_1',
        'deathCircle_2',
        'deathCircle_3',
        'deathCircle_4',
        'gates_0',
        'gates_1',
        'gates_2',
        'gates_3',
        'gates_4',
        'gates_5',
        'gates_6',
        'gates_7',
        'gates_8',
        'hyang_0',
        'hyang_1',
        'hyang_3',
        'hyang_4',
        'hyang_5',
        'hyang_6',
        'hyang_7',
        'hyang_8',
        'hyang_9',
        'little_0',
        'little_1',
        'little_2',
        'little_3',
        'nexus_0',
        'nexus_1',
        'nexus_2',
        'nexus_3',
        'nexus_4',
        'nexus_5',
        'nexus_6',
        'nexus_7',
        'nexus_8',
        'nexus_9',
        'quad_0',
        'quad_1',
        'quad_2',
        'quad_3',

        'Exp_1_run_1',
        'Exp_1_run_2',
        'Exp_1_run_3',
        'Exp_1_run_4',
        'Exp_2_run_1',
        'Exp_2_run_2',
        'Exp_2_run_3',
        'Exp_2_run_4',
        'Exp_2_run_5',
        'Exp_3_run_1',
        'Exp_3_run_2',
        'Exp_3_run_3',
        'Exp_3_run_4',
    ]

    DATASETS = DATASETS \
             + [ str(i) for i in range(40) ] \
             + [ '42', '44' ] \
             + [ '43', '47', '48', '49' ]


    vector_field_dir = "./datasets/vectorfield"
    image_dir = "./datasets/image"

    for dataset in DATASETS:
        vector_field_path = os.path.join(vector_field_dir, f"{dataset}_vector_field.npy")
        image_path = os.path.join(image_dir, f"{dataset}_map.png")
        visualize_vector_field_sample(vector_field_path, image_path)

if __name__ == "__main__":
    main()
