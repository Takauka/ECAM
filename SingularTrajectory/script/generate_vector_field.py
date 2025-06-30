import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


def compute_vector_field(img):
    """
    Compute vector field pointing to nearest non-zero pixel.
    Returns coordinates in the same format as the original implementation.
    """

    h, w = img.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # For each zero pixel, we want to find the nearest non-zero pixel.
    mask = ~img.astype(bool)  # Mask of zero pixels.

    # Compute distance transform for both original and inverted image.
    # This gives us the nearest walkable (zero) point to each foreground point.
    _, indices = distance_transform_edt(mask, return_indices=True)

    # Get the coordinates of nearest non-zero pixels.
    nearest_y = indices[0]
    nearest_x = indices[1]

    # For non-zero pixels, they should point to themselves.
    nearest_y[~mask] = y_coords[~mask]
    nearest_x[~mask] = x_coords[~mask]

    # Stack coordinates to match original format.
    vector_field = np.stack([nearest_y, nearest_x], axis=-1)

    return vector_field


def main(id):
    print(id)

    # Load image.
    import PIL.Image as Image
    img = Image.open(f'./datasets/image/{id}_map.png')
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, 0]
    img = img > 0.5
    img = img.astype(np.int32)

    # Pad the image.
    img_padded = np.pad(img, ((img.shape[0] // 2,) * 2, (img.shape[1] // 2,) * 2),
                        'constant', constant_values=0)
    print(img.shape, img_padded.shape)

    # Compute vector field.
    vector_field = compute_vector_field(img_padded)

    # Save the result.
    np.save(f"./datasets/vectorfield/{id}_vector_field.npy", vector_field)


if __name__ == "__main__":
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


    for dataset in DATASETS:
        main(id=dataset)
