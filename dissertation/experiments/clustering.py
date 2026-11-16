import os
import shutil
import numpy as np
from PIL import Image
import imagehash
from sklearn.cluster import KMeans

def cluster_images_by_hash(input_dir, output_dir_base, max_clusters=10):
    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    print(f"Found {len(images)} files in the directory.")
    valid_images = [img for img in images if Image.open(img).format in ['JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF']]
    print(f"Found {len(valid_images)} valid image files.")
    hashes = [imagehash.average_hash(Image.open(img)) for img in valid_images]
    hash_vectors = np.array([np.array(hash.hash, dtype=int).flatten() for hash in hashes])
    print(f"Generated {hash_vectors.shape[0]} valid hash vectors.")
    
    if hash_vectors.shape[0] == 0:
        print("No images found in the directory.")
        return []

    kmeans = KMeans(n_clusters=min(max_clusters, len(images)), random_state=0).fit(hash_vectors)
    labels = kmeans.labels_

    for label in set(labels):
        output_dir = os.path.join(output_dir_base, f"cluster_{label}")
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(images):
            if labels[i] == label:
                shutil.copy(img, os.path.join(output_dir, os.path.basename(img)))

    print(f"Created {len(set(labels))} clusters.")

cluster_images_by_hash("data/initial_capitals/specific", "data/initial_capitals/specific")
