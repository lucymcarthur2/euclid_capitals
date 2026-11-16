import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

model_path = r"data\models\best_1000_images.h5"
model = load_model(model_path)



# Example usage
layer_names = ['conv2d_4', 'conv2d_6']  # Selected layers to visualize
img_path = r"data\extracted_letters\L\extracted_object_32811_188.jpg_1_processed.png"




import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def plot_selected_activations(model, img_path, layer_names, filter_indices, img_size=(128, 128)):
    # Load the image
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create a model that outputs the activations from the specified layers
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get the activations
    activations = activation_model.predict(img_array)
    
    for layer_name, activation in zip(layer_names, activations):
        plt.figure(figsize=(20, 10))
        for i, filter_index in enumerate(filter_indices):
            filter_img = activation[0, :, :, filter_index]
            filter_img -= filter_img.mean()
            filter_img /= (filter_img.std() + 1e-7)
            filter_img *= 64
            filter_img += 128
            filter_img = np.clip(filter_img, 0, 255).astype('uint8')
            
            plt.subplot(1, len(filter_indices), i + 1)
            plt.title(f"{layer_name} - Filter {filter_index}", fontsize=46)
            plt.imshow(filter_img, cmap='viridis')
            plt.axis('on')
            plt.xlabel('Feature Map Width', fontsize=40)
            
            #
            plt.ylabel('Feature Map Height', fontsize=40)

            plt.xticks(fontsize=40)
            plt.yticks(fontsize=40)
        plt.tight_layout()
        plt.show()

filter_indices = [0, 1]  # Indices of the filters to visualize

plot_selected_activations(model, img_path, layer_names, filter_indices)
