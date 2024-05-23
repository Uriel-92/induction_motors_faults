 # Grad cam algorithm
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
import cv2
import numpy as np
import torch

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.Model(model.input, 
                             model.get_layer(last_conv_layer_name).output)
    last_conv_layer_output = grad_model(img_array)
    preds = model(img_array)
    
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    if(pred_index is None):
        pred_index = torch.argmax(preds[0])
    preds[:, pred_index].backward()
    
    
    grads = [v.value.grad for v in model.trainable_weights]
    grads = grads[4]
    pooled_grads = torch.mean(grads, dim=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    for i in range(last_conv_layer_output.shape[-1]):
        last_conv_layer_output[:,:,i] *= pooled_grads[i]
    
    heatmap = torch.mean(last_conv_layer_output, dim=-1).detach().cpu().numpy()
    
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= np.max(heatmap)
    
    return heatmap
    
    
   
        
model = keras.saving.load_model("best_model_torch.keras")
img = cv2.imread("faults/A10/032.bmp")
img = cv2.resize(img, (150, 150))
img = np.expand_dims(img, axis=0)
layer_name = "conv2d_35"

heatmap = make_gradcam_heatmap(img, model, layer_name)
heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.8 + img
cv2.imwrite('map.jpg', superimposed_img[0])