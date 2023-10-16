import argparse
import numpy as np
import torch
from PIL import Image
import json

class Predict_class :
    @staticmethod
    def process_image(img_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
        img = Image.open(img_path)
        width,height = img.size
        aspect_ratio = width / height 
        if aspect_ratio > 1 :
            img = img.resize((round(aspect_ratio*256),256))
        else:
            img = img.resize((256,round(256/aspect_ratio)))
    
        #Crop
        width, height =img.size
        n_width = 224
        n_height = 224
        top = (height - n_height)/2
        right = (width + n_width)/2
        bottom = (height + n_height) / 2
        left = (width - n_width)/2
        img = img.crop((round(left),round(top),round(right),round(bottom)))
    
        # Convert channels, normalize, reorder dimmensions 
        np_img = np.array(img) /225
        np_img = (np_img - np.array([0.485,0.456,0.406])/np.array([0.229,0.224,0.225]))
        np_img = np_img.transpose((2,0,1))
        return np_img
    @staticmethod
    def predict(np_image, model,gpu,topk=5):
        ''' Predict the class (or classes) of an image using a trained machine learning model.'''
    
        device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    
        with torch.no_grad():
            imgs = torch.from_numpy(np_image)
            imgs = imgs.unsqueeze(0)
            imgs = imgs.type(torch.FloatTensor)
            imgs = imgs.to(device)
            out = model.forward(imgs)
            ps = torch.exp(out)
            pbs, inds = torch.topk(ps,topk)
            pbs = [float(pb) for pb in pbs[0]]
            inv_map = {val:key for key, val in model.class_to_idx.items()}
            clss = [inv_map[int(idx)] for idx in inds[0]]
        return pbs, clss

# Get the command line input
parser = argparse.ArgumentParser()


parser.add_argument('image_path', action='store',
                    default = 'flowers/test/1/image_06743.jpg',
                    help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')

parser.add_argument('checkpoint', action='store',
                    default = '.',
                    help='Directory of saved checkpoints, e.g., "assets"')

# Return top-k most likely classes
parser.add_argument('--top_k', action='store',
                    default = 5,
                    dest='top_k',
                    help='Return top KK most likely classes, e.g., 5')

# Use a mapping of categories to real names
parser.add_argument('--category_names', action='store',
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='File name of the mapping of flower categories to real names, e.g., "cat_to_name.json"')

# Use GPU for inference
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for inference, set a switch to true')

parse_results = parser.parse_args()

image_path = parse_results.image_path
checkpoint = parse_results.checkpoint
top_k = int(parse_results.top_k)
category_names = parse_results.category_names
gpu = parse_results.gpu

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
filepath = checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath, map_location='cpu')
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])

# Create an object of class predict
pred_obj = Predict_class()

# Image preprocessing
np_image = pred_obj.process_image(image_path)

# Predict class and probabilities
print(f"Predicting top {top_k} most likely flower names from image {image_path}.")

probs, classes = pred_obj.predict(np_image, model,gpu, top_k )
classes_name = [cat_to_name[class_i] for class_i in classes]

print("\nFlower name (probability): ")
print("*********")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("")