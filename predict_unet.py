import argparse
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet_model import UNet 
import cv2

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=2)  # Asegúrate de que n_channels y n_classes sean correctos
    checkpoint = torch.load(model_path, map_location=device)
    
    # Eliminar la clave `mask_values` si está presente
    if 'mask_values' in checkpoint:
        del checkpoint['mask_values']
    
    model.load_state_dict(checkpoint)  # Carga el checkpoint en el modelo
    model.to(device)  # Mueve el modelo al dispositivo
    model.eval()  # Cambia el modelo al modo evaluación
    return model


def preprocess_image(image, scale_factor=1.0):
    img = image.convert('RGB')  
    transform = transforms.Compose([
        transforms.Resize((int(img.height * scale_factor), int(img.width * scale_factor))),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0) 
    return img_tensor

def predict_img(net, full_img, scale_factor=1.0, out_threshold=0.5, device='cuda'):
    img = preprocess_image(full_img, scale_factor) 
    img = img.to(device=device, dtype=torch.float32) 

    with torch.no_grad():
        output = net(img)  
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear', align_corners=True)  
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().cpu().squeeze().numpy()  

def save_mask(mask, output_filename):
    out_img = Image.fromarray(mask.astype(np.uint8))
    out_img.save(output_filename)

def visualize_prediction(image_path, prediction, alpha=0.5):
    img = cv2.imread(image_path) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    img_resized = cv2.resize(prediction, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    pred_mask_rgb = np.zeros_like(img_rgb) 
    pred_mask_rgb[img_resized > 0] = [255, 0, 0] 

    img_rgb = img_rgb.astype(np.uint8)
    pred_mask_rgb = pred_mask_rgb.astype(np.uint8)

    img_with_mask = cv2.addWeighted(img_rgb, 1 - alpha, pred_mask_rgb, alpha, 0)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Imagen Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title("Máscara de Segmentación")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_with_mask)
    plt.title("Imagen con Máscara Superpuesta")
    plt.axis('off')

    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Archivo del modelo')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Imágenes de entrada', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Imágenes de salida')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualizar las imágenes')
    parser.add_argument('--no-save', '-n', action='store_true', help='No guardar las máscaras')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.8, help='Umbral de la máscara')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Factor de escala para las imágenes')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Número de clases')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    in_files = args.input
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Cargando modelo {args.model}')
    model = load_model(args.model, device)
    logging.info('¡Modelo cargado!')
    
    for i, filename in enumerate(in_files):
        logging.info(f'Prediciendo la imagen {filename} ...')
        
        img = Image.open(filename)
        
        mask = predict_img(model, img, scale_factor=args.scale, out_threshold=args.mask_threshold, device=device)
        
        if args.viz:
            logging.info(f'Visualizando los resultados para {filename}, cierre para continuar...')
            visualize_prediction(filename, mask)
