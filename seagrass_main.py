# -*- coding: utf-8 -*-
"""
Training module for seagrass image recognition.

next step: tuning hyperparams.
"""


from training import train_seagrass_model
import os
    
def gpu_check():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available. Training will run on CPU.")
    
def set_precision(level):
    torch.set_float32_matmul_precision(level)


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def main():
    model_name = "densenet-201"
    batch_size = 128
    learning_rate =2e-5
    max_epochs = 20
    train_transects = [5,9,10,14,20]
    test_transects = [12]
    data_folder = "C:\\Users\\obgib\\Documents\\CERI\\final_dataset\\data_by_transect"

    train_seagrass_model(model_name, batch_size, learning_rate, max_epochs, 
                        train_transects, test_transects, data_folder)
    
if __name__ == '__main__':
    main()
    
