import torch

def load_model (model_path, map_location = torch.device('cpu')) :

    model = torch.load(model_path, map_location=map_location)
    model.eval()

    return model
