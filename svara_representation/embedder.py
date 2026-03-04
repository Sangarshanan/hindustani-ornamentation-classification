import numpy as np
import torch

class Embedder:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def __call__(self, data_loader):
        embeddings = self._propagate(data_loader)
        return embeddings

    def _propagate(self, data_loader):
        self.model.eval()

        embeddings = np.array([])
        for i, (input) in enumerate(data_loader):
            embedding = self._embed(input).detach().cpu().numpy()
            embeddings = np.concatenate((embeddings, embedding), axis=0) if embeddings.size else embedding

        return embeddings

    def _embed(self, pitch):
        silence_mask = (torch.isnan(pitch)).float()
        pitch = torch.nan_to_num(pitch, nan=0)
        input = torch.cat([pitch, silence_mask], dim=1)

        return self.model(input)