import os
import torch
import librosa
import numpy as np
from swift_f0 import SwiftF0
from svara_representation import Model, Embedder, smooth_pitch_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 48
depth = 5
num_classes = 7

# Initialize model and load encoder weights
model = Model(embed_dim=embed_dim, num_classes=num_classes, depth=depth).to(device)
model.encoder.load(os.path.join("svara_representation", "checkpoint", "svara_encoder.pth"), device)
model.eval()

# Create a simple logger for the embedder
class Logger:
    def info(self, msg):
        print(msg)

logger = Logger()
embedder = Embedder(model, logger)


def create_pitch_curve(audio_path=None, time_series=None, pitch_series=None, tonic=None):
    """
    Create a pitch curve from raw pitch data or process it from audio.
    
    Args:
        audio_path: Path to audio file (not implemented - use external pitch extraction)
        time_series: Array of time values
        pitch_series: Array of pitch values (Hz)
        tonic: Tonic frequency for normalization
    
    Returns:
        Processed pitch curve in cents relative to tonic
    """
    if time_series is not None and pitch_series is not None:
        # Smooth the pitch curve
        smoothed_pitch = smooth_pitch_curve(time_series, pitch_series, smoothing_factor=0.5, min_points=4)
        
        # Convert to cents relative to tonic if provided
        if tonic is not None:
            pitch_cents = 1200 * np.log2(np.array(smoothed_pitch) / tonic)
        else:
            pitch_cents = smoothed_pitch
            
        return pitch_cents
    else:
        raise ValueError("Please provide time_series and pitch_series")


def extract_embedding(pitch_curve, device=device):
    """
    Extract embedding from a single pitch curve.
    
    Args:
        pitch_curve: Numpy array of pitch values
        device: torch device
    
    Returns:
        Embedding vector
    """
    model.eval()
    with torch.no_grad():
        # Prepare input: [batch, channels, sequence]
        pitch_tensor = torch.tensor(pitch_curve, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Create silence mask (NaN values indicate silence)
        silence_mask = torch.isnan(pitch_tensor).float()
        pitch_tensor = torch.nan_to_num(pitch_tensor, nan=0)
        
        # Concatenate pitch and silence mask
        input_tensor = torch.cat([pitch_tensor, silence_mask], dim=1)
        
        # Get embedding
        embedding = model(input_tensor)
        
    return embedding.cpu().numpy()


# Example usage
if __name__ == "__main__":
    # Load audio from Data/Yaman/yaman-scale_1.mp3
    audio_path = "Data/Yaman/yaman-scale_1.mp3"
    y, sr = librosa.load(audio_path, sr=None)

    # Extract pitch using SwiftF0
    detector = SwiftF0(
        fmin=46.875, 
        fmax=2093.75,
        confidence_threshold=0.85
    )
    pitch_result = detector.detect_from_array(y, sr)

    # Get time and pitch series from result
    time_series = pitch_result.timestamps
    pitch_series = pitch_result.pitch_hz

    # Tonic for Yaman (typically around D4, adjust as needed)
    tonic = 293.66  # D4

    # Create pitch curve
    pitch_curve = create_pitch_curve(
        time_series=time_series,
        pitch_series=pitch_series,
        tonic=tonic
    )
    
    print(f"Pitch curve shape: {pitch_curve.shape}")
    
    # Extract embedding
    embedding = extract_embedding(pitch_curve)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")
