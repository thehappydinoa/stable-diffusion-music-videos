import os.path
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    output_path: Path = Field(
        default=os.path.join(os.path.dirname(__file__), "..", "outputs")
    )
    audio_path: Path = Field(
        default=os.path.join(os.path.dirname(__file__), "..", "audio")
    )

    default_sample_rate: int = Field(default=22050)
    stable_diffusion_model: str = Field(
        default="CompVis/stable-diffusion-v1-4", env="STABLE_DIFFUSION_MODEL"
    )

    genius_access_token: Optional[str] = Field(env="GENIUS_ACCESS_TOKEN")
    spotify_client_id: Optional[str] = Field(env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: Optional[str] = Field(env="SPOTIFY_CLIENT_SECRET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
