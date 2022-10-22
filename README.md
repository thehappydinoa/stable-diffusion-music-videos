# Stable Diffusion Music Videos

This is a work in progress. It is not in a usable state yet.

## What is this?

This is a [gradio](https://gradio.app/) interface for stable diffusion generated music videos.
This demo is based on the [stable diffusion videos](https://github.com/nateraw/stable-diffusion-videos) project by [Nathan Raw](https://github.com/nateraw).
I have added a few features to the original project to automate the process.

## Features

- [x] Download songs from Spotify
- [x] Download lyrics from Genius
- [x] Analyze percussive content (e.g. drums) of songs
- [ ] Turn lyrics into a set of prompts
- [ ] Generate music videos from prompts
- [ ] Generate music videos from prompts combined with percussive content (Creates the effect of a music video with lyrics that match the beat of the song)

## Requirements

- [Python 3.10](https://www.python.org/downloads/)
- [Anaconda](https://www.anaconda.com/products/individual)
- [Genius Access Token](https://genius.com/api-clients)
- [Spotify Client ID and Secret](https://developer.spotify.com/dashboard/applications)

## Installation

### Clone

```bash
git clone ...
cd stable-diffusion-music-videos
```

### Setup

```bash
conda env create --file env.yml
conda activate sdmv
```

## Usage

To start the Gradio app, run:

```bash
python -m stable_diffusion_music_videos
```

## Contributing

### Setup

```bash
conda env create --file env.yml
conda activate sdmv
pre-commit install
```

### Live Reload

```bash
gradio stable_diffusion_music_videos/app.py
```

### Testing

```bash
pytest
```

### Linting

```bash
flake8
```

### Formatting

```bash
black .
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- [Gradio](https://gradio.app/)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Stable Diffusion Videos](https://github.com/nateraw/stable-diffusion-videos)

## Contact

Created by [@thehappydinoa](https://twitter.com/thehappydinoa) - feel free to contact me on Twitter!
