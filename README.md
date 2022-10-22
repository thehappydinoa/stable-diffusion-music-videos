# Stable Diffusion Music Videos

## Features

<!-- TODO: Add features here -->

## Requirements

- [Python 3.10](https://www.python.org/downloads/)
- [Anaconda](https://www.anaconda.com/products/individual)

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

Created by [@thehappydinoa](https://twitter.com/thehappydinoa) - feel free to contact me!
