import random
import re
from pathlib import Path
from typing import Optional

import gradio as gr
import librosa
import lyricsgenius
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from diffusers.schedulers import LMSDiscreteScheduler
from spotdl import Spotdl
from spotdl.types import Song
from spotdl.utils.formatter import create_file_name
from stable_diffusion_videos import StableDiffusionWalkPipeline

from .paths import static_dir
from .settings import Settings
from .stable_diffusion import NoCheck

css = None
with open(static_dir / "styles.css") as f:
    css = f.read()

settings = Settings()  # type: ignore

if settings.spotify_client_id is None or settings.spotify_client_secret is None:
    print("Spotify client ID and secret not set")
    exit(1)


spotify = Spotdl(
    client_id=settings.spotify_client_id,
    client_secret=settings.spotify_client_secret,
    output=str(settings.audio_path),
)
genius = lyricsgenius.Genius(
    access_token=settings.genius_access_token, remove_section_headers=True
)

is_cuda = torch.cuda.is_available()
if not is_cuda:
    print("CUDA not available. Exiting.")
    exit(1)


def get_spec_norm(wav, sr, n_mels=512, hop_length=704):
    """Obtain maximum value for each time-frame in Mel Spectrogram,
    and normalize between 0 and 1
    Borrowed from lucid sonic dreams repo. In there, they programmatically determine
    hop length but I really didn't understand what was going on so I removed it and
    hard coded the output.
    """

    # Generate Mel Spectrogram
    spec_raw = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, hop_length=hop_length
    )

    # Obtain maximum value per time-frame
    spec_max = np.amax(spec_raw, axis=0)

    # Normalize all values between 0 and 1
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    return spec_norm


def get_timesteps_arr(wav, sr, duration, fps=30, margin=(1.0, 5.0)):
    """Get the array that will be used to determine how much to interpolate between images.
    Normally, this is just a linspace between 0 and 1 for the number of frames to
    generate. In this case,we want to use the amplitude of the audio to determine
    how much to interpolate between images.
    So, here we:
        1. Load the audio file
        2. Split the audio into harmonic and percussive components
        3. Get the normalized amplitude of the percussive component, resized to the
           number of frames
        4. Get the cumulative sum of the amplitude array
        5. Normalize the cumulative sum between 0 and 1
        6. Return the array
    I honestly have no clue what I'm doing here. Suggestions welcome.
    """
    _, wav_percussive = librosa.effects.hpss(wav, margin=margin)

    # Apparently n_mels is supposed to be input shape but I don't think it matters here?
    frame_duration = int(sr / fps)
    wav_norm = get_spec_norm(wav_percussive, sr, n_mels=512, hop_length=frame_duration)
    amplitude_arr = np.resize(wav_norm, int(duration * fps))
    T = np.cumsum(amplitude_arr)
    T /= T[-1]
    T[0] = 0.0
    return wav_norm, T


def plot_data(
    audio_file: str,
    plot: px.line,  # type: ignore
    start_time: int,
    end_time: int,
    sr: int = settings.default_sample_rate,
):
    """Updates plot and appropriate state variable when audio is uploaded or deleted."""
    # Default duration to 0 seconds
    duration_sec = 0
    # If the current audio file is deleted
    if audio_file is None:
        # Replace the state variable for the audio source with placeholder values
        sample_rate, audio_data = [sr, np.array([])]
        # Update the plot to be empty
        plot = px.line(labels={"x": "Time (s)", "y": ""})
        # Update the start and end time to be -1 (Gets updated later)
        start_time = -1
        end_time = -1
    # If new audio is uploaded
    else:
        audio_data, sample_rate = librosa.load(audio_file, mono=True, sr=sr)
        # Replace the current state variable with new
        duration_sec = len(audio_data) / sample_rate
        wav_norm, audio_data = get_timesteps_arr(audio_data, sample_rate, duration_sec)
        # Time array
        time = np.linspace(0, duration_sec, len(audio_data))

        # Plot the new data (fix the x-axis to be time)
        plot.add_trace(go.Scatter(y=audio_data, x=time, mode="lines", name="Audio"))
        plot.add_trace(
            go.Scatter(
                y=wav_norm,
                x=time,
                mode="lines",
                name="Normalized",
                line=go.scatter.Line(color="blue"),
            )
        )
        if start_time == -1:
            start_time = 0
        duration_sec = int(duration_sec)
        if end_time == -1:
            end_time = duration_sec
        plot.update_layout(xaxis_range=[start_time, end_time])

    # Update the plot component and data state variable
    return [
        plot,
        [sample_rate, audio_data],
        plot,
        gr.Slider.update(start_time, minimum=0, maximum=duration_sec - 1),
        gr.Slider.update(end_time, minimum=1, maximum=duration_sec),
    ]


def get_lyrics(song_name: str) -> tuple[str, str]:
    """Get lyrics from Genius API"""
    song = genius.search_song(song_name)
    if song is None:
        return "No lyrics found", song_name
    lyrics = song.lyrics
    # If the first line is .* Lyrics, remove it
    if lyrics.splitlines()[0].endswith("Lyrics"):
        song_name = lyrics.splitlines()[0].split(" Lyrics")[0]
        lyrics = "\n".join(lyrics.splitlines()[1:])
    # Remove any instance of r"\d+Embed"
    lyrics = re.sub(r"\d+Embed", "", lyrics)
    return lyrics, song_name


def search_songs(song_search_query: str) -> tuple[list[Song], dict]:
    """Search for songs using Genius API"""
    if song_search_query == "":
        return [], gr.Dropdown.update(choices=[])
    search_results = spotify.search([song_search_query])
    song_names: list[str] = []
    for song in search_results:
        song_names.append(f"{song.name} - {song.artist}")
    if len(song_names) == 0:
        return [], gr.Dropdown.update(choices=[])
    return search_results, gr.Dropdown.update(value=song_names[0], choices=song_names)


def get_random_seeds_for_prompts(prompts: list[str]) -> list[int]:
    return [random.randint(0, 2**32 - 1) for _ in prompts]


def get_interpolation_steps_from_audio_offsets(
    audio_offsets: tuple[int, int], fps: int = 30
) -> list[int]:
    return [(b - a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]


def generate_music_video(
    song_name: str,
    prompts: list[str],
    audio_file_path: str,
    audio_offsets: tuple[int, int],
    fps: int = 30,
    height: int = 680,
    width: int = 480,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 100,
    seed: Optional[int] = None,
    seeds: Optional[list[int]] = None,
    prompt_template: str = "{}",
) -> str:
    """Generate a music video using the given prompts and audio offsets"""
    sd_pipeline = StableDiffusionWalkPipeline.from_pretrained(
        settings.stable_diffusion_model,
        use_auth_token=True,
        torch_dtype=torch.float16,
        revision="fp16",
        scheduler=LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
    ).to("cuda")
    sd_pipeline.safety_checker = NoCheck().cuda()

    # Get the number of frames to interpolate between
    interpolation_steps = get_interpolation_steps_from_audio_offsets(audio_offsets, fps)

    # Format the prompts
    prompts = [prompt_template.format(prompt) for prompt in prompts]

    # Get the seeds for each prompt
    if seeds is None:
        if seed is None:
            # Get the random seeds for each prompt
            seeds = get_random_seeds_for_prompts(prompts)
        else:
            # Get the same seed for each prompt
            seeds = [seed] * len(prompts)

    # Generate the music video
    video_path = sd_pipeline.walk(
        name=song_name,
        prompts=prompts,
        seeds=seeds,
        num_interpolation_steps=interpolation_steps,
        audio_filepath=audio_file_path,
        audio_start_sec=audio_offsets[0],
        fps=fps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_dir=settings.output_path,
    )
    return video_path


with gr.Blocks(title="Stable Diffusion Music Videos", css=css) as demo:
    selected_song = gr.State(None)
    with gr.Row():
        # Display the selected song
        selected_song_display = gr.Markdown("## Selected Song: None\n*Genres:* None")

    with gr.Tab("Search for Song"):
        with gr.Row():
            song_search_query = gr.Textbox(
                "Do I Wanna Know?",
                label="Search for song (Spotify URL or song name)",
                placeholder="Blink 182 - All the Small Things",
            )
            song_search_button = gr.Button("Search", variant="primary")
            song_search_button.style(full_width=False)
        song_results = gr.State([])
        song_results_dropdown = gr.Dropdown(
            choices=[], label="Song Results", interactive=True
        )
        song_search_button.click(
            search_songs,
            inputs=[song_search_query],
            outputs=[song_results, song_results_dropdown],
        )

        selected_song_gallery = gr.Gallery(label="Selected Song")
        selected_song_gallery.style(grid=[2])  # type: ignore

        with gr.Row():
            # get_lyrics_button = gr.Button("Get Lyrics", variant="secondary")
            # get_lyrics_button.style(full_width=False)
            download_button = gr.Button("Download", variant="secondary")
            download_button.style(full_width=False)
            song_file = gr.File(label="Song File")

        with gr.Row():
            lyrics = gr.TextArea(
                label="Lyrics",
                placeholder="No lyrics found",
            )

        def download_song(
            selected_song: Optional[Song],
        ) -> tuple[Optional[Song], Optional[Path], Optional[str]]:
            """Download song"""
            if selected_song is None:
                return None, None, None
            selected_song, song_file_path = spotify.download(selected_song)
            if song_file_path is None:
                song_file_path = settings.audio_path / create_file_name(
                    selected_song, "{artists} - {title}.{output-ext}", "mp3"
                )
            lyrics, _ = get_lyrics(selected_song.name)
            return selected_song, song_file_path, lyrics

        download_button.click(
            download_song,
            inputs=[selected_song],
            outputs=[selected_song, song_file, lyrics],
        )

    def set_selected_song(
        song_results, song_results_dropdown
    ) -> tuple[Optional[Song], dict, dict, dict]:
        if song_results_dropdown is None:
            return (
                None,
                gr.Gallery.update([]),
                gr.Button.update(variant="secondary"),
                gr.Markdown.update("## Selected Song: None\n*Genres:* None"),
            )
        for song in song_results:
            song_name = f"{song.name} - {song.artist}"
            if song_name == song_results_dropdown:
                genres = ", ".join(song.genres)
                return (
                    song,
                    gr.Gallery.update([song.cover_url]),
                    gr.Button.update(variant="primary"),
                    gr.Markdown.update(
                        f"## Selected Song: {song_name}\n*Genres:* {genres}"
                    ),
                )
        return (
            None,
            gr.Gallery.update([]),
            gr.Button.update(variant="secondary"),
            gr.Markdown.update("## Selected Song: None\n*Genres:* None"),
        )

    song_results_dropdown.change(
        set_selected_song,
        inputs=[song_results, song_results_dropdown],
        outputs=[
            selected_song,
            selected_song_gallery,
            download_button,
            selected_song_display,
        ],
    )

    with gr.Tab("Audio"):
        plot = gr.State(px.line(labels={"x": "Time (s)", "y": ""}))
        file_data = gr.State(
            [settings.default_sample_rate, []]
        )  # [sample rate, [data]]

        # Audio file upload
        audio_file = gr.Audio(type="filepath", label="Audio File")

        # Plot the audio file
        audio_wave = gr.Plot(plot.value)

        # Select the start and end time of the audio file
        start_time = gr.Slider(value=-1, label="Start Time (s)")
        end_time = gr.Slider(value=-1, label="End Time (s)")

        # Refresh Plot button
        refresh_plot_btn = gr.Button("Refresh Plot")

        audio_file.change(
            fn=plot_data,
            inputs=[audio_file, plot, start_time, end_time],
            outputs=[audio_wave, file_data, plot, start_time, end_time],
        )
        refresh_plot_btn.click(
            fn=plot_data,
            inputs=[audio_file, plot, start_time, end_time],
            outputs=[audio_wave, file_data, plot, start_time, end_time],
        )

    with gr.Tab("Music Video"):
        song_name = gr.Textbox(
            "Do I Wanna Know?",
            label="Song Name",
            placeholder="Blink 182 - All the Small Things",
        )
        # Audio file upload
        audio_file = gr.Audio(type="filepath", label="Audio File")
        prompts = gr.Textbox(
            "I'm so happy",
            label="Prompts",
            placeholder="I'm so happy",
        )
        seeds = gr.Textbox(
            "1",
            label="Seeds",
            placeholder="1",
        )
        fps = gr.Slider(
            value=30,
            minimum=1,
            maximum=90,
            step=1,
            label="FPS",
        )
        width = gr.Slider(
            value=512,
            minimum=256,
            maximum=1024,
            step=32,
            label="Width",
        )
        height = gr.Slider(
            value=512,
            minimum=256,
            maximum=1024,
            step=32,
            label="Height",
        )
        start_time = gr.Number(
            value=0,
            label="Start Time (s)",
        )
        end_time = gr.Number(
            value=0,
            label="End Time (s)",
        )
        guidance_scale = gr.Slider(
            value=7.5,
            minimum=0,
            maximum=20,
            step=0.5,
            label="Guidance Scale",
        )
        num_inference_steps = gr.Slider(
            value=100,
            minimum=10,
            maximum=300,
            step=10,
            label="Num Inference Steps",
        )
        prompt_template = gr.Textbox(
            "{}",
            label="Prompt Template",
            placeholder="{}",
        )

        # Music Video button
        music_video_btn = gr.Button("Generate Music Video")

        video_output = gr.Video(
            label="Video Output",
            interactive=False,
        )

        def generate_music_video_fn(
            song_name: str,
            audio_file: str,
            prompts: str,
            seeds: str,
            fps: int,
            width: int,
            height: int,
            start_time: float,
            end_time: float,
            guidance_scale: float,
            num_inference_steps: int,
            prompt_template: str,
        ) -> str:
            """Generate Music Video"""
            video_path = generate_music_video(
                song_name=song_name,
                prompts=prompts.split("\n"),
                audio_file_path=audio_file,
                audio_offsets=[start_time, end_time],
                fps=fps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                prompt_template=prompt_template,
                seeds=seeds.split("\n"),
            )
            return video_path

        music_video_btn.click(
            generate_music_video_fn,
            inputs=[
                song_name,
                audio_file,
                prompts,
                seeds,
                fps,
                width,
                height,
                start_time,
                end_time,
                guidance_scale,
                num_inference_steps,
                prompt_template,
            ],
            outputs=[video_output],
        )
