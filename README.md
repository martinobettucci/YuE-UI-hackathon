# YuE - UI

Welcome to YuE - UI, an interface for the YuE music generation model.\
The focus is on creativity, with an incremental song generation workflow using batch, select and continue.\
Low VRAM 6-8 GB is supported as well!

The project uses YuE-exllamav2 for music generation [sgsdxzy/YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2).\
The interface was started from [alisson-anjos/YuE-Interface](https://github.com/alisson-anjos/YuE-Interface).\
The official YuE repo can be found here [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
 
## Features
- Batch, select & continue
- Rewind
- Choose whether to generate Stage 1, 2, 1+2 or preview cache
- Import vocal & instrumental tracks
- Visualize generated song segments
- Control song segment token distribution
- Mute selected segments to steer the generation process and save VRAM
- Load & save

## UI preview
![preview ui](/preview.png)

## Installation
Windows:
```bash
git clone https://github.com/joeljuvel/YuE-UI/
cd YuE-UI
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
python -m venv venv
./venv/scripts/activate
pip install -r requirements.txt
```

Linux:
```bash
git clone https://github.com/joeljuvel/YuE-UI/
cd YuE-UI
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running
Place your stage 1 & 2 model folders in the YuE exl2 `models/` folder.\
E.g.\
`models/YuE-s1-7B-anneal-en-cot-exl2/`\
`models/YuE-s2-1B-general-exl2/`

Windows:\
Run `start_windows.bat` to start the gradio server

Linux:\
Run `./start_linux.sh` to start the gradio server

After starting the gradio server open http://127.0.0.1:7860 in your browser to access the UI.

## Generating music
### Genre tags
Try to describe the music as accurately as possible using genre tags. A stable prompt should typically contain 5 components: genre, instrument, mood, gender, and timbre.\
There's a provided list of 200 common tags, of course you can try your own tags as well! The order of tags is flexible.\
You can also try adding or removing tags as you generate different sections of the song to change the outcome.

### Lyrics
The lyrics must be structured in named segments. Each segment starts with a name that must be surrounded by square brackets \[\].\
An example \[verse\] segment
```
[verse]
Running in the night
My heart beats like a drum
I'm searching for the light
In this city so cold
```
You don't need the lyrics for the entire song from the start, add new lyrics segments as you go. The important thing to keep in mind is that you are always moving forward.\
Avoid removing earlier segments since this will bring the lyrics out of sync with the already generated song.\
You can always change lyrics or even rename segments since this won't change the past structure.

https://github.com/user-attachments/assets/a161d135-f14d-43fa-855e-3c4b64c6784a

You can find the above example song in the `examples/` folder.

## Tips & tricks
Try to maximize the stage 1 cache for best quality. If it's too small, earlier data will not be available to the model. But if it's too large, the generation will be excessively slow.
On a 8 GB card you can use a setting somewhere between 7000-7500 with a 4 bit quantized model+cache, this will allow you to fit around 60s of audio.

Switch between models e.g. you can use the ICL model with an audio prompt to generate one section, then disable the audio prompt and switch to another model.\
On the stage 1 settings tab you have several familiar control options e.g. CFG.

If you're trying to end the song with an outro but it keeps going, try using an \[end\] segment.

You can modify the start and stop time of segments in the timeline view. Optionally you can delete the last segment or split it in two.\
Any segment, except the last one, can be muted by selecting it in the timeline and toggle mute. This can be used to guide the generation towards a specific result.\
It will also help you keep VRAM under control by only having the necessary segments loaded.

If you only like the start of a clip, fear not, you can still accept it and move the end of the last segment to remove the unwanted part before the next generation. You can otionally use rewind as well.

Stage 1 is great for generating drafts since it's much faster than 1+2, although much lower quality. But remember to run stage 2 now and then to avoid surprises in the end 😁

The stage 2 refiner works in block of 6 seconds. The last section of the song might sound a bit off if the song length isn't exactly divisible by 6.

It's recommended to always use the dual track mode when working with audio prompts for best quality. You can substitute vocal or instrumental with a silence track if necessary.
If you need to split stems you can use a tool such as [deezer/spleeter](https://github.com/deezer/spleeter) to separate the vocal & instrumental tracks.

## Troubleshooting
If the gradio server fails start with `errno 111` on linux. You can try using an alternative ip address other than localhost.\
Replace n.n.n.n with the new address and optionally change the port in the command below.
```bash
./start_linux.sh --server_name n.n.n.n --server_port 7860
```

## Credits
- YuE Official [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
- YuE-exllamav2 [sgsdxzy/YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2)
- YuE Interface [alisson-anjos/YuE-Interface](https://github.com/alisson-anjos/YuE-Interface)
