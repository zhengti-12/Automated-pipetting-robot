# Automated Pipetting Robot — Well Detection

Computer-vision program that finds the wells in a microplate from a camera image, so an automated pipetting robot knows where each well's opening is.

Given a photo (or a live camera frame) of a well plate, it returns the pixel position and radius of every well opening, labelled with the plate's own well names (A1, A2, …), plus an annotated image for checking the result.

It is designed to work on **clear plastic plates**, where each well shows two rims (the top opening and the floor) and where a camera that isn't exactly overhead shifts the openings by perspective.

## How it works

When the plate layout is known (`--rows` / `--cols`), detection runs in four stages:

1. **Grid search.** Builds a "circular edge" map of the image and searches over well size, spacing, rotation and orientation for the grid that best matches it. Fitting the whole plate at once makes it robust to glare and reflections on individual wells.
2. **Per-well refinement.** Refines each well at higher resolution using one shared radius, since every well on a plate is the same size.
3. **Perspective fit (floors).** Fits a perspective transform through the refined wells, rejecting outliers, so all positions are geometrically consistent.
4. **Openings.** Finds the larger top-opening rim of each well. The perspective shift between floor and opening is measured per well, then smoothed across the plate with a robust linear model.

Without a layout, it falls back to plain Hough circle detection. That's fine for opaque, high-contrast plates, but unreliable for clear ones.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/zhengti-12/Automated-pipetting-robot.git
cd Automated-pipetting-robot
pip install -r requirements.txt
```

`pillow-heif` is only needed for iPhone `.HEIC` photos. Without it, `heic_reader.py` falls back to decoding with `ffmpeg` if that is installed.

## Usage

### From an image file

```bash
python detect_wells.py plate.jpg --rows 8 --cols 6
```

For our 48-well plate as photographed (letters A–F along the bottom edge, row numbers counting up from the bottom):

```bash
python detect_wells.py plate.jpg --rows 8 --cols 6 --letters-on cols --flip-rows
```

### From a live camera

```bash
python detect_wells.py --camera 0 --rows 8 --cols 6 --letters-on cols --flip-rows --save-frame shot.jpg
```

`--camera` takes a device index (`0` = first camera) or a stream URL. The program grabs a single frame after letting auto-exposure settle, then runs detection on it.

### From Python (robot code)

```python
from detect_wells import capture_frame, find_wells

frame = capture_frame(0)   # or any OpenCV/numpy BGR image
wells = find_wells(frame, rows=8, cols=6, letters_on="cols", flip_rows=True)

for w in wells:
    print(w["label"], w["x"], w["y"])   # opening centre in pixels
```

### Options

| Option | Description |
|---|---|
| `--rows`, `--cols` | Wells per column / row **as seen in the image**. The swapped orientation is also tried automatically (disable with `--no-transpose`). |
| `--letters-on rows\|cols` | Which image axis carries the plate's letter labels. |
| `--flip-rows`, `--flip-cols` | Reverse numbering direction to match the plate's markings. |
| `--camera` | Capture from a camera index or stream URL instead of a file. |
| `--resolution WxH` | Requested camera resolution, e.g. `1920x1080`. |
| `--save-frame FILE` | Save the captured camera frame. |
| `--min-r`, `--max-r` | Limit the well radius search (original-image pixels). |
| `--out FILE` | Output file, `.json` (default `wells.json`) or `.csv`. |
| `--annotated FILE` | Annotated preview image (default `wells_annotated.jpg`). |
| `--show` | Open a preview window. |

## Output

Each well in the JSON/CSV output:

| Field | Meaning |
|---|---|
| `label` | Plate well name, e.g. `A1` |
| `row`, `col` | Grid position in the image |
| `x`, `y`, `r` | **Top opening** centre and radius (pixels in the original image) |
| `floor_x`, `floor_y`, `floor_r` | Well floor centre and radius |
| `fit_ok` | `true` if the well's own measurement agreed with the fitted grid |

In the annotated image, green circles are wells whose measurement agreed with the grid. Orange circles are wells placed mainly from the grid (for example because of glare). Grey circles are the well floors.

## Camera setup tips

- Mount the camera **perpendicular to the table and centred over the plate**. This removes most of the double-rim effect.
- Mount it as high as practical and zoom or crop in, so edge wells are seen closer to straight-on.
- Use diffuse lighting, or a backlight under the plate, and a matte dark surface to reduce glare.
- Detection takes a few seconds on typical camera resolutions (1–5 MP), and longer on 24 MP phone photos.

## Roadmap

- Pixel → robot coordinate calibration (jog the tip to a few known wells, fit a transform to mm)
- Live preview mode with on-demand detection
- Presets for common plate formats (6/12/24/48/96/384-well)

## Files

- `detect_wells.py`: detection program (CLI + importable functions)
- `heic_reader.py`: reads iPhone `.HEIC` photos
- `requirements.txt`: Python dependencies
