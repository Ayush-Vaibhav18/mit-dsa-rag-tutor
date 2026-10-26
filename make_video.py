
from pathlib import Path
import fitz
from PIL import Image
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

BASE = Path(__file__).parent
PDF = BASE / "slides" / "Randomized_Quick_Sort.pdf"
AUDIO = BASE / "audio"
OUTPUT = BASE / "final_video" / "Randomized_Quick_Sort_Hindi.mp4"
TEMP = BASE / "final_video" / "rendered_slides"

TEMP.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Render each PDF page as a high-quality image
doc = fitz.open(PDF)
slide_images = []

for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
    image_path = TEMP / f"slide_{i + 1}.png"
    pix.save(str(image_path))
    slide_images.append(image_path)

doc.close()

clips = []

for i, image_path in enumerate(slide_images, start=1):
    # Your files are named Slide-1.mp3, Slide-2.mp3, etc.
    audio_path = AUDIO / f"Slide-{i}.mp3"

    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    audio = AudioFileClip(str(audio_path))
    image = Image.open(image_path).convert("RGB")
    
    # Fit the slide into a 1920x1080 video without stretching
    image.thumbnail((1920, 1080))
    canvas = Image.new("RGB", (1920, 1080), "white")
    x = (1920 - image.width) // 2
    y = (1080 - image.height) // 2
    canvas.paste(image, (x, y))

    canvas_path = TEMP / f"video_slide_{i}.jpg"
    canvas.save(canvas_path, quality=95)

    clip = (
        ImageClip(str(canvas_path))
        .with_duration(audio.duration)
        .with_audio(audio)
    )
    clips.append(clip)

print("Rendering video... Please wait.")
final_video = concatenate_videoclips(clips, method="compose")
final_video.write_videofile(
    str(OUTPUT),
    fps=24,
    codec="libx264",
    audio_codec="aac"
)

final_video.close()
for clip in clips:
    clip.close()

print(f"\nVIDEO READY: {OUTPUT}")
