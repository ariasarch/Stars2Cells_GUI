"""
Generate platform icon files (Windows .ico, macOS .icns) from S2C_logo.png.

Run automatically by the build scripts; outputs land in packaging/icons/
(which is gitignored — icons are derived artifacts).
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = Path(__file__).resolve().parent / "icons"

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def load_square_logo(size: int = 1024) -> Image.Image:
    """Load the logo, pad it to a centered square on a transparent canvas."""
    logo = Image.open(ROOT / "S2C_logo.png").convert("RGBA")
    side = max(logo.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    square = load_square_logo()

    ico_path = ICON_DIR / "S2C_logo.ico"
    square.save(ico_path, sizes=ICO_SIZES)
    print(f"Wrote {ico_path}")

    icns_path = ICON_DIR / "S2C_logo.icns"
    square.save(icns_path)
    print(f"Wrote {icns_path}")


if __name__ == "__main__":
    main()
