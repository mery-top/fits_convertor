# FITS Converter 🌌

**FITS Converter** is a Python-based utility to convert satellite images in the **FITS (Flexible Image Transport System)** format into visually interpretable **PNG images**. The tool includes functionality for RGB coloring, image enhancement, and graph generation for detailed data visualization.

---

## Features

- Converts **FITS files** to **PNG images**.
- Supports **RGB coloring** for enhanced visualization.
- Expands FITS image details for improved readability.
- Generates **graphs** for detailed analysis of the FITS image data.
- Easy-to-use command-line interface.

---

## Requirements

Make sure you have the following Python libraries installed:

- `astropy`
- `numpy`
- `matplotlib`

You can install them using:

```bash
pip install astropy numpy matplotlib
```

## Usage

### Conversion Steps

1. Place your `.fits` file in the same directory as `fits.py`.
2. Run the script:
   ```bash
   python fits.py <your_fits_file.fits>
