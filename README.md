# GraphViz
This repository contains different examplex for python based graph visualizations
![Graph](Graphs/animation.gif?raw=true)
# NetworkX Animation

This repository contains a Python script that generates an animation of a 3D network visualization using NetworkX and Matplotlib.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- `networkx`
- `matplotlib`
- `Pillow`
- `tqdm`

You can install these dependencies using pip:
pip install networkx matplotlib Pillow tqdm


## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Florian2Richtere/GraphViz.git

Change into the repository directory:
   ```bash
   cd networkx-animation

2. Run the Python script:
   ```bash
   python animation.py

3. The script will generate a series of PNG images in the "output_images" directory and then create an animated GIF named animation.gif.

## Customization

    You can modify the visualization options and parameters in the create_axes and _format_axes functions to adjust the appearance of the 3D network plot.
    To change the animation duration, modify the duration argument in the images[0].save() function.




