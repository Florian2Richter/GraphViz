graph_viz - A Package for Generating Colored 2-D and 3-D Plots of Network Visualizations
===============================================================================

[![License](https://img.shields.io/github/license/Florian2Richter/graph_viz)](https://github.com/Florian2Richter/graph_viz/blob/main/LICENSE)
![Graph](animations/animation_football_3.gif?raw=true)

Overview
--------

`graph_viz` is a Python package that provides functionalities to create animated GIFs of network visualizations, including 2-D and 3-D plots of graphs, which are in  standard networkx format. It leverages various Python libraries such as `networkx`, `numpy`, `matplotlib`, `Pillow`, and `tqdm` to generate dynamic visualizations.

Installation
------------

To install the `graph_viz` package, follow the steps below:

1. Clone the repository:

   ```git clone https://github.com/Florian2Richter/graph_viz.git
      cd graph_viz
   ```

2. Create a virtual enivronment (optional but recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate # On Windows: env\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install .
   ```

4. You can now check whether everything works by running the 'graph_animation.py' with the default settings:
   ```bash
   cd graph_viz
   python graph_animation.py
   ```

## Customization
 You can change parameters of the script inside the main function, e.g. changing the graphs as well as the dimension.



