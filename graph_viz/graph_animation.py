"""

This script utilizes various libraries, such as networkx, matplotlib, PIL, and tqdm, to generate
animated GIFs of network visualizations. It provides functions to load network datasets,
color nodes based on specified features, create 2D or 3D visualizations of the graph,
and generate an animated GIF by varying the azimuth angle of the visualization.

Libraries:
    - os
    - urllib.request
    - io
    - zipfile
    - cProfile
    - argparse
    - typing
    - numpy
    - networkx
    - matplotlib
    - PIL
    - tqdm

Constants:
    - OUTPUT_DIR: Output directory for the generated images.

Functions:
    - load_dataset: Load the specified network dataset.
    - color_nodes: Color the nodes of the input graph based on a specified feature.
    - graph_coordinates: Utilizes the nx.spring_layout() function to generate the position
                         of the edges either in 2-D or 3-D.
    - create_axes: Create a 2D or 3D axis with visual elements such as nodes and edges.
    - generate_image: Generate a PIL image of the network visualization.
    - main: Generate an animated GIF of a network visualization.

Description:
    The script provides a main function that loads a network dataset, colorizes the nodes based
    on a specified feature, and generates an animated GIF by rotating the 3D visualization
    from 0 to 360 degrees of azimuth angle. The GIF showcases the evolution of the network
    visualization as the azimuth angle changes.

    The script can be executed from the command line with optional profiling capabilities.

Note:
    - The script assumes that the specified feature for coloring the nodes has discrete and
      hashable values.
    - The script requires the 'tqdm' library for progress bars during the image generation loop.

Dependencies:
    - networkx
    - numpy
    - matplotlib
    - PIL (Python Imaging Library)
    - tqdm

Usage:
    The script can be executed directly, and it generates an animated GIF for the specified dataset
    and dimension (2D or 3D) in the `OUTPUT_DIR`.

    Example command: python script_name.py

    Additionally, profiling can be enabled by passing the `-p` or `--profile` argument:

    Example command with profiling: python script_name.py -p

"""
import os
import urllib.request
import io
import zipfile
import cProfile
import argparse
from typing import Union
import numpy as np
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Output directory for the images
OUTPUT_DIR = "../animations"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_dataset(key: str) -> nx.Graph:
    """
    Load the specified network dataset.

    Parameters:
        key (str): The key representing the dataset to load.

    Returns:
        nx.Graph: The loaded networkx Graph.

    Raises:
        ValueError: If an invalid dataset key is provided.
    """

    if key == "karate":
        return nx.karate_club_graph()
    if key == "football":
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        with urllib.request.urlopen(url) as sock, io.BytesIO(sock.read()) as stream:
            with zipfile.ZipFile(stream) as zip_file, zip_file.open(
                "football.gml"
            ) as gml_file:
                gml = gml_file.read().decode()  # read gml data
                # throw away bogus first line with # from mejn files
                gml = gml.split("\n")[1:]
                graph = nx.parse_gml(gml)  # parse gml data
                return graph

    raise ValueError("you have to choose a valid dataset key")


def color_nodes(graph: nx.Graph) -> list[int]:
    """
    Color the nodes of the input graph based on a specified feature.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        list: A list containing the colors of the nodes.
          The colors are represented by integer indices
          corresponding to the unique values of a specified
          feature in the graph.

    Raises:
        ValueError: If there is more than one feature in the graph node data.

    Description:
        This function takes a graph G and colors its nodes based on a specified feature. The graph
        should be represented using the NetworkX library.

        Note:
        - The input graph should have node attributes containing the specified feature to be
          used for coloring.
        - The function assumes that the specified feature has discrete and hashable values.
    """

    community_map = {}
    # obtain all possible keys from the networkx nodes dictionary
    nodes = graph.nodes(data=True)
    # obtain the features in the graph of keys
    data_keys = list({key for node in nodes for key in node[1].keys()})

    if len(data_keys) != 1:
        raise ValueError("There is more than one feature in the graph.")

    # obtain the values of the features
    data_vals = list({val for node in nodes for val in node[1].values()})

    # create the community map
    for node in graph.nodes(data=True):
        for val in data_vals:
            if node[1][data_keys[0]] == val:
                community_map[node[0]] = data_vals.index(val)

    # create node coloring according to an index, i.e. colors have value 0, 1, 2, 3, 4
    node_color = [community_map[node] for node in graph.nodes()]
    return node_color


def graph_coordinates(
    graph: nx.Graph, plot_settings
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Utilizes the nx.spring_layout() function to generate the position of the edges either in 2-D
    or 3-D

    Parameters:
        graph (networkx.Graph): The input graph.
        optimal_dist (float): Optimal distance between nodes in Fruchterman-Reingold (default=0.15).

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing nodes and edges (as pair of nodes)

    """

    pos = nx.spring_layout(
        graph,
        iterations=plot_settings["max_iterations"],
        dim=plot_settings["dimension"],
        seed=1721,
        k=plot_settings["optimal_dist"],
    )

    # Extract node and edge positions from the layout
    nodes = np.array([pos[v] for v in graph])
    edges = np.array([(pos[u], pos[v]) for u, v in graph.edges()])

    return nodes, edges


def create_axes(
    axis: Union[Axes3D, plt.Axes],
    node_data,
    dim: int,
    print_label: bool = False,
    azi: float = 20,
) -> None:
    """
    Create a 2D or 3D axis with visual elements such as nodes and edges.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to which the 3D axis will be added.
        nodes (numpy.ndarray): A 2D or 3D numpy array representing the coordinates of nodes.
        edges (list[numpy.ndarray]): A list of 2D or 3D numpy arrays representing the coordinates
                                     of edges.
        node_color (list[int]): A list containing the colors of the nodes.
        dim (int): The dimension of the plot. Use 2 for 2D plots or 3 for 3D plots.
        print_label (bool, optional): If True, labels will be printed next to the nodes.
            Default is False.
        azi (float, optional): The azimuthal viewing angle of the 3D plot. Default is 20.

    Returns:
        Union[mpl_toolkits.mplot3d.axes3d.Axes3D, matplotlib.axes._subplots.AxesSubplot]:
            The 3D or 2D axis object, depending on the specified dimension.

    Note:
        - The 'nodes' and 'edges' should be compatible numpy arrays, with 2D or 3D coordinates
            depending on the 'dim' parameter.
        - The 'node_color' list should correspond to the colors of nodes in 'nodes'.
    """
    nodes, edges, node_color = node_data
    if dim == 3:
        # optical fine tuning
        axis.view_init(elev=50.0, azim=azi)
        radius = 1.25  # Control this value for axis limits of the plot
        axis.set_xlim3d(-radius / 2, radius / 2)
        axis.set_zlim3d(-radius / 2, radius / 2)
        axis.set_ylim3d(-radius / 2, radius / 2)
        if print_label:
            # Add labels
            label = range(len(node_color))
            for i in range(len(node_color)):
                axis.text(nodes[i][0], nodes[i][1], nodes[i][2], label[i])

    if dim == 2:
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)

        # rotate points according to azimuth
        theta = np.radians(azi)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        # transform nodes and edges
        nodes = np.array([rotation_matrix @ node for node in nodes])
        edges = np.array([(rotation_matrix @ u, rotation_matrix @ v) for u, v in edges])
        # Add labels
        if print_label:
            label = range(len(node_color))
            for i in range(len(node_color)):
                axis.text(nodes[i][0], nodes[i][1], label[i])

    # create the plot using matplotlib's scatter function
    axis.scatter(*nodes.T, s=100, ec="w", c=node_color)

    # Plot the edges
    for vizedge in edges:
        axis.plot(*vizedge.T, color="tab:gray", linewidth=0.15)

    # Turn gridlines off
    axis.grid(False)
    axis.axis("off")

    return axis


def _convert_fig_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    image = Image.frombytes("RGB", canvas.get_width_height(), renderer.tostring_rgb())
    return image


def generate_image(nodes, edges, node_color, plot_settings, azi=0):
    """
    Generate a PIL image of the network visualization.

    Parameters:
        nodes (numpy.ndarray): A 2D or 3D numpy array representing the coordinates of nodes.
        edges (list[numpy.ndarray]): A list of 2D or 3D numpy arrays representing the
                                     coordinates of edges.
        node_color (list[int]): A list containing the colors of the nodes.
        dimension (int, optional): The dimension of the plot. Use 2 for 2D plots or 3 for 3D plots.
                                   Default is 3.
        print_label (bool, optional): If True, labels will be printed next to the nodes.
                                      Default is False.
        azi (float, optional): The azimuthal viewing angle of the 3D plot. Default is 0.

    Returns:
        PIL.Image: The generated PIL image of the network visualization.
    """
    fig = plt.figure()
    axis = fig.add_subplot(
        111, projection="3d" if plot_settings["dimension"] == 3 else None
    )
    node_data = (nodes, edges, node_color)
    create_axes(
        axis,
        node_data,
        plot_settings["dimension"],
        plot_settings["print_label"],
        azi=azi,
    )
    # plt.clf()
    image = _convert_fig_image(fig)
    plt.close()
    return image


def main():
    """
    Generate an animated GIF of a network visualization.

    Available graphs:
        - "football": Football network
        - "karate": Karate club network

    Description:
        The `main` function first defines available graphs, such as the football and
        karate networks. It sets the parameters for the plot, including the dimension
        (2D or 3D), the optimal Fruchterman-Reingold distance for spring_layout
        between two nodes, the maximum number of iterations before the iteration algorithm stops,
        and whether to print labels near the nodes.

        The function then loads the selected dataset from the available graphs,
        colorizes the nodes based on their community, and generates initial coordinates
        for the nodes and edges. Subsequently, it creates an animation by varying the
        azimuth angle from 0 to 360 degrees in a loop.

        The generated animation is saved as a GIF file in the specified output
        directory. The GIF showcases the evolution of the network visualization as the
        azimuth angle changes.

    """
    # available graphs
    graphs = {"football": "football", "karate": "karate"}

    # loaded_graph
    dataset_key = "football"

    # parameters of the plot
    plot_settings = {
        "dimension": 2,
        "optimal_dist": 0.15,  # optimal Fruchterman-Reingold distance
        "max_iterations": 100,  # maximal number of iterations for the algortihm
        "print_label": False,
    }

    # determines the maximal angle for azimutahl camera track
    azimuth_max = 360

    # load dataset
    graph = load_dataset(graphs[dataset_key])

    # color nodes according to their community
    node_color = color_nodes(graph)

    # generate the initial data
    nodes, edges = graph_coordinates(graph, plot_settings)

    # Generate animation here
    images = []
    for azi in tqdm(range(0, azimuth_max), desc="Generating Images"):
        image = generate_image(nodes, edges, node_color, plot_settings, azi=azi)
        images.append(image)

    # Save the animation as a GIF
    output_filename = (
        f"{OUTPUT_DIR}/animation_{dataset_key}_{plot_settings['dimension']}.gif"
    )
    images[0].save(
        output_filename, save_all=True, append_images=images[1:], duration=100, loop=0
    )
    print(f"successfully generated animation at {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate either 2-D or 3-D animation of network graph using spring layout"
    )
    parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    if args.profile:
        # Profile the main function
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        profiler.print_stats(sort="cumtime")
    else:
        main()
