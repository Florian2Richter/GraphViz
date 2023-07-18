import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import urllib.request
import io
import zipfile
from tqdm import tqdm

# Output directory for the images
output_dir = '../output_images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def load_dataset(key: str) -> nx.Graph:
    if key == 'karate':
        return nx.karate_club_graph()
    if key == 'football':
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)  # parse gml data
        return G
    else:
        raise ValueError("you have to choose a valid dataset key")

def color_nodes(G: nx.Graph) -> list[int]:
    """
    Color the nodes of the input graph G based on a specified feature.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        list: A list containing the colors of the nodes. The colors are represented by
              integer indices corresponding to the unique values of a specified feature in the graph.

    Raises:
        ValueError: If there is more than one feature in the graph node data.

    Description:
        This function takes a graph G and colors its nodes based on a specified feature. The graph
        should be represented using the NetworkX library. 
        
        Note:
        - The input graph G should have node attributes containing the specified feature to be used for coloring.
        - The function assumes that the specified feature has discrete and hashable values.
    """

    community_map = {}
    # obtain all possible keys from the networkx nodes dictionary
    nodes = G.nodes(data=True)
    # obtain the features in the graph of keys
    data_keys = list(set([key for node in nodes for key in node[1].keys()]))

    if len(data_keys) != 1:
        raise ValueError("There is more than one feature in the graph.")

    # obtain the values of the features
    data_vals = list(set([val for node in nodes for val in node[1].values()]))

    # create the community map
    for node in G.nodes(data=True):
        for val in data_vals:
            if node[1][data_keys[0]] == val:
                community_map[node[0]] = data_vals.index(val)

    # create node coloring according to an index, i.e. colors have value 0, 1, 2, 3, 4
    node_color = [community_map[node] for node in G.nodes()]
    return node_color

    
def graph_coordinates(G: nx.Graph,dim: int = 3 , optimal_dist: float = 0.15, max_iterations: int = 10) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Utilizes the nx.spring_layout() function to generate the position of the edges either in 2-D or 3-D

    Parameters:
        G (networkx.Graph): The input graph.
        k (float): Optimal distance between nodes in the spring layout (default=0.15).

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: A tuple containing node_xyz and edge_xyz.

    Description:
        This function creates a 3D visualization of the input graph using the spring layout algorithm.

    """
    
    pos = nx.spring_layout(G, iterations = max_iterations, dim=dim, seed=1721, k=optimal_dist)

    # Extract node and edge positions from the layout
    nodes = np.array([pos[v] for v in G])
    edges = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
    return nodes, edges

def create_axes(fig: plt.figure,
                nodes: np.ndarray,
                edges: list[np.ndarray],
                node_color: list[int],
                dim: int,
                print_label: bool = False,
                azi: float = 20) -> matplotlib.axes.Axes:
    """
    Create a 2D or 3D axis with visual elements such as nodes and edges.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to which the 3D axis will be added.
        node_xyz (numpy.ndarray): A 3D numpy array representing the coordinates of nodes.
        edge_xyz (list[numpy.ndarray]): A list of 3D numpy arrays representing the coordinates of edges.
        node_color (list[int]): A list containing the colors of the nodes.
        azim (float): The azimuthal viewing angle of the 3D plot.

    Returns:
        mpl_toolkits.mplot3d.axes3d.Axes3D: The 3D axis object.

    
    Note:
        - The node_xyz and edge_xyz should be compatible 3D numpy arrays.
        - The node_color list should correspond to the colors of nodes in node_xyz.
    """
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        # optical fine tuning
        ax.view_init(elev=50.0, azim=azi)
        RADIUS = 1.25  # Control this value.
        ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
        ax.set_zlim3d(-RADIUS / 2, RADIUS / 2)
        ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)
        if print_label:
            # Add labels
            label = range(len(node_color))
            for i in range(len(node_color)):
                ax.text(nodes[i][0], nodes[i][1], nodes[i][2], label[i])

    if dim == 2:
        ax = fig.add_subplot(111)
        # Add labels
        if print_label:
            label = range(len(node_color))
            for i in range(len(node_color)):
                ax.text(nodes[i][0], nodes[i][1], label[i])
    
    # create the plot using matplotlib's scatter function
    ax.scatter(*nodes.T, s=100, ec="w", c=node_color)

    # Plot the edges
    for vizedge in edges:
        ax.plot(*vizedge.T, color="tab:gray", linewidth=0.15)

    # Turn gridlines off
    ax.grid(False)
    ax.axis('off')
    
    return ax

# available graphs
graphs = { 'football' : 'football',
              'karate' : 'karate'}


# constants
dimension = 3
optimal_dist = 0.15
max_iterations = 100
print_label = False

# load dataset
G = load_dataset(graphs['football'])

# color nodes according to their community
node_color = color_nodes(G)

# generate coordinates
nodes, edges = graph_coordinates(G, dimension, optimal_dist, max_iterations)

#print(nodes)
# intialize plot
fig = plt.figure()
azi = 20 # standard value for 3-D Plotting

# create the plot
ax = create_axes(fig, nodes, edges, node_color, dimension, print_label, azi=azi)

# show the plot
plt.show()


# def generate_animation(graph, node_color, dimension=2, spring_constant=0.15, azimuth_max)
#     if dimension == 2:
#         # 2-D code here

#         fig, ax = plt.subplots(figsize=(15, 9))
#         plot_options = {"node_size": 100, "with_labels": True, "width": 0.15}
#         pos = nx.spring_layout(G, iterations=20, seed=1721, k=0.1)
#         ax.axis("off")
#         nx.draw_networkx(G, pos=pos, ax=ax, node_color = node_color, labels = node_labels ,**plot_options)
    
#     if dimension == 3:
#         #3-D code here
    
#     else:
#         raise ValueError("we only support 2-D and 3-D plots")



# fig = plt.figure()
# azimuth_max = 360
# images = []
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# for azi in tqdm(range(0, azimuth_max), desc='Generating Images'):
#     ax = create_axes(fig, node_xyz, edge_xyz, node_color, azi)
#     fig.tight_layout()
#     # Convert the plot to an image object
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     renderer = canvas.get_renderer()
#     image = Image.frombytes('RGB', canvas.get_width_height(), renderer.tostring_rgb())
#     images.append(image)
#     plt.clf()

# # Save the animation as a GIF
# output_filename = f'{output_dir}/animation.gif'
# images[0].save(output_filename, save_all=True, append_images=images[1:], duration=100, loop=0)

# fig = plt.figure()
# azimuth_max = 360
# images = []
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# for k in tqdm(np.arange(0.01,10,0.1), desc='Generating Images'):
#     # generating 3D representation of graph
#     node_xyz, edge_xyz = graph_3D(G, k)

#     # generate axes object
#     ax = create_axes(fig, node_xyz, edge_xyz, node_color, 20)
#     fig.tight_layout()

#     # Convert the plot to an image object and append to list
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     renderer = canvas.get_renderer()
#     image = Image.frombytes('RGB', canvas.get_width_height(), renderer.tostring_rgb())
#     images.append(image)
    
#     # Clear the plot for performance reasons
#     plt.clf()

# # Save the animation as a GIF
# output_filename = f'{output_dir}/animation.gif'
# images[0].save(output_filename, save_all=True, append_images=images[1:], duration=100, loop=0)
# print(f'successfully generated animation at {output_filename}')