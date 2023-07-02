import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
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

def color_nodes(G):
    community_map = {}
    # create community map
    for node in G.nodes(data=True):
        if node[1]["club"] == "Mr. Hi":
            community_map[node[0]] = 0
        else:
            community_map[node[0]] = 1
    #create node coloring according to commuinty map
    node_color = []
    color_map = {0: 0, 1: 1}
    node_color = [color_map[community_map[node]] for node in G.nodes()]
    return node_color
def color_nodes_general(G):
    community_map = {}
    # obtain all possible keys from the networkx nodes dictionary
    nodes = G.nodes(data=True)
    # obtain the features in the graph of keys
    data_keys = list(set([key for node in nodes for key in node[1].keys()]))
    if len(data_keys) != 1:
        raise ValueError("There is more than one feature int the graph.")
    # obtain the values of the features
    data_vals = list(set([val for node in nodes for val in node[1].values()]))
    # create the community map
    for node in G.nodes(data=True):
        for val in data_vals:
            if node[1][data_keys[0]] == val:
                community_map[node[0]] = data_vals.index(val)
    #create node coloring according to an index , i.e. colors have value 0,1,2,3,4
    node_color = [community_map[node] for node in G.nodes()]
    return node_color


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    ax.axis('off')
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])

    # Set axes labels
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    #ax.set_zlabel("z")

def create_axes(fig,node_xyz,edge_xyz,node_color, azim):
    ax = fig.add_subplot(111, projection="3d")
    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w",c=node_color)
    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")
    ax.view_init(elev=50., azim=azim)
    _format_axes(ax)
    
    return ax
# import graph

#G = nx.karate_club_graph()

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

# color nodes according to their community
node_color = color_nodes_general(G)

# generate data for scatter plot
#  3d spring layout
pos = nx.spring_layout(G, dim=3, seed=779)
# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])


azimuth_max = 360
fig = plt.figure()
images = []
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
for azi in tqdm(range(0, azimuth_max), desc='Generating Images'):
    ax = create_axes(fig, node_xyz, edge_xyz, node_color, azi)
    fig.tight_layout()
    # Convert the plot to an image object
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    image = Image.frombytes('RGB', canvas.get_width_height(), renderer.tostring_rgb())
    images.append(image)
    plt.clf()

# Save the animation as a GIF
output_filename = f'{output_dir}/animation.gif'
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=100, loop=0)