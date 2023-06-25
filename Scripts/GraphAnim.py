import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image
from pylab import show
import numpy as np
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

G = nx.karate_club_graph()
# color nodes according to their community
node_color = color_nodes(G)

# generate data for scatter plot
#  3d spring layout
pos = nx.spring_layout(G, dim=3, seed=779)
# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])


azimuth_max = 360
fig = plt.figure()
for azi in tqdm(range(0, azimuth_max), desc='Generating Images'):
    ax = create_axes(fig,node_xyz, edge_xyz, node_color,azi)
    fig.tight_layout()
    filename = f'{output_dir}/image{str(azi).zfill(3)}.png'
    plt.savefig(filename)
    plt.clf()


# Create an animation from the PNG images
images = []
for azi in tqdm(range(0, azimuth_max), desc='Generating Images'):
    filename = f'{output_dir}/image{str(azi).zfill(3)}.png'
    image = Image.open(filename)
    images.append(image)

# Save the animation as a GIF
output_filename = 'animation.gif'
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=100, loop=0)