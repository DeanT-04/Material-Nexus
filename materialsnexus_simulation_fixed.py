import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
fig.suptitle('MaterialsNexus: Federated Materials Intelligence Network', fontsize=24, fontweight='bold')

# Define a grid layout
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1.2, 1])

# 1. Network Visualization (top left)
ax1 = plt.subplot(gs[0, 0])
ax1.set_title('Federated Learning Network', fontsize=18)

# Create a network graph
G = nx.Graph()

# Add nodes for different types of institutions
institutions = {
    'Research Lab 1': {'type': 'academic', 'size': 300, 'pos': (0.2, 0.8)},
    'University 1': {'type': 'academic', 'size': 250, 'pos': (0.8, 0.9)},
    'Industry Partner 1': {'type': 'industry', 'size': 280, 'pos': (0.9, 0.3)},
    'National Lab': {'type': 'government', 'size': 350, 'pos': (0.5, 0.5)},
    'Research Lab 2': {'type': 'academic', 'size': 220, 'pos': (0.1, 0.3)},
    'University 2': {'type': 'academic', 'size': 240, 'pos': (0.3, 0.1)},
    'Industry Partner 2': {'type': 'industry', 'size': 260, 'pos': (0.7, 0.1)}
}

# Add nodes to the graph
for institution, attrs in institutions.items():
    G.add_node(institution, **attrs)

# Add edges (connections between institutions)
connections = [
    ('Research Lab 1', 'University 1', 0.7),
    ('Research Lab 1', 'National Lab', 0.9),
    ('University 1', 'National Lab', 0.8),
    ('Industry Partner 1', 'National Lab', 0.9),
    ('Research Lab 2', 'National Lab', 0.7),
    ('University 2', 'National Lab', 0.6),
    ('Industry Partner 2', 'National Lab', 0.8),
    ('Research Lab 2', 'University 2', 0.5),
    ('Industry Partner 1', 'Industry Partner 2', 0.4),
    ('University 1', 'University 2', 0.3),
    ('Research Lab 1', 'Research Lab 2', 0.2)
]

for source, target, weight in connections:
    G.add_edge(source, target, weight=weight)

# Get node positions
pos = nx.spring_layout(G, pos={n: attrs['pos'] for n, attrs in institutions.items()}, fixed=institutions.keys())

# Define node colors based on type
node_colors = []
for node in G.nodes():
    if institutions[node]['type'] == 'academic':
        node_colors.append('#1f77b4')  # blue
    elif institutions[node]['type'] == 'industry':
        node_colors.append('#2ca02c')  # green
    else:  # government
        node_colors.append('#d62728')  # red

# Draw the network
nx.draw_networkx_nodes(G, pos, 
                      node_size=[institutions[node]['size'] for node in G.nodes()],
                      node_color=node_colors, 
                      alpha=0.8,
                      ax=ax1)

# Draw edges with varying thickness based on weight
edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax1)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax1)

# Add a legend
legend_elements = [
    Patch(facecolor='#1f77b4', label='Academic Institution'),
    Patch(facecolor='#2ca02c', label='Industry Partner'),
    Patch(facecolor='#d62728', label='Government Lab')
]
ax1.legend(handles=legend_elements, loc='upper right')
ax1.set_axis_off()

# 2. Materials Property Space (top right)
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('Materials Property Space Exploration', fontsize=18)

# Generate synthetic materials data
np.random.seed(42)
n_materials = 200

# Create properties for different material classes
material_classes = ['Battery Materials', 'Catalysts', 'Structural Materials', 'Electronic Materials']
class_properties = {
    'Battery Materials': {'energy_density': (800, 200), 'cycle_life': (1000, 300), 'cost': (50, 20)},
    'Catalysts': {'activity': (0.8, 0.2), 'selectivity': (0.9, 0.1), 'cost': (80, 30)},
    'Structural Materials': {'strength': (1200, 300), 'density': (5, 2), 'cost': (30, 10)},
    'Electronic Materials': {'conductivity': (600, 150), 'band_gap': (2, 0.5), 'cost': (60, 25)}
}

# Create a DataFrame to store material properties
materials_data = []

for mat_class in material_classes:
    n_class = n_materials // len(material_classes)
    props = class_properties[mat_class]
    
    for i in range(n_class):
        material = {
            'class': mat_class,
            'id': f"{mat_class.split()[0][:3]}-{i:03d}"
        }
        
        # Add properties with random values based on the class
        for prop, (mean, std) in props.items():
            material[prop] = max(0, np.random.normal(mean, std))
            
        # Add some common properties
        material['toxicity'] = np.random.uniform(0, 1)
        material['synthesis_complexity'] = np.random.uniform(1, 10)
        
        materials_data.append(material)

# Convert to DataFrame
df = pd.DataFrame(materials_data)

# Prepare data for dimensionality reduction
# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numerical_cols].values

# Ensure no NaN values in the data
X = np.nan_to_num(X)  # Replace NaNs with zeros

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Add t-SNE coordinates to the DataFrame
df['x'] = X_tsne[:, 0]
df['y'] = X_tsne[:, 1]

# Create a colormap for material classes
class_colors = {
    'Battery Materials': '#ff7f0e',
    'Catalysts': '#2ca02c',
    'Structural Materials': '#1f77b4',
    'Electronic Materials': '#9467bd'
}

# Plot the materials in t-SNE space
for mat_class in material_classes:
    subset = df[df['class'] == mat_class]
    ax2.scatter(subset['x'], subset['y'], 
                c=class_colors[mat_class], 
                label=mat_class, 
                alpha=0.7, 
                s=80)

# Add a legend
ax2.legend(loc='upper right')
ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)

# 3. Discovery Pipeline (bottom)
ax3 = plt.subplot(gs[1, :])
ax3.set_title('Materials Discovery Pipeline Simulation', fontsize=18)

# Set up the pipeline stages
stages = ['Literature\nExtraction', 'Property\nPrediction', 'Candidate\nRanking', 'Experiment\nDesign', 'Validation']
stage_positions = np.linspace(0.1, 0.9, len(stages))
y_position = 0.5

# Draw the pipeline stages
for i, (stage, x_pos) in enumerate(zip(stages, stage_positions)):
    circle = plt.Circle((x_pos, y_position), 0.08, color='#1f77b4', alpha=0.7)
    ax3.add_artist(circle)
    ax3.text(x_pos, y_position, str(i+1), ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax3.text(x_pos, y_position - 0.15, stage, ha='center', va='center', fontsize=12)

# Connect the stages with arrows
for i in range(len(stages) - 1):
    ax3.annotate('', 
                xy=(stage_positions[i+1] - 0.08, y_position),
                xytext=(stage_positions[i] + 0.08, y_position),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Add a feedback loop arrow
ax3.annotate('', 
            xy=(stage_positions[0] - 0.02, y_position + 0.05),
            xytext=(stage_positions[-1] + 0.02, y_position + 0.05),
            arrowprops=dict(arrowstyle='->', lw=2, color='#d62728', connectionstyle='arc3,rad=-0.3'))
ax3.text(0.5, y_position + 0.15, 'Continuous Learning Feedback Loop', ha='center', va='center', fontsize=12, color='#d62728')

# Add some example materials flowing through the pipeline
n_examples = 5
example_materials = df.sample(n_examples)
example_colors = [class_colors[mat_class] for mat_class in example_materials['class']]

# Add material flow visualization
for i, (_, material) in enumerate(example_materials.iterrows()):
    y_offset = 0.25 + i * 0.05
    
    # Draw material points at each stage
    for stage_idx, x_pos in enumerate(stage_positions):
        # Vary the size to show "confidence" increasing through the pipeline
        size = 80 + stage_idx * 20
        alpha = 0.6 + stage_idx * 0.08
        
        ax3.scatter(x_pos, y_position - y_offset, 
                   s=size, 
                   color=class_colors[material['class']], 
                   alpha=alpha,
                   edgecolor='black',
                   linewidth=1)
        
        # Add material ID to the first point
        if stage_idx == 0:
            ax3.text(x_pos - 0.05, y_position - y_offset, 
                    material['id'], 
                    ha='right', va='center', 
                    fontsize=10)

# Add annotations explaining the process
annotations = [
    (0.1, 0.2, "1. Extract material properties\nfrom scientific literature"),
    (0.3, 0.2, "2. Predict unknown properties\nusing ML models"),
    (0.5, 0.2, "3. Rank candidates based on\ndesired properties"),
    (0.7, 0.2, "4. Design optimal experiments\nfor validation"),
    (0.9, 0.2, "5. Validate predictions and\nupdate models")
]

for x, y, text in annotations:
    ax3.text(x, y, text, ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_axis_off()

# Add MaterialsNexus logo reference
plt.figtext(0.02, 0.02, "© MaterialsNexus - Connecting Intelligence for Materials Discovery", fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the visualization
plt.savefig('/home/ubuntu/materials_science_project/images/materialsnexus_simulation.png', dpi=300, bbox_inches='tight')

print("MaterialsNexus simulation visualization created successfully!")
