import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
import os
import random
import seaborn as sns

# Define all parameters
PARAMS = {
    'volume_size': 150,
    'tc_radius': 20,
    'te_radius': 30,
    'vessel_start_distance': 30,
    'step_size_max': 5,
    'min_step_size': 0.5,
    'branching_prob': 0.1,
    'iterations': 1500,
    'threshold': 0.45,
    'sigmoid_slope': 25
}

def print_progress(message, detail=""):
    """Print progress message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    if detail:
        print(f"[{timestamp}] {message}: {detail}")
    else:
        print(f"[{timestamp}] {message}")

def get_distance_based_step_size(current_point, start_position, step_size, te_radius, bias_factor, reached_te, min_step_size):
    """Calculate step size that decreases with distance from start position"""
    distance_from_start = np.linalg.norm(np.subtract(current_point, start_position))
    distance_factor = np.exp(-distance_from_start / (2 * te_radius))
    
    if reached_te and random.random() < bias_factor:
        base_step = step_size * 0.3
    else:
        base_step = step_size * random.uniform(0.8, 1.2)
    
    final_step_size = max(min_step_size, base_step * (0.3 + 0.7 * distance_factor))
    return final_step_size

def get_valid_biased_target(vessel_points, te_radius):
    """Generate valid biased target point"""
    max_attempts = 10
    for _ in range(max_attempts):
        biased_point = random.choice(vessel_points)
        point_distance = np.linalg.norm(biased_point)
        
        if point_distance <= te_radius:
            perturbation_scale = 2.0
        else:
            perturbation_scale = 2.0 * np.exp(-(point_distance - te_radius) / (0.1 * te_radius))
        
        perturbation = np.random.normal(scale=perturbation_scale, size=3)
        target_point = tuple(np.add(biased_point, perturbation))
        
        target_dist = np.linalg.norm(target_point)
        if target_dist <= 1.1 * te_radius:
            return target_point
            
    return None

def modified_grow_vessel_network(volume_size, tc_radius, te_radius, vessel_start_distance,
                               step_size_max, min_step_size, branching_prob, iterations, 
                               bias_factor, seed=42):
    """Modified vessel growth with distance-dependent step size"""
    print_progress(f"Starting vessel growth simulation", f"bias = {bias_factor:.1f}")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize the 3D grid and masks
    x = np.linspace(-volume_size/2, volume_size/2, volume_size)
    y = np.linspace(-volume_size/2, volume_size/2, volume_size)
    z = np.linspace(-volume_size/2, volume_size/2, volume_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distances from the center
    distance_from_center = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Initialize masks
    tumor_core_mask = distance_from_center <= tc_radius
    tumor_edge_mask = (distance_from_center > tc_radius) & (distance_from_center <= te_radius)
    vessel_mask = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    # Define the main vessel's starting position
    vessel_start_position = (0, te_radius + vessel_start_distance, 0)
    
    # Track points and connections
    vessel_points = [vessel_start_position]
    vessel_connections = []
    x_idx = int(vessel_start_position[0] + volume_size/2)
    y_idx = int(vessel_start_position[1] + volume_size/2)
    z_idx = int(vessel_start_position[2] + volume_size/2)
    vessel_mask[x_idx, y_idx, z_idx] = True
    
    # Precompute indices for tumor edge mask
    te_indices = np.argwhere(tumor_edge_mask)
    
    # Variable to check if TE has been reached
    reached_te = False
    
    print_progress("Starting growth iterations")
    
    for iter_count in range(iterations):
        if iter_count % 100 == 0:  # Progress update every 100 iterations
            print_progress("Growth progress", f"{iter_count}/{iterations} iterations")
            
        # Check if any vessel point has reached the tumor edge radius
        if not reached_te:
            for vp in vessel_points:
                if np.linalg.norm(vp) >= te_radius:
                    reached_te = True
                    print_progress("Reached tumor edge")
                    break
        
        if reached_te and random.random() < bias_factor:
            target_point = get_valid_biased_target(vessel_points, te_radius)
            if target_point is None:
                target_idx = te_indices[np.random.choice(te_indices.shape[0])]
                target_point = (target_idx[0] - volume_size/2,
                              target_idx[1] - volume_size/2,
                              target_idx[2] - volume_size/2)
        else:
            target_idx = te_indices[np.random.choice(te_indices.shape[0])]
            target_point = (target_idx[0] - volume_size/2,
                          target_idx[1] - volume_size/2,
                          target_idx[2] - volume_size/2)
        
        # Find the closest existing vessel point
        distances = [np.linalg.norm(np.subtract(target_point, vp)) for vp in vessel_points]
        closest_point = vessel_points[np.argmin(distances)]
        
        # Calculate step size based on distance from start
        step_size = get_distance_based_step_size(closest_point, vessel_start_position, 
                                               step_size_max, te_radius, bias_factor, 
                                               reached_te, min_step_size)
        
        # Calculate the direction vector
        direction = np.subtract(target_point, closest_point)
        distance_to_target = np.linalg.norm(direction)
        if distance_to_target == 0:
            continue
        direction_normalized = direction / distance_to_target
        
        # Calculate the new point
        new_point = np.add(closest_point, direction_normalized * step_size)
        
        # Round the indices for array indexing
        x_idx = int(round(new_point[0] + volume_size/2))
        y_idx = int(round(new_point[1] + volume_size/2))
        z_idx = int(round(new_point[2] + volume_size/2))
        
        # Check if the new point is within bounds and not already occupied
        if (0 <= x_idx < volume_size) and (0 <= y_idx < volume_size) and (0 <= z_idx < volume_size):
            if (tumor_edge_mask[x_idx, y_idx, z_idx] or not vessel_mask[x_idx, y_idx, z_idx]):
                vessel_points.append(tuple(new_point))
                vessel_connections.append((closest_point, new_point))
                vessel_mask[x_idx, y_idx, z_idx] = True
                
                # Branching with distance-dependent step size
                if random.random() < branching_prob:
                    random_direction = np.random.normal(size=3)
                    random_direction /= np.linalg.norm(random_direction)
                    branch_point = np.add(new_point, random_direction * step_size)
                    
                    bx_idx = int(round(branch_point[0] + volume_size/2))
                    by_idx = int(round(branch_point[1] + volume_size/2))
                    bz_idx = int(round(branch_point[2] + volume_size/2))
                    
                    if (0 <= bx_idx < volume_size) and (0 <= by_idx < volume_size) and (0 <= bz_idx < volume_size):
                        if (tumor_edge_mask[bx_idx, by_idx, bz_idx] or not vessel_mask[bx_idx, by_idx, bz_idx]):
                            vessel_points.append(tuple(branch_point))
                            vessel_connections.append((new_point, branch_point))
                            vessel_mask[bx_idx, by_idx, bz_idx] = True
    
    print_progress("Vessel growth completed", f"Generated {len(vessel_points)} points")
    return vessel_points, vessel_connections, vessel_mask

def add_vessel_properties(vessel_points, vessel_connections, te_radius, threshold=0.45, sigmoid_slope=25):
    """Add properties to vessels based only on local vessel density"""
    print_progress("Calculating vessel properties")
    
    vessel_properties = {}
    detection_radius = 1.0
    
    # First pass: gather densities for normalization
    densities = []
    points_dict = {}
    
    for i, point in enumerate(vessel_points):
        if i % 100 == 0:  # Progress update
            print_progress("Processing densities", f"{i}/{len(vessel_points)} points")
            
        point_key = tuple(point) if isinstance(point, np.ndarray) else point
        
        # Calculate local density
        weighted_density = 0
        for other_point in vessel_points:
            dist = np.linalg.norm(np.subtract(point, other_point))
            if dist < detection_radius:
                weighted_density += np.exp(-(dist**2)/(2*(detection_radius/3)**2))
        
        densities.append(weighted_density)
        points_dict[point_key] = {
            'density': weighted_density
        }
    
    # Calculate density statistics for normalization
    mean_density = np.mean(densities)
    std_density = np.std(densities)
    
    print_progress("Calculating final vessel properties")
    
    # Second pass: calculate properties using normalized values
    for point in vessel_points:
        point_key = tuple(point) if isinstance(point, np.ndarray) else point
        point_data = points_dict[point_key]
        
        # Normalize density score
        density_score = (point_data['density'] - mean_density) / (std_density + 1e-6)
        
        # Modified sigmoid with stronger distinction between normal and abnormal
        density_contribution = 1 / (1 + np.exp(-(density_score - threshold) * sigmoid_slope))
        
        # New permeability calculation
        base_permeability = 1.0
        max_additional_permeability = 2.0
        
        if density_contribution < 0.3:  # Normal vessels
            permeability = base_permeability + (0.1 * density_contribution)
        else:  # Abnormal vessels
            permeability = base_permeability + (max_additional_permeability * density_contribution)
        
        vessel_properties[point_key] = {
            'permeability': permeability,
            'blood_flow': 1.0/permeability,
            'oxygen_delivery': 1.0/permeability,
            'local_density': density_score,
            'is_glomeruloid': permeability > 2.0
        }
    
    return vessel_properties
def analyze_oxygen_distribution(oxygen_levels, volume_size, tc_radius, te_radius):
    """
    Calculate statistics for oxygen distribution in tumor shell
    """
    # Create tumor shell mask
    x = np.linspace(-volume_size/2, volume_size/2, volume_size)
    y = np.linspace(-volume_size/2, volume_size/2, volume_size)
    z = np.linspace(-volume_size/2, volume_size/2, volume_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    distance_from_center = np.sqrt(X**2 + Y**2 + Z**2)
    tumor_shell_mask = (distance_from_center > tc_radius) & (distance_from_center <= te_radius)
    
    # Get oxygen values in tumor shell
    shell_oxygen = oxygen_levels[tumor_shell_mask]
    
    stats = {
        # Overall oxygen delivery
        'total_oxygen': np.sum(shell_oxygen),
        'mean_oxygen': np.mean(shell_oxygen),
        
        # Measures of distribution homogeneity
        'std_oxygen': np.std(shell_oxygen),  # Standard deviation
        'cv_oxygen': np.std(shell_oxygen) / np.mean(shell_oxygen),  # Coefficient of variation
        'gini_coefficient': calculate_gini(shell_oxygen),  # Gini coefficient
        'entropy': calculate_entropy(shell_oxygen),  # Shannon entropy
        
        # Additional distribution characteristics
        'min_oxygen': np.min(shell_oxygen),
        'max_oxygen': np.max(shell_oxygen),
        'median_oxygen': np.median(shell_oxygen),
        'hypoxic_fraction': np.mean(shell_oxygen < np.mean(shell_oxygen) * 0.1)  # Fraction below 10% of mean
    }
    
    return stats

def calculate_gini(array):
    """
    Calculate Gini coefficient - measure of inequality
    Ranges from 0 (perfect equality) to 1 (perfect inequality)
    """
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # Make all values non-negative
    array += 0.0000001  # Ensure no zero values
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def calculate_entropy(array):
    """
    Calculate Shannon entropy - measure of randomness/uniformity
    Higher values indicate more uniform distribution
    """
    # Normalize array to probabilities
    array = array.flatten()
    array = array / np.sum(array)
    # Remove zero values to avoid log(0)
    array = array[array > 0]
    return -np.sum(array * np.log2(array))

def analyze_oxygen_distribution_by_region(oxygen_levels, volume_size, tc_radius, te_radius):
    """
    Calculate statistics for oxygen distribution in different tumor regions
    """
    # Create masks for different regions
    x = np.linspace(-volume_size/2, volume_size/2, volume_size)
    y = np.linspace(-volume_size/2, volume_size/2, volume_size)
    z = np.linspace(-volume_size/2, volume_size/2, volume_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    distance_from_center = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Define regions
    tumor_core_mask = distance_from_center <= tc_radius
    tumor_shell_mask = (distance_from_center > tc_radius) & (distance_from_center <= te_radius)
    tumor_total_mask = distance_from_center <= te_radius  # Both core and shell
    
    # Dictionary to store regional statistics
    regional_stats = {}
    
    # Calculate statistics for each region
    for region_name, mask in [
        ('tumor_core', tumor_core_mask),
        ('tumor_shell', tumor_shell_mask),
        ('tumor_total', tumor_total_mask)
    ]:
        region_oxygen = oxygen_levels[mask]
        
        regional_stats[region_name] = {
            'volume_mm3': np.sum(mask),  # Each voxel could represent a mm³
            'total_oxygen': np.sum(region_oxygen),
            'mean_oxygen': np.mean(region_oxygen),
            'median_oxygen': np.median(region_oxygen),
            'std_oxygen': np.std(region_oxygen),
            'cv_oxygen': np.std(region_oxygen) / np.mean(region_oxygen),
            'gini_coefficient': calculate_gini(region_oxygen),
            'entropy': calculate_entropy(region_oxygen),
            'hypoxic_fraction': np.mean(region_oxygen < np.mean(region_oxygen) * 0.5),
            'min_oxygen': np.min(region_oxygen),
            'max_oxygen': np.max(region_oxygen),
            # Add radial analysis for shell
            'radial_profile': calculate_radial_profile(oxygen_levels, distance_from_center, mask) 
                if region_name == 'tumor_shell' else None
        }
    
    return regional_stats

def calculate_radial_profile(oxygen_levels, distance_map, mask):
    """
    Calculate average oxygen levels at different radial distances within the region
    """
    unique_distances = np.unique(np.round(distance_map[mask], decimals=1))
    profile = []
    
    for dist in unique_distances:
        dist_mask = (np.abs(distance_map - dist) < 0.1) & mask
        if np.any(dist_mask):
            mean_oxygen = np.mean(oxygen_levels[dist_mask])
            profile.append((dist, mean_oxygen))
    
    return profile

def calculate_oxygen_distribution(vessel_points, vessel_properties, volume_size):
    """
    Calculate oxygen distribution from vessel network
    """
    print_progress("Calculating oxygen distribution")
    
    # Initialize oxygen level array
    oxygen_levels = np.zeros((volume_size, volume_size, volume_size))
    
    # Calculate oxygen distribution from vessels
    for point in vessel_points:
        point_key = tuple(point) if isinstance(point, np.ndarray) else point
        x, y, z = [int(p + volume_size/2) for p in point]
        oxygen_delivery = vessel_properties[point_key]['oxygen_delivery']
        
        # Add oxygen in a sphere around each vessel point
        radius = 10  # Diffusion radius
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < volume_size and 
                            0 <= ny < volume_size and 
                            0 <= nz < volume_size):
                            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                            # Exponential decay of oxygen with distance
                            oxygen_levels[nx, ny, nz] += oxygen_delivery * np.exp(-distance/5)
    
    return oxygen_levels

def plot_oxygen_analysis(vessel_points, vessel_properties, volume_size, tc_radius, te_radius, bias, save_dir):
    """
    Create comprehensive oxygen analysis plots and save them
    """
    # Calculate oxygen distribution
    oxygen_levels = calculate_oxygen_distribution(vessel_points, vessel_properties, volume_size)
    
    # Get regional statistics
    stats = analyze_oxygen_distribution_by_region(oxygen_levels, volume_size, tc_radius, te_radius)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Middle slice oxygen distribution
    ax1 = fig.add_subplot(gs[0, 0])
    middle_slice = oxygen_levels[:, :, volume_size//2]
    im = ax1.imshow(middle_slice.T, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax1, label='Oxygen Level')
    ax1.set_title('Oxygen Distribution (Middle Slice)')
    
    # Plot 2: Radial profile
    ax2 = fig.add_subplot(gs[0, 1])
    if stats['tumor_shell']['radial_profile']:
        distances, o2_levels = zip(*stats['tumor_shell']['radial_profile'])
        ax2.plot(distances, o2_levels)
        ax2.set_xlabel('Distance from Center')
        ax2.set_ylabel('Mean Oxygen Level')
        ax2.set_title('Radial Oxygen Profile')
    
    # Plot 3: Statistics text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create formatted statistics text
    stats_text = f"Oxygen Distribution Analysis (Bias: {bias:.1f})\n\n"
    
    for region in ['tumor_core', 'tumor_shell', 'tumor_total']:
        stats_text += f"\n{region.replace('_', ' ').title()}:\n"
        region_stats = stats[region]
        stats_text += f"Mean O₂: {region_stats['mean_oxygen']:.2f}\n"
        stats_text += f"CV: {region_stats['cv_oxygen']:.2f}\n"
        stats_text += f"Gini: {region_stats['gini_coefficient']:.2f}\n"
        stats_text += f"Entropy: {region_stats['entropy']:.2f}\n"
        stats_text += f"Hypoxic Fraction: {region_stats['hypoxic_fraction']*100:.1f}%\n"
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the figure
    plot_filename = f'oxygen_analysis_bias_{bias:.1f}.png'
    plot_filepath = os.path.join(save_dir, plot_filename)
    fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return stats

# Then modify the compare_bias_values function:
def compare_bias_values(seed=42):
    """Compare different bias values"""
    print_progress("Starting bias comparison")
    
    # Create directory for saving plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'bias_comparison_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print_progress(f"Created output directory", save_dir)
    
    # Store all statistics for later analysis
    all_stats = []
    bias_values = np.arange(0, 1.0, 0.1)
    
    for bias in bias_values:
        print_progress(f"\nProcessing bias = {bias:.1f}")
        
        # Run vessel growth simulation
        vessel_points, vessel_connections, vessel_mask = modified_grow_vessel_network(
            volume_size=PARAMS['volume_size'],
            tc_radius=PARAMS['tc_radius'],
            te_radius=PARAMS['te_radius'],
            vessel_start_distance=PARAMS['vessel_start_distance'],
            step_size_max=PARAMS['step_size_max'],
            min_step_size=PARAMS['min_step_size'],
            branching_prob=PARAMS['branching_prob'],
            iterations=PARAMS['iterations'],
            bias_factor=bias,
            seed=seed
        )
        
        # Calculate vessel properties
        vessel_properties = add_vessel_properties(vessel_points, vessel_connections, 
                                                PARAMS['te_radius'],
                                                PARAMS['threshold'],
                                                PARAMS['sigmoid_slope'])
        
        # Get oxygen analysis (now returns only stats)
        stats = plot_oxygen_analysis(vessel_points, vessel_properties, 
                                   PARAMS['volume_size'], 
                                   PARAMS['tc_radius'], 
                                   PARAMS['te_radius'], 
                                   bias, 
                                   save_dir)
        
        all_stats.append(stats)
        
    # Create summary plots of statistics vs bias
    plot_stats_summary(all_stats, bias_values, save_dir)
    
    print_progress("Completed all bias values")
    return all_stats

def plot_stats_summary(all_stats, bias_values, save_dir):
    """Create summary plots of how statistics vary with bias"""
    metrics = ['mean_oxygen', 'cv_oxygen', 'gini_coefficient', 'entropy', 'hypoxic_fraction']
    regions = ['tumor_core', 'tumor_shell', 'tumor_total']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for region in regions:
            values = [stats[region][metric] for stats in all_stats]
            plt.plot(bias_values, values, 'o-', label=region.replace('_', ' ').title())
        
        plt.xlabel('Bias Factor')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'{metric}_vs_bias.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_permeability_histogram(vessel_properties, bias):
    """Create histogram of vessel permeability values"""
    print_progress("Creating permeability histogram")
    
    permeability_values = [p['permeability'] for p in vessel_properties.values()]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permeability_values, bins=30, kde=True)
    
    plt.axvline(x=1.0, color='g', linestyle='--', label='Normal vessel baseline')
    plt.axvline(x=2.0, color='r', linestyle='--', label='Glomeruloid threshold')
    
    plt.xlabel('Vessel Permeability')
    plt.ylabel('Count')
    plt.title(f'Distribution of Vessel Permeability (Bias: {bias:.1f})')
    
    stats_text = f'Mean: {np.mean(permeability_values):.2f}\n'
    stats_text += f'Median: {np.median(permeability_values):.2f}\n'
    stats_text += f'Std Dev: {np.std(permeability_values):.2f}\n'
    stats_text += f'% Glomeruloid: {(sum(p > 2.0 for p in permeability_values)/len(permeability_values)*100):.1f}%'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def visualize_3d_network(vessel_points, vessel_connections, vessel_properties, 
                        tc_radius, te_radius, volume_size, bias):
    """Create 3D visualization of vessel network"""
    print_progress("Creating 3D visualization")
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(1, 20)
    ax = fig.add_subplot(gs[0, :19], projection='3d')
    
    # Plot vessels with permeability coloring
    permeability_values = []
    for i, (start_point, end_point) in enumerate(vessel_connections):
        if i % 100 == 0:  # Progress update
            print_progress("Plotting vessels", f"{i}/{len(vessel_connections)} connections")
            
        start_key = tuple(start_point) if isinstance(start_point, np.ndarray) else start_point
        end_key = tuple(end_point) if isinstance(end_point, np.ndarray) else end_point
        
        perm_start = vessel_properties[start_key]['permeability']
        perm_end = vessel_properties[end_key]['permeability']
        avg_perm = (perm_start + perm_end) / 2
        permeability_values.append(avg_perm)
        
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=plt.cm.coolwarm((avg_perm - 1.0) / 1.5),
                linewidth=1)
    
    print_progress("Adding tumor geometry")
    
    # Add tumor spheres
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    
    # Plot tumor core
    x = tc_radius * np.outer(np.cos(u), np.sin(v))
    y = tc_radius * np.outer(np.sin(u), np.sin(v))
    z = tc_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='red', alpha=0.1)
    
    # Plot tumor edge
    x = te_radius * np.outer(np.cos(u), np.sin(v))
    y = te_radius * np.outer(np.sin(u), np.sin(v))
    z = te_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='orange', alpha=0.1)
    
    # Set equal aspect ratio
    max_range = max(te_radius, volume_size/2) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Add colorbar
    cax = fig.add_subplot(gs[0, -1])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                              norm=plt.Normalize(vmin=1.0, vmax=2.5))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='Vessel Permeability')
    
    plt.title(f'Vessel Network (Bias: {bias:.1f})')
    plt.tight_layout()
    return fig

def compare_bias_values(seed=42):
    """Compare different bias values"""
    print_progress("Starting bias comparison")
    
    # Create directory for saving plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'bias_comparison_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print_progress(f"Created output directory", save_dir)
    
    # Store all statistics for later analysis
    all_stats = []
    bias_values = np.arange(0, 1.0, 0.1)
    
    for bias in bias_values:
        print_progress(f"\nProcessing bias = {bias:.1f}")
        
        # Run vessel growth simulation
        vessel_points, vessel_connections, vessel_mask = modified_grow_vessel_network(
            volume_size=PARAMS['volume_size'],
            tc_radius=PARAMS['tc_radius'],
            te_radius=PARAMS['te_radius'],
            vessel_start_distance=PARAMS['vessel_start_distance'],
            step_size_max=PARAMS['step_size_max'],
            min_step_size=PARAMS['min_step_size'],
            branching_prob=PARAMS['branching_prob'],
            iterations=PARAMS['iterations'],
            bias_factor=bias,
            seed=seed
        )
        
        # Calculate vessel properties
        vessel_properties = add_vessel_properties(vessel_points, vessel_connections, 
                                                PARAMS['te_radius'],
                                                PARAMS['threshold'],
                                                PARAMS['sigmoid_slope'])
        
        # Create and save all plots
        oxygen_stats = plot_oxygen_analysis(vessel_points, vessel_properties, 
                                          PARAMS['volume_size'], 
                                          PARAMS['tc_radius'], 
                                          PARAMS['te_radius'], 
                                          bias,
                                          save_dir)  # Added save_dir here
        
        all_stats.append(oxygen_stats)
        
    # Create summary plots
    plot_stats_summary(all_stats, bias_values, save_dir)
    
    print_progress("Completed all bias values")
    return all_stats

def create_interactive_vessel_plotly():
    # Generate vessel network with bias 0
    vessel_points, vessel_connections, vessel_mask = modified_grow_vessel_network(
        volume_size=PARAMS['volume_size'],
        tc_radius=PARAMS['tc_radius'],
        te_radius=PARAMS['te_radius'],
        vessel_start_distance=PARAMS['vessel_start_distance'],
        step_size_max=PARAMS['step_size_max'],
        min_step_size=PARAMS['min_step_size'],
        branching_prob=PARAMS['branching_prob'],
        iterations=PARAMS['iterations'],
        bias_factor=0.0,
        seed=42
    )
    
    # Calculate vessel properties
    vessel_properties = add_vessel_properties(
        vessel_points, 
        vessel_connections, 
        PARAMS['te_radius'],
        PARAMS['threshold'],
        PARAMS['sigmoid_slope']
    )
    
    # Create figure
    fig = go.Figure()
    
    # Add vessels as line segments
    for start_point, end_point in vessel_connections:
        start_key = tuple(start_point) if isinstance(start_point, np.ndarray) else start_point
        end_key = tuple(end_point) if isinstance(end_point, np.ndarray) else end_point
        
        perm_start = vessel_properties[start_key]['permeability']
        perm_end = vessel_properties[end_key]['permeability']
        avg_perm = (perm_start + perm_end) / 2
        
        # Convert permeability to color
        color = f'rgb{tuple(int(x*255) for x in plt.cm.coolwarm((avg_perm - 1.0) / 1.5)[:3])}'
        
        fig.add_trace(go.Scatter3d(
            x=[start_point[0], end_point[0]],
            y=[start_point[1], end_point[1]],
            z=[start_point[2], end_point[2]],
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='text',
            hovertext=f'Permeability: {avg_perm:.2f}',
            showlegend=False
        ))
    
    # Add tumor core sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    
    x = PARAMS['tc_radius'] * np.outer(np.cos(u), np.sin(v)).flatten()
    y = PARAMS['tc_radius'] * np.outer(np.sin(u), np.sin(v)).flatten()
    z = PARAMS['tc_radius'] * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,
            color='red',
            opacity=0.1
        ),
        name='Tumor Core'
    ))
    
    # Add tumor edge sphere
    x = PARAMS['te_radius'] * np.outer(np.cos(u), np.sin(v)).flatten()
    y = PARAMS['te_radius'] * np.outer(np.sin(u), np.sin(v)).flatten()
    z = PARAMS['te_radius'] * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,
            color='orange',
            opacity=0.1
        ),
        name='Tumor Edge'
    ))
    
    # Add buttons for different zoom levels
    max_range = PARAMS['volume_size']/2
    
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-max_range, max_range]),
            yaxis = dict(range=[-max_range, max_range]),
            zaxis = dict(range=[-max_range, max_range]),
            aspectmode='cube'
        ),
        title='Interactive Vessel Network Visualization',
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        args=[{"scene.camera": dict(
                            eye=dict(x=max_range*zoom, y=max_range*zoom, z=max_range*zoom)
                        )}],
                        label=f"{zoom}x Zoom",
                        method="relayout"
                    ) for zoom in [1.0, 0.4, 0.2, 0.1]
                ],
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ],
        # Add a color bar
        coloraxis=dict(
            colorbar=dict(
                title="Vessel Permeability",
                ticktext=["Normal (1.0)", "Abnormal (2.0)", "Highly Abnormal (2.5)"],
                tickvals=[1.0, 2.0, 2.5],
            ),
            colorscale="RdBu",
            reversescale=True,
            cmin=1.0,
            cmax=2.5,
        )
    )
    
    return fig

# Run the comparison
if __name__ == "__main__":
    compare_bias_values(seed=42)
