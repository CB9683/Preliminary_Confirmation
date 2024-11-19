import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from vessel_growth_utils import (
    PARAMS, 
    modified_grow_vessel_network, 
    add_vessel_properties
)

def create_surface_mesh(radius, resolution=30):
    """Create a spherical mesh for tumor boundaries"""
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(-np.pi/2, np.pi/2, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.sin(theta)
    
    return x, y, z

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
        bias_factor=0.5,
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
    
    # Add tumor core as transparent surface
    x_core, y_core, z_core = create_surface_mesh(PARAMS['tc_radius'])
    fig.add_trace(go.Surface(
        x=x_core, y=y_core, z=z_core,
        colorscale=[[0, 'red'], [1, 'red']],
        opacity=0.2,
        showscale=False,
        name='Tumor Core',
        hoverinfo='skip'
    ))
    
    # Add tumor edge as transparent surface
    x_edge, y_edge, z_edge = create_surface_mesh(PARAMS['te_radius'])
    fig.add_trace(go.Surface(
        x=x_edge, y=y_edge, z=z_edge,
        colorscale=[[0, 'orange'], [1, 'orange']],
        opacity=0.1,
        showscale=False,
        name='Tumor Edge',
        hoverinfo='skip'
    ))
    
    # Add buttons for different zoom levels
    max_range = PARAMS['volume_size']/2
    
    # Calculate number of vessel traces
    n_vessel_traces = len(vessel_connections)
    
    # Create visibility buttons for tumor regions and background
# Create visibility buttons for tumor regions and background
    updatemenus = [
        # Zoom buttons
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
        ),
        # Visibility buttons for tumor regions and vessels
        dict(
            type="buttons",
            buttons=[
                dict(
                    args=[{"visible": [True]*n_vessel_traces + [True, True]}],
                    label="Show All",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [True]*n_vessel_traces + [True, False]}],
                    label="Core Only",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [True]*n_vessel_traces + [False, True]}],
                    label="Edge Only",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [True]*n_vessel_traces + [False, False]}],
                    label="Vessels Only",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [False]*n_vessel_traces + [True, True]}],
                    label="Tumor Only",
                    method="restyle"
                ),
            ],
            direction="right",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.1,
            yanchor="top"
        ),
        # Background toggle button
        dict(
            type="buttons",
            buttons=[
                dict(
                    args=[{
                        "scene.xaxis.showgrid": True,
                        "scene.yaxis.showgrid": True,
                        "scene.zaxis.showgrid": True,
                        "scene.xaxis.showline": True,
                        "scene.yaxis.showline": True,
                        "scene.zaxis.showline": True,
                        "scene.xaxis.zeroline": True,
                        "scene.yaxis.zeroline": True,
                        "scene.zaxis.zeroline": True,
                        "scene.xaxis.showbackground": True,
                        "scene.yaxis.showbackground": True,
                        "scene.zaxis.showbackground": True,
                    }],
                    label="Show Background",
                    method="relayout"
                ),
                dict(
                    args=[{
                        "scene.xaxis.showgrid": False,
                        "scene.yaxis.showgrid": False,
                        "scene.zaxis.showgrid": False,
                        "scene.xaxis.showline": False,
                        "scene.yaxis.showline": False,
                        "scene.zaxis.showline": False,
                        "scene.xaxis.zeroline": False,
                        "scene.yaxis.zeroline": False,
                        "scene.zaxis.zeroline": False,
                        "scene.xaxis.showbackground": False,
                        "scene.yaxis.showbackground": False,
                        "scene.zaxis.showbackground": False,
                    }],
                    label="Hide Background",
                    method="relayout"
                ),
            ],
            direction="right",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.9,
            xanchor="right",
            y=1.1,
            yanchor="top"
        )
    ]
    
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-max_range, max_range]),
            yaxis = dict(range=[-max_range, max_range]),
            zaxis = dict(range=[-max_range, max_range]),
            aspectmode='cube'
        ),
        title='Interactive Vessel Network Visualization',
        updatemenus=updatemenus,
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

# In your notebook, run:
fig = create_interactive_vessel_plotly()
fig.show()