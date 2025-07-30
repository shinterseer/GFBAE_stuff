import numpy as np


def plotly_plot():
    import plotly.graph_objects as go

    # Define the grid
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    x, y = np.meshgrid(x, y)
    z = x * y

    # Create the 3D surface
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])

    # Update layout for better control
    fig.update_layout(title=r'$f(x, y) = x \cdot y$',
                      scene=dict(
                          xaxis_title='x',
                          yaxis_title='y',
                          zaxis_title='f(x,y)'
                      ))

    # Show plot in browser or notebook
    fig.show()


def matplotlib_plot():
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D  # Needed to register 3D projection

    # Create a grid
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    x, y = np.meshgrid(x, y)
    z = x * y

    # Use the Qt backend for interactivity (optional in scripts)
    plt.switch_backend('qt5agg')  # or 'qt6agg' if you have Qt6

    # Create figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    # Titles and labels (with LaTeX in title only)
    ax.set_title(r'$f\hspace{.1}(x, y) = x \cdot y$', fontsize=14)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f\hspace{.1}(x, y)$')

    # Show the plot (opens a rotatable window)
    plt.show()


if __name__ == '__main__':
    matplotlib_plot()
