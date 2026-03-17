import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def bfs_multi_source(grid, sources):
    """
    grid: 2D numpy array of 0=free, 1=obstacle
    sources: list of (r,c) goal coordinates to start BFS from
    Returns:
      dist: 2D array of distances (np.inf for unreachable)
      parent: dict mapping (r,c) -> parent (r,c) along shortest path to a source
    """
    H, W = grid.shape
    dist = np.full((H, W), np.inf)
    parent = {}
    q = deque()
    for s in sources:
        r, c = s
        if grid[r, c] == 0:
            dist[r, c] = 0
            q.append((r, c))
            parent[(r, c)] = None

    moves = [(1,0),(-1,0),(0,1),(0,-1)]  # 4-neighbour grid
    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                if dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    return dist, parent

def reconstruct_path(parent, start, is_goal_fn):
    """
    Reconstruct path from start following parent pointers until a cell whose parent is None
    (a source). If start is unreachable (not in parent) return [].
    """
    if start not in parent:
        return []
    path = []
    cur = start
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    # path goes from start -> ... -> goal source, reverse if you want goal-first
    return path

def plot_bfs_heatmap(grid, agents, goals, figsize=(8,8), cmap='viridis', savepath=None, transform='scale', scale_factor=1.0, normalize=True):
    H, W = grid.shape
    dist, parent = bfs_multi_source(grid, goals)

    # Convert distances to a proximity score so that "hotter" means closer.
    # We use proximity = 1 / (1 + dist) which maps dist=0 -> 1.0 (hottest)
    # and large dist -> small proximity. Unreachable cells become NaN and are
    # shown in a neutral gray color.
    finite = np.isfinite(dist)
    if np.any(finite):
        proximity = np.full_like(dist, np.nan, dtype=float)
        proximity[finite] = 1.0 / (1.0 + dist[finite]/1500)

        # Apply optional transform: 'none' (identity), 'scale' (multiply by
        # scale_factor), or 'log' (log(1 + scale_factor * proximity)). The
        # log transform compresses large differences and slows color changes.
        tf = (transform or 'none').lower()
        if tf == 'scale':
            proximity[finite] = proximity[finite] * float(scale_factor)
        elif tf == 'log':
            # Use log1p for stability; keep NaNs where unreachable
            proximity[finite] = np.log1p(proximity[finite] * float(scale_factor))

        # Debug: print numeric ranges to help diagnose flat colormap issues
        finite_vals = proximity[finite]
        print(f"dist: min={float(np.nanmin(dist[finite]))}, max={float(np.nanmax(dist[finite]))}")
        print(f"proximity before transform: min={float(np.nanmin(1.0/(1.0+dist[finite]))):.6f}, max={float(np.nanmax(1.0/(1.0+dist[finite]))):.6f}")
        print(f"proximity after transform: min={float(np.nanmin(finite_vals)):.6e}, max={float(np.nanmax(finite_vals)):.6e}")

        # Optional normalization to 0..1 across finite cells (helps colormap
        # interpretation when transforms change dynamic range)
        if normalize:
            vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
            vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
            if vmax > vmin:
                proximity[finite] = (proximity[finite] - vmin) / (vmax - vmin)

        masked = np.ma.masked_invalid(proximity)
    else:
        masked = np.ma.masked_all_like(dist)

    plt.figure(figsize=figsize)
    # Use a perceptually-uniform "hot" style cmap by default. Allow caller to
    # override via the `cmap` argument.
    cmap_obj = plt.get_cmap(cmap)
    # Color for bad/masked cells (unreachable)
    cmap_obj.set_bad(color='lightgray')
    # Plot the proximity heatmap (no interpolation so grid is clear)
    im = plt.imshow(masked, origin='lower', cmap=cmap_obj, interpolation='nearest')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # Label reflects transform and scale
    if transform == 'scale':
        cbar.set_label(f'Proximity * {scale_factor} (higher = closer)')
    elif transform == 'log':
        cbar.set_label(f'log(1 + {scale_factor} * proximity) (higher = closer)')
    else:
        cbar.set_label('Proximity to nearest goal (higher = closer)')

    # Overlay obstacles
    obs_y, obs_x = np.where(grid == 1)
    # plt.scatter(obs_x, obs_y, marker='s', color='dimgray', s=10, label='obstacle')

    # Plot goals and agents
    for i, g in enumerate(goals):
        gr, gc = g
        plt.scatter(gc, gr, marker='x', color='red', s=80, linewidths=2, label='goal' if i==0 else None)

    for i, a in enumerate(agents):
        ar, ac = a
        plt.scatter(ac, ar, marker='o', color='blue', s=40, label='agent' if i==0 else None)

    # Reconstruct and plot path for each agent to nearest goal using parent pointers
    for idx, a in enumerate(agents):
        ar, ac = a
        path = reconstruct_path(parent, (ar, ac), None)
        if len(path) == 0:
            # unreachable
            plt.text(ac, ar, 'X', color='black', fontsize=12, ha='center', va='center')
            continue
        # path is start->...->goal; convert to x,y arrays for plotting
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys, linewidth=2, label=f'path_agent_{idx}' if idx==0 else None)

    plt.legend(loc='upper right')
    plt.title('BFS distance heatmap (steps to nearest goal)')
    plt.xlim(-0.5, W-0.5)
    plt.ylim(-0.5, H-0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

# Example usage: create a random obstacle grid, place agents and goals
if __name__ == '__main__':
    np.random.seed(0)
    H, W = 7, 7
    grid = np.zeros((H, W), dtype=int)

    # add some random obstacles
    # obstacle_prob = 0.18
    # grid[np.random.rand(H, W) < obstacle_prob] = 1

    # carve a clear start area (optional)
    grid[0:3, 0:3] = 0

    # define agents and goals (row, col)
    agents = [(3, 0)]
    goals = [(3, 6)]

    # ensure agents and goals are on free cells
    for a in agents:
        if grid[a] == 1:
            grid[a] = 0
    for g in goals:
        if grid[g] == 1:
            grid[g] = 0

    plot_bfs_heatmap(grid, agents, goals, savepath='bfs_heatmap.png')