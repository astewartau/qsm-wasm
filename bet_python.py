"""
BET (Brain Extraction Tool) Implementation - VECTORIZED
Based on: Smith, S.M. (2002) "Fast robust automated brain extraction"
Human Brain Mapping, 17(3):143-155

Optimized with NumPy vectorization for speed.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates


def create_icosphere(subdivisions=4):
    """
    Create a tessellated icosphere mesh.
    subdivisions=4 gives 2562 vertices, subdivisions=3 gives 642
    """
    phi = (1 + np.sqrt(5)) / 2

    vertices = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=np.float64)

    vertices /= np.linalg.norm(vertices[0])

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    for _ in range(subdivisions):
        vertices, faces = _subdivide_icosphere(vertices, faces)

    return vertices, faces


def _subdivide_icosphere(vertices, faces):
    """Subdivide each triangle into 4 triangles."""
    edge_midpoints = {}
    new_faces = []
    vertices = list(vertices)

    def get_midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (np.array(vertices[i1]) + np.array(vertices[i2])) / 2
        mid /= np.linalg.norm(mid)
        idx = len(vertices)
        vertices.append(mid)
        edge_midpoints[key] = idx
        return idx

    for f in faces:
        v0, v1, v2 = f
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_faces.extend([
            [v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20]
        ])

    return np.array(vertices, dtype=np.float64), np.array(new_faces, dtype=np.int32)


def build_neighbor_matrix(n_vertices, faces, max_neighbors=6):
    """
    Build a padded neighbor index matrix for vectorized neighbor lookups.
    Returns: (n_vertices, max_neighbors) array of neighbor indices, -1 for padding
    """
    neighbor_lists = [[] for _ in range(n_vertices)]

    for f in faces:
        for i in range(3):
            v1, v2 = f[i], f[(i + 1) % 3]
            if v2 not in neighbor_lists[v1]:
                neighbor_lists[v1].append(v2)
            if v1 not in neighbor_lists[v2]:
                neighbor_lists[v2].append(v1)

    # Find actual max neighbors
    actual_max = max(len(n) for n in neighbor_lists)
    max_neighbors = max(max_neighbors, actual_max)

    # Build padded matrix
    neighbor_matrix = np.full((n_vertices, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_vertices, dtype=np.int32)

    for i, neighs in enumerate(neighbor_lists):
        neighbor_counts[i] = len(neighs)
        neighbor_matrix[i, :len(neighs)] = neighs

    return neighbor_matrix, neighbor_counts


def compute_vertex_normals_vectorized(vertices, faces):
    """Compute outward-pointing normals at each vertex - vectorized."""
    n_vertices = len(vertices)
    normals = np.zeros((n_vertices, 3), dtype=np.float64)

    # Get face vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face normals
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    face_normals /= norms

    # Accumulate at vertices
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    normals /= norms

    return normals


def estimate_brain_parameters(data, voxel_size=(1, 1, 1)):
    """Estimate initial brain parameters from the image."""
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        nonzero = data.flatten()

    t2 = np.percentile(nonzero, 2)
    t98 = np.percentile(nonzero, 98)
    t = t2 + 0.1 * (t98 - t2)

    mask = data > t
    indices = np.array(np.where(mask)).T

    if len(indices) == 0:
        cog = np.array(data.shape) / 2
    else:
        weights = data[mask]
        cog = np.average(indices, axis=0, weights=weights)

    n_voxels = np.sum(mask)
    voxel_volume = np.prod(voxel_size)
    brain_volume = n_voxels * voxel_volume
    radius = (3 * brain_volume / (4 * np.pi)) ** (1/3)

    return t2, t98, t, cog, radius


def sample_intensities_vectorized(data, points, normals, voxel_size, max_dist=10, n_samples=10):
    """
    Sample intensities along inward normal for ALL vertices at once.

    Returns:
        I_min: (n_vertices,) minimum intensity for each vertex
        I_max: (n_vertices,) maximum intensity for each vertex
    """
    n_vertices = len(points)
    distances = np.linspace(0, max_dist, n_samples)

    # Scale normals by voxel size for proper mm->voxel conversion
    voxel_size = np.array(voxel_size)

    # Initialize
    I_min = np.full(n_vertices, np.inf)
    I_max = np.full(n_vertices, -np.inf)

    for d in distances:
        # Sample points along inward normal (all vertices at once)
        sample_points = points - d * normals / voxel_size  # (n_vertices, 3)

        # Clamp to valid range
        sample_points = np.clip(sample_points, 0, np.array(data.shape) - 1.001)

        # Use map_coordinates for fast interpolation (expects [z, y, x] coords)
        coords = [sample_points[:, 0], sample_points[:, 1], sample_points[:, 2]]
        intensities = map_coordinates(data, coords, order=1, mode='nearest')

        # Update min/max
        I_min = np.minimum(I_min, intensities)
        I_max = np.maximum(I_max, intensities)

    return I_min, I_max


def bet_surface_evolution(data, voxel_size=(1, 1, 1),
                          fractional_intensity=0.5,
                          n_iterations=1000,
                          subdivisions=4,
                          progress_callback=None):
    """
    Main BET algorithm - VECTORIZED version.
    """
    print(f"BET: Starting brain extraction (vectorized)...")
    print(f"  Image shape: {data.shape}")
    print(f"  Voxel size: {voxel_size} mm")

    voxel_size = np.array(voxel_size, dtype=np.float64)

    # Step 1: Estimate brain parameters
    print("BET: Estimating brain parameters...")
    t2, t98, t, cog, radius = estimate_brain_parameters(data, voxel_size)
    print(f"  t2={t2:.1f}, t98={t98:.1f}, threshold={t:.1f}")
    print(f"  COG: {cog}, radius: {radius:.1f} mm")

    # Step 2: Create icosphere
    print(f"BET: Creating icosphere (subdivisions={subdivisions})...")
    unit_vertices, faces = create_icosphere(subdivisions)
    n_vertices = len(unit_vertices)
    print(f"  {n_vertices} vertices, {len(faces)} faces")

    # Scale and position sphere
    initial_radius_vox = (radius * 0.5) / voxel_size
    vertices = unit_vertices * initial_radius_vox + cog

    # Build neighbor structure for vectorized operations
    neighbor_matrix, neighbor_counts = build_neighbor_matrix(n_vertices, faces)

    # BET parameters (from FSL)
    bt = fractional_intensity
    rmin = 3.33  # mm - minimum local radius of curvature
    rmax = 10.0  # mm - maximum local radius of curvature
    E = (1.0/rmin + 1.0/rmax) / 2.0  # ≈ 0.2
    F = 6.0 / (1.0/rmin - 1.0/rmax)  # ≈ 30
    normal_max_update_fraction = 0.5
    lambda_fit = 0.1

    # Compute mean edge length
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    edge_lengths = np.linalg.norm((v1 - v0) * voxel_size, axis=1)
    l = np.mean(edge_lengths)
    print(f"  Initial mean edge length: {l:.2f} mm")

    # Step 3: Iterative surface evolution - VECTORIZED
    print(f"BET: Evolving surface ({n_iterations} iterations)...")

    for iteration in range(n_iterations):
        if progress_callback and iteration % 25 == 0:
            progress_callback(iteration, n_iterations)

        # Compute all vertex normals at once
        normals = compute_vertex_normals_vectorized(vertices, faces)

        # Compute mean neighbor positions for all vertices
        padded_vertices = np.vstack([vertices, vertices[0]])  # Add dummy for -1 indices
        neighbor_idx = np.where(neighbor_matrix >= 0, neighbor_matrix, n_vertices)
        neighbor_pos = padded_vertices[neighbor_idx]  # (n_vertices, max_neighbors, 3)

        # Mask for valid neighbors
        valid_mask = (neighbor_matrix >= 0).astype(np.float64)[:, :, np.newaxis]

        # Mean neighbor position (difference vector in FSL terms)
        mean_neighbor = np.sum(neighbor_pos * valid_mask, axis=1) / np.maximum(neighbor_counts[:, np.newaxis], 1)

        # Vector from vertex to mean neighbor (dv in FSL)
        dv = mean_neighbor - vertices  # (n_vertices, 3)

        # Decompose into normal and tangential components
        # tmp = dv|n (dot product, can be positive or negative)
        dv_dot_n = np.sum(dv * normals, axis=1, keepdims=True)
        sn = dv_dot_n * normals  # Normal component (preserves sign!)
        st = dv - sn  # Tangential component

        # === Force 1: Tangential (vertex spacing) ===
        u1 = st * 0.5

        # === Force 2: Normal (smoothness) ===
        # Inverse radius of curvature: rinv = 2 * |sn| / l^2
        sn_mag = np.abs(dv_dot_n)  # |sn·n| = |sn|
        rinv = (2.0 * sn_mag) / (l * l)

        # Sigmoid function for smoothing factor
        f2 = (1.0 + np.tanh(F * (rinv - E))) * 0.5

        # u2 = f2 * sn (NOT f2 * |sn| * n !)
        u2 = f2 * sn

        # === Force 3: Intensity-based (vectorized) ===
        I_min, I_max = sample_intensities_vectorized(
            data, vertices, normals, voxel_size, max_dist=7, n_samples=15
        )

        # Clamp Imin and Imax to valid range
        I_min = np.maximum(I_min, t2)
        I_max = np.minimum(I_max, t98)

        # Local threshold: tl = (Imax - t2) * bt + t2
        t_l = (I_max - t2) * bt + t2

        # f3 = 2 * (Imin - tl) / (Imax - t2)
        denom = I_max - t2
        denom = np.where(denom > 0, denom, 1.0)  # Avoid division by zero
        f3 = 2.0 * (I_min - t_l) / denom

        # Scale by FSL constants
        f3 = f3 * normal_max_update_fraction * lambda_fit * l

        u3 = f3[:, np.newaxis] * normals

        # === Combined update ===
        u = u1 + u2 + u3
        vertices += u

        # Update edge length periodically
        if iteration % 100 == 0:
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            edge_lengths = np.linalg.norm((v1 - v0) * voxel_size, axis=1)
            l = np.mean(edge_lengths)
            print(f"  Iteration {iteration}/{n_iterations}, edge length: {l:.2f} mm")

    if progress_callback:
        progress_callback(n_iterations, n_iterations)

    print("BET: Surface evolution complete")

    # Step 4: Convert surface to binary mask
    print("BET: Creating binary mask from surface...")
    mask = surface_to_mask_fast(vertices, faces, data.shape, voxel_size)

    # Fill holes
    mask = ndimage.binary_fill_holes(mask)

    coverage = 100 * np.sum(mask) / mask.size
    print(f"BET: Complete. Mask coverage: {np.sum(mask)}/{mask.size} voxels ({coverage:.1f}%)")

    return vertices, faces, mask


def surface_to_mask_fast(vertices, faces, shape, voxel_size):
    """
    Convert surface to mask using FSL BET's approach:
    1. Draw mesh surface onto grid (mark as 0)
    2. Flood fill from CENTER of mesh (mark as 0)
    3. Return inverted mask

    NOTE: Our vertices are in VOXEL coordinates (not mm like FSL).
    """
    from collections import deque

    # Step increment: 0.5 voxels (vertices are already in voxel coords)
    mininc = 0.5

    # Start with all 1s (will become "outside")
    grid = np.ones(shape, dtype=np.uint8)

    # Step 1: Draw mesh surface as 0s
    # For each triangle, step along one edge and draw lines to opposite vertex
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

        # Edge from v1 to v0
        edge = v0 - v1
        edge_len = np.linalg.norm(edge)

        if edge_len < 0.001:
            continue

        edge_dir = edge / edge_len
        n_edge_steps = int(np.ceil(edge_len / mininc)) + 1

        for j in range(n_edge_steps):
            # Point along edge v1 -> v0
            p_edge = v1 + (j * mininc) * edge_dir
            if j * mininc > edge_len:
                p_edge = v0

            # Draw segment from p_edge to v2
            seg = v2 - p_edge
            seg_len = np.linalg.norm(seg)

            if seg_len < 0.001:
                # Just mark the point
                ix, iy, iz = int(round(p_edge[0])), int(round(p_edge[1])), int(round(p_edge[2]))
                if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                    grid[ix, iy, iz] = 0
                continue

            seg_dir = seg / seg_len
            n_seg_steps = int(np.ceil(seg_len / mininc)) + 1

            for k in range(n_seg_steps):
                p = p_edge + (k * mininc) * seg_dir
                if k * mininc > seg_len:
                    p = v2

                # Round to voxel index (vertices are in voxel coords)
                ix = int(round(p[0]))
                iy = int(round(p[1]))
                iz = int(round(p[2]))

                if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                    grid[ix, iy, iz] = 0

    print(f"  Surface voxels marked: {np.sum(grid == 0)}")

    # Step 2: Flood fill from CENTER of mesh (like FSL)
    center = np.mean(vertices, axis=0)
    cx, cy, cz = int(round(center[0])), int(round(center[1])), int(round(center[2]))

    # Clamp to valid range
    cx = max(0, min(cx, shape[0] - 1))
    cy = max(0, min(cy, shape[1] - 1))
    cz = max(0, min(cz, shape[2] - 1))

    print(f"  Flood fill from center: ({cx}, {cy}, {cz})")

    # Check if center is on surface (shouldn't be, but check anyway)
    if grid[cx, cy, cz] == 0:
        print("  WARNING: Center is on surface, finding nearby interior point")
        # Search nearby for a non-surface voxel
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                for dz in range(-5, 6):
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                        if grid[nx, ny, nz] == 1:
                            cx, cy, cz = nx, ny, nz
                            break
                else:
                    continue
                break
            else:
                continue
            break

    # BFS flood fill from center, marking interior as 0
    queue = deque([(cx, cy, cz)])
    grid[cx, cy, cz] = 0

    while queue:
        x, y, z = queue.popleft()

        # Check 6-connected neighbors
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if grid[nx, ny, nz] == 1:  # Not yet visited and not surface
                    grid[nx, ny, nz] = 0  # Mark as inside
                    queue.append((nx, ny, nz))

    # Mask: 0 = brain (inside + surface), 1 = outside
    # Invert so True = brain
    mask = (grid == 0)

    print(f"  Final mask voxels: {np.sum(mask)}")

    return mask


def run_bet(magnitude_data, voxel_size=(1, 1, 1), fractional_intensity=0.5,
            gradient_threshold=0.0, iterations=1000, subdivisions=4,
            progress_callback=None):
    """
    Main entry point for BET brain extraction.

    Parameters:
    -----------
    magnitude_data : ndarray
        3D magnitude image data
    voxel_size : tuple
        Voxel dimensions in mm (z, y, x)
    fractional_intensity : float
        Fractional intensity threshold (0.0-1.0). Smaller = larger brain estimate.
    gradient_threshold : float
        Gradient threshold (currently unused)
    iterations : int
        Number of surface evolution iterations (default: 1000)
    subdivisions : int
        Icosphere subdivision level. 3=642 vertices, 4=2562 vertices (default: 4)
    progress_callback : callable
        Optional callback function(iteration, total) for progress updates
    """
    vertices, faces, mask = bet_surface_evolution(
        magnitude_data,
        voxel_size=voxel_size,
        fractional_intensity=fractional_intensity,
        n_iterations=iterations,
        subdivisions=subdivisions,
        progress_callback=progress_callback
    )

    return mask.astype(np.float32)
