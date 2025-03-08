import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse


# Method 1: Matrix-vector Multiplication

def get_A(N):
    """
    Args:
        N: Number of grid points in each dimension.

    Returns:
        A: 2D finite difference matrix.
    """
    n = N * N
    diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
    
    # Set boundary conditions
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0

    A = np.diag(diagonals[0])  # Main diagonal
    A += np.diag(diagonals[1], 1) +  np.diag(diagonals[2], -1) # right and left neighbor
    A += np.diag(diagonals[3], N) +  np.diag(diagonals[4], -N) # bottom and top neighbor
    
    return A

def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep using matrix-vector multiplication.
    Args:
        A: 2D finite difference matrix. 
        u: N x N grid state at timestep k.
        epsilon: stability constant.

    Returns:
        N x N Grid state at timestep k+1.
    """
    N = u.shape[0]
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u


# Method 2: Sparse Matrix Multiplication with JAX

def get_sparse_A(N):
    """
    Args:
        N: Number of grid points in each dimension.  

    Returns:
        A_sp_matrix: Sparse 2D finite difference matrix in BCOO format.
    """
    n = N * N
    rows, cols, values = [], [], []  # Initialize lists to store nonzero entries
    
    for i in range(n):
        rows.append(i)
        cols.append(i)
        values.append(-4)  # Main diagonal
        
        # Left neighbor
        if i % N != 0:
            rows.append(i)
            cols.append(i - 1)
            values.append(1)
        
        # Right neighbor
        if (i + 1) % N != 0:
            rows.append(i)
            cols.append(i + 1)
            values.append(1)
        
        # Top neighbor
        if i >= N:
            rows.append(i)
            cols.append(i - N)
            values.append(1)
        
        # Bottom neighbor
        if i < n - N:
            rows.append(i)
            cols.append(i + N)
            values.append(1)
    
    # Convert lists to arrays
    idx = np.column_stack((rows, cols))
    values = np.array(values)
    
    # Create sparse matrix in BCOO format
    A_sp = sparse.BCOO((values, idx), shape=(n, n))
    return A_sp


@jax.jit
def advance_time_matvecmul_sparse(A_sp, u_flat, epsilon):
    """Advances the simulation by one timestep using sparse matrix-vector multiplication
    Args:
        A_sp: Sparse 2D finite difference matrix. 
        u_flat: Flattened N x N grid state at timestep k.
        epsilon: Stability constant.

    Returns:
        Flattened grid state at timestep k+1.
    """
    return u_flat + epsilon * (A_sp @ u_flat)


# Method 3: Direct Operation with NumPy

def advance_time_numpy(u, epsilon):
    """Advances the simulation by one timestep using direct NumPy operations.  
    Args:
        u: N x N grid state at timestep k.
        epsilon: Stability constant.
        
    Returns:
        N x N Grid state at timestep k+1.
    """
    u_down = np.roll(u, -1, axis=0)  # Shift down (i+1, j)
    u_up = np.roll(u, 1, axis=0)     # Shift up (i-1, j)
    u_right = np.roll(u, -1, axis=1) # Shift right (i, j+1)
    u_left = np.roll(u, 1, axis=1)   # Shift left (i, j-1)
    
    # Apply zero boundary conditions
    u_down[-1, :] = 0  # bottom boundary
    u_up[0, :] = 0 
    u_right[:, -1] = 0 # right boundary
    u_left[:, 0] = 0
    
    u_new = u + epsilon * (
        u_down +
        u_up +
        u_right +
        u_left -
        4 * u
    )
    
    return u_new


# Method 4: JIT-Compiled Direct Operation with JAX

@jax.jit
def advance_time_jax(u, epsilon):
    """Advances the simulation by one timestep using JAX with JIT compilation.
    Args:
        u: N x N grid state at timestep k.
        epsilon: Stability constant.
        
    Returns:
        N x N Grid state at timestep k+1.
    """
    u_down = jnp.roll(u, -1, axis=0)
    u_up = jnp.roll(u, 1, axis=0)
    u_right = jnp.roll(u, -1, axis=1)
    u_left = jnp.roll(u, 1, axis=1)
    
    # Apply zero boundary conditions
    u_down = u_down.at[-1, :].set(0)
    u_up = u_up.at[0, :].set(0)
    u_right = u_right.at[:, -1].set(0)
    u_left = u_left.at[:, 0].set(0)
    
    u_new = u + epsilon * (
        u_down +
        u_up +
        u_right +
        u_left -
        4 * u
    )
    
    return u_new