import torch


def fftfreq(n, d, device='cpu', dtype=torch.float64):

    val = 1.0 / (n * d)
    pos = torch.arange(0, (n + 1) // 2, device=device, dtype=dtype)
    neg = torch.arange(-n // 2, 0, device=device, dtype=dtype)
    freqs = torch.cat((pos, neg)) * val
    return freqs * 2 * torch.pi  # Convert to angular frequencies
def vol(points, vol, device='cpu', eps=1e-3):

  
    
    centroid = torch.mean(points, dim=0)
    c = torch.floor(centroid)
    points_shifted = points - c
    normals = -points_shifted / torch.norm(points_shifted, dim=1, keepdim=True)
    
    xmin = torch.floor(torch.min(points_shifted[:, 0]) - 5).item()
    xmax = torch.ceil(torch.max(points_shifted[:, 0]) + 5).item()
    ymin = torch.floor(torch.min(points_shifted[:, 1]) - 5).item()
    ymax = torch.ceil(torch.max(points_shifted[:, 1]) + 5).item()
    zmin = torch.floor(torch.min(points_shifted[:, 2]) - 5).item()
    zmax = torch.ceil(torch.max(points_shifted[:, 2]) + 5).item()

    # Create axis ranges ensuring shapes are even
    x_range = torch.arange(xmin, xmax, device=device, dtype=torch.float32)
    if x_range.shape[0] % 2 != 0:
        x_range = torch.arange(xmin, xmax + 1, device=device, dtype=torch.float32)

    y_range = torch.arange(ymin, ymax, device=device, dtype=torch.float32)
    if y_range.shape[0] % 2 != 0:
        y_range = torch.arange(ymin, ymax + 1, device=device, dtype=torch.float32)

    z_range = torch.arange(zmin, zmax, device=device, dtype=torch.float32)
    if z_range.shape[0] % 2 != 0:
        z_range = torch.arange(zmin, zmax + 1, device=device, dtype=torch.float32)

    # Create meshgrid with proper axis ordering (X, Y, Z)
    X, Y, Z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    # Get correct grid dimensions (nx=X, ny=Y, nz=Z)
    nx, ny, nz = X.shape[0], Y.shape[1], Z.shape[2]

    # Calculate voxel sizes using CORRECT dimensions
    dx = 1
    dy = 1
    dz = 1
    
    indices = torch.zeros_like(points, dtype=torch.long)
    
    indices[:, 0] = ((points_shifted[:, 0] - xmin) / (xmax - xmin) * nx).clamp(0, nx-1).long()
    
    indices[:, 1] = ((points_shifted[:, 1] - ymin) / (ymax - ymin) * ny).clamp(0, ny-1).long()
    
    indices[:, 2] = ((points_shifted[:, 2] - zmin) / (zmax - zmin) * nz).clamp(0, nz-1).long()

    # Initialize gradient field with CORRECT dimensions (nx, ny, nz)
    N_field = torch.zeros((nx, ny, nz, 3), dtype=torch.complex128, device=device)
    
    for idx in range(points.shape[0]):
        i, j, k = indices[idx]
        N_field[i, j, k] += normals[idx].to(device).type(torch.complex128)

    # FFT calculations remain similar but with corrected dimensions
    N_hat = torch.fft.fftn(N_field, dim=(0, 1, 2))
    # Generate frequency grids with PROPER dimension alignment
    kx = fftfreq(nx, dx, device=device)
    ky = fftfreq(ny, dy, device=device)
    kz = fftfreq(nz, dz, device=device)
    
    # Create meshgrid with corrected axis ordering
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K_sq = KX**2 + KY**2 + KZ**2
    # Compute characteristic function coefficients
    with torch.no_grad():
        filter_coeff = 1j / (K_sq + eps)
        filter_coeff[K_sq < eps] = 0
        chi_hat = filter_coeff * (KX * N_hat[..., 0] + KY * N_hat[..., 1] + KZ * N_hat[..., 2])


    # Inverse FFT and return real component
    chi = -torch.fft.ifftn(chi_hat, dim=(0, 1, 2)).real
    
    chi = torch.where(chi > 0, torch.tensor(1.0), torch.tensor(0.0))
    
    d2 = torch.zeros_like(vol, dtype=torch.float32)
    c_int = c.to(torch.int64)
    
    x_start = int(xmin + c_int[0].item())
    x_end   = x_start + chi.shape[0]
    
    y_start = int(ymin + c_int[1].item())
    y_end   = y_start + chi.shape[1]
    
    z_start = int(zmin + c_int[2].item())
    z_end   = z_start + chi.shape[2]

    #d2[x_start:x_end, y_start:y_end, z_start:z_end] = chi
    return chi
