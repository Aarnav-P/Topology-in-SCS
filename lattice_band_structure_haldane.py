# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 19:15:52 2025

@author: arceu
"""

import numpy as np
import matplotlib.pyplot as plt

#TODO - make colourmap of sigma for two-axes plot of t2 and M
# %% DEFINE LATTICE
# Pauli matrices
sigma0 = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
sigma1 = np.array([[0.0, 1.0],
                   [1.0, 0.0]])
sigma2 = np.array([[0.0, -1.0j],
                   [1.0j,  0.0]])
sigma3 = np.array([[1.0, 0.0],
                   [0.0, -1.0]])

# Primitive vectors 
a1 = np.array([1.5, np.sqrt(3)/2.0])
a2 = np.array([1.5, -np.sqrt(3)/2.0])

# Reciprocal vectors
b1 = np.array([2*np.pi/3, 2*np.pi*np.sqrt(3)/3]) 
b2 = np.array([2*np.pi/3, -2*np.pi*np.sqrt(3)/3])

# All reciprocal lattice vectors for Brillouin Zone
# (Used to test points in k-space to see if they are within the BZ or not)
G = [
      b1, -b1,
      b2, -b2,
      b2 + b1,
      -(b2 + b1)]

# nearest-neighbor vectors (a_i) connecting A -> B 
delta1 = np.array([0.5, np.sqrt(3)/2])
delta2 = np.array([0.5, -np.sqrt(3)/2])
delta3 = np.array([-1.0, 0.0])

nn = [delta1,delta2,delta3]

# next-nearest-neighbor vectors (b_i) on the same sublattice
nnn = [
    nn[2] - nn[0],  # delta3 - delta1
    nn[1] - nn[2],   # delta2 - delta
    nn[0] - nn[1]  # delta1 - delta2
]

#print(nn)
#print(nnn)

# %% DEFINE FUNCTIONS
def H_of_k(kvec, t1, t2, phi, M):
    """
    Return the 2x2 Bloch Hamiltonian H(k) for a given k-vector.

    Parameters
    ----------
    kvec : array-like, shape (2,)
        Crystal momentum (kx, ky).
    t1 : float
        Nearest-neighbour hopping amplitude (real).
    t2 : float
        Next-nearest-neighbour hopping amplitude (real).
    phi : float
        Complex phase for the n.n.n. hopping (real).
    M : float
        On-site mass (staggered potential).

    Returns
    -------
    H : ndarray, shape (2,2), dtype complex
        The Hamiltonian matrix at kvec.
    """
    k = np.asarray(kvec, dtype=float)

    # sums over nearest neighbors (a_i)
    cos_ka = sum(np.cos(np.dot(k, ai)) for ai in nn)
    sin_ka = sum(np.sin(np.dot(k, ai)) for ai in nn)

    d1 = t1 * cos_ka
    d2 = t1 * sin_ka
    
    # sums over next-nearest neighbors (b_i)
    cos_kb = sum(np.cos(np.dot(k, bi)) for bi in nnn)
    sin_kb = sum(np.sin(np.dot(k, bi)) for bi in nnn)

    # scalar prefactors
    d0 = 2.0 * t2 * np.cos(phi) * cos_kb
    d3 = M - 2.0 * t2 * np.sin(phi) * sin_kb
    
    # combine
    H = (d0*sigma0
         + d1*sigma1
         + d2*sigma2
         + d3*sigma3)

    return H

# Convenience: eigenvalues and eigenvectors
def bands_at_k(kvec, **hparams):
    """Return eigenvalues (sorted ascending) and eigenvectors of H(k)."""
    H = H_of_k(kvec, **hparams)
    eigvals, eigvecs = np.linalg.eigh(H)
    # eigvals ascending; eigvecs columns correspond to eigenvectors
    return eigvals, eigvecs

# Optional: compute bands on a k-grid (mesh) and return eigenvalues arrays
def bands_on_mesh(kx_array, ky_array, **hparams):
    """
    Compute two bands on the grid defined by kx_array x ky_array.
    Returns eigs shape (2, len(kx), len(ky)) with eigs[0] = lower band, eigs[1] = upper band.
    """
    kx = np.asarray(kx_array)
    ky = np.asarray(ky_array)
    NX = kx.size
    NY = ky.size
    eigs = np.zeros((2, NX, NY), dtype=float)
    for i, kxi in enumerate(kx):
        for j, kyj in enumerate(ky):
            vals, _ = bands_at_k([kxi, kyj], **hparams)
            eigs[:, i, j] = np.sort(np.real(vals))
    return eigs


# %% DEFINE PARAMETERS AND CONSTANTS
# Parameters
Nk = 50  # grid resolution
t1 = 1.0
t2 = 0.03
phi = np.pi/2
M = 0.1
k_lim = 1*np.pi
# constants (set to 1 for natural units)
hbar = 1.0   # set 1.0 for natural units; put 1.054571817e-34 for SI
e_charge = 1.0  # set 1.602176634e-19 for SI
E_F = 0
# %% BANDSTRUCTURE AND FULL BZ SAMPLING
def evaluate_band_structure(Nk,t1,t2,phi,M,k_lim):
    # sample full BZ (kx,ky) grid and get band energies
    kxs = np.linspace(-k_lim, k_lim, Nk)
    kys = np.linspace(-k_lim, k_lim, Nk)
    energies = np.zeros((Nk, Nk, 2), dtype=float)
    
    def gap_at_dirac_points():
        """Test for energy difference in gap at K,K' """
        
        K = np.array([2*np.pi/3, 2*np.pi/(3*np.sqrt(3))])
        K_prime = np.array([2*np.pi/3, -2*np.pi/(3*np.sqrt(3))])
        E1 = H_of_k(K, t1=t1, t2=t2, phi=phi, M=M)
        evals1 = np.linalg.eigvalsh(E1)
        energies1 = np.sort(evals1.real)
        E2 = H_of_k(K_prime, t1=t1, t2=t2, phi=phi, M=M)
        evals2 = np.linalg.eigvalsh(E2)
        energies2 = np.sort(evals2.real)
        print(f"Energies are: {energies1} at K, {energies2} at K'")
        return 0 
    
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            H = H_of_k(np.array([kx, ky]), t1=t1, t2=t2, phi=phi, M=M)
            evals = np.linalg.eigvalsh(H)
            energies[ix, iy, :] = np.sort(evals.real)
    
    # 1D cut: energy vs kx for fixed ky = cut_val
    cut_val = (2*np.pi)/(3*np.sqrt(3))  # typical cut through K
    kcut = np.linspace(-k_lim, k_lim, Nk)
    lower_band, upper_band = [], []
    
    for kx in kcut:
        H = H_of_k(np.array([kx, cut_val]), t1=t1, t2=t2, phi=phi, M=M)
        e = np.linalg.eigvalsh(H)
        lower_band.append(e[0].real)
        upper_band.append(e[1].real)
    
    phi_str = r'$\phi=$'
    t1_str = r'$t_1=$'
    t2_str = r'$t_2=$'
    M_str = r'$M=$'
    
    plt.figure(figsize=(6,4))
    plt.plot(kcut, lower_band, label="lower band")
    plt.plot(kcut, upper_band, label="upper band")
    plt.xlabel(fr'$k_x$  (for $k_y={cut_val:4.3f}$)')
    plt.ylabel('Energy')
    plt.title(f'Honeycomb lattice {phi_str}{phi:3.2f}, {t1_str}{t1:3.2f}, {t2_str}{t2:3.2f}, {M_str}{M:3.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2D band maps (full BZ)
    KX, KY = np.meshgrid(kxs, kys, indexing='ij')
    E_lower = energies[:,:,0]
    E_upper = energies[:,:,1]
    
    # =============================================================================
    # # lower band contour
    # plt.figure(figsize=(6,5))
    # plt.contourf(KX, KY, E_lower, levels=80, cmap='RdBu_r')
    # plt.colorbar(label='Energy')
    # plt.title(f'Lower band (E₋) {phi_str}{phi:3.2f}, {t1_str}{t1:3.2f}, {t2_str}{t2:3.2f}, {M_str}{M:3.2f}')
    # plt.xlabel(r'$k_x$')
    # plt.ylabel(r'$k_y$')
    # plt.tight_layout()
    # plt.show()
    # 
    # # upper band contour
    # plt.figure(figsize=(6,5))
    # plt.contourf(KX, KY, E_upper, levels=80, cmap='RdBu_r')
    # plt.colorbar(label='Energy')
    # plt.title(f'Upper band (E₊) {phi_str}{phi:3.2f}, {t1_str}{t1:3.2f}, {t2_str}{t2:3.2f}, {M_str}{M:3.2f}')
    # plt.xlabel(r'$k_x$')
    # plt.ylabel(r'$k_y$')
    # plt.tight_layout()
    # plt.show()
    # =============================================================================
    
    # 3D surface plots of both bands
# =============================================================================
#     fig = plt.figure(figsize=(12,5))
#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     ax1.plot_surface(KX, KY, E_lower, cmap='viridis', edgecolor='none', alpha=0.9)
#     ax1.plot_surface(KX, KY, E_upper, cmap='plasma', edgecolor='none', alpha=0.6)
#     ax1.set_title(f'Honeycomb lattice bands {phi_str}{phi:3.2f}, {t1_str}{t1:3.2f}, {t2_str}{t2:3.2f}, {M_str}{M:3.2f}')
#     ax1.set_xlabel(r'$k_x$')
#     ax1.set_ylabel(r'$k_y$')
#     ax1.set_zlabel('Energy')
#     
# =============================================================================
    plt.tight_layout()
    plt.show()
# %% BERRY CURVATURE

def d_coeffs_and_derivs(kvec, t1, t2, phi, M):
    """Return d0,d1,d2,d3 and their derivatives wrt kx, ky at kvec."""
    k = np.asarray(kvec, dtype=float)
    # initialise arrays, leaving sum until later because of derivatives
    cos_ka = np.array([np.cos(np.dot(k, ai)) for ai in nn])
    sin_ka = np.array([np.sin(np.dot(k, ai)) for ai in nn])
    cos_kb = np.array([np.cos(np.dot(k, bi)) for bi in nnn])
    sin_kb = np.array([np.sin(np.dot(k, bi)) for bi in nnn])

    # d's
    d0 = 2.0 * t2 * np.cos(phi) * cos_kb.sum()
    d1 = t1 * cos_ka.sum()
    d2 = t1 * sin_ka.sum()
    d3 = M - 2.0 * t2 * np.sin(phi) * sin_kb.sum()

    # derivatives wrt kx and ky
    # For nearest neighbors:
    a_x = np.array([ai[0] for ai in nn])
    a_y = np.array([ai[1] for ai in nn])
    # For next-nearest neighbors:
    b_x = np.array([bi[0] for bi in nnn])
    b_y = np.array([bi[1] for bi in nnn])

    # derivatives are done analytically and then typed up
    dd0_dkx = 2.0 * t2 * np.cos(phi) * (- (b_x * sin_kb).sum())
    dd0_dky = 2.0 * t2 * np.cos(phi) * (- (b_y * sin_kb).sum())

    dd1_dkx = t1 * ( - (a_x * sin_ka).sum() )
    dd1_dky = t1 * ( - (a_y * sin_ka).sum() )

    dd2_dkx = t1 * ( (a_x * cos_ka).sum() )
    dd2_dky = t1 * ( (a_y * cos_ka).sum() )

    dd3_dkx = -2.0 * t2 * np.sin(phi) * ( (b_x * cos_kb).sum() )
    dd3_dky = -2.0 * t2 * np.sin(phi) * ( (b_y * cos_kb).sum() )
    
    # overall arrays. Once multiplied by appropriate factors these will give
    # H, J_x, J_y etc.
    d = np.array([d0, d1, d2, d3], dtype=complex)
    dd_dkx = np.array([dd0_dkx, dd1_dkx, dd2_dkx, dd3_dkx], dtype=complex)
    dd_dky = np.array([dd0_dky, dd1_dky, dd2_dky, dd3_dky], dtype=complex)

    return d, dd_dkx, dd_dky

def dmatrix_from_coeffs(coeffs):
    """H = Sum over j of (d_j * sigma_j)"""
    return coeffs[0]*sigma0 + coeffs[1]*sigma1 + coeffs[2]*sigma2 + coeffs[3]*sigma3

def check_in_BZ(kx,ky):
    """Takes point in k space and compares distance to origin with reciprocal
    lattice vectors to see if inside Wigner-Seitz cell (i.e. first BZ)
    Uses eq26, Kittel's SSP"""
    k = np.array([kx,ky])
   
    for vec in G:
        if np.dot(k, vec) >= 0.5 * np.dot(vec, vec):
            return False
    return True

def compute_sigma_xy_kubo(Nk, t1, t2, phi, M, use_hbar, use_e):
    """
    Compute sigma_xy using the Kubo formula provided in the David Tong notes.
    Returns sigma_xy in units where e and hbar are 1 if use_e=use_hbar=1.
    """
    k_lim_x = 2*np.pi/3
    k_lim_y = 4*np.pi/(3*np.sqrt(3))
    kxs = np.linspace(-k_lim_x, k_lim_x, Nk)
    kys = np.linspace(-k_lim_y, k_lim_y, Nk)
    dkx = kxs[1]-kxs[0] # kx length element
    dky = kys[1]-kys[0] # ky length element
    area_element = dkx*dky
# =============================================================================
#     # Visualisation of k space and Brillouin zone
#     inside_x, inside_y = [], []
#     outside_x, outside_y = [], []
# 
#     for x in kxs:
#         for y in kys:
#             if check_in_BZ(x, y):
#                 inside_x.append(x)
#                 inside_y.append(y)
#             else:
#                 outside_x.append(x)
#                 outside_y.append(y)
# 
#     fig, ax = plt.subplots(figsize=(8,8))
#     KX, KY = np.meshgrid(kxs, kys)
#     ax.scatter(KX, KY)
#     ax.scatter(inside_x, inside_y, color='green', s=20, label='Inside 1st BZ')
#     ax.scatter(outside_x, outside_y, color='blue', s=20, label='Outside 1st BZ')
#     ax.set_xlabel(r'$k_x$')
#     ax.set_ylabel(r'$k_y$')
#     ax.set_xlim(-k_lim_x-0.2,k_lim_x+0.2)
#     ax.set_ylim(-k_lim_y-0.2,k_lim_y+0.2)
#     ax.set_aspect('equal')
#     ax.plot([0,b1[0]], [0,b1[1]], 'ro--', label=r'$\mathbf{b}_1$')
#     ax.plot([0,b2[0]], [0,b2[1]], 'ro--', label=r'$\mathbf{b}_2$')
#     ax.legend()
#     ax.set_title('k-space grid with First Brillouin Zone')
#     plt.show()
# =============================================================================
    # initialise sigma_xy
    sigma_sum = 0.0 + 0.0j
    eps_BC = 1e-8 # filter out tiny denominators
    Omega_map = np.zeros((Nk,Nk)) 
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            if check_in_BZ(kx,ky):
                k = np.array([kx, ky])
                d, dd_dkx, dd_dky = d_coeffs_and_derivs(k, t1=t1, t2=t2, phi=phi, M=M)
                Hk = dmatrix_from_coeffs(d)
                # derivative matrices
                dHdkx = dmatrix_from_coeffs(dd_dkx)
                dHdky = dmatrix_from_coeffs(dd_dky)
    
                #J_i = -e/hbar * dH/dki
                Jx = (use_e/use_hbar) * dHdkx 
                Jy = (use_e/use_hbar) * dHdky
    
                # diagonalize
                evals, evecs = np.linalg.eigh(Hk)  # evals ascending; evecs columns
                # Now technically there will never be a swapping but this code is written
                # generally, such that the eigenvectors *could* behave in any way really,
                # and any band could be filled or unfilled.
                
                # pick occupied/unoccupied indices according to E_F
                occ_inds = np.where(evals < E_F-eps_BC)[0]
                unocc_inds = np.where(evals > E_F+eps_BC)[0]
    
                # if there are no occupied or no unoccupied states, skip
                if occ_inds.size == 0 or unocc_inds.size == 0:
                    continue
    
                # for 2x2 if single band occupied this reduces to single sum
                for a in occ_inds:
                    for b in unocc_inds:
                        Ea = evals[a]
                        Eb = evals[b]
                        denom = (Eb - Ea)**2
                        if abs(Eb-Ea) < eps_BC:
                            continue
                        ua = evecs[:, a]
                        ub = evecs[:, b]
    
                        # matrix elements
                        Jy_ab = np.vdot(ua, Jy.dot(ub))   # <u^a | J_y | u^b >
                        Jx_ba = np.vdot(ub, Jx.dot(ua))   # <u^b | J_x | u^a >
                        Jx_ab = np.vdot(ua, Jx.dot(ub))
                        Jy_ba = np.vdot(ub, Jy.dot(ua))
    
                        num = (Jy_ab * Jx_ba) - (Jx_ab * Jy_ba)
                        Omega_map[ix, iy] = -np.imag(num/(Eb - Ea)**2)

                        sigma_k = 1j * use_hbar * num / denom # i*hbar prefactor as in Tong
                        sigma_sum += sigma_k * area_element # numerical integral performed over BZ
            else:
                Omega_map[ix,iy] = 0
                
    plt.figure(figsize=(6,5))
    plt.contourf(kxs, kys, Omega_map.T, levels=80, cmap='RdBu') 
    plt.colorbar(label=r'$\Omega_z(\mathbf{k})$')
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.title(f'Berry curvature of lower Haldane band, φ={phi:.2f}, M={M:.2f}')
    plt.show()
    # prefactor from integral normalization is 1/(2pi)^2
    sigma_xy = sigma_sum / (2.0 * np.pi)**2
    # sigma_xy is complex numerically; physical value is its real part (imaginary should be ~0)
    return np.real(sigma_xy)

def compute_chern_number_FHS(Nk, t1, t2, phi, M, plot):
    """
    Compute the Chern number of a band using the FHS method.
    
    Parameters
    ----------
    Nk : int
        Number of k-points in each direction (NxN grid)
    t1, t2, phi, M : floats
        Haldane model parameters.
    plot: bool
        Decides whether or not to create a plot.
    Returns
    -------
    C : float
        Chern number of the lower band.
    Omega_FHS : 2D array
        Lattice of Berry fluxes on the k-grid 
    """
    # Generate k-grid
    kxs = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    kys = np.linspace(-np.pi, np.pi, Nk, endpoint=False)

    # Storage for link variables and Berry fluxes
    Ux = np.zeros((Nk, Nk), dtype=complex)
    Uy = np.zeros((Nk, Nk), dtype=complex)
    Omega_FHS = np.zeros((Nk, Nk))

    # compute eigenvectors of lower band
    psi = np.zeros((Nk, Nk, 2), dtype=complex)
    for i, kx in enumerate(kxs):
        for j, ky in enumerate(kys):
            _, evecs = bands_at_k([kx, ky], t1=t1, t2=t2, phi=phi, M=M)
            psi[i,j,:] = evecs[:,0]  # lower band eigenvector

    # Compute link variables
    for i in range(Nk):
        for j in range(Nk):
            ip = (i+1) % Nk
            jp = (j+1) % Nk

            # Link variable by direction
            Ux[i,j] = np.vdot(psi[i,j,:], psi[ip,j,:])
            Uy[i,j] = np.vdot(psi[i,j,:], psi[i,jp,:])

            # Normalize
            Ux[i,j] /= np.abs(Ux[i,j])
            Uy[i,j] /= np.abs(Uy[i,j])

    # Compute "lattice field strength" (Berry flux)
    for i in range(Nk):
        for j in range(Nk):
            ip = (i+1) % Nk
            jp = (j+1) % Nk
            
            # Simplest plaquette
            F = np.log(Ux[i,j]*Uy[ip,j]/(Ux[i,jp]*Uy[i,j]))
            Omega_FHS[i,j] = np.imag(F)

    # Total Chern number
    C = np.sum(Omega_FHS) / (2*np.pi)
    
    
    # To make this work for your bands, I think it would be something like 
    # for n in range (bands):
    #    for i, kx in enumerate(kxs):
    #        for j, ky in enumerate(kys):
    #            _, evecs = bands_at_k([kx, ky], t1=t1, t2=t2, phi=phi, M=M)
    #            psi[i,j,:] = evecs[:,n]  # lower band eigenvector    
    # 
    # ... Repeat calculation of link variables as previous
    # Then the Chern number becomes C_n = np.sum(Omega_FHS_n) / (2*np.pi)
    # And C = np.sum(C_n)
    # You would want to implement a check to see which bands are occupied at below the Fermi_energy
    # I have absolutely no idea how this would work for a partially occupied band.
    
    if plot:
        # Plot FHS Berry flux
        plt.figure(figsize=(10,8))
        plt.contourf(kxs, kys, Omega_FHS.T, levels=80, cmap='RdBu_r')
        plt.colorbar(label='FHS Berry curvature')
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$k_y$')
        plt.title(f'FHS Berry curvature lower Haldane band, C={C:.2f}, φ={phi:.2f}, M={M:.2f}, t2={t2:.2f}')
        plt.show()
    
    def round_if_close(x, eps):
        """
        Round to the nearest integer if the value is within eps of it.
        Works on scalars or numpy arrays.
        """
        x = np.asarray(x)
        nearest = np.round(x)
        mask = np.abs(x - nearest) < eps
        result = np.where(mask, nearest, x)
        # preserve input type
        return result.item() if np.isscalar(x) else result
    
    C = round_if_close(C,1e-8)
    
    return C


#%% MAIN

def main():
    
    #evaluate_band_structure(Nk,t1,t2,phi,M,k_lim)
    
    #sigma_xy = compute_sigma_xy_kubo(Nk,t1,t2,phi,M,hbar,e_charge)
    #print(f"sigma_xy is {sigma_xy:4.3f}")
    
    #C = compute_chern_number_FHS(Nk, t1, val_t2, phi, val_M, plot = True)
    
    M_range = [0.5,2]
    NM = 50
    t2_range = [0.15,0.3]
    Nt2 = 10
    M_vals = np.linspace(M_range[0],M_range[1],NM)
    t2_vals = np.linspace(t2_range[0],t2_range[1],Nt2)
    C_map= np.zeros((NM,Nt2))

    for p, val_M in enumerate(M_vals):
        for q, val_t2 in enumerate(t2_vals):
            #evaluate_band_structure(Nk, t1, val_t2, phi, val_M, k_lim)
            C = compute_chern_number_FHS(Nk, t1, val_t2, phi, val_M, plot = True)

            C_map[p,q] = C
            

    ratio = (M_range[1] - M_range[0])/(t2_range[1]-t2_range[0])
    x_ticks = np.linspace(M_range[0],M_range[1],NM+1)
    y_ticks =  np.linspace(t2_range[0],t2_range[1],Nt2+1)
    fig, ax = plt.subplots(figsize=(8,8))
    colourplot = ax.imshow(C_map.T, aspect=ratio, extent = (M_range[0], M_range[1], t2_range[0], t2_range[1]), cmap='RdBu')
#    plt.contourf(M_vals, t2_vals, C_map.T, cmap='RdBu') 
    #.T because contourf assumes the array is [y,x]
    fig.colorbar(colourplot, ax=ax, label=r'$\Omega_z(\mathbf{k})$')
    ax.set_xticks(x_ticks[::(int(np.round(NM/5)))])
    ax.set_yticks(y_ticks[::(int(np.round(Nt2/5)))])
    ax.set_xlabel(r'$M$')
    ax.set_ylabel(r'$t2$')
    ax.set_title(f'Parameter space of honeyomb lattice Chern number, φ={phi:.2f}')
    ax.grid()
    plt.show()
    return 0 

main()

