# %% -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 19:37:26 2025

@author: arceu

This code is written to numerically evaluate the self-consistent equations from
the Supplementary Information from Wagner et al. (2024) on Edge Zeroes and 
Boundary Spinons. The procedure is outlined below equation (17).

The idea is to guess a value for Q_X, which is defined in terms of the Green's
function for ψ (G_ψ) in equations (13). This means guessing some numerical 
inputs for G_ψ.

The value of the Lagrange multiplier rho is then determined via its constraint 
in equation (12), which is used to determine the value of Z - the condensate 
fraction of G_X.

Once this is done, G_X can be defined via the guessed values for Q_X and the 
determined value for rho. 

This allows us to define Q_ψ in terms of G_X via equations (14), which allows 
us to define G_ψ again, but this time having optimised via the constraint 
enforced by rho. 

We repeat this process to converge on stable values.

The schematic structure of the code is to store all these variables inside a 
class so we can dynamically update them as the process iterates and track
the iterations by storing states in the object.

> There will be a class which encodes the geometry of the situation
> And a class which solves the Green's functions using the geometry attributes.

This allows the code to stay cleanly defined, clearly updated and keeps 
separate parts that should not interact. It also allows the code to execute on
different geometries easily, and also to solve for different systems with the 
same geometries.
"""
# %% Imports
import numpy as np
import scipy.optimize
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import ast
from matplotlib.colors import TwoSlopeNorm
import time
from contextlib import contextmanager

@contextmanager
def timer(label, timing_dict=None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if timing_dict is not None:
        timing_dict[label] = timing_dict.get(label, 0.0) + dt
    else:
        print(f"{label}: {dt:.6f} s")

# %% Definition of Lattice parameters

LATTICE_CONSTANT = 1
# CALCULATIONS IN PAPER PERFORMED WITH λ/t = 0.5 and T/t = 0.01
t = 1 # hopping parameter
λ = 0.5*t # SOC parameter
U = 10*t # sometimes they use U/t = 12
T = 0.01 # Temperature
Ns = 30 # Number of unit cells
n = 1000
mixing = 0.5 # 0 <mixing <= 1, Q_new = Qn + mixing*Qn+1

# lattice vectors
a1 = np.array([np.sqrt(3),1])/2 * LATTICE_CONSTANT
a2 = np.array([-np.sqrt(3),1])/2 * LATTICE_CONSTANT

def rotate_vec(vec, angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return R @ vec

# nearest neighbours (nn's)
u1 = np.array([1/np.sqrt(3),0]) * LATTICE_CONSTANT
u2 = rotate_vec(u1, 2*np.pi/3) 
u3 = rotate_vec(u1, 4*np.pi/3) 

# next nearest neighbours (nnn's)
y1 = a1+a2
y2 = rotate_vec(y1, 2*np.pi/3) 
y3 = rotate_vec(y1, 4*np.pi/3) 

area = a1[0]*a2[1] - a1[1]*a2[0]

b1 = 2*np.pi * np.array([ a2[1], -a2[0] ]) / area
b2 = 2*np.pi * np.array([ -a1[1], a1[0] ]) / area

# Wigner–Seitz needs all reciprocal vectors pointing to nearest lattice points
G = [b1, b2, -(b1), -(b2), (b1-b2), -(b1-b2)]
# %% Useful misc functions

def fmt(x):
    # format a scalar (possibly complex) to 2dp scientific notation
    if np.iscomplexobj(x):
        return f"{x.real:.2e}{x.imag:+.2e}j"
    else:
        return f"{x:.2e}"

def fmt_array(A):
    # apply fmt elementwise and preserve array shape
    return np.vectorize(fmt)(A)
# %% Definition of Lattice class
class Lattice:
    def __init__(self,lattice_vectors,nn,nnn,SOC,hopping):
        """
        This class requires you to calculate the nn,nnn beforehand.
        
        lattice_vectors : *ARRAY* of shape (n,m)
                          e.g. here n=1, m=2 i.e. [a1, a2] where a1, a2 are 2D vectors
        nn             : *ARRAY* of NN displacement vectors u_j, precompiled.
        nnn            : *ARRAY* of NNN displacement vectors gamma_j, precompiled.
        SOC            : lambda 
        hopping        : t
        """
        
        self.a1, self.a2 = np.array(lattice_vectors[0]), np.array(lattice_vectors[1])
        self.nn  = [np.array(v) for v in nn]
        self.nnn = [np.array(v) for v in nnn]
        self.λ = SOC
        self.t = hopping
        self.epsk_all = []
        self.epskAσ_all = []
        self.epskBσ_all = []
        self.Vk_all = []
        self.epsk_plot = []
        self.epskAσ_plot = []
        self.epskBσ_plot = []
        self.Vk_plot = []
        self.k_grid = []
    
    # TODO, make this work with only a1,a2, number of bases in unit cell
    def eps_k(self, k):
        sum_term = sum(np.cos(np.dot(vec, k)) for vec in self.nnn)
        return 2*self.λ * sum_term
     
    def eps_ksσ(self, k, s, σ):
        """
        k - float
        s - str. should be 'a' or 'b'
        σ - int. Should be 1 or -1 (for up or down)
        Handles arrays of k values.
        """
        k = np.asarray(k)
        if k.ndim == 1:
            k = k[None,:]
        if σ not in (1,-1):
            raise ValueError("σ must be ±1")
        nnn = np.asarray(self.nnn)
        dot_product = np.tensordot(k, nnn, axes=([1],[1]))
        if s == 'a':
            phase = dot_product + np.pi*σ/2
        elif s == 'b':
            phase = dot_product - np.pi*σ/2
        else:
            raise ValueError("s must be 'a' or 'b'")
        sum_term = np.sum(np.cos(phase), axis=-1)
        result = (2*self.λ * sum_term)
        if result.shape[0] == 1:
            return result[0]
        else:
            return result
    
# =============================================================================
#     def Vk(self, k):
#         k = np.asarray(k)
#         if k.ndim == 1:
#             k = k[None,:]
#         nn = np.asarray(self.nn)
#         phase = np.tensordot(k, nn, axes=([1],[1]))
#         
#         result = (np.sum(np.exp(1j*phase), axis=-1))
#         
#         if result.shape[0] == 1:
#             return result[0]
#         else:
#             return result
# =============================================================================
    
    def Vk(self, k):
        k = np.asarray(k)
        if k.ndim == 1:
            k = k[None, :]
            
        # Use the periodic gauge to drop the non-periodic exp(i k . u1) phase
        # V(k) = 1 + exp(i k . a2) + exp(-i k . a1)
        phase_a1 = np.tensordot(k, self.a1, axes=([1], [0]))
        phase_a2 = np.tensordot(k, self.a2, axes=([1], [0]))
        
        result = (1.0 + np.exp(-1j * phase_a1) + np.exp(1j * phase_a2))
        
        if result.shape[0] == 1:
            return result[0]
        else:
            return result
        
    def dVk_dk(self, k, axis='x'):
        # axis is 'x' or 'y'. 
        idx = 0 if axis == 'x' else 1
        
        k = np.asarray(k)
        if k.ndim == 1: k = k[None, :]
            
        phase_a1 = np.tensordot(k, self.a1, axes=([1], [0]))
        phase_a2 = np.tensordot(k, self.a2, axes=([1], [0]))
        
        # Chain rule pulls down the x or y component of the lattice zvectors
        term1 = -1j * self.a1[idx] * np.exp(-1j * phase_a1)
        term2 =  1j * self.a2[idx] * np.exp( 1j * phase_a2)
        
        result = term1 + term2
        return result[0] if result.shape[0] == 1 else result
    
    def deps_ksσ_dk(self, k, s, σ, axis='x'):
        idx = 0 if axis == 'x' else 1
        k = np.asarray(k)
        if k.ndim == 1: k = k[None, :]
        nnn_array = np.array(self.nnn)    
        # Assuming self.nnn holds your 3 gamma vectors
        dot_product = np.tensordot(k, nnn_array, axes=([1],[1]))
        
        phase = dot_product + (np.pi*σ/2 if s == 'a' else -np.pi*σ/2)
        
        # pull down the x or y components of the gamma vectors
        gamma_components = nnn_array[:, idx] 
        
        deriv_terms = -gamma_components * np.sin(phase)
        result = 2 * self.λ * np.sum(deriv_terms, axis=-1)  
        
        return result[0] if result.shape[0] == 1 else result

    def lattice_k_mesh(self, N_BZ1, N_BZ2):
        """
        Generate k-points in the Brillouin zone spanned by reciprocal vectors b1, b2.
        Returns array of shape (N_BZ1*N_BZ2, 2) of 2D k-vectors.
        """        
        k_list = []
        delta = 0.5
        
        for i in range(N_BZ1):
            x = (i+delta)/N_BZ1
            for j in range(N_BZ2):                
                y = (j+delta)/N_BZ2
                
                k = x*b1 + y*b2
#                if self.check_in_BZ(k[0],k[1]):
                k_list.append(k)
            
        return np.array(k_list) - ((b1+b2)/2)
    
    def plot_BZ(self, N_BZ1 , N_BZ2):
        grid = self.lattice_k_mesh(N_BZ1, N_BZ2)
        plt.rcParams["font.size"] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.scatter(grid[:,0],grid[:,1], s=2)
        plt.arrow(0,0,b1[0],b1[1], head_width=0.5, color = 'k', length_includes_head=True)
        plt.text(b1[0]-1,b1[1]-0.1, r"$\vec{b_1}$")
        plt.arrow(0,0,b2[0],b2[1], head_width=0.5, color = 'k', length_includes_head=True)
        plt.text(b2[0]+0.3,b2[1]-0.1, r"$\vec{b_2}$")
        gamma = r"$\Gamma$"
        locs = [[0,0]] # b1, b2, b1+b2
        for point in locs:
            plt.text(point[0]*1.1,point[1]-0.1, gamma, color='r')
        plt.gca().set_aspect('equal')

        plt.grid(True)
        #plt.xlim(b2[0]-1, b1[0]+1)
        #plt.ylim(-0.5, 2*b1[1]+0.5)
        plt.title("BZ for Honeycomb")
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.show()        
        plt.close()
        
    def calculate_structure_functions(self, N_BZ1, N_BZ2):
        """
        Produces contour plots of eps_k(k), eps_ksσ and |V_k(k)| over the BZ mesh used
        by `lattice_k_mesh`.
        """

        self.epskAσ_all = []
        self.epskBσ_all = []
        self.Vk_all = []
        # need to reshape list into 2D to plot.
        k_grid = self.lattice_k_mesh(N_BZ1, N_BZ2)     # shape → (N_BZ1*N_BZ2, 2)
        self.k_grid = k_grid
        kx = k_grid[:,0].reshape(N_BZ1, N_BZ2)
        ky = k_grid[:,1].reshape(N_BZ1, N_BZ2)
    
        # Evaluate variables
        epsk = np.zeros((N_BZ1, N_BZ2))
        Vk = np.zeros((N_BZ1, N_BZ2), dtype=np.complex128)
        epskAσ = np.zeros((2,N_BZ1, N_BZ2))
        epskBσ = np.zeros((2,N_BZ1, N_BZ2))
        
        for i in range(N_BZ1):
            for j in range(N_BZ2):
                k = np.array([kx[i,j], ky[i,j]])
                epsk[i,j] = self.eps_k(k)
                Vk[i,j] = self.Vk(k)
                epskAσ[0,i,j] = self.eps_ksσ(k, 'a', 1)
                epskAσ[1,i,j] = self.eps_ksσ(k, 'a', -1)
                epskBσ[0,i,j] = self.eps_ksσ(k, 'b', 1)
                epskBσ[1,i,j] = self.eps_ksσ(k, 'b', -1)

        self.epsk_plot = epsk
        self.epskAσ_plot = epskAσ
        self.epskBσ_plot = epskBσ
        self.Vk_plot = Vk
        
        # Flatten all arrays into 1D lists matching k_grid indexing
        self.epsk_all = epsk.reshape(N_BZ1*N_BZ2)
        self.Vk_all = Vk.reshape(N_BZ1*N_BZ2)
        # For σ-dependent eps_kAσ and eps_kBσ keep the σ dim.
        self.epskAσ_all = epskAσ.reshape(2, N_BZ1*N_BZ2)
        self.epskBσ_all = epskBσ.reshape(2, N_BZ1*N_BZ2)
        
    def plot_structure_functions(self, N_BZ1, N_BZ2):
        #PLOTTING

        epsk = self.epsk_plot
        Vk = self.Vk_plot
        epskAσ = self.epskAσ_plot
        epskBσ = self.epskBσ_plot
        Vk_abs = np.abs(Vk)
        Vk_arg = np.angle(Vk)
        k_grid = self.k_grid 
        kx = k_grid[:,0].reshape(N_BZ1, N_BZ2)
        ky = k_grid[:,1].reshape(N_BZ1, N_BZ2)
        
        # contour plots 
        #plt.gca().set_aspect('equal')
        fig, ax = plt.subplots(1, 2, figsize=(7,5))
        plt.gca().set_aspect('equal')
        # epsk
        try:
            norm_eps = TwoSlopeNorm(vmin=np.min(epsk), vcenter=0.0, vmax=np.max(epsk))
        except:
            norm_eps=None
            
        ctr1 = ax[0].contourf(kx, ky, Vk_arg, levels=40, cmap="RdBu_r", norm=norm_eps)
        ax[0].set_title(r"Vk_arg")
        ax[0].set_xlabel(r"$k_x$")
        ax[0].set_ylabel(r"$k_y$")
        fig.colorbar(ctr1, ax=ax[0])

        # |Vk|
        ctr2 = ax[1].contourf(kx, ky, Vk_abs, levels=20, cmap="RdBu_r")
        ax[1].set_title(r"$|V_k|$")
        ax[1].set_xlabel(r"$k_x$")
        ax[1].set_ylabel(r"$k_y$")
        fig.colorbar(ctr2, ax=ax[1])
    
        plt.tight_layout()
        plt.show()
        plt.close()
        
        fig2, ax2 = plt.subplots(2, 2, figsize=(12,12))
        try:
            norm_epsksσ = TwoSlopeNorm(vmin=np.min(epskAσ[0]), vcenter=0.0, vmax=np.max(epskAσ[0]))
        except:
            norm_epsksσ=None
        plt.rcParams['font.size'] = 30 
        plt.rcParams['axes.titlesize'] = 28  
        ctra0 = ax2[0,0].contourf(kx,ky,epskAσ[0], levels = 40, cmap = 'RdBu_r', norm = norm_epsksσ)
        ctra1 = ax2[0,1].contourf(kx,ky,epskAσ[1], levels = 40, cmap = 'RdBu_r', norm = norm_epsksσ)
        ctrb0 = ax2[1,0].contourf(kx,ky,epskBσ[0], levels = 40, cmap = 'RdBu_r', norm = norm_epsksσ)
        ctrb1 = ax2[1,1].contourf(kx,ky,epskBσ[1], levels = 40, cmap = 'RdBu_r', norm = norm_epsksσ)
        ax2[0,0].set_title(r"$\epsilon_{kA+}$")
        ax2[0,0].set_xlabel(r"$k_x$")
        ax2[0,0].set_ylabel(r"$k_y$")
        fig2.colorbar(ctra0, ax=ax2[0,0])
        ax2[0,1].set_title(r"$\epsilon_{kA-}$")
        ax2[0,1].set_xlabel(r"$k_x$")
        ax2[0,1].set_ylabel(r"$k_y$")
        fig2.colorbar(ctra1, ax=ax2[0,1])
        ax2[1,0].set_title(r"$\epsilon_{kB+}$")
        ax2[1,0].set_xlabel(r"$k_x$")
        ax2[1,0].set_ylabel(r"$k_y$")
        fig2.colorbar(ctrb0, ax=ax2[1,0])
        ax2[1,1].set_title(r"$\epsilon_{kB-}$")
        ax2[1,1].set_xlabel(r"$k_x$")
        ax2[1,1].set_ylabel(r"$k_y$")
        fig2.colorbar(ctrb1, ax=ax2[1,1])
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
        Vk = self.Vk_plot # Shape (N_BZ1, N_BZ2)
        k_grid = self.k_grid 
        kx = k_grid[:, 0].reshape(N_BZ1, N_BZ2)
        ky = k_grid[:, 1].reshape(N_BZ1, N_BZ2)
        
        
        
        tol = 1e-1
        
        # Iterate through the grid
        differences = np.zeros(((N_BZ1,N_BZ2)))
        differences_phase =  np.zeros(((N_BZ1,N_BZ2)))
        for i in range(N_BZ1):
            for j in range(N_BZ2):
                if kx[i, j] < -tol:  # Focus on the left side
                    # Target mirrored coordinates
                    target_kx = -kx[i, j]
                    target_ky = ky[i, j]
                    
                    # Find the index of the mirrored point (kx > 0)
                    mirror_idx = np.where(
                        (np.abs(kx - target_kx) < tol) & 
                        (np.abs(ky - target_ky) < tol)
                    )
                    
                    if len(mirror_idx[0]) > 0:
                        m_i, m_j = mirror_idx[0][0], mirror_idx[1][0]
                        
                        # Compare complex values
#                        is_symmetric = np.abs(Vk[i, j] - Vk[m_i, m_j]) < tol
                        differences[i,j] = np.abs(Vk[i, j]) - np.abs(Vk[m_i, m_j])   
                        differences_phase[i,j] = np.angle(Vk[i, j]) - np.angle(Vk[m_i, m_j])                                            
                        #color = 'green' if is_symmetric else 'red'
                        
                        # Plot both points
                        #plt.scatter(kx[i, j], ky[i, j], c=color, s=10)
                        #plt.scatter(kx[m_i, m_j], ky[m_i, m_j], c=color, s=10)
                if kx[i, j] > tol:  # Focus on the left side
                    # Target mirrored coordinates
                    target_kx = -kx[i, j]
                    target_ky = ky[i, j]
                    
                    # Find the index of the mirrored point (kx > 0)
                    mirror_idx = np.where(
                        (np.abs(kx - target_kx) < tol) & 
                        (np.abs(ky - target_ky) < tol)
                    )
                    
                    if len(mirror_idx[0]) > 0:
                        m_i, m_j = mirror_idx[0][0], mirror_idx[1][0]
                        
                        # Compare complex values
#                        is_symmetric = np.abs(Vk[i, j] - Vk[m_i, m_j]) < tol
                        differences[i,j] = np.abs(Vk[i, j]) - np.abs(Vk[m_i, m_j])  
                        differences_phase[i,j] = np.angle(Vk[i, j]) - np.angle(Vk[m_i, m_j])                                             
                        #color = 'green' if is_symmetric else 'red'
                        
                        # Plot both points
                        #plt.scatter(kx[i, j], ky[i, j], c=color, s=10)
                        #plt.scatter(kx[m_i, m_j], ky[m_i, m_j], c=color, s=10)    
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14 
        fig, ax = plt.subplots(1, 2, figsize=(8,5))
        plt.gca().set_aspect('equal')
        ctr1 = ax[0].contourf(kx, ky, differences, levels=20, cmap="RdBu_r")
        ax[0].set_title(r"$|V_k|$ differences")
        ax[0].set_xlabel(r"$k_x$")
        ax[0].set_ylabel(r"$k_y$")
        ax[0].axvline(0, color='black', linestyle='--', alpha=0.5)
        fig.colorbar(ctr1, ax=ax[0])
        
        ctr2 = ax[1].contourf(kx, ky, differences_phase, levels=20, cmap="RdBu_r")
        ax[1].set_title(r"arg($V_k$) differences")
        ax[1].set_xlabel(r"$k_x$")
        ax[1].set_ylabel(r"$k_y$")
        ax[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        fig.colorbar(ctr2, ax=ax[1])
        
        
        plt.show()
        return epsk, Vk_abs
# %% Definition of SlaveRotorSolver class

class SlaveRotorSolver:
    def __init__(self, lattice, T, n, Qψ_guess, max_iterations, tolerance,
                 hubbard_pot):
        """
        This class is a house for the iterative process of numerically solving
        for the Green's Functions. It stores the value of each variable and keeps
        a track of the full state at each loop.
        
        T - float
        n - int, range of values (-n,n) over which to sum Matsubara frequencies.
        Qψ_guess - 2x2 array
        """
        # Define the matrix elements of the bond variables
        self.lat = lattice
        self.QX = np.zeros((2,2))
        self.Qψ = Qψ_guess
        self.GX = np.zeros((2,2))
        self.Gψ = np.zeros((2,2))
        # it appears we can ignore a constraint by setting h=0
        self.h = 0                       # lagrange multiplier - left for posterity
        self.rho = 1.6   # lagrange multiplier, TODO: fix rho!
        self.Z = np.zeros(2)     # Quasiparticle weight        
        self.T = T
        self.iteration = 1
        self.n = n
        self.omega = (2*np.arange(-self.n, self.n+1) +1)*np.pi*self.T
        self.nu = (2*np.arange(-self.n, self.n+1))*np.pi*self.T
        self.k_grid = self.lat.k_grid
        self.N_BZ = len(self.k_grid)
        self.data = []
        self.max_iter = max_iterations
        self.tol = tolerance # convergence tolerance
        self.epsX_min = 0
        self.optm_GX_sum = 0
        self.bos_poles = 0
        self.ferm_poles = 0
        self.MX00 = 0
        self.MX11 = 0
        self.MX01 = 0
        self.MX10 = 0
        self.rho_min = 0
        self.U = hubbard_pot
        self.rotor_gap = 0
    
    def nB(self, w, T):
        cond = w/T
        cond = np.clip(cond,-200,200)
        return 1.0 / (np.exp(cond) - 1.0)
    
    def nF(self, w, T):
        cond = w/T
        cond = np.clip(cond,-200,200)
        return 1.0 / (np.exp(cond) + 1.0)

    def compute_Gψ_analytic(self):
        """
        Computes the sum over k, iw of G(k,iw) multiplied by T, without the 
        k=0 contribution. The solution for the sum over iw is given analytically,
        and the sum over k is computed.
        
        i.e. T*sum_{k}sum_{iw}{G(k,iw)}
        """
        N_BZ = self.N_BZ
        # Initialise empty array
        Gψ_sum = np.zeros((2,2), dtype=np.complex128)
        # Import pre-computed values
        epskAσ_all = self.lat.epskAσ_all
        epskBσ_all = self.lat.epskBσ_all
        # n.b. because epsksσ is in (2,N_BZ) format, everything else is too.
        Vk_all = self.lat.Vk_all[np.newaxis,:] # from (N_BZ) to (1, N_BZ)
        M00 = self.Qψ[0,0]*epskAσ_all
        M01 = -self.lat.t*self.Qψ[0,1]*Vk_all
        M11 = self.Qψ[1,1]*epskBσ_all
        M10 = -self.lat.t*self.Qψ[1,0]*Vk_all.conjugate()

        Tr  = M00 + M11
        det = M00*M11 - M01*M10   # (2,N_BZ) - (1,N_BZ), possible source of error?
        disc = Tr**2 - 4*det
        energies = np.empty((2,2,N_BZ), dtype=np.complex128)
        energies[0] = 0.5*((Tr-2*self.h) + np.sqrt(disc))
        energies[1] = 0.5*((Tr-2*self.h) - np.sqrt(disc))
        
        self.ferm_poles = energies

        nF1 = self.nF(energies[0], self.T)                          
        nF2 = self.nF(energies[1], self.T)
        delta_occ = nF1 - nF2
        delta_e = energies[0]-energies[1]
        
        Gψ00 = ((energies[0]*nF1 - energies[1]*nF2) + delta_occ*(self.h-M11)) / delta_e
        Gψ01 = (delta_occ*+M01) / delta_e
        Gψ11 = ((energies[0]*nF1 - energies[1]*nF2) + delta_occ*(self.h-M00)) / delta_e
        Gψ10 = (delta_occ*+M10) / delta_e
        
        Gψ_sum = np.array([[Gψ00, Gψ01], [Gψ10, Gψ11]])
        # Gψ_sum should have the shape [2,2,2,N_BZ] (complex128)
        # The indices are [2,2] for the Green's function matrix in an A,B (site) basis
        # then [2] for each spin index 
        # then [N_BZ] for each k value
        Gψ_sum = np.transpose(Gψ_sum, (2,3,0,1)) # in (2,N_BZ,2,2)
        return Gψ_sum/self.T
    
    def compute_Gψ(self): 
        """
        Compute spinon Green function Gψ[σ,i_k] (2×2 matrix).
        Stores result in self.Gψ as: self.Gψ[σ,i_k,2,2]
        
        i_w is an index to range over w_n values.
        N_BZ, Nw are the number of k vectors and (fermionic) Matsubara frequencies.
        σ takes values +- 1 and G is different for each one.
        The result output is the NORMAL Green's function, NOT INVERSE.
        
        We need for our calculations to preserve the k and spin indices but not w.
        Therefore, we only store the final object as (2,N_BZ,2,2)
        """
        N_BZ = self.N_BZ
        epsA = self.lat.epskAσ_all 
        epsB = self.lat.epskBσ_all
        # n.b. epsksσ is shape (2,N_BZ) 
        Vk = self.lat.Vk_all      
        VkC = np.conj(Vk)
        Qψ = self.Qψ
        t = self.lat.t
        h = self.h

        Gψ = np.zeros((2, N_BZ, 2, 2), dtype=np.complex128)
        
        for iσ in (0,1):
            epsA_σ = epsA[iσ]
            epsB_σ = epsB[iσ]
            
            # Mψ entries (N_BZ)
            M00 = Qψ[0,0] * epsA_σ
            M11 = Qψ[1,1] * epsB_σ
            M01 = -t * Qψ[0,1] * Vk
            M10 = -t * Qψ[1,0] * VkC
    
            # Dummies for sum over w (N_BZ)
            S00 = np.zeros(N_BZ, dtype=np.complex128)
            S01 = np.zeros(N_BZ, dtype=np.complex128)
            S10 = np.zeros(N_BZ, dtype=np.complex128)
            S11 = np.zeros(N_BZ, dtype=np.complex128)
            # aqaq
            eps = 0.001
            # Matsubara sum
            for w in self.omega:
                z = 1j*w + h  # the scalar bit
    
                # Ginv entries as [[a,b],[c,d]]. 
                # Gψ_inv = zI - M
                a = (z - M00) 
                d = (z - M11) 
                b = -M01 
                c = -M10 
    
                det = a*d - b*c
    
                # G = 1/det * [[d, -b], [-c, a]]
                S00 += (d / det)*np.exp(1j*w*eps)
                S01 += ((-b) / det) * np.exp(1j*w*eps) 
                S10 += ((-c) / det) * np.exp(1j*w*eps) 
                S11 += (a / det) * np.exp(1j*w*eps) 
    
            Gψ[iσ, :, 0, 0] = S00
            Gψ[iσ, :, 0, 1] = S01
            Gψ[iσ, :, 1, 0] = S10
            Gψ[iσ, :, 1, 1] = S11
        
         
        self.Gψ = Gψ
        return self.Gψ
    
    def precompute_GX(self):
        """
        We compute GX many times in the rho_optimisation loop. Saving as much time
        as we can greatly reduces the length of operation. To do this we can 
        perform the calculations that don't change in advance. For GX, this is 
        all of the M elements that are N_BZ*N_BZ operations
        """
        eps = self.lat.epsk_all
        Vk  = self.lat.Vk_all
        t   = self.lat.t
        QX  = self.QX
    
        self.MX00 = QX[0,0] * eps
        self.MX11 = QX[1,1] * eps
        self.MX01 = -t * QX[0,1] * Vk
        self.MX10 = -t * QX[1,0] * np.conj(Vk)
        return 0 
                    
    def compute_GX_analytic(self):
        """
        Computes the sum over k, iv of G(k,iv) multiplied by T, removing the 
        k=0 contribution. The solution for the sum over iv is given analytically,
        and the sum over k is computed.
        
        i.e. T*sum_{k}sum_{iv}{G(k,iv)}
        """
        U = self.U
        T = self.T
        N_BZ = self.N_BZ
        rho = self.rho
        h = self.h
        
        M00 = self.MX00
        M11 = self.MX11
        M01 = self.MX01
        M10 = self.MX10
        Tr  = M00 + M11
        det = M00*M11 - M01*M10
        disc = Tr**2 - 4*det
        alpha_plus = (-(2*rho + Tr) + np.sqrt(disc))/2
        alpha_minus = (-(2*rho + Tr) - np.sqrt(disc))/2
        # ei shape: (4, N_BZ)
        ei = np.empty((4, N_BZ), dtype=np.complex128)
        ei[0] = -h + np.sqrt(h**2 - alpha_plus*U)
        ei[1] = -h + np.sqrt(h**2 - alpha_minus*U)
        ei[2] = -h - np.sqrt(h**2 - alpha_plus*U)
        ei[3] = -h - np.sqrt(h**2 - alpha_minus*U)
    
        # store representative poles if you still want a 1D array
        self.bos_poles = ei  #(4,N_BZ)
        
        nBi = self.nB(ei, T)
    
        def Delta(i, j):
            return ei[i] - ei[j]          # (N_BZ,)
    
        D12 = Delta(0,1); D13 = Delta(0,2); D14 = Delta(0,3)
        D23 = Delta(1,2); D24 = Delta(1,3); D34 = Delta(2,3)

        def f_scalar(e, X):
            return -(e**2 + 2*e*h)/U + rho + X
        
        def F_matrix(e):
            F00 = -f_scalar(e, M11)
            F11 = -f_scalar(e, M00)
            F01 = M01
            F10 = M10
            return np.array([[F00, F01],
                             [F10, F11]], dtype=np.complex128)
        F1 = F_matrix(ei[0])*nBi[0]
        F2 = F_matrix(ei[1])*nBi[1]
        F3 = F_matrix(ei[2])*nBi[2]
        F4 = F_matrix(ei[3])*nBi[3]

        term1 = (D23*D24*D34) * F1
        term2 = (D13*D14*D34) * F2
        term3 = (D12*D14*D24) * F3
        term4 = (D12*D13*D23) * F4
        Delta_total = D12*D13*D14*D23*D24*D34

        GX_of_k = ((term1 - term2 + term3 - term4)*(U**2)) / Delta_total   # (2,2,N_BZ)
        GX_of_k = np.transpose(GX_of_k, (2,0,1)) 
        return GX_of_k/T
    
    def compute_GX(self):
        """
        Compute spinon Green function GX[i_k,i_v] (2×2 matrix).
        Stores result in self.Gψ as: self.Gx[i_k,2,2]
        N.B. There is no NNN interaction in the roton, so no sigma dependence.
        
        i_k,i_v are indices to range over k and v_n values.
        N_BZ, Nv are the number of k vectors and (bosonic) Matsubara frequencies.
        The result output is the NORMAL Green's function, NOT INVERSE.
        """
        N_BZ = self.N_BZ
        h = self.h
        U = self.U
        rho = self.rho
        
        GX = np.zeros((N_BZ, 2, 2), dtype=np.complex128)
      
        # Mψ entries (N_BZ)
        M00 = self.MX00
        M11 = self.MX11
        M01 = self.MX01
        M10 = self.MX10
    
        # Dummies for sum over v (N_BZ)
        S00 = np.zeros(N_BZ, dtype=np.complex128)
        S01 = np.zeros(N_BZ, dtype=np.complex128)
        S10 = np.zeros(N_BZ, dtype=np.complex128)
        S11 = np.zeros(N_BZ, dtype=np.complex128)
        
        
        # Matsubara sum
        for v in self.nu:
            z = v**2/U - 2*v*h*1j/U + rho  # the scalar bit
    
            # Ginv entries as [[a,b],[c,d]]. 
            # GX_inv = zI + M
            a = z + M00
            d = z + M11
            b = M01       
            c = M10
    
            det = a*d - b*c
    
            # G = 1/det [[d, -b], [-c, a]]
            S00 += d / det
            S01 += (-b) / det
            S10 += (-c) / det
            S11 += a / det
    
        out = np.zeros((N_BZ, 2, 2), dtype=np.complex128)
        out[:, 0, 0] = S00
        out[:, 0, 1] = S01
        out[:, 1, 0] = S10
        out[:, 1, 1] = S11
        GX = out
    
        self.GX = GX
        return self.GX       
   
    def update_QX(self):
        """
        Updates QX using the guessed/computed value of Gψ.
        In order to do this, we loop through computing the sum over all elements
        of Gψ using the self-consistency equations (13). We reset QX each full loop.
        """
        λ = self.lat.λ
        new_QX = np.zeros((2,2), dtype=np.complex128)
        N_BZ = self.N_BZ
        new_QX_of_k = np.zeros((N_BZ,2,2), dtype=np.complex128)
        epsA = self.lat.epskAσ_all       # (2, N_BZ)
        epsB = self.lat.epskBσ_all       # (2, N_BZ)
        Vk   = self.lat.Vk_all           # (N_BZ,)
        Gψ = self.compute_Gψ_analytic() #(2,N_BZ,2,2)
        T = self.T
        
        # Multiply by σ,k-dependent structure functions, then sum over σ       
        Q00 = (epsA * Gψ[:, :, 0, 0]).sum(axis=0) # Gψ[:, :, 0, 0] is a (2,N_BZ) array
        Q11 = (epsB * Gψ[:, :, 1, 1]).sum(axis=0)
        Q01 = (Vk[None,:] * Gψ[:, :, 1, 0]).sum(axis=0)
        #Q10 = np.sum(np.conj(Vk[None, :]) * Gψ[:, :, 1, 0])
        #NOTE: fiddle with this later to see effect
        Q10 = np.conj(Q01)
        
        # Multiply by pre-factors
        new_QX_of_k[:,0,0] = (T * Q00.real) / (6*λ)
        new_QX_of_k[:,1,1] = (T * Q11.real) / (6*λ)
        new_QX_of_k[:,0,1] = (T * Q01) / 3
        new_QX_of_k[:,1,0] = (T * Q10) / 3
        # This is now QX as a function of k, store it. 
        self.QX_of_k = new_QX_of_k
        
        # Sum over k to get the final, fully-summed QX.
        new_QX[0,0] = (new_QX_of_k[:,0,0]/N_BZ).sum(axis=0)
        new_QX[1,1] = (new_QX_of_k[:,1,1]/N_BZ).sum(axis=0)
        new_QX[0,1] = (new_QX_of_k[:,0,1]/N_BZ).sum(axis=0)
        #Q10 = np.sum(np.conj(Vk[None, :]) * Gψ[:, :, 1, 0])
        #NOTE: fiddle with this later to see effect
        new_QX[1,0] = np.conj(new_QX[0,1])
        
        return new_QX
    
    def update_Qψ(self):
        """
        Updates Qψ using the computed value of GX.
        In order to do this, we loop through computing the sum over all elements
        of GX using the self-consistency equations (14). We reset Qψ each full loop.

        Note that the SI gives below (14) that Z^{ss'} = \sqrt{ Z^s * Z^{s'}}
        """
        Z_a, Z_b = self.Z
        Z_ab = np.sqrt(Z_a*Z_b)
        λ = self.lat.λ
        new_Qψ = np.zeros((2,2), dtype=np.complex128)
        N_BZ = self.N_BZ
        eps = self.lat.epsk_all #(N_BZ)
        Vk = self.lat.Vk_all
        T = self.T
        
        GX = self.compute_GX_analytic() #(N_BZ,2,2)

        # sum k
        Q00 = np.sum(eps * GX[:, 0, 0])
        Q11 = np.sum(eps * GX[:, 1, 1])
        Q01 = np.sum(Vk * GX[:, 1, 0]) 
        #Q10 = np.sum(np.conj(Vk) * GX[]:, 1, 0])
        #NOTE: fiddle with this later to see effect
        Q10 = np.conj(Q01)
                
        new_Qψ[0,0] = Z_a + Q00.real*T /(6*λ*N_BZ)
        new_Qψ[1,1] = Z_b + Q11.real*T /(6*λ*N_BZ)
        new_Qψ[0,1] = Z_ab + Q01*T /(3*N_BZ)
        new_Qψ[1,0] = Z_ab + Q10*T /(3*N_BZ)
        
        return new_Qψ
    
    def get_hsp_path(self, num_pts=1000):
        # Define High Symmetry Points in terms of reciprocal vectors
        Gamma = np.array([0, 0])
        M = 0.5 * b1
        K = (b1 + b2)/3
        points = [Gamma, M, K, Gamma]
        labels = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
        
        full_path = []
        ticks = [0]
        
        for i in range(len(points)-1):
            segment = np.linspace(points[i], points[i+1], num_pts)
            full_path.append(segment)
            ticks.append(ticks[-1] + num_pts)
            
        return np.vstack(full_path), ticks, labels
    
    def get_hsp_path_2(self, num_pts=100):
        # Define High Symmetry Points in terms of reciprocal vectors
        def R(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle),  np.cos(angle)]])
        
        Gamma = np.array([0, 0])
        Gamma_top = b1+b2
        #M = (b1+b2)/2
        K = (b1 + b2)/3
        #Kp = (b1+b2)*(2/3)
        Kp = -K

        K2 = K@R((2*np.pi)/3)
        Kp2 = K@R((2*np.pi)/6)
        M = np.array([K2[0],0])
        points = [K, Gamma, Kp, K2, M, Gamma]
        labels = [r'K', r'$\Gamma$', r"K'", 'K', 'M', r'$\Gamma$']
        #points = [Gamma, K, M, Kp, Gamma_top]
        #labels = [r'$\Gamma$', 'K', 'M', r"K'", r'$\Gamma_2$']
        full_path = []
        ticks = [0]
        
        for i in range(len(points)-1):
            segment = np.linspace(points[i], points[i+1], num_pts, endpoint=False)
            full_path.append(segment)
            ticks.append(ticks[-1] + num_pts)
            
        return np.vstack(full_path), ticks, labels
    
    def find_rho_constraint(self):
        N_BZ = self.N_BZ
        t = self.lat.t
        QX_mesh = self.QX  
        eps_mesh = self.lat.epsk_all 
        Vk_mesh = self.lat.Vk_all

        H_mesh = np.zeros((N_BZ,2,2), dtype=np.complex128)
        H_mesh[:,0,0] = QX_mesh[0,0]*eps_mesh
        H_mesh[:,0,1] = -t*QX_mesh[0,1]*Vk_mesh
        H_mesh[:,1,0] = -t*QX_mesh[1,0]*Vk_mesh.conjugate()
        H_mesh[:,1,1] = QX_mesh[1,1]*eps_mesh
        energies_mesh = np.linalg.eigvalsh(H_mesh)
        energies_flat = np.reshape(energies_mesh, 2*N_BZ)
        self.rho_min = -1.01*min(energies_flat)

        # TODO: Make this its own function.
        # Plot high-symmetry path for eigenvalues
#        if self.iteration == 25:
            #-----------------------------------------------------------------
# =============================================================================
#             # Plot BZ for eigenvalues
#             kx = self.k_grid[:, 0].reshape(Ns, Ns)
#             ky = self.k_grid[:, 1].reshape(Ns, Ns)
#             band1 = energies_mesh[:, 0].reshape(Ns, Ns) + self.rho_min
#             band2 = energies_mesh[:, 1].reshape(Ns, Ns) + self.rho_min
#             vmin = min(band1.min(), band2.min())
#             vmax = max(band1.max(), band2.max())
#             shared_levels = np.linspace(vmin, vmax, 50)
# 
#             fig, ax = plt.subplots(1, 2, figsize=(10, 6))
#             # Lower Band
#             c1 = ax[0].contourf(kx, ky, band1, levels=shared_levels,
#                                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
#             ax[0].set_title(r"Lower Rotor Band $E_1(k)$")
#             ax[0].set_aspect('equal')
#             # Upper Band
#             c2 = ax[1].contourf(kx, ky, band2,levels=shared_levels, 
#                                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
#             ax[1].set_title(r"Upper Rotor Band $E_2(k)$")
#             ax[1].set_aspect('equal')
#                         
#             fig.colorbar(c1, ax=ax[1], orientation='vertical', label=r'Energy $E(k)$')
#             
# # =============================================================================
# #             for a in ax:
# #                 a.set_xlabel(r"$k_x$")
# #                 a.set_ylabel(r"$k_y$")
# #                 K = (b1 + b2) / 3.0
# #                 a.scatter([K[0], -K[0]], [K[1], -K[1]], color='red', s=10, label="K / K\'")
# # =============================================================================
#             
#             plt.tight_layout()
#             plt.show()
#             plt.close()
#             #-----------------------------------------------------------------
#             # 2. Setup 3D Plot
#             fig = plt.figure(figsize=(13, 6))
#             ax = fig.add_subplot(111, projection='3d')
#         
#             # Define common color scale
#             vmax = np.max(band2)
#             vmin = np.min(band1)
#             # 2.5. Set Camera Angle
#             # elev: elevation angle in degrees (0 is looking horizontally, 90 is looking straight down)
#             # azim: azimuthal angle (rotation around the z-axis)
#             ax.view_init(elev=15, azim=25)
#             # 3. Plot Surfaces
#             # Lower Band (Viridis)
#             surf1 = ax.plot_surface(kx, ky, band1, cmap='viridis', 
#                                     antialiased=True, alpha=0.8, vmin=vmin, vmax=vmax)
#             
#             # Upper Band (Magma/Plasma)
#             surf2 = ax.plot_surface(kx, ky, band2, cmap='viridis', 
#                                     antialiased=True, alpha=0.7, vmin=vmin, vmax=vmax)
#             
#             z_zero = np.zeros_like(kx)
#             ax.plot_surface(kx, ky, z_zero, color='gray', alpha=0.2, shade=False, zorder=0)
#             
#             # 4. Ground the Energy Axis at Zero
#             ax.set_zlim(0, vmax) 
#             
#             # 5. Labeling
#             ax.set_xlabel(r'$k_x$')
#             ax.set_ylabel(r'$k_y$')
#             ax.set_zlabel(r'Energy $E(k)$')
#             ax.set_title('3D Rotor Band Structure')
#         
#             # Add a colorbar for reference
#             # 4. Create the Global Colorbar
#             # This creates a mapping that spans the actual data range [vmin, vmax]
#             mappable = cm.ScalarMappable(cmap='viridis') # Choose the primary colormap
#             mappable.set_array([vmin, vmax])
#             mappable.set_clim(vmin, vmax)
#             
#             cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
#             cbar.set_label(f'Energy')
#             plt.title('Rotor Dispersion')
#             plt.show()
# =============================================================================
        return self.rho_min
            
    def optimise_for_rho(self):
        """
        Optimise the constraint equation for rho, assuming that the quasiparticle
        weight Z is zero. If the equation can converge on a value of rho, we 
        preserve the value of Z^ss=0 and rho. If not, we proceed to minimise the 
        value of [ T*Σ_ν (G^ss_X(iν,r-r')) -1 ] to set the value of rho, and 
        rearrange the constraint equation to set a finite value for Z. Doing 
        this allows for the computation of Qpsi again.
                                                
        It is important to note that the presence of ^ss means that we should 
        only consider the diagonal elements of G_X.
        
        We label the constraint equation rearranged so the RHS = 0 (i.e. the
        minimised expression) as C_rho = T*Σ_ν{G^ss_X(iν,r-r')} -1
                                                       
        Summary:
        Takes updated QX
        Computes GX with a given rho[s], optimises (12b)
        updates self.rho
        updates self.Z
        
        # nb. this is assuming rho_A!=rho_B. It seems the SI assumes the 
        # opposite due to the symmetry of the system in bulk and this is 
        # backed up by the fact that the SI does not index rho.
        # For finite geometries it appears a real-space solution of rho
        # as a function or r may be needed.
        """
        
        N_BZ = self.N_BZ
        U = self.U
        T = self.T 
        
        # TODO: Make this work for vector rho s.t. rho_A, rho_B, independent    
        
        def C(rho):
            """
            This calculates the constraint expression, but for each s.
            """
            # TODO: vary only component s, keep the other fixed at current value
            rho_backup = self.rho
            self.rho = rho
            GX_trial = self.compute_GX_analytic() # return GX as (N_BZ,2,2)
            GX_trial_sum = GX_trial.sum(axis=0)
            C = T/N_BZ * GX_trial_sum
            self.rho = rho_backup
            return C
        
        # choose a rho scanning interval, sample to see whether C_rho_s crosses 1        
        rho_min_theoretical = self.find_rho_constraint()
        #print('rho_min = ' , rho_min_theoretical)
        if self.iteration <= 2:
            rho_min_phys =  1
        else: 
            rho_min_phys = rho_min_theoretical
        
        rho_max_phys =  5.0 + self.iteration/4  # 
        
        rho_min = max(self.rho - 30/self.iteration**0.5, rho_min_phys)
        rho_max = min(self.rho + 30/self.iteration**0.5, rho_max_phys)
        rho_sample_vals = np.linspace(rho_min, rho_max, 50)

        C_vals = [C(m) for m in rho_sample_vals]      
        #print("sample constraints computed")
# =============================================================================
#         fig,ax = plt.subplots(1,1, figsize=(8,8))
#         plt.title(r"$1-T/N_BZ \sum_{k,i\nu} G^X_{ss}(k,iv)$ vs $\rho$, " 
#                   + f"Iteration {self.iteration}, "
#                   + f"U/t = {U/t:3.2f}")
#         plt.ylabel(r"$1-\sum_{k,i\nu} G^X_{ss}(k,iv)$")
#         plt.xlabel(r"$\rho$")
#         plt.xlim(0, rho_max_phys)
#         plt.axvline(rho_min_theoretical)
#         plt.grid(True)
# =============================================================================
        newZ = np.empty(2)
        for s in(0,1):
            
            diag_vals = np.array([C_val[s,s].real for C_val in C_vals])
# =============================================================================
#             if s == 0:
#                 plt.errorbar(rho_sample_vals, 1-diag_vals,
#                              color = 'b', label = r'$\rho_A$')
#             elif s==1:
#                 plt.errorbar(rho_sample_vals, 1-diag_vals,
#                              color = 'r', label = r'$\rho_B$')
# =============================================================================
            # -------- Mott attempt: find rho s.t. C(rho) = 1 (Z=0) --------
            # Look for sign changes of C-1 (indicates there is a value for which
            # C=0) (technically this fails if C<1 for all inputs but it clearly is
            # not if you look at the computations - worth being aware of)
            f_vals = diag_vals - 1.0
            sign_changes = np.where(f_vals[:-1] * f_vals[1:] < 0)[0]
            if len(sign_changes) > 0:
                # take the first bracket where a crossing exists
                i = sign_changes[-1]
                a = rho_sample_vals[i]
                b = rho_sample_vals[i+1]
            
                # refine to root inside [a,b] 
                rho_solution = scipy.optimize.brentq(
                    lambda m: C(m)[s,s].real - 1.0,
                    a, b
                )
            
                new_rho = rho_solution
                newZ[s] = 0.0           
                
            # ---------- Metallic: pick rho_s that minimises C_s-1 ----------
            # minimise |C-1| or minimise C-1 ? 
            # The former finds the smallest value of Z, the latter the largest.
            else:
                idx = int(np.argmin(1-diag_vals))
                best_rho = rho_sample_vals[idx]
                C_best = diag_vals[idx]
                if C_best > 1.1: # allow for a bit of wiggle room for overshoot  
                    #print(C_vals)    
                    print("C_s is bigger than 1")
            
                #print(f"alt minimisation: rho = {rho_sample_vals[idx_alt]}"
                #      + f"Z = {1-diag_vals[idx_alt]}")
            
                new_rho = best_rho
                # Z_s = 1 - C_s(best_rho), clipped to ≥0 
                newZ[s] = 1-C_best
        #print(f"minimisation: rho = {self.rho}, Z = {self.Z}")
# =============================================================================
#         plt.show()
#         plt.close()
# =============================================================================
        return new_rho, newZ
    
    def calculate_rotor_gap(self, plot = False):
        N_BZ = self.N_BZ
        t = self.lat.t
        QX_mesh = self.QX  
        eps_mesh = self.lat.epsk_all 
        Vk_mesh = self.lat.Vk_all

        H_mesh = np.zeros((N_BZ,2,2), dtype=np.complex128)
        H_mesh[:,0,0] = QX_mesh[0,0]*eps_mesh
        H_mesh[:,0,1] = -t*QX_mesh[0,1]*Vk_mesh
        H_mesh[:,1,0] = -t*QX_mesh[1,0]*Vk_mesh.conjugate()
        H_mesh[:,1,1] = QX_mesh[1,1]*eps_mesh
        energies_mesh = np.linalg.eigvalsh(H_mesh)
        energies_flat = np.reshape(energies_mesh, 2*N_BZ)
        
        if plot == True:
            # Plot BZ for eigenvalues
            kx = self.k_grid[:, 0].reshape(Ns, Ns)
            ky = self.k_grid[:, 1].reshape(Ns, Ns)
            band1 = energies_mesh[:, 0].reshape(Ns, Ns)
            band2 = energies_mesh[:, 1].reshape(Ns, Ns)
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Lower Band
            c1 = ax[0].contourf(kx, ky, band1, levels=50, cmap='viridis')
            ax[0].set_title(r"Lower Rotor Band $E_1(k)$")
            ax[0].set_aspect('equal')
            fig.colorbar(c1, ax=ax[0])
            
            # Upper Band
            c2 = ax[1].contourf(kx, ky, band2, levels=50, cmap='magma')
            ax[1].set_title(r"Upper Rotor Band $E_2(k)$")
            ax[1].set_aspect('equal')
            
            print(f"Min energy = {min(energies_flat):4.3f}, ρ = {self.rho:4.3f}")
            print(f"ΔX = ρ + min(ε_X) = {self.rho + min(energies_flat):4.3f}")
            
            # Generate the 1D path
            path_k, ticks, labels = self.get_hsp_path_2(num_pts=500) # Use helper from above
            
            # Calculate structure functions along THIS path
            path_eps = np.array([self.lat.eps_k(k) for k in path_k])
            path_Vk = np.array([self.lat.Vk(k) for k in path_k])
            # Build H along the path using the CURRENT mean-field QX values
            H_path = np.zeros((len(path_k), 2, 2), dtype=np.complex128)
            H_path[:,0,0] = self.QX[0,0] * path_eps
            H_path[:,1,1] = self.QX[1,1] * path_eps
            H_path[:,0,1] = -t * self.QX[0,1] * path_Vk
            H_path[:,1,0] = -t * self.QX[1,0] * path_Vk.conjugate()
            
            path_energies = np.linalg.eigvalsh(H_path) + self.rho
    
            # Plotting 
            plt.figure(figsize=(6, 5))
            plt.plot(path_energies[:, 0], label=r'$\epsilon_{k-}$', color='b')
            plt.plot(path_energies[:, 1], label=r'$\epsilon_{k+}$', color='r')
            
            # Formatting the path
            for t in ticks:
                plt.axvline(t, color='k', linestyle='--', alpha=0.3)
            plt.xticks(ticks, labels)
            plt.axhline(0, color='k', linestyle='--', alpha=0.3)
            plt.ylabel('Energy')
            plt.title(f'U/t = {self.U}, Rotor Dispersion')
            #plt.ylim(-2.2,1.5)
            plt.legend()
            plt.show()
            plt.close()
            fig.colorbar(c2, ax=ax[1])
            
            for a in ax:
                a.set_xlabel(r"$k_x$")
                a.set_ylabel(r"$k_y$")
                K = (b1 + b2) / 3.0
                a.scatter([K[0], -K[0]], [K[1], -K[1]], color='red', s=10, label="K / K\'")
            
            plt.tight_layout()
            plt.show()
            plt.close()
            
        return self.rho + min(energies_flat)
  
    def calculate_spinon_gap(self, plot= False):
        N_BZ = self.N_BZ
        t = self.lat.t
        Qψ_mesh = self.Qψ  
        epskAσ_mesh = self.lat.epskAσ_all
        epskBσ_mesh = self.lat.epskBσ_all
        Vk_mesh = self.lat.Vk_all
        
        H_mesh = np.zeros((N_BZ,2,2), dtype=np.complex128)
        H_mesh[:,0,0] = Qψ_mesh[0,0]*epskAσ_mesh[0,:]
        H_mesh[:,0,1] = -t*Qψ_mesh[0,1]*Vk_mesh
        H_mesh[:,1,0] = -t*Qψ_mesh[1,0]*Vk_mesh.conjugate()
        H_mesh[:,1,1] = Qψ_mesh[1,1]*epskBσ_mesh[0,:]
        energies_mesh = np.linalg.eigvalsh(H_mesh)
        spinon_gap = min(energies_mesh[:,1])
        
        if plot:
            print("Spinon gap: ", spinon_gap)
            path_k, ticks, labels = self.get_hsp_path_2(num_pts=500) # Use helper from above
            
            # Calculate structure functions along THIS path
            path_epsA = np.array([self.lat.eps_ksσ(k,'a',1) for k in path_k])
            path_epsB = np.array([self.lat.eps_ksσ(k,'b',1) for k in path_k])
            path_Vk = np.array([self.lat.Vk(k) for k in path_k])
            # Build H along the path using the CURRENT mean-field QX values
            H_path = np.zeros((len(path_k), 2, 2), dtype=np.complex128)
            H_path[:,0,0] = self.Qψ[0,0] * path_epsA
            H_path[:,1,1] = self.Qψ[1,1] * path_epsB
            H_path[:,0,1] = -t * self.Qψ[0,1] * path_Vk
            H_path[:,1,0] = -t * self.Qψ[1,0] * path_Vk.conjugate()
            
            path_energies = np.linalg.eigvalsh(H_path)
    
            # Plotting 
            plt.figure(figsize=(6,5))
            plt.plot(path_energies[:, 0], label=r'$\epsilon_{k-}$', color='b')
            plt.plot(path_energies[:, 1], label=r'$\epsilon_{k+}$', color='r')
            
            # Formatting the path
            for t in ticks:
                plt.axvline(t, color='k', linestyle='--', alpha=0.3)
            plt.xticks(ticks, labels)
            
            plt.ylabel('Energy')
            plt.title(f'U/t = {self.U}, Spinon Dispersion')
            plt.legend()
            plt.savefig("spinon_dispersion_U=10.png", dpi=500)
            plt.show()
            plt.close()
        
        return spinon_gap
    
    def dump_state(self, csv_file = 'solver_log.csv'):
        """
        Dump the state of the solver to a file after each iteration.
        Stores iteration number, Qψ, QX, rho, Z, and optionally norms.
        """
        self.rho = np.array(self.rho)
        # Make dict
        row = {
            "iteration": self.iteration,
            "rho": self.rho.tolist(),
            "Z": self.Z.tolist(),
            "Qpsi": self.Qψ.tolist(),
            "QX": self.QX.tolist(),
            "GXsum": self.optm_GX_sum.tolist(),
            "rho_min": self.rho_min.tolist(),
            "rotor_gap": self.rotor_gap.tolist()
        }
        # Store in RAM history
        self.data.append(row)
        # append to CSV
        mode = "w" if self.iteration == 1 else "a"

        with open(csv_file, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if self.iteration == 1:
                writer.writeheader()
            writer.writerow(row)

        return 0
    
    def mix(self, old, new, α=0.5):
        return (1-α)*old + α*new
    
    def loop(self):
        """
        Full self-consistency loop calling the methods in the correct order:
    
            2. update_QX      (needs Gpsi)
            3. optimise_for_rho (needs QX via GX)
            4. update_Qψ       (needs GX and Z)
    
        After each iteration the new state is dumped to file.
        Convergence is checked from changes in (Qψ, QX, rho, Z).
    
        Returns when converged or when max_iter reached.
        """
        
        # initialize previous values for convergence check
# =============================================================================
#         print("Check for k=0 value being removed correctly:")
#         print(self.k_grid[0])
#         print("If the above value is not [0,0] something is not right.\n\n")
#         # the reason this is no longer needed is because the k=0 is no longer
#         # inluded in the generation of the grid.
# =============================================================================
        
        prev_Qpsi = self.Qψ.copy()
        prev_QX   = self.QX.copy()
        prev_rho  = self.rho
        prev_Z    = self.Z.copy()
        for it in range(self.max_iter):
            self.iteration = it + 1
            timings = {}
            # ---- main loop steps ----
            with timer("update_QX", timings):
                new_QX = self.update_QX()            
            if it == 0:
                self.QX = new_QX
            else:
                self.QX = self.mix(self.QX, new_QX, mixing)
                
            self.precompute_GX()
            
            #print(f"\nIteration #{self.iteration}, U/t = {self.U}")
            #print("-----------------------------------")
            #print("previous rho", fmt_array(self.rho), "previous Z", fmt_array(self.Z))
            GX_of_k = self.compute_GX_analytic()
            # print("GX sum for previous rho\n", fmt_array(GX_of_k.sum(axis=0)))
            C = (self.T / len(self.k_grid)) * GX_of_k.sum(axis=0)
            #print("1 - C_diag = Z:", 1 - C[0,0].real)
            with timer("optimise_for_rho", timings):
                new_rho, newZ = self.optimise_for_rho()        
            if it == 0:
                self.rho = new_rho
                self.Z = newZ
            else:
                self.rho = self.mix(self.rho, new_rho, mixing)
                self.Z = self.mix(self.Z, newZ, mixing)
            with timer("update_Qψ", timings):
                new_Qψ = self.update_Qψ()
            if it == 0:
                self.Qψ = new_Qψ
            else:
                self.Qψ = self.mix(self.Qψ, new_Qψ, mixing)
            #print("QX:", fmt_array(self.QX))
            #print("Qψ:", fmt_array(self.Qψ))
            with timer("compute_GX_analytic", timings):
                        GX_opt = self.compute_GX_analytic()
                        self.optm_GX_sum = GX_opt.sum(axis=0)
            with timer("compute_Gψ_analytic", timings):
                        Gpsi_a = (self.compute_Gψ_analytic()).sum(axis=0)
                        Gpsi_a = Gpsi_a.sum(axis=0)
# =============================================================================
#             print(f"\nIteration {self.iteration} timing:")
#             for k, v in timings.items():
#                 print(f"  {k:20s} {v:9.4f} s")
#             print("-----------------------------------")
#             print("\n\n")
# =============================================================================
           
            # compute spinon and rotor dispersion
            self.rotor_gap = self.calculate_rotor_gap()
            self.spinon_gap = self.calculate_spinon_gap()
            
            # ---- plot band structure Qpsi and QX ----
            # ------------------------ dump state -----------------------------
            self.dump_state()
            
            # -------------------- convergence check --------------------------
            dQpsi = np.max(np.abs(self.Qψ - prev_Qpsi))
            dQX   = np.max(np.abs(self.QX - prev_QX))
            dRho  = abs(self.rho - prev_rho)
            dZ    = np.max(np.abs(self.Z - prev_Z))
    
            delta = max(dQpsi, dQX, dRho, dZ)
    
            # --------------------- update "previous" -------------------------
            prev_Qpsi = self.Qψ.copy()
            prev_QX   = self.QX.copy()
            prev_rho  = self.rho
            prev_Z    = self.Z.copy()
            if delta < self.tol:
                print(f"Converged after {it+1} iterations (Δ = {delta:.3e})")
                #self.plot_fermi_level_bz(σ=1, eta=0.01)
                #self.plot_spectral_function(σ=1, omega_range=[-8, 8])
                #self.calculate_N3_invariant(σ=1, omega_range=[-12,12])
                return
    
        print(f"Reached max_iter = {self.max_iter} without converging.")
        #print("GX sum \n", fmt_array(GX_of_k.sum(axis=0)))
        #print("Gψ sum \n", fmt_array(.sum(axis=0)))
        #self.plot_fermi_level_bz(σ=1, eta=0.01)
        #self.plot_spectral_function(σ=1, omega_range=[-8, 8])
        #self.calculate_N3_invariant(σ=1, omega_range=[-12,12])
        return 0
 
    def single_particle_GF(self, σ, k, ω):
        """Now that the Q values are found, the Green's function can be calculated.
        The expression for Gpsi can be calculated and inverted. The convolution of
        Gpsi and GX needs to be calculated as a sum over q and iv, which is done 
        using an analytical expression for the sum over iv and by summing numerically
        over q and hence the BZ. 
        
        This calculates the single-particle Green's Function as a function of k, and iw
        This function can be called with k as an array to then build a map of the single-
        particle GF, which is referred to as G.
                 
        G_ss' is a function of σ, k, w
        G^ss'_σ (k,iw) = Z^{ss'}G^{ss'}_σψ (k, iw) 
                         + T/N Σ_(q,iv) [G^{ss'}_σψ(k-q, iw-iv) G^{ss'}_X(q, iv)]
        
        k - array of k points for the high symmetry point path, shape (N_path)
        ω - frequency, shape (Nω) or (1), complex due to small offset η in analytic cont.
        """
        ω = np.asarray(ω)
        if ω.ndim == 0:
            ω = ω[None]
        Nω = len(ω)
        N_path = len(k)
        N_BZ = self.N_BZ
        U = self.U
        ω = ω[:,None,None]  # (Nω,1,1)
        term1 = 0
        # -------------------------------------------------------------------------
        # TERM 1
        # -------------------------------------------------------------------------
        # note that term1 is always 0 in the Mott phase as Z = 0
        ω1 = ω[:,:,0]
        Mψ00_1 = self.k_path_Mψ00[None,:]
        Mψ01_1 = self.k_path_Mψ01[None,:] 
        Mψ10_1 = self.k_path_Mψ10[None,:] 
        Mψ11_1 = self.k_path_Mψ11[None,:] 
        aψ_1 = ω1 - Mψ00_1
        bψ_1 = Mψ01_1
        cψ_1 = Mψ10_1
        dψ_1 = ω1 - Mψ11_1
        detψ_1 = aψ_1*dψ_1 - bψ_1*cψ_1
        Gψ_k_iω = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        Gψ_k_iω[:,:,0,0] =  dψ_1/detψ_1
        Gψ_k_iω[:,:,0,1] =  bψ_1/detψ_1
        Gψ_k_iω[:,:,1,0] =  cψ_1/detψ_1
        Gψ_k_iω[:,:,1,1] =  aψ_1/detψ_1
        term1 = self.Z*Gψ_k_iω
        # ---------------------------------------------------------
        # TERM 2 
        # ---------------------------------------------------------
        energy_f1 = self.energy_f1_kmq[None,:,:]
        energy_f2 = self.energy_f2_kmq[None,:,:]
        energy_b1 = self.bos_poles[0,:][None,None,:]
        energy_b2 = self.bos_poles[1,:][None,None,:]
        energy_b3 = self.bos_poles[2,:][None,None,:]
        energy_b4 = self.bos_poles[3,:][None,None,:]
        Atildeψ = ω - self.Mψ11_kmq
        Dtildeψ = ω - self.Mψ00_kmq
        AtildeX = self.AtildeX_const
        DtildeX = self.DtildeX_const 
        # Elementwise multiplication        
        Aprime = Atildeψ * AtildeX
        Bprime = self.Bprime_kmq
        Cprime = self.Cprime_kmq 
        Dprime = Dtildeψ * DtildeX
        term2f = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        term2b = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        # ---------------------------------------------------------
        # Fermionic poles
        # ---------------------------------------------------------
        f_poles = [energy_f1, energy_f2]
        b_poles = [energy_b1,energy_b2,energy_b3,energy_b4]
        nF_precomputed = [self.nF_ef1[None, :, :], self.nF_ef2[None, :, :]] 
        for idx, ef in enumerate(f_poles):
            other_ef = f_poles[1-idx]
            w_p = ω - ef
            denom = -1*(w_p-energy_b1)*(w_p-energy_b2)*(w_p-energy_b3)*(w_p-energy_b4)*(ef-other_ef)
            w = w_p
            w2 = w*w
            w3 = w2*w
            A_G = -w*AtildeX -w2*Atildeψ/U + w3/U + Aprime
            #B_G = w*bX - w2*bψ/U + Bprime
            B_G = Bprime
            #C_G = w*cX - w2*cψ/U + Cprime
            C_G = Cprime
            D_G = -w*DtildeX  -w2*Dtildeψ/U + w3/U + Dprime
            pref = -nF_precomputed[idx]/denom
            term2f[:,:,0,0] += np.sum(A_G*pref,axis=2)
            term2f[:,:,0,1] += np.sum(B_G*pref,axis=2)
            term2f[:,:,1,0] += np.sum(C_G*pref,axis=2)
            term2f[:,:,1,1] += np.sum(D_G*pref,axis=2)
        # ---------------------------------------------------------
        # Bosonic poles
        # ---------------------------------------------------------
        for idx, eb in enumerate(b_poles):
            w_p = eb
            denomψ = (ω - w_p - energy_f1)*(ω - w_p - energy_f2)
            denom_X = self.eb_denom_X[idx,:][None, None, :] 
            denom = denomψ*denom_X
            w = w_p
            w2 = w*w
            w3 = w2*w
            A_G = -w*AtildeX -w2*Atildeψ/U + w3/U + Aprime
            #B_G = w*bX - w2*bψ/U + Bprime
            B_G = Bprime*(np.ones_like(w))
            #C_G = w*cX - w2*cψ/U + Cprime
            C_G = Cprime*np.ones_like(w)
            D_G = -w*DtildeX  -w2*Dtildeψ/U + w3/U + Dprime
            nB_val = self.nB_eb[idx, :][None, None, :]
            pref = nB_val/denom 
            term2b[:,:,0,0] += np.sum(A_G*pref,axis=2)
            term2b[:,:,0,1] += np.sum(B_G*pref,axis=2)
            term2b[:,:,1,0] += np.sum(C_G*pref,axis=2)
            term2b[:,:,1,1] += np.sum(D_G*pref,axis=2)

        #term2 = (-U**2)*(term2f+term2b)/N_BZ
    
        return (-U**2)*(term2f)/N_BZ, (-U**2)*(term2b)/N_BZ, term1
    
    def precompute_single_particle_GF(self, k_path, σ):
        """
        Precomputes the things which are not omega dependent.
        This means the poles for the spinon and rotor GFs in term 2.
        Even though Gψ takes iω-iν as an argument, the poles only depend on Mψ.
        These terms either depend on k-q, or just q.
        """
        Qψ = self.Qψ
        N_BZ = self.N_BZ
        N_path = len(k_path)
        k = k_path
        
        # Compute spinon dispersion 
        self.k_path_epskaσ = self.lat.eps_ksσ(k, 'a', σ)
        self.k_path_epskbσ = self.lat.eps_ksσ(k, 'b', σ)
        self.k_path_Vk = self.lat.Vk(k)
        self.k_path_VkC = np.conj(self.k_path_Vk)
        Qψ  = self.Qψ
        self.k_path_Mψ00 = Qψ[0,0]*self.k_path_epskaσ
        self.k_path_Mψ01 = -t*Qψ[0,1]*self.k_path_Vk
        self.k_path_Mψ10 = -t*Qψ[1,0]*self.k_path_VkC
        self.k_path_Mψ11 = Qψ[1,1]*self.k_path_epskbσ

        H_path = np.zeros((N_path, 2, 2), dtype=np.complex128)
        H_path[:,0,0] = Qψ[0,0] * self.k_path_epskaσ
        H_path[:,1,1] = Qψ[1,1] * self.k_path_epskbσ
        H_path[:,0,1] = t * -Qψ[0,1] * self.k_path_Vk 
        H_path[:,1,0] = t * -Qψ[1,0] * self.k_path_VkC
        
        self.Gψ_path_energies = np.linalg.eigvalsh(H_path) # (N_path,2)
        
        ######################
        
        # Need to recompute Gψ for (k-q) instead of (k)      
        q_grid = np.array(self.k_grid)
        k_minus_q = k[:,None,:] - q_grid[None,:,:] # (N_path,N_BZ,2) object
        
        kq_flat = k_minus_q.reshape(-1,2)
        epskaσ_kmq = self.lat.eps_ksσ(kq_flat,'a',σ)
        epskbσ_kmq = self.lat.eps_ksσ(kq_flat,'b',σ)
        Vk_kmq  = self.lat.Vk(kq_flat)
        VkC_kmq = np.conj(Vk_kmq)
        
# =============================================================================
#         kx = kq_flat[:,0]
#         ky = kq_flat[:,1]
#         def R(angle):
#             return np.array([[np.cos(angle), -np.sin(angle)],
#                              [np.sin(angle),  np.cos(angle)]])
#         Gamma = np.array([0, 0])
#         #Gamma_top = b1+b2
#         #M = (b1+b2)/2
#         K = (b1 + b2)/3
#         #Kp = (b1+b2)*(2/3)
#         Kp = -K
#         K2 = K@R((2*np.pi)/3)
#         M = np.array([K2[0],0])
#         
#         plt.figure(figsize=(5,8))
#         plt.gca().set_aspect('equal')
#         plt.grid(True)
#         i=100
#         plt.scatter(kx[(1225*i):(1225*(i+1))], ky[(1225*i):(1225*(i+1))],
#                     c=np.real(epskaσ_kmq[(1225*i):(1225*(i+1))]),s=1,cmap="seismic")
#         plt.xlabel("kx")
#         plt.ylabel("ky")
#         plt.colorbar(label=r"Re $ε_{ka+}(k-q)$")
#         plt.scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         plt.title(r" $ε_{ka+}(k-q)$ sampled at k=0")
#         plt.show()
#         
#         plt.figure(figsize=(5,8))
#         plt.gca().set_aspect('equal')
#         plt.grid(True)
#         plt.scatter(
#         kx,ky,c=np.abs(Vk_kmq),s=1,cmap="RdBu_r")
#         plt.xlabel("kx")
#         plt.ylabel("ky")
#         plt.title(r"$V_k(k-q)$")
#         plt.colorbar(label=r'$|V_k|$')
#         plt.scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         plt.show()
# =============================================================================

        Mψ00_kmq = Qψ[0,0]*epskaσ_kmq
        Mψ01_kmq = -t*Qψ[0,1]*Vk_kmq
        Mψ10_kmq = -t*Qψ[1,0]*VkC_kmq
        Mψ11_kmq = Qψ[1,1]*epskbσ_kmq
        
        Mψ00_kmq = Mψ00_kmq.reshape(N_path,N_BZ)
        Mψ01_kmq = Mψ01_kmq.reshape(N_path,N_BZ)
        Mψ10_kmq = Mψ10_kmq.reshape(N_path,N_BZ)
        Mψ11_kmq = Mψ11_kmq.reshape(N_path,N_BZ)
        
        Trψ  = Mψ00_kmq + Mψ11_kmq
        detψ = Mψ00_kmq*Mψ11_kmq  - Mψ01_kmq*Mψ10_kmq
        discψ = Trψ**2 - 4*detψ
        energy_f1 = 0.5*((Trψ-2*self.h) + np.sqrt(discψ))
        energy_f2 = 0.5*((Trψ-2*self.h) - np.sqrt(discψ))
        self.energy_f1_kmq = energy_f1
        self.energy_f2_kmq = energy_f2
        self.Mψ00_kmq = Mψ00_kmq
        self.Mψ01_kmq = Mψ01_kmq
        self.Mψ10_kmq = Mψ10_kmq
        self.Mψ11_kmq = Mψ11_kmq
        
# =============================================================================
#         
#         fig, ax = plt.subplots(1, 2, figsize=(8,5))
#         ax[0].set_aspect('equal')
#         ax[0].grid(True)
#         ctr1 = ax[0].scatter(kx, ky, c=np.real(energy_f1), s=1, cmap="RdBu_r")
#         ax[0].set_title(r"Re(ε+)")
#         ax[0].set_xlabel(r"$k_x$")
#         ax[0].set_ylabel(r"$k_y$")
#         fig.colorbar(ctr1, ax=ax[0])
#         ax[0].scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         ctr2 = ax[1].scatter(kx, ky, c=np.real(energy_f2), s=1, cmap="RdBu_r")
#         ax[1].set_aspect('equal')
#         ax[1].grid(True)
#         ax[1].set_title(r"Re(ε-)")
#         ax[1].set_xlabel(r"$k_x$")
#         ax[1].set_ylabel(r"$k_y$")
#         fig.colorbar(ctr2, ax=ax[1])
#         ax[1].scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         plt.tight_layout()
#         plt.show()
# =============================================================================
        
        # Can compute bosonic pole structures since they are independent of k
        # (path) and ω.
        eb = self.bos_poles
        
        denom_X = np.ones_like(eb)
        for i in range(4):
            for j in range(4):
                if i == j: continue
                denom_X[i, :] *= (eb[i, :] - eb[j, :])
        # This denom_X array contains all possible Π_j (eb_i-eb_j), i≠j
        self.eb_denom_X = denom_X # (4, N_BZ)
        self.nB_eb = self.nB(eb, self.T)
        
        
# =============================================================================
#         fig, ax = plt.subplots(2, 2, figsize=(10,8))
#         for axis in ax.flatten():
#             axis.set_aspect('equal')
#             axis.grid(True)
#             axis.set_xlabel(r"$k_x$")
#             axis.set_ylabel(r"$k_y$")
#         plt.grid(True)
#         kx = q_grid[:,0]
#         ky = q_grid[:,1]
#         #eb = eb.reshape((4,35,35))
#         ctr1 = ax[0,0].scatter(kx,ky, c=np.real(eb[0,:]), s=1, cmap="RdBu_r")
#         ax[0,0].set_title(r"$ε_1$")
#         fig.colorbar(ctr1, ax=ax[0,0])
#         ax[0,0].scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         ctr2 = ax[0,1].scatter(kx, ky, c=np.real(eb[1,:]), s=1, cmap="RdBu_r")
#         ax[0,1].set_title(r"$ε_2$")
#         fig.colorbar(ctr2, ax=ax[0,1])
#         ax[0,1].scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         ctr3 = ax[1,0].scatter(kx, ky, c=np.real(eb[2,:]), s=1, cmap="RdBu_r")
#         ax[1,0].set_title(r"$ε_3$")
#         fig.colorbar(ctr3, ax=ax[1,0])
#         ax[1,0].scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         ctr4 = ax[1,1].scatter(kx, ky, c=np.real(eb[3,:]), s=1, cmap="RdBu_r")
#         ax[1,1].set_title(r"$ε_4$")
#         fig.colorbar(ctr4, ax=ax[1,1])
#         ax[1,1].scatter([Gamma[0], K2[0], M[0], K[0], Kp[0]],
#                     [Gamma[1], K2[1], M[1], K[1], Kp[1]],
#                     s=20, c='green')
#         plt.tight_layout()
#         plt.show()
# =============================================================================
        # =========================================================
        # spatial derivatives (x and y)
        # =========================================================
        # Structure functions (for term 1)
        dVk_k_x = self.lat.dVk_dk(k, axis='x')
        dVk_k_y = self.lat.dVk_dk(k, axis='y')
        deps_a_k_x = self.lat.deps_ksσ_dk(k, 'a', σ, axis='x')
        deps_a_k_y = self.lat.deps_ksσ_dk(k, 'a', σ, axis='y')
        deps_b_k_x = self.lat.deps_ksσ_dk(k, 'b', σ, axis='x')
        deps_b_k_y = self.lat.deps_ksσ_dk(k, 'b', σ, axis='y')

        self.dAψ_1_x = -(Qψ[0,0] * deps_a_k_x)[None, :]
        self.dBψ_1_x = (-t * Qψ[0,1] * dVk_k_x)[None, :]
        self.dCψ_1_x = (-t * Qψ[1,0] * np.conj(dVk_k_x))[None, :]
        self.dDψ_1_x = -(Qψ[1,1] * deps_b_k_x)[None, :]

        self.dAψ_1_y = -(Qψ[0,0] * deps_a_k_y)[None, :]
        self.dBψ_1_y = (-t * Qψ[0,1] * dVk_k_y)[None, :]
        self.dCψ_1_y = (-t * Qψ[1,0] * np.conj(dVk_k_y))[None, :]
        self.dDψ_1_y = -(Qψ[1,1] * deps_b_k_y)[None, :]

        # 2. Structure Functions (for k-q i.e. term 2)
        dVk_flat_x = self.lat.dVk_dk(kq_flat, axis='x')
        dVk_flat_y = self.lat.dVk_dk(kq_flat, axis='y')
        deps_a_flat_x = self.lat.deps_ksσ_dk(kq_flat, 'a', σ, axis='x')
        deps_a_flat_y = self.lat.deps_ksσ_dk(kq_flat, 'a', σ, axis='y')
        deps_b_flat_x = self.lat.deps_ksσ_dk(kq_flat, 'b', σ, axis='x')
        deps_b_flat_y = self.lat.deps_ksσ_dk(kq_flat, 'b', σ, axis='y')

        dVk_kmq_x = dVk_flat_x.reshape(N_path, N_BZ)
        dVk_kmq_y = dVk_flat_y.reshape(N_path, N_BZ)
        deps_a_kmq_x = deps_a_flat_x.reshape(N_path, N_BZ)
        deps_a_kmq_y = deps_a_flat_y.reshape(N_path, N_BZ)
        deps_b_kmq_x = deps_b_flat_x.reshape(N_path, N_BZ)
        deps_b_kmq_y = deps_b_flat_y.reshape(N_path, N_BZ)

        # 3. Matrix element derivatives
        self.dAψ_x = Qψ[0,0] * deps_a_kmq_x
        self.dBψ_x = -t * Qψ[0,1] * dVk_kmq_x
        self.dCψ_x = -t * Qψ[1,0] * np.conj(dVk_kmq_x)
        self.dDψ_x = Qψ[1,1] * deps_b_kmq_x

        self.dAψ_y = Qψ[0,0] * deps_a_kmq_y
        self.dBψ_y = -t * Qψ[0,1] * dVk_kmq_y
        self.dCψ_y = -t * Qψ[1,0] * np.conj(dVk_kmq_y)
        self.dDψ_y = Qψ[1,1] * deps_b_kmq_y

        safe_disc = np.maximum(discψ, 1e-12)

        # X-axis poles
        dTr_x = self.dAψ_x + self.dDψ_x
        dDet_x = self.dAψ_x*Mψ11_kmq + Mψ00_kmq*self.dDψ_x - (self.dBψ_x*Mψ10_kmq + Mψ01_kmq*self.dCψ_x)
        ddisc_x = 2*Trψ*dTr_x - 4*dDet_x
        self.def1_x_kmq = 0.5 * (dTr_x + 0.5 * ddisc_x / np.sqrt(safe_disc))
        self.def2_x_kmq = 0.5 * (dTr_x - 0.5 * ddisc_x / np.sqrt(safe_disc))

        # Y-axis poles
        dTr_y = self.dAψ_y + self.dDψ_y
        dDet_y = self.dAψ_y*Mψ11_kmq + Mψ00_kmq*self.dDψ_y - (self.dBψ_y*Mψ10_kmq + Mψ01_kmq*self.dCψ_y)
        ddisc_y = 2*Trψ*dTr_y - 4*dDet_y
        self.def1_y_kmq = 0.5 * (dTr_y + 0.5 * ddisc_y / np.sqrt(safe_disc))
        self.def2_y_kmq = 0.5 * (dTr_y - 0.5 * ddisc_y / np.sqrt(safe_disc))
        
        # More miscellaneous independent quantities
        self.Bprime_kmq = -self.Mψ01_kmq * self.MX01
        self.Cprime_kmq = -self.Mψ10_kmq * self.MX10
        self.AtildeX_const = self.rho + self.MX11
        self.DtildeX_const = self.rho + self.MX00

        # 3. Fermionic Pole Differences (f5)
        self.f5_1 = self.energy_f1_kmq - self.energy_f2_kmq
        self.f5_2 = self.energy_f2_kmq - self.energy_f1_kmq
        
        # 4. Fermi-Dirac occ func
        self.nF_ef1 = self.nF(-self.energy_f1_kmq, self.T)
        self.nF_ef2 = self.nF(-self.energy_f2_kmq, self.T)
        return 0 
    
    def plot_spectral_function(self, σ, omega_range, eta=0.01):
        """
        Reproduces the spectral map of |det(G)| along the high-symmetry path.
        """
        # 1. Get the k-path from your lattice object
        # Assuming self.lat.get_hsp_path() returns (k_points, labels, tick_indices)
        k_path, ticks, labels = self.get_hsp_path_2(num_pts=100)
        k_path = np.array(k_path)
        N_path = len(k_path)
        # 2. Setup frequency grid
        omegas = np.linspace(omega_range[0], omega_range[-1],200)
        intensity = np.zeros((len(omegas),N_path))
        intensityf = np.zeros((len(omegas),N_path))
        intensityb = np.zeros((len(omegas),N_path))
        
        # 2b Calculate spinon poles:
        print("Precomputing variables...")
        self.precompute_single_particle_GF(k_path, σ)

        # 3. Calculate |det(G(k, w + i*eta))|
        print("Starting detG plot...")
        iterable=0
        batch = 20
        for start in range(0,len(omegas),batch):
            end = min(start+batch,len(omegas))
            ω_batch = omegas[start:end] + 1j*eta
            term2f, term2b, term1 = self.single_particle_GF(σ, k_path, ω_batch)
            intensity[start:end] = np.abs(np.linalg.det(term2f+term2b+term1))
            intensityf[start:end] = np.abs(np.linalg.det(term2f))
            intensityb[start:end] = np.abs(np.linalg.det(term2b))
            iterable+=1
            print(f"{iterable} batches complete. ({min(batch*iterable,len(omegas))}/{len(omegas)})")
       
        # 4. Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid('on')
        # Use LogNorm for the intensity to match the logarithmic colorbar in your image
        from matplotlib.colors import LogNorm
        from matplotlib.lines import Line2D
        im = ax.imshow(intensityf, extent=[0, len(k_path), omegas[0], omegas[-1]], 
                       aspect='auto', origin='lower', cmap='bwr_r',
                       norm=LogNorm(vmin=1e-6, vmax=1e1))
        ax.plot(np.arange(N_path), self.Gψ_path_energies[:,0]*10, color='black')
        ax.plot(np.arange(N_path), self.Gψ_path_energies[:,1]*10, color='black')
        # Formatting
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\omega / t$")
        
        cbar = plt.colorbar(im, label=r"$|\det(G)|$", orientation='horizontal',
                     location = 'top', pad=0.02)
        cbar.set_label(r"$|\det(G)|$, term 2 fermionic part"+f", U/t={self.U}, λ/t = {self.lat.λ}") # , term 2 fermionic part
        legend_handle = Line2D([0], [0], color='black', linewidth=1.0)
        ax.legend(
            handles=[legend_handle],
            labels=[f'spinon dispersion, multiplied by 10. U/t = {self.U}, λ/t = {self.lat.λ}'],
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
            ncol=1)
        
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use LogNorm for the intensity to match the logarithmic colorbar in your image
        im = ax.imshow(intensityb, extent=[0, len(k_path), omegas[0], omegas[-1]], 
                       aspect='auto', origin='lower', cmap='bwr_r',
                       norm=LogNorm(vmin=1e-3, vmax=1e1))
        
        ax.plot(np.arange(N_path), self.Gψ_path_energies[:,0]*10, color='black')
        ax.plot(np.arange(N_path), self.Gψ_path_energies[:,1]*10, color='black')
        # Formatting
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\omega / t$")
        
        cbar = plt.colorbar(im, label=r"$|\det(G)|$", orientation='horizontal',
                     location = 'top', pad=0.02)
        cbar.set_label(r"$|\det(G)|$, term 2 bosonic part"+f", U/t={self.U}, λ/t = {self.lat.λ}")
        legend_handle = Line2D([0], [0], color='black', linewidth=1.0)
        ax.legend(
            handles=[legend_handle],
            labels=['spinon dispersion, multiplied by 10'],
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
            ncol=1)
        
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use LogNorm for the intensity to match the logarithmic colorbar in the paper
        im = ax.imshow(intensity, extent=[0, len(k_path), omegas[0], omegas[-1]], 
                       aspect='auto', origin='lower', cmap='bwr_r',
                       norm=LogNorm(vmin=1e-6, vmax=1e1))
        
        #ax.plot(np.arange(N_path), self.Gψ_path_energies[:,0]*10, color='black')
        #ax.plot(np.arange(N_path), self.Gψ_path_energies[:,1]*10, color='black')
        # Formatting
        ax.set_xticks(ticks)
        ax.set_ylabel(r"$\omega / t$")
        
        cbar = plt.colorbar(im, label=r"$|\det(G)|$", orientation='horizontal',
                     location = 'top', pad=0.02)
        cbar.set_label(r"$|\det(G(\mathbf{k},ω+i0^+))|$"+f", U/t={self.U}, λ/t = {self.lat.λ}")
        legend_handle = Line2D([0], [0], color='black', linewidth=1.0)
        #ax.legend(
        #    handles=[legend_handle],
        #    labels=['spinon dispersion, multiplied by 10'],
        #    loc='lower center',
        #    bbox_to_anchor=(0.5, -0.15),
        #    frameon=False,
        #    ncol=1)
        
        plt.tight_layout()
        plt.show()
        return 0 
    
    def plot_fermi_level_bz(self, σ=1, eta=0.01, N_BZ = Ns**2):
        """
        Plots |det(G(k, w=0))| over the full 2D Brillouin Zone.
        """
        # 1. Prepare the full k-grid and frequency (w=0)
        # k_grid = np.array(self.k_grid)  # Shape (N_BZ, 2)
        # N = int(np.sqrt(N_BZ))  
        N = 50
        k_grid = self.lat.lattice_k_mesh(N, N)     # shape → (N*N, 2)
        omega = np.array([1.0 + 1j*eta])
        
        # 2. Precompute variables for the full grid
        # We temporarily hijack precompute_single_particle_GF to use the full grid
        print(f"Computing Green's function for {len(k_grid)} k-points at ω={np.real(omega)[0]}...")
        self.precompute_single_particle_GF(k_grid, σ)
        
        # 3. Calculate the Green's function components
        # single_particle_GF returns (term2f, term2b, term1)
        term2f, term2b, term1 = self.single_particle_GF(σ, k_grid, omega)
        
        # 4. Calculate |det(G)| for the combined GF
        # Note: single_particle_GF output shapes are (N_omega, N_k, 2, 2)
        G_total = term2f[0] + term2b[0] + term1[0]  # Take first index for w=0
        det_G = np.abs(np.linalg.det(G_total))
        
        # 5. Reshape for contour plotting
        # Assuming k_grid was generated as a square mesh of size sqrt(N_BZ)
        kx = k_grid[:, 0].reshape(N,N)
        ky = k_grid[:, 1].reshape(N,N)
        z = det_G.reshape(N, N)
        
        # 6. Plotting
        fig, ax = plt.subplots(figsize=(7, 6))
        from matplotlib.colors import LogNorm 
        
        cp = ax.contourf(kx, ky, z, levels=50, cmap='bwr_r', norm=LogNorm(vmin=1e-6, vmax=1e1))
        # draw Bz
        origin = -(b1 + b2) / 2
        corners = np.array([
            origin,               # (0,0)
            origin + b1,          # (1,0)
            origin + b1 + b2,     # (1,1)
            origin + b2,          # (0,1)
            origin                # Close the loop
        ])
        
        ax.plot(corners[:, 0], corners[:, 1], color='black', linewidth=1.5, linestyle='--')
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        #title_string = r'$|\\det(G(\\mathbf{{k}}, \\omega=0))|$, $U/t=$' + f'{self.U}'
        #ax.set_title(r'$|\\det(G(\\mathbf{{k}}, \\omega=0))|$, $U/t={self.U}$')
        
        cbar = plt.colorbar(cp, ax=ax)
        cbar.set_label(r'$|\det(G)|$')
        
        plt.tight_layout()
        plt.show()
        return 0
    def get_G_inv(self, G):
        """
        Fast, analytical 2x2 matrix inversion for batched arrays.
        Expects G shape: (..., 2, 2)
        """
        a = G[..., 0, 0]
        b = G[..., 0, 1]
        c = G[..., 1, 0]
        d = G[..., 1, 1]

        det = a*d - b*c
        
        # Clamp the determinant to avoid division by exact zero at Dirac points
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)

        G_inv = np.zeros_like(G)
        G_inv[..., 0, 0] =  d / det
        G_inv[..., 0, 1] = -b / det
        G_inv[..., 1, 0] = -c / det
        G_inv[..., 1, 1] =  a / det

        return G_inv
    
    def dG_dω(self, σ, k, ω):
        """
        Calculates the exact analytical derivative of the interacting Green's function
        with respect to the external Matsubara frequency ω.      
        Note: z = iω
        """
        ω = np.asarray(ω)
        if ω.ndim == 0:
            ω = ω[None]
        Nω = len(ω)
        N_path = len(k)
        N_BZ = self.N_BZ
        U = self.U
        T = self.T
        rho = self.rho
        
        # Transform ω to make shape match for braodcasting; define z.
        ω = ω[:,None,None]
        z = 1j * ω 
        
        # ---------------------------------------------------------
        # Term 1 
        # ---------------------------------------------------------
        z_1 = z[:,:,0]
        Mψ00_1 = self.k_path_Mψ00[None,:]
        Mψ01_1 = self.k_path_Mψ01[None,:] 
        Mψ10_1 = self.k_path_Mψ10[None,:] 
        Mψ11_1 = self.k_path_Mψ11[None,:] 

        Aψ_1 = z_1 - Mψ00_1
        Bψ_1 = Mψ01_1
        Cψ_1 = Mψ10_1
        Dψ_1 = z_1 - Mψ11_1
        detψ_1 = Aψ_1*Dψ_1 - Bψ_1*Cψ_1
        
        # Logarithmic derivative of the Term 1 determinant
        d_det_dw = 1j * (Aψ_1 + Dψ_1)
        S_t1 = d_det_dw / detψ_1
        dGψ_k_dw = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        
        # Quotient Rule: (N' - N*(D'/D))/ D
        dGψ_k_dw[:,:,0,0] = (1j - Dψ_1 * S_t1) / detψ_1
        dGψ_k_dw[:,:,0,1] = ( 0 - Bψ_1 * S_t1) / detψ_1
        dGψ_k_dw[:,:,1,0] = ( 0 - Cψ_1 * S_t1) / detψ_1
        dGψ_k_dw[:,:,1,1] = (1j - Aψ_1* S_t1) / detψ_1

        dω_term1 = self.Z * dGψ_k_dw
        
        # ---------------------------------------------------------
        # Term 2 Setup
        # ---------------------------------------------------------
        energy_f1 = self.energy_f1_kmq[None,:,:]
        energy_f2 = self.energy_f2_kmq[None,:,:]
        energy_b1 = self.bos_poles[0,:][None,None,:]
        energy_b2 = self.bos_poles[1,:][None,None,:]
        energy_b3 = self.bos_poles[2,:][None,None,:]
        energy_b4 = self.bos_poles[3,:][None,None,:]
        
        Mψ00_kmq = self.Mψ00_kmq
        Mψ01_kmq = self.Mψ01_kmq
        Mψ10_kmq = self.Mψ10_kmq
        Mψ11_kmq = self.Mψ11_kmq
        
        aX = self.MX00
        bX = self.MX01
        cX = self.MX10
        dX = self.MX11
        
        Atildeψ = z - Mψ11_kmq
        Dtildeψ = z - Mψ00_kmq
        
        bψ = Mψ01_kmq
        cψ = Mψ10_kmq
        
        AtildeX = rho + dX
        DtildeX = rho + aX
        
        Aprime = Atildeψ * AtildeX
        Bprime = -bψ * bX
        Cprime = -cψ * cX
        Dprime = Dtildeψ * DtildeX

        term2f_dG_dw = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        term2b_dG_dw = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        
        # ---------------------------------------------------------
        # Fermionic poles
        # ---------------------------------------------------------
        f_poles = [energy_f1, energy_f2]
        b_poles = [energy_b1, energy_b2, energy_b3, energy_b4]
        
        # Fast local precomputation of exp() to avoid 3D array loop calls
        nF_vals = [-self.nF(-energy_f1, T), -self.nF(-energy_f2, T)]
        
        for idx, ef in enumerate(f_poles):
            other_ef = f_poles[1-idx]
            w = z - ef
            w2 = w*w
            w3 = w2*w
            
            # Inline denoms (Memory Safe)
            denom = -1 * (w - energy_b1) * (w - energy_b2) * (w - energy_b3) * (w - energy_b4) * (ef - other_ef)
            S_f = 1j * (1/(w - energy_b1) + 1/(w - energy_b2) + 1/(w - energy_b3) + 1/(w - energy_b4))

            A_G = -w*AtildeX - w2*Atildeψ/U + w3/U + Aprime
            B_G = Bprime 
            C_G = Cprime 
            D_G = -w*DtildeX - w2*Dtildeψ/U + w3/U + Dprime
            
            # Numerator Derivatives
            dAg_dw_f = 1j * (2*w / U) * (w - Atildeψ) 
            dBg_dw_f = 0.0 
            dCg_dw_f = 0.0
            dDg_dw_f = 1j * (2*w / U) * (w - Dtildeψ)

            pref = nF_vals[idx] / denom
            
            term2f_dG_dw[:,:,0,0] += np.sum((dAg_dw_f - A_G * S_f) * pref, axis=2)
            term2f_dG_dw[:,:,0,1] += np.sum((dBg_dw_f - B_G * S_f) * pref, axis=2)
            term2f_dG_dw[:,:,1,0] += np.sum((dCg_dw_f - C_G * S_f) * pref, axis=2)
            term2f_dG_dw[:,:,1,1] += np.sum((dDg_dw_f - D_G * S_f) * pref, axis=2)
            
        # ---------------------------------------------------------
        # Bosonic poles
        # ---------------------------------------------------------
        for idx, eb in enumerate(b_poles):
            w = eb
            w2 = w*w
            w3 = w2*w
            
            # Local scalar evaluation for denom_X
            denom_X = 1.0
            for jdx, other_eb in enumerate(b_poles):
                if idx == jdx:
                    continue
                denom_X *= (eb - other_eb)
                
            denom = (z - w - energy_f1) * (z - w - energy_f2) * denom_X
            S_b = 1j * (1/(z - w - energy_f1) + 1/(z - w - energy_f2))
            
            A_G = -w*AtildeX - w2*Atildeψ/U + w3/U + Aprime
            B_G = Bprime 
            C_G = Cprime 
            D_G = -w*DtildeX - w2*Dtildeψ/U + w3/U + Dprime
            
            dAg_dw_b = 1j * (AtildeX - w2/U)
            dBg_dw_b = 0.0
            dCg_dw_b = 0.0
            dDg_dw_b = 1j * (DtildeX - w2/U)
            
            pref = self.nB(w, T) / denom
            
            term2b_dG_dw[:,:,0,0] += np.sum((dAg_dw_b - A_G * S_b) * pref, axis=2)
            term2b_dG_dw[:,:,0,1] += np.sum((dBg_dw_b - B_G * S_b) * pref, axis=2)
            term2b_dG_dw[:,:,1,0] += np.sum((dCg_dw_b - C_G * S_b) * pref, axis=2)
            term2b_dG_dw[:,:,1,1] += np.sum((dDg_dw_b - D_G * S_b) * pref, axis=2)

        dω_term2 = (-U**2)*(term2f_dG_dw + term2b_dG_dw) / N_BZ
        
        return dω_term2 + dω_term1
    
    def dG_dk_xy(self, σ, k, ω):
        """
        Calculates the exact analytical spatial derivative (kx and ky) of the 
        electron Green's function.
        Note: z = iω
        Note: k should have the shape of N_BZ
        Returns: (dG_dkx, dG_dky)
        """
        ω = np.asarray(ω)
        if ω.ndim == 0:
            ω = ω[None]
        Nω = len(ω)
        N_path = len(k)
        N_BZ = self.N_BZ
        U = self.U
        T = self.T
        rho = self.rho
        
        # Transform ω to make shape match for braodcasting; define z.
        ω = ω[:,None,None]
        z = 1j * ω 
        
        # ---------------------------------------------------------
        # Term 1 
        # ---------------------------------------------------------
        z_1 = z[:,:,0]
        Aψ_1 = z_1 - self.k_path_Mψ00[None,:]
        Bψ_1 = self.k_path_Mψ01[None,:]
        Cψ_1 = self.k_path_Mψ10[None,:]
        Dψ_1 = z_1 - self.k_path_Mψ11[None,:]
        detψ_1 = Aψ_1*Dψ_1 - Bψ_1*Cψ_1
        
        # X-Axis Term 1
        d_det_dk_x = self.dAψ_1_x*Dψ_1 + Aψ_1*self.dDψ_1_x - (self.dBψ_1_x*Cψ_1 + Bψ_1*self.dCψ_1_x)
        S_1_x = d_det_dk_x / detψ_1
        dGψ_k_dkx = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        dGψ_k_dkx[:,:,0,0] = (self.dDψ_1_x - Dψ_1 * S_1_x) / detψ_1
        dGψ_k_dkx[:,:,0,1] = (self.dBψ_1_x - Bψ_1 * S_1_x) / detψ_1
        dGψ_k_dkx[:,:,1,0] = (self.dCψ_1_x - Cψ_1 * S_1_x) / detψ_1
        dGψ_k_dkx[:,:,1,1] = (self.dAψ_1_x - Aψ_1 * S_1_x) / detψ_1
        term1_dG_dkx = self.Z * dGψ_k_dkx

        # Y-Axis Term 1
        d_det_dk_y = self.dAψ_1_y*Dψ_1 + Aψ_1*self.dDψ_1_y - (self.dBψ_1_y*Cψ_1 + Bψ_1*self.dCψ_1_y)
        S_1_y = d_det_dk_y / detψ_1
        dGψ_k_dky = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        dGψ_k_dky[:,:,0,0] = (self.dDψ_1_y - Dψ_1 * S_1_y) / detψ_1
        dGψ_k_dky[:,:,0,1] = (self.dBψ_1_y - Bψ_1 * S_1_y) / detψ_1
        dGψ_k_dky[:,:,1,0] = (self.dCψ_1_y - Cψ_1 * S_1_y) / detψ_1
        dGψ_k_dky[:,:,1,1] = (self.dAψ_1_y - Aψ_1 * S_1_y) / detψ_1
        term1_dG_dky = self.Z * dGψ_k_dky

        # ---------------------------------------------------------
        # Term 2 Setup
        # ---------------------------------------------------------
        energy_f1 = self.energy_f1_kmq[None,:,:]
        energy_f2 = self.energy_f2_kmq[None,:,:]
        energy_b1 = self.bos_poles[0,:][None,None,:]
        energy_b2 = self.bos_poles[1,:][None,None,:]
        energy_b3 = self.bos_poles[2,:][None,None,:]
        energy_b4 = self.bos_poles[3,:][None,None,:]
        
        Aψ = self.Mψ00_kmq
        Dψ = self.Mψ11_kmq
        bψ = self.Mψ01_kmq
        cψ = self.Mψ10_kmq
        
        aX = self.MX00
        bX = self.MX01
        cX = self.MX10
        dX = self.MX11
        
        Atildeψ = z - Dψ
        Dtildeψ = z - Aψ
        AtildeX = rho + dX
        DtildeX = rho + aX
        
        Aprime = Atildeψ * AtildeX
        Bprime = -bψ * bX
        Cprime = -cψ * cX
        Dprime = Dtildeψ * DtildeX
        
        term2f_dG_dkx = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        term2b_dG_dkx = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        term2f_dG_dky = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        term2b_dG_dky = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)

        # ---------------------------------------------------------
        # Fermionic poles
        # ---------------------------------------------------------
        f_poles = [energy_f1, energy_f2]
        d_f_poles_x = [self.def1_x_kmq[None,:,:], self.def2_x_kmq[None,:,:]]
        d_f_poles_y = [self.def1_y_kmq[None,:,:], self.def2_y_kmq[None,:,:]]
        
        # Fast local precomputation of exp() to avoid 3D array loop calls
        nF_vals = [-self.nF(-energy_f1, T), -self.nF(-energy_f2, T)]
        
        for idx, ef in enumerate(f_poles):
            other_ef = f_poles[1-idx]
            w = z - ef
            w2 = w*w
            w3 = w2*w

            # Inline Denominator
            denom = -1*(w - energy_b1)*(w - energy_b2)*(w - energy_b3)*(w - energy_b4)*(ef - other_ef)
            pref = nF_vals[idx] / denom
            
            A_G = -w*AtildeX - w2*Atildeψ/U + w3/U + Aprime
            B_G = Bprime 
            C_G = Cprime 
            D_G = -w*DtildeX - w2*Dtildeψ/U + w3/U + Dprime
            
            # --- Evaluate X-Axis ---
            def_x = d_f_poles_x[idx]
            def_other_x = d_f_poles_x[1-idx]
            
            # Inline S_f_x calculation
            S_f_x = (-def_x)*(1/(w - energy_b1) + 1/(w - energy_b2) + 1/(w - energy_b3) + 1/(w - energy_b4)) + ((def_x - def_other_x) / (ef - other_ef))
            
            dAg_x = self.dDψ_x*(-AtildeX + w2/U) + def_x*(AtildeX - (w/U)*(2*Dψ - 2*ef + w))
            dBg_x = -self.MX01 * self.dBψ_x
            dCg_x = -self.MX10 * self.dCψ_x
            dDg_x = self.dAψ_x*(-DtildeX + w2/U) + def_x*(DtildeX - (w/U)*(2*Aψ - 2*ef + w))

            term2f_dG_dkx[:,:,0,0] += np.sum((dAg_x - A_G * S_f_x) * pref, axis=2)
            term2f_dG_dkx[:,:,0,1] += np.sum((dBg_x - B_G * S_f_x) * pref, axis=2)
            term2f_dG_dkx[:,:,1,0] += np.sum((dCg_x - C_G * S_f_x) * pref, axis=2)
            term2f_dG_dkx[:,:,1,1] += np.sum((dDg_x - D_G * S_f_x) * pref, axis=2)

            # --- Evaluate Y-Axis ---
            def_y = d_f_poles_y[idx]
            def_other_y = d_f_poles_y[1-idx]
            
            # Inline S_f_y calculation
            S_f_y = (-def_y)*(1/(w - energy_b1) + 1/(w - energy_b2) + 1/(w - energy_b3) + 1/(w - energy_b4)) + ((def_y - def_other_y) / (ef - other_ef))
            
            dAg_y = self.dDψ_y*(-AtildeX + w2/U) + def_y*(AtildeX - (w/U)*(2*Dψ - 2*ef + w))
            dBg_y = -self.MX01 * self.dBψ_y
            dCg_y = -self.MX10 * self.dCψ_y
            dDg_y = self.dAψ_y*(-DtildeX + w2/U) + def_y*(DtildeX - (w/U)*(2*Aψ - 2*ef + w))

            term2f_dG_dky[:,:,0,0] += np.sum((dAg_y - A_G * S_f_y) * pref, axis=2)
            term2f_dG_dky[:,:,0,1] += np.sum((dBg_y - B_G * S_f_y) * pref, axis=2)
            term2f_dG_dky[:,:,1,0] += np.sum((dCg_y - C_G * S_f_y) * pref, axis=2)
            term2f_dG_dky[:,:,1,1] += np.sum((dDg_y - D_G * S_f_y) * pref, axis=2)

        # ---------------------------------------------------------
        # Bosonic poles
        # ---------------------------------------------------------
        b_poles = [energy_b1, energy_b2, energy_b3, energy_b4]
        
        for idx, eb in enumerate(b_poles):
            w = eb
            w2 = w*w
            w3 = w2*w
            
            # Local scalar evaluation for denom_X
            denom_X = 1.0
            for jdx, other_eb in enumerate(b_poles):
                if idx == jdx:
                    continue
                denom_X *= (eb - other_eb)
            
            denom = (z - w - energy_f1) * (z - w - energy_f2) * denom_X
            pref = self.nB(w, T) / denom
            
            A_G = -w*AtildeX - w2*Atildeψ/U + w3/U + Aprime
            B_G = Bprime 
            C_G = Cprime 
            D_G = -w*DtildeX - w2*Dtildeψ/U + w3/U + Dprime

            # --- Evaluate X-Axis ---
            S_b_x = (-d_f_poles_x[0] / (z - w - energy_f1)) + (-d_f_poles_x[1] / (z - w - energy_f2))
            
            dAg_b_x = self.dDψ_x * (-AtildeX + w2/U)
            dBg_b_x = -self.MX01 * self.dBψ_x
            dCg_b_x = -self.MX10 * self.dCψ_x
            dDg_b_x = self.dAψ_x * (-DtildeX + w2/U)

            term2b_dG_dkx[:,:,0,0] += np.sum((dAg_b_x - A_G * S_b_x) * pref, axis=2)
            term2b_dG_dkx[:,:,0,1] += np.sum((dBg_b_x - B_G * S_b_x) * pref, axis=2)
            term2b_dG_dkx[:,:,1,0] += np.sum((dCg_b_x - C_G * S_b_x) * pref, axis=2)
            term2b_dG_dkx[:,:,1,1] += np.sum((dDg_b_x - D_G * S_b_x) * pref, axis=2)

            # --- Evaluate Y-Axis ---
            S_b_y = (-d_f_poles_y[0] / (z - w - energy_f1)) + (-d_f_poles_y[1] / (z - w - energy_f2))
            
            dAg_b_y = self.dDψ_y * (-AtildeX + w2/U)
            dBg_b_y = -self.MX01 * self.dBψ_y
            dCg_b_y = -self.MX10 * self.dCψ_y
            dDg_b_y = self.dAψ_y * (-DtildeX + w2/U)

            term2b_dG_dky[:,:,0,0] += np.sum((dAg_b_y - A_G * S_b_y) * pref, axis=2)
            term2b_dG_dky[:,:,0,1] += np.sum((dBg_b_y - B_G * S_b_y) * pref, axis=2)
            term2b_dG_dky[:,:,1,0] += np.sum((dCg_b_y - C_G * S_b_y) * pref, axis=2)
            term2b_dG_dky[:,:,1,1] += np.sum((dDg_b_y - D_G * S_b_y) * pref, axis=2)

        # Combine
        dterm2_x = (-U**2)*(term2f_dG_dkx + term2b_dG_dkx) / N_BZ
        dterm2_y = (-U**2)*(term2f_dG_dky + term2b_dG_dky) / N_BZ
        
        return dterm2_x + term1_dG_dkx, dterm2_y + term1_dG_dky
    
    def calculate_N3_invariant(self, σ, omega_range):
        """
        Calculates the interacting topological winding number N_3.
        omega_grid: 1D array of real number.
        """
        timings = {}
        print("Precomputing grid arrays")
        with timer("Precomputing grid arrays", timings):
            omegas = np.linspace(omega_range[0], omega_range[1], 400)
            # For the integration, the 'path' is just the flattened Brillouin Zone
            k_grid_flat = np.array(self.k_grid) 
            self.precompute_single_particle_GF(k_grid_flat, σ)
            
            N_BZ = self.N_BZ
            dω = omegas[1] - omegas[0]

        total_sum = 0.0
        iterable = 0
        batch = 5
        for start in range(0,len(omegas),batch):
            end = min(start + batch, len(omegas))
            ω_batch = omegas[start:end]
            matsubara_batch = 1j * ω_batch
            print("Calculating Base G")
            with timer("Calculating Base G", timings):
                # single_particle_GF should evaluate with imaginary omega as argument.
                term2f, term2b, term1 = self.single_particle_GF(σ, k_grid_flat, matsubara_batch)
                G = term1 + term2f + term2b
            #print("Inverting G")            
            with timer("Inverting G", timings):
                G_inv = self.get_G_inv(G)
            #print("Derivatives")
            with timer("Calculating ω Derivative", timings):
                # Pass real grid here, because I wrote these to use z=1j*ω and cba to change it.
                dG_dw = self.dG_dω(σ, k_grid_flat, ω_batch)
            with timer("Calculating k Derivatives", timings):
                dG_dkx, dG_dky = self.dG_dk_xy(σ, k_grid_flat, ω_batch)
            #print("Blocks")                
            with timer("Building G^-1∂μG blocks", timings):
                # Matrix multiply these blocks. 
                B_ω = G_inv @ dG_dw
                B_x = G_inv @ dG_dkx
                B_y = G_inv @ dG_dky
            #print("Trace")
            with timer("Calculating Levi-Civita Trace", timings):
                # 6 permutations of ε_{abc} B_a B_b B_c
                term1_tr = B_ω @ B_x @ B_y   #abc +
                term2_tr = B_ω @ B_y @ B_x   #acb -
# =============================================================================
#                 term3_tr = B_y @ B_x @ B_ω   #cab + # this was mistaken, it's acc. cba
#                 term4_tr = B_y @ B_ω @ B_x   #cba - # this too. it's actually cab
#                 term5_tr = B_x @ B_y @ B_ω   #bca +
#                 term6_tr = B_x @ B_ω @ B_y   #bac -
# =============================================================================
                # because the trace is equivalent under cyclic permutation, terms
                # 1,3, 5 and terms 2,4,6 are equivalent.
                
                integrand = 3*(term1_tr - term2_tr)
                # Trace over the 2x2 matrix indices (the last two dimensions)
                trace_array = np.trace(integrand, axis1=-2, axis2=-1)
            #print("Integrating")
            with timer("Integrating over Frequency and Brillouin Zone", timings):
                # Sum over the flattened BZ (axis 1) and Frequency (axis 0)
                total_sum += np.sum(trace_array)
            
            iterable+=1
            print(f"{iterable} batches complete. ({min(batch*iterable,len(omegas))}/{len(omegas)})")
            print(f"\n Loop {iterable} timing:")
            total_time_loop = 0
            cum_total_time_loop = 0
            for k, v in timings.items():
                print(f"  {k:20s} {(v/iterable):9.4f} s")
                total_time_loop += v/iterable
                cum_total_time_loop += v
            mins = int(total_time_loop // 60)
            secs = total_time_loop % 60
            cum_mins = int(cum_total_time_loop // 60)
            cum_secs = total_time_loop % 60
            #print(f"Loop runtime: {mins} mins {secs:.2f} secs")
            #print(f"Total elapsed runtime: {cum_mins} mins {cum_secs:.2f} secs")
            #print("-----------------------------------")
            #print("\n\n")
        
        #print() # Clear the carriage return line

        # Note: Added self.lat.area assuming 'area' is an attribute of your lattice object
        prefactor = dω/(6 * area * N_BZ)
        N3 = prefactor * total_sum 
        
        print(f"\n U/t = {self.U}, timing:")
        total_time = 0
        for k, v in timings.items():
            #print(f"  {k:20s} {v:9.4f} s")
            total_time += v
        mins = int(total_time // 60)
        secs = total_time % 60
        print(f"Runtime: {mins} mins {secs:.2f} secs")
        print("-----------------------------------")
        print(f"\n N_3 = {N3}")
        
        return np.real(N3)
# %% Looper over U
def run_vs_U(
    U_over_t_values,
    base_lattice,
    T, Ns, n,
    Qpsi_guess,
    max_iter,
    tol):
    
    rows = []
    # update lattice interaction
    lattice = Lattice(
        lattice_vectors=[base_lattice.a1, base_lattice.a2],
        nn=base_lattice.nn,
        nnn=base_lattice.nnn,
        SOC=base_lattice.λ,
        hopping=base_lattice.t
    )
    
    #lattice.plot_BZ(Ns, Ns)
    lattice.calculate_structure_functions(Ns,Ns)
    #lattice.plot_structure_functions(Ns, Ns)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14 
    for U_over_t in U_over_t_values:
        print(f"Running for U/t = {U_over_t}")
        
        solver = SlaveRotorSolver(
            lattice, T, n,
            Qpsi_guess.copy(),
            max_iter,
            tol,
            U_over_t * base_lattice.t
        )

        solver.loop()

        # take last state
        QX = solver.QX
        Qpsi = solver.Qψ

        rows.append({
            "U_over_t": U_over_t,

            # QX
            "QX_00_real": QX[0,0].real,
            "QX_00_imag": QX[0,0].imag,
            "QX_11_real": QX[1,1].real,
            "QX_11_imag": QX[1,1].imag,
            "QX_01_real": QX[0,1].real,
            "QX_01_imag": QX[0,1].imag,
            "QX_10_real": QX[1,0].real,
            "QX_10_imag": QX[1,0].imag,

            # Qpsi
            "Qpsi_00_real": Qpsi[0,0].real,
            "Qpsi_00_imag": Qpsi[0,0].imag,
            "Qpsi_11_real": Qpsi[1,1].real,
            "Qpsi_11_imag": Qpsi[1,1].imag,
            "Qpsi_01_real": Qpsi[0,1].real,
            "Qpsi_01_imag": Qpsi[0,1].imag,
            "Qpsi_10_real": Qpsi[1,0].real,
            "Qpsi_10_imag": Qpsi[1,0].imag,
            
            "Z_A": solver.Z[0],
            "Z_B": solver.Z[1],
            
            "rho": solver.rho,
            "rho_min": solver.rho_min.real,
            "rotor_gap": solver.rotor_gap,
            "spinon_gap": solver.spinon_gap
        })

    return pd.DataFrame(rows)

# %% Looper over λ

def run_vs_λ(
    λ_over_t_values,
    base_lattice,
    T, Ns, n,
    Qpsi_guess,
    max_iter,
    tol):
    
    rows = []


    for λ_over_t in λ_over_t_values:
        print(f"Running for λ/t = {λ_over_t}")
        # update lattice interaction
        
        lattice = Lattice(
            lattice_vectors=[base_lattice.a1, base_lattice.a2],
            nn=base_lattice.nn,
            nnn=base_lattice.nnn,
            SOC=λ_over_t,
            hopping=base_lattice.t
        )
        #lattice.plot_BZ(Ns, Ns)
        lattice.calculate_structure_functions(Ns,Ns)
        #lattice.plot_structure_functions(Ns, Ns)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14 
        
        solver = SlaveRotorSolver(
            lattice, T, n,
            Qpsi_guess.copy(),
            max_iter,
            tol,
            U*base_lattice.t
        )

        solver.loop()

        # take last state
        QX = solver.QX
        Qpsi = solver.Qψ

        rows.append({
            "U_over_t": U,
            "λ_over_t": λ_over_t,
            
            # QX
            "QX_00_real": QX[0,0].real,
            "QX_00_imag": QX[0,0].imag,
            "QX_11_real": QX[1,1].real,
            "QX_11_imag": QX[1,1].imag,
            "QX_01_real": QX[0,1].real,
            "QX_01_imag": QX[0,1].imag,
            "QX_10_real": QX[1,0].real,
            "QX_10_imag": QX[1,0].imag,

            # Qpsi
            "Qpsi_00_real": Qpsi[0,0].real,
            "Qpsi_00_imag": Qpsi[0,0].imag,
            "Qpsi_11_real": Qpsi[1,1].real,
            "Qpsi_11_imag": Qpsi[1,1].imag,
            "Qpsi_01_real": Qpsi[0,1].real,
            "Qpsi_01_imag": Qpsi[0,1].imag,
            "Qpsi_10_real": Qpsi[1,0].real,
            "Qpsi_10_imag": Qpsi[1,0].imag,
            
            "Z_A": solver.Z[0],
            "Z_B": solver.Z[1],
            
            "rho": solver.rho,
            "rho_min": solver.rho_min.real
        })

    return pd.DataFrame(rows)


# %% Main Code
# KMH for Kane-Mele-Hubbard
KMH = Lattice(
    lattice_vectors=[a1, a2],
    nn=[u1, u2, u3],
    nnn=[y1, y2, y3],
    SOC=λ,
    hopping=t,
)

Qψ_guess = np.array([ [0.2,   0.4],
                     [ 0.4,  0.2]])

Uvals = np.linspace(2,10,25)
#Uvals = Uvals[1:]
#Uvals = np.array([10])
df = run_vs_U(
    Uvals,
    KMH,
    T, Ns, n,
    Qψ_guess,
    max_iter=100,
    tol=1e-7
)
#λvals = np.array([0,1])
# =============================================================================
# λvals = np.linspace(-1,1,4)
# df  = run_vs_λ(
#     λvals,
#     KMH,
#     T, Ns, n,
#     Qψ_guess,
#     max_iter=50,
#     tol=1e-7
# )
# =============================================================================
# %% Main Plot

x_variable = "U"
df_string = x_variable+"_over_t"

plt.figure(figsize=(6,5))
plt.plot(df[df_string], df["Z_A"], label=r"$Z$", color='k')
plt.plot(df[df_string], df["Qpsi_01_real"], label=r"$Q_\psi^{t}$", 
         color = 'r', ls = '--')
plt.plot(df[df_string], df["Qpsi_00_real"], label=r"$Q_\psi^{s}$", 
         color = 'c', ls = '--')
plt.plot(df[df_string], df["QX_01_real"], label=r"$Q_X^{t}$", color = 'g')
plt.plot(df[df_string], df["QX_00_real"], label=r"$Q_X^{s}$", color = 'b')
#plt.plot(df[df_string], df["rho"], label=r"$\rho$", color = 'b')
plt.xlim(np.array(df[df_string])[0], np.array(df[df_string])[-1])
plt.xlabel(x_variable+"/t", fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title(r"Bond Variables and Condensate Fraction, $T/t=0.01$", fontsize=16)
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
#plt.legend(loc = 'upper right')
plt.grid(True)
plt.tight_layout()
#plt.savefig("highT0_1.png",dpi=1000)
plt.show()

# =============================================================================
# plt.figure(figsize=(7,5))
# plt.ylabel(r'$\rho$')
# plt.xlabel(x_variable+"/t")
# plt.title("Physical minimum and converged values of ρ")
# plt.plot(df[df_string], df["rho"], label=r"$\rho$", color = 'b')
# plt.plot(df[df_string], df["rho_min"], label=r"$\rho_{min}$", color = 'r') 
# plt.savefig("highT0_1_rho.png",dpi=400)
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
# plt.grid(True)
# plt.legend()
# plt.show()
# =============================================================================
plt.figure(figsize=(7,5))
offset = np.array(df["rotor_gap"])[0]
plt.plot(df[df_string], df["rotor_gap"]-offset, label=r"$\Delta X$", color = 'y')
plt.plot(df[df_string], df["spinon_gap"], label=r"$\Delta \psi$", color = 'm')
plt.xlim(np.array(df[df_string])[0], np.array(df[df_string])[-1])
plt.xlabel(x_variable+"/t", fontsize=14)
plt.ylabel("Energy/t", fontsize=14)
plt.legend(loc='upper left')
#plt.title(r"$T/t=0.01$", fontsize=16)
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.savefig("quasiparticle_disp",dpi=600)
#%% Plotting
# =============================================================================
# 
# #import csv file
# 
# def parse_array(s):
#     """Parse repr(...) arrays stored in CSV."""
#     return np.array(ast.literal_eval(s), dtype = np.complex128)
# 
# df = pd.read_csv("solver_log.csv")
# # NOTE eventually rho should be changed back into being a shape (2) array
# 
# df["rho"]       = pd.to_numeric(df["rho"])
# df["rho_min"]       = pd.to_numeric(df["rho_min"])
# df["Z"]         = df["Z"].apply(parse_array)
# df["Qpsi"]      = df["Qpsi"].apply(parse_array)
# df["QX"]        = df["QX"].apply(parse_array)
# df["GXsum"]     = df["GXsum"].apply(parse_array)
# 
# # Unpack
# #df["rho_A"] = df["rho"].apply(lambda x: x[0])
# #df["rho_B"] = df["rho"].apply(lambda x: x[1])
# df["Z_A"] = df["Z"].apply(lambda x: x[0])
# df["Z_B"] = df["Z"].apply(lambda x: x[1])
# for name in ["QX", "Qpsi"]:
#     df[f"{name}_00_real"] = df[name].apply(lambda M: M[0,0].real)
#     df[f"{name}_00_imag"] = df[name].apply(lambda M: M[0,0].imag)
# 
#     df[f"{name}_01_real"] = df[name].apply(lambda M: M[0,1].real)
#     df[f"{name}_01_imag"] = df[name].apply(lambda M: M[0,1].imag)
# 
#     df[f"{name}_10_real"] = df[name].apply(lambda M: M[1,0].real)
#     df[f"{name}_10_imag"] = df[name].apply(lambda M: M[1,0].imag)
# 
#     df[f"{name}_11_real"] = df[name].apply(lambda M: M[1,1].real)
#     df[f"{name}_11_imag"] = df[name].apply(lambda M: M[1,1].imag)
#     
# df["GXsum_det"] = df["GXsum"].apply(lambda M: np.linalg.det(M))
# # way too big? right variable being written?
#     
# plt.figure(figsize=(7,5))
# #plt.plot(df["iteration"], df["rho_A"], label=r"$\rho_A$")
# #plt.plot(df["iteration"], df["rho_B"], label=r"$\rho_B$")
# plt.plot(df["iteration"], df["rho"], label=r"$\rho$")
# plt.plot(df["iteration"], df["Z_A"], label=r"$Z_A$")
# plt.plot(df["iteration"], df["Z_B"], label=r"$Z_B$")
# plt.xlabel("Iteration")
# plt.ylabel(r"$\rho$")
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.close()
# 
# #%% QX plots
# 
# plt.figure(figsize=(7,5))
# plt.title("QX components, Real part")
# plt.plot(df["iteration"], df["QX_00_real"], label=r"$Q_X^{AA}$")
# plt.plot(df["iteration"], df["QX_11_real"], label=r"$Q_X^{BB}$")
# plt.plot(df["iteration"], df["QX_01_real"], label=r"$Q_X^{AB}$")
# plt.plot(df["iteration"], df["QX_10_real"], label=r"$Q_X^{BA}$")
# plt.xlabel("Iteration")
# plt.ylabel(r"$\mathrm{Re}\,Q_X$")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# 
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
# 
# # --- Diagonal: real ---
# axes[0, 0].set_title("QX Diagonal components (Real part)")
# axes[0, 0].plot(df["iteration"], df["QX_00_real"], label=r"$Q_X^{AA}$")
# axes[0, 0].plot(df["iteration"], df["QX_11_real"], label=r"$Q_X^{BB}$")
# axes[0, 0].set_ylabel(r"$\mathrm{Re}\,Q_X^{ss}$")
# axes[0, 0].legend()
# axes[0, 0].grid(True)
# 
# # --- Diagonal: imaginary ---
# axes[0, 1].set_title("QX Diagonal components (Imaginary part)")
# axes[0, 1].plot(df["iteration"], df["QX_00_imag"], label=r"$\Im Q_X^{AA}$")
# axes[0, 1].plot(df["iteration"], df["QX_11_imag"], label=r"$\Im Q_X^{BB}$")
# axes[0, 1].set_ylabel(r"$\mathrm{Im}\,Q_X^{ss}$")
# axes[0, 1].legend()
# axes[0, 1].grid(True)
# 
# # --- Off-diagonal: real ---
# axes[1, 0].set_title("QX Off-diagonal components (Real part)")
# axes[1, 0].plot(df["iteration"], df["QX_01_real"], label=r"$Q_X^{AB}$")
# axes[1, 0].plot(df["iteration"], df["QX_10_real"], label=r"$Q_X^{BA}$")
# axes[1, 0].set_xlabel("Iteration")
# axes[1, 0].set_ylabel(r"$\mathrm{Re}\,Q_X^{s\neq s'}$")
# axes[1, 0].legend()
# axes[1, 0].grid(True)
# 
# # --- Off-diagonal: imaginary ---
# axes[1, 1].set_title("QX Off-diagonal components (Imaginary part)")
# axes[1, 1].plot(df["iteration"], df["QX_01_imag"], label=r"$\Im Q_X^{AB}$")
# axes[1, 1].plot(df["iteration"], df["QX_10_imag"], label=r"$\Im Q_X^{BA}$")
# axes[1, 1].set_xlabel("Iteration")
# axes[1, 1].set_ylabel(r"$\mathrm{Im}\,Q_X^{s\neq s'}$")
# axes[1, 1].legend()
# axes[1, 1].grid(True)
# 
# plt.show()
# #%% Qψ plots
# fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
# 
# # --- Diagonal: real ---
# axes2[0, 0].set_title(r"$Q_\psi$ Diagonal components (Real part)")
# axes2[0, 0].plot(df["iteration"], df["Qpsi_00_real"], label=r"$Q_\psi^{AA}$")
# axes2[0, 0].plot(df["iteration"], df["Qpsi_11_real"], label=r"$Q_\psi^{BB}$")
# axes2[0, 0].set_ylabel(r"$\mathrm{Re}\,Q_\psi^{ss}$")
# axes2[0, 0].legend()
# axes2[0, 0].grid(True)
# 
# # --- Diagonal: imaginary ---
# axes2[0, 1].set_title(r"$Q_\psi$ Diagonal components (Imaginary part)")
# axes2[0, 1].plot(df["iteration"], df["Qpsi_00_imag"], label=r"$\Im Q_\psi^{AA}$")
# axes2[0, 1].plot(df["iteration"], df["Qpsi_11_imag"], label=r"$\Im Q_\psi^{BB}$")
# axes2[0, 1].set_ylabel(r"$\mathrm{Im}\,Q_\psi^{ss}$")
# axes2[0, 1].legend()
# axes2[0, 1].grid(True)
# 
# # --- Off-diagonal: real ---
# axes2[1, 0].set_title(r"$Q_\psi$ Off-diagonal components (Real part)")
# axes2[1, 0].plot(df["iteration"], df["Qpsi_01_real"], label=r"$Q_\psi^{AB}$")
# axes2[1, 0].plot(df["iteration"], df["Qpsi_10_real"], label=r"$Q_\psi^{BA}$")
# axes2[1, 0].set_xlabel("Iteration")
# axes2[1, 0].set_ylabel(r"$\mathrm{Re}\,Q_\psi^{s\neq s'}$")
# axes2[1, 0].legend()
# axes2[1, 0].grid(True)
# 
# # --- Off-diagonal: imaginary ---
# axes2[1, 1].set_title(r"$Q_\psi$ Off-diagonal components (Imaginary part)")
# axes2[1, 1].plot(df["iteration"], df["Qpsi_01_imag"], label=r"$\Im Q_\psi^{AB}$")
# axes2[1, 1].plot(df["iteration"], df["Qpsi_10_imag"], label=r"$\Im Q_\psi^{BA}$")
# axes2[1, 1].set_xlabel("Iteration")
# axes2[1, 1].set_ylabel(r"$\mathrm{Im}\,Q_\psi^{s\neq s'}$")
# axes2[1, 1].legend()
# axes2[1, 1].grid(True)
# 
# plt.show()
# 
# 
# plt.figure(figsize=(7,5))
# plt.plot(df["iteration"], df["Qpsi_00_real"], label=r"$Q_\psi^{AA}$")
# plt.plot(df["iteration"], df["Qpsi_11_real"], label=r"$Q_\psi^{BB}$")
# plt.plot(df["iteration"], df["Qpsi_01_real"], label=r"$Q_\psi^{AB}$")
# plt.plot(df["iteration"], df["Qpsi_10_real"], label=r"$Q_\psi^{BA}$")
# plt.xlabel("Iteration")
# plt.ylabel(r"$\mathrm{Re}\,Q_\psi$")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# 
# # Recent iteration plot
# 
# plt.figure(figsize=(7,5))
# plt.plot(df["iteration"], df["Z_A"], label=r"$Z$", color='k')
# plt.plot(df["iteration"], df["Qpsi_01_real"], label=r"$Q_\psi^{t}$", 
#          color = 'r', ls = '--')
# plt.plot(df["iteration"], df["QX_01_real"], label=r"$Q_X^{t}$", color = 'g')
# plt.xlabel("Iteration")
# plt.ylabel(r"$\mathrm{Re}\,Q_psi$")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# 
# =============================================================================



# =============================================================================
#             if not hasattr(self, 'plotted_integrand'):
#                 # Pick a frequency near the bands and a point midway along the path
#                 w_idx = Nω // 2  
#                 k_idx = N_path // 2 
#                 
#                 # Extract the un-summed 2D array for the top-left matrix element
#                 integrand = (A_G * pref)[w_idx, k_idx, :]
#                 N_side = int(np.sqrt(N_BZ))
#                 
#                 plt.figure(figsize=(6, 5))
#                 plt.contourf(self.k_grid[:,0].reshape(N_side, N_side), 
#                              self.k_grid[:,1].reshape(N_side, N_side), 
#                              np.real(integrand).reshape(N_side, N_side), 
#                              levels=50, cmap='RdBu_r')
#                 plt.colorbar(label="Re(Integrand)")
#                 plt.title(f"Integrand in q-space (Phase coherence check)")
#                 plt.show()
#                 self.plotted_integrand = True
# =============================================================================
