# %% -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 19:37:26 2025
Last updated: 13/05/26 - AP
    
@author: Aarnav Panda

WARNINGS:
-->The point of this code is to adapt it for your purposes and save you time, BUT:
--> YOU WILL NEED TO MODIFY THIS CODE! Your Hamiltonian will not have the same parameters
and functions unless it is the Kane-Mele-Hubbard model.
-->THIS IS THE MOST STREAMLINED VERSION OF THE CODE. ALL THE DIAGNOSTIC GRAPHS AND
TOOLS ARE OMITTED. If you need to make a new graph, you can check the 'testbed.py'
or just try it yourself.
-------------------------------------------------------------------------------
This code is written to evaluate the self-consistent equations from
the Supplementary Information from Wagner et al. (2024) on Edge Zeroes and 
Boundary Spinons. The procedure is outlined below equation (17).

The idea is to guess a value for Q_X, which is defined in terms of the Green's
function for ψ (G_ψ) in equations (13). This means guessing some numerical 
inputs for G_ψ.

The value of the Lagrange multiplier ρ is then determined via its constraint 
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
U = 10*t  # Interaction Strength
T = 0.01*t # Temperature
Ns = 35 # Number of unit cells
N_BZ = Ns**2
n = 1000 # Number of Matsubara frequencies to sum over
mixing = 0.5 # 0 < mixing <= 1, Q_new = Qn + mixing*Qn+1
# Lattice vectors
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
# Other
area = a1[0]*a2[1] - a1[1]*a2[0]
b1 = 2*np.pi * np.array([ a2[1], -a2[0] ]) / area
b2 = 2*np.pi * np.array([ -a1[1], a1[0] ]) / area
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
        If you want to, consider using the lattpy library to do so.
        IMPORTANT: Check the SI of Wagner et al. 2024 to see the definitions! Your lattice may vary.
        lattice_vectors : *ARRAY* of shape (n,m)
                          e.g. here n=1, m=2 i.e. [a1, a2] where a1, a2 are 2D vectors
        nn             : *ARRAY* of NN displacement vectors u_j, precompiled.
        nnn            : *ARRAY* of NNN displacement vectors gamma_j, precompiled.
        SOC            : lambda 
        hopping        : t
        """
        
        self.a1, self.a2 = np.array(lattice_vectors[0]), np.array(lattice_vectors[1])
        self.nn  = np.array([np.array(v) for v in nn])
        self.nnn = np.array([np.array(v) for v in nnn])
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
    
    def eps_k(self, k):
        """
        k - 2D array
        Spin independent structure function on the off-diagonals.
        """
        sum_term = sum(np.cos(np.dot(vec, k)) for vec in self.nnn)
        return 2*self.λ * sum_term
     
    def eps_ksσ(self, k, s, σ):
        """
        k - 2D array
        s - str. should be 'a' or 'b'
        σ - int. Should be 1 or -1 (for up or down)
        """
        k = np.asarray(k)
        if k.ndim == 1:
            k = k[None,:]
        if σ not in (1,-1):
            raise ValueError("σ must be ±1")
        dot_product = np.tensordot(k, self.nnn, axes=([1],[1]))
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
      
    def Vk(self, k):
        k = np.asarray(k)
        if k.ndim == 1:
            k = k[None, :]
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
        dot_product = np.tensordot(k, nnn_array, axes=([1],[1]))
        phase = dot_product + (np.pi*σ/2 if s == 'a' else -np.pi*σ/2)
        # pull down the x or y components of the gamma vectors
        gamma_components = nnn_array[:, idx] 
        deriv_terms = -gamma_components * np.sin(phase)
        result = 2 * self.λ * np.sum(deriv_terms, axis=-1)  
        return result[0] if result.shape[0] == 1 else result

    def lattice_k_mesh(self, N_BZ):
        """
        Generate k-points in the Brillouin zone spanned by reciprocal vectors b1, b2.
        Returns array of shape (N_BZ, 2) of 2D k-vectors.
        """        
        k_list = []
        Ns = int(np.sqrt(N_BZ)) # Number of points to sample along each 'axis'
        delta = 0.5 # offset for sampling points.
        for i in range(Ns):
            x = (i+delta)/Ns
            for j in range(Ns):                
                y = (j+delta)/Ns
                k = x*b1 + y*b2
                k_list.append(k)
        return np.array(k_list) - ((b1+b2)/2)
    
    def plot_BZ(self, N_BZ):
        grid = self.lattice_k_mesh(N_BZ)
        plt.rcParams["font.size"] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.scatter(grid[:,0],grid[:,1], s=2)
        plt.arrow(0,0,b1[0],b1[1], head_width=0.5, color = 'k', length_includes_head=True)
        plt.text(b1[0]-1,b1[1]-0.1, r"$\vec{b_1}$")
        plt.arrow(0,0,b2[0],b2[1], head_width=0.5, color = 'k', length_includes_head=True)
        plt.text(b2[0]+0.3,b2[1]-0.1, r"$\vec{b_2}$")
        locs = [[0,0]] # b1, b2, b1+b2
        for point in locs:
            plt.text(point[0]*1.1,point[1]-0.1, r"$\Gamma$", color='r')
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.title("BZ for Honeycomb")
        plt.xlabel(r"$k_x$")
        plt.ylabel(r"$k_y$")
        plt.show()        
        plt.close()

    def calculate_structure_functions(self, N_BZ):
        """
        Calculates of eps_k(k), eps_ksσ and |V_k(k)| over the BZ mesh used
        by `lattice_k_mesh`.
        """
        Ns = int(np.sqrt(N_BZ))
        self.epskAσ_all = []
        self.epskBσ_all = []
        self.Vk_all = []
        # need to reshape list into 2D to plot.
        k_grid = self.lattice_k_mesh(N_BZ)     # shape → (N_BZ, 2)
        self.k_grid = k_grid
        kx = k_grid[:,0].reshape(Ns, Ns)
        ky = k_grid[:,1].reshape(Ns, Ns)
        # Evaluate variables
        epsk = np.zeros((Ns, Ns))
        Vk = np.zeros((Ns, Ns), dtype=np.complex128)
        epskAσ = np.zeros((2,Ns, Ns))
        epskBσ = np.zeros((2,Ns, Ns))
        for i in range(Ns):
            for j in range(Ns):
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
        self.epsk_all = epsk.reshape(N_BZ)
        self.Vk_all = Vk.reshape(N_BZ)
        # For σ-dependent eps_kAσ and eps_kBσ keep the σ dim.
        self.epskAσ_all = epskAσ.reshape(2, N_BZ)
        self.epskBσ_all = epskBσ.reshape(2, N_BZ)
        
    def plot_structure_functions(self, N_BZ):
        """
        Plots the Vk, ε_k and ε_ksσ functions
        """
        epsk = self.epsk_plot
        Vk = self.Vk_plot
        epskAσ = self.epskAσ_plot
        epskBσ = self.epskBσ_plot
        Vk_abs = np.abs(Vk)
        Vk_arg = np.angle(Vk)
        k_grid = self.k_grid 
        kx = k_grid[:,0].reshape(Ns, Ns)
        ky = k_grid[:,1].reshape(Ns, Ns)
        # Contour plots 
        fig, ax = plt.subplots(1, 2, figsize=(7,5))
        plt.gca().set_aspect('equal')
        # ϵ_k
        try:
            norm_eps = TwoSlopeNorm(vmin=np.min(epsk), vcenter=0.0, vmax=np.max(epsk))
        except:
            norm_eps=None
        ctr1 = ax[0].contourf(kx, ky, epsk, levels=40, cmap="RdBu_r", norm=norm_eps)
        ax[0].set_title(r"$\epsilon_k$")
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
        # ϵ_ksσ        
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
        K = (b1 + b2)/3
        Kp = -K
        K2 = K@R((2*np.pi)/3)
        Kp2 = K@R((2*np.pi)/6)
        M = np.array([K2[0],0])
        points = [K, Gamma, Kp, K2, M, Gamma]
        labels = [r'K', r'$\Gamma$', r"K'", 'K', 'M', r'$\Gamma$']
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
        self.rotor_energies_mesh = energies_mesh
        energies_flat = np.reshape(energies_mesh, 2*N_BZ)
        self.rho_min = -1.01*min(energies_flat)
        return self.rho_min
    
    def plot_rotor_dispersion_hsp(self):
        # Generate the 1D path
        path_k, ticks, labels = self.get_hsp_path(num_pts=500) # Use helper from above
        
        # Calculate structure functions along THIS path
        path_eps = np.array([self.lat.eps_k(k) for k in path_k])
        path_Vk = np.array([self.lat.Vk(k) for k in path_k])
        # Build H along the path using the CURRENT mean-field QX values
        H_path = np.zeros((len(path_k), 2, 2), dtype=np.complex128)
        H_path[:,0,0] = self.QX[0,0] * path_eps
        H_path[:,1,1] = self.QX[1,1] * path_eps
        H_path[:,0,1] = -self.lat.t * self.QX[0,1] * path_Vk
        H_path[:,1,0] = -self.lat.t * self.QX[1,0] * path_Vk.conjugate()
        
        path_energies = np.linalg.eigvalsh(H_path)

        # Plotting 
        plt.figure(figsize=(8, 5))
        plt.plot(path_energies[:, 0], label=r'$\epsilon_{k-}$', color='b')
        plt.plot(path_energies[:, 1], label=r'$\epsilon_{k+}$', color='r')
        
        # Formatting the path
        for t in ticks:
            plt.axvline(t, color='k', linestyle='--', alpha=0.3)
        plt.xticks(ticks, labels)
        
        plt.ylabel('Energy')
        plt.title(f'U/t = {self.U}, Rotor Dispersion')
        plt.ylim(-2.2,1.5)
        plt.legend()
        plt.show()
        plt.close()

    def plot_rotor_dispersion_BZ(self):
        # Plot BZ for eigenvalues
        kx = self.k_grid[:, 0].reshape(Ns, Ns)
        ky = self.k_grid[:, 1].reshape(Ns, Ns)
        band1 = self.rotor_energies_mesh[:, 0].reshape(Ns, Ns)
        band2 = self.rotor_energies_mesh[:, 1].reshape(Ns, Ns)
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        # Lower Band
        c1 = ax[0].contourf(kx, ky, band1, levels=50, cmap='viridis')
        ax[0].set_title(r"Lower Rotor Band $E_1(k)$")
        ax[0].set_aspect('equal')
        ax[0].set_ylim(bottom=0)
        fig.colorbar(c1, ax=ax[0])
        # Upper Band
        c2 = ax[1].contourf(kx, ky, band2, levels=50, cmap='magma')
        ax[1].set_title(r"Upper Rotor Band $E_2(k)$")
        ax[1].set_aspect('equal')
        ax[1].set_ylim(bottom=0)
        fig.colorbar(c2, ax=ax[1])
        
        for a in ax:
            a.set_xlabel(r"$k_x$")
            a.set_ylabel(r"$k_y$")
            K = (b1 + b2) / 3.0
            a.scatter([K[0], -K[0]], [K[1], -K[1]], color='red', s=10, label="K / K\'")
        
        plt.tight_layout()
        plt.show()
        plt.close()
        # 2. Setup 3D Plot
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111, projection='3d')
        # Define common color scale
        vmax = np.max(band2)
        vmin = np.min(band1)
        # 3. Plot Surfaces
        # Lower Band 
        surf1 = ax.plot_surface(kx, ky, band1, cmap='viridis', 
                                antialiased=True, alpha=0.8, vmin=vmin, vmax=vmax)
        # Upper Band 
        surf2 = ax.plot_surface(kx, ky, band2, cmap='viridis', 
                                antialiased=True, alpha=0.7, vmin=vmin, vmax=vmax)
        z_zero = np.zeros_like(kx)
        ax.plot_surface(kx, ky, z_zero, color='gray', alpha=0.2, shade=False, zorder=0)
        # 4. Ground the Energy Axis at Zero
        ax.set_zlim(vmin, vmax) 
        # 5. Labeling
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_zlabel(r'Energy $E(k)$')
        ax.set_title('3D Rotor Band Structure')
        # 4. Colorbar
        # This creates a mapping that spans the actual data range [vmin, vmax]
        mappable = cm.ScalarMappable(cmap='viridis') # Choose the primary colormap
        mappable.set_array([vmin, vmax])
        mappable.set_clim(vmin, vmax)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'Energy')
        plt.title('Rotor Dispersion')
        plt.show()
        return 0
   
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
            This calculates the constraint expression
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
        print('rho_min = ' , rho_min_theoretical)
        if self.iteration <= 2:
            rho_min_phys =  1
        else: 
            rho_min_phys = rho_min_theoretical
        
        rho_max_phys =  5.0 + self.iteration/4  # 
        rho_min = max(self.rho - 30/self.iteration**0.5, rho_min_phys)
        rho_max = min(self.rho + 30/self.iteration**0.5, rho_max_phys)
        rho_sample_vals = np.linspace(rho_min, rho_max, 50)
        C_vals = [C(m) for m in rho_sample_vals]      
        newZ = np.empty(2)

        for s in(0,1):
            
            diag_vals = np.array([C_val[s,s].real for C_val in C_vals])

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
                
            # ---------- Condensate: pick rho_s that minimises C_s-1 ----------
            # minimise |C-1| or minimise C-1 ? 
            # The former finds the smallest value of Z, the latter the largest.
            else:
                idx = int(np.argmin(1-diag_vals))
                best_rho = rho_sample_vals[idx]
                C_best = diag_vals[idx]
                if C_best > 1.1: # allow for a bit of wiggle room for overshoot  
                    print(C_vals)    
                    raise ValueError("C_s is bigger than 1")
            
                new_rho = best_rho
                newZ[s] = 1-C_best
        print(f"minimisation: rho = {self.rho}, Z = {self.Z}")
        return new_rho, newZ
    
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
            "rho_min": self.rho_min.tolist()
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
            
            print(f"\nIteration #{self.iteration}, U/t = {self.U}")
            #print("-----------------------------------")
            #print("previous rho", fmt_array(self.rho), "previous Z", fmt_array(self.Z))
            GX_of_k = self.compute_GX_analytic()
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
#             if self.iteration == (self.max_iter - 5):
#                 with timer("plot rotor dispersion", timings):
#                     self.plot_rotor_dispersion_hsp()
#                     self.plot_rotor_dispersion_BZ()
# =============================================================================
# =============================================================================
#             print(f"\nIteration {self.iteration} timing:")
#             for k, v in timings.items():
#                 print(f"  {k:20s} {v:9.4f} s")
#             print("-----------------------------------")
#             print("\n\n")
# =============================================================================
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
                self.plot_spectral_function(σ=1, omega_range=[-8, 8])
                return
        print(f"Reached max_iter = {self.max_iter} without converging.")
        self.plot_spectral_function(σ=1, omega_range=[-8, 8])
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
        T = self.T
        rho = self.rho
    
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
    
        Mψ00_kmq = self.Mψ00_kmq
        Mψ01_kmq = self.Mψ01_kmq
        Mψ10_kmq = self.Mψ10_kmq
        Mψ11_kmq = self.Mψ11_kmq
    
        energy_b1 = self.bos_poles[0,:][None,None,:]
        energy_b2 = self.bos_poles[1,:][None,None,:]
        energy_b3 = self.bos_poles[2,:][None,None,:]
        energy_b4 = self.bos_poles[3,:][None,None,:]
    
        aX = self.MX00
        bX = self.MX01
        cX = self.MX10
        dX = self.MX11
    
        Atildeψ = ω - Mψ11_kmq
        Dtildeψ = ω - Mψ00_kmq
    
        bψ = Mψ01_kmq
        cψ = Mψ10_kmq
    
        AtildeX = rho + dX
        DtildeX = rho + aX

        # Elementwise multiplication        
        Aprime = Atildeψ * AtildeX
        Bprime = -bψ * bX
        Cprime = -cψ * cX
        Dprime = Dtildeψ * DtildeX
        
        term2f = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
        term2b = np.zeros((Nω, N_path, 2,2), dtype=np.complex128)
    
        # ---------------------------------------------------------
        # Fermionic poles
        # ---------------------------------------------------------
    
        f_poles = [energy_f1, energy_f2]
        b_poles = [energy_b1,energy_b2,energy_b3,energy_b4]
        
        for idx, ef in enumerate(f_poles):
    
            other_ef = f_poles[1-idx]
            w_p = ω - ef
    
            denom = -1*(w_p-energy_b1)*(w_p-energy_b2)*(w_p-energy_b3)*(w_p-energy_b4)*(ef-other_ef)
    
            w = w_p
            w2 = w*w
            w3 = w2*w
            A_G = -w*AtildeX -w2*Atildeψ/U + w3/U + Aprime
            #B_G = w*bX - w2*bψ/U + Bprime
            B_G = Bprime*(np.ones_like(w))
            #C_G = w*cX - w2*cψ/U + Cprime
            C_G = Cprime*np.ones_like(w)
            D_G = -w*DtildeX  -w2*Dtildeψ/U + w3/U + Dprime
            
            pref = -self.nF(-ef,T)/denom
                
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
            denom_X = 1.0
            for jdx, other_eb in enumerate(b_poles):
                if idx == jdx:
                    continue
                denom_X *= (eb - other_eb)
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

            pref = self.nB(w_p,T)/denom
            term2b[:,:,0,0] += np.sum(A_G*pref,axis=2)
            term2b[:,:,0,1] += np.sum(B_G*pref,axis=2)
            term2b[:,:,1,0] += np.sum(C_G*pref,axis=2)
            term2b[:,:,1,1] += np.sum(D_G*pref,axis=2)

        term2 = (-U**2)*(term2f+term2b)/N_BZ
    
        return (-U**2)*(term2f)/N_BZ, (-U**2)*(term2b)/N_BZ, term1
    
    def precompute_single_particle_GF(self, k_path, σ):
        """
        Precomputes the things which are not omega dependent.
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
        
        # Need to recompute Gψ for (k-q) instead of (k)        
        q_grid = np.array(self.k_grid)
        k_minus_q = k[:,None,:] - q_grid[None,:,:] # (N_path,N_BZ,2) object
        
        kq_flat = k_minus_q.reshape(-1,2)
        epskaσ_kmq = self.lat.eps_ksσ(kq_flat,'a',σ)
        epskbσ_kmq = self.lat.eps_ksσ(kq_flat,'b',σ)
        Vk_kmq  = self.lat.Vk(kq_flat)
        VkC_kmq = np.conj(Vk_kmq)
        
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
        
        # Can compute bosonic pole structures since they are independent of k,ω
        eb = self.bos_poles
        denom_X = np.ones_like(eb)
        for i in range(4):
            for j in range(4):
                if i == j: continue
                denom_X[i, :] *= (eb[i, :] - eb[j, :])
        # This denom_X array contains all possible Π_j (eb_i-eb_j), i≠j
        self.eb_denom_X = denom_X # (4, N_BZ)
        self.nB_eb = self.nB(eb, self.T)
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
        omegas = np.linspace(omega_range[0], omega_range[-1],500)
        intensity = np.zeros((len(omegas),N_path))
        # 2b Calculate omega independent variables to avoid repetition:
        print("Precomputing variables...")
        self.precompute_single_particle_GF(k_path, σ)
        # 3. Calculate |det(G(k, w + i*eta))|
        print("Starting detG plot...")
        iterable=0
        # the aim now is to broadcast this over omega 
        batch = 40
        for start in range(0,len(omegas),batch):
            end = min(start+batch,len(omegas))
            ω_batch = omegas[start:end] + 1j*eta
            term2f, term2b, term1 = self.single_particle_GF(σ, k_path, ω_batch)
            intensity[start:end] = np.abs(np.linalg.det(term2f+term2b+term1))
            iterable+=1
            print(f"{iterable} batches complete. ({min(batch*iterable,len(omegas))}/{len(omegas)})")
       # 4. Plotting
        from matplotlib.colors import LogNorm
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(intensity, extent=[0, len(k_path), omegas[0], omegas[-1]], 
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
        cbar.set_label(r"$|\det(G(\mathbf{k},ω+i0^+))|$"+f", U/t={self.U}")
        legend_handle = Line2D([0], [0], color='black', linewidth=1.0)
        ax.plot(np.arange(N_path), self.Gψ_path_energies[:,0]*10, color='black')
        ax.plot(np.arange(N_path), self.Gψ_path_energies[:,1]*10, color='black')
        ax.legend(handles=[legend_handle],labels=['spinon dispersion, multiplied by 10'],
            loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=1)
        plt.tight_layout()
        plt.show()
        return 0 
    
   
        
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
#%% Looper over U
def run_vs_U(
    U_over_t_values,
    base_lattice,
    T, N_BZ, n,
    Qpsi_guess,
    max_iter,
    tol):
    
    rows = []
    for U_over_t in U_over_t_values:
        print(f"Running for U/t = {U_over_t}")

        # update lattice interaction
        lattice = Lattice(
            lattice_vectors=[base_lattice.a1, base_lattice.a2],
            nn=base_lattice.nn,
            nnn=base_lattice.nnn,
            SOC=base_lattice.λ,
            hopping=base_lattice.t
        )
        
        #lattice.plot_BZ(N_BZ)
        lattice.calculate_structure_functions(N_BZ)
        #lattice.plot_structure_functions(N_BZ)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14 
        
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
            "rho_min": solver.rho_min
        })

    return pd.DataFrame(rows)

Uvals = np.linspace(9, 10, 2)  
Uvals = Uvals[1:]
df_U = run_vs_U(Uvals,KMH,T, N_BZ, n,Qψ_guess,max_iter=30,tol=1e-7)

# %% Main Plot

# =============================================================================
# plt.figure(figsize=(7,5))
# plt.plot(df_U["U_over_t"], df_U["Z_A"], label=r"$Z$", color='k')
# plt.plot(df_U["U_over_t"], df_U["Qpsi_01_real"], label=r"$Q_\psi^{t}$", 
#          color = 'r', ls = '--')
# plt.plot(df_U["U_over_t"], df_U["Qpsi_00_real"], label=r"$Q_\psi^{s}$", 
#          color = 'c', ls = '--')
# plt.plot(df_U["U_over_t"], df_U["QX_01_real"], label=r"$Q_X^{t}$", color = 'g')
# plt.plot(df_U["U_over_t"], df_U["QX_00_real"], label=r"$Q_X^{s}$", color = 'b')
# #plt.plot(df_U["U_over_t"], df_U["rho"], label=r"$\rho$", color = 'b')
# plt.xlabel(r"$U/t$")
# plt.ylabel(r"Amplitude")
# plt.title("Bond Variables and Condensate Fraction against Hubbard Potential")
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
# plt.legend(loc = 'upper right')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# 
# plt.figure(figsize=(7,5))
# plt.ylabel(r'$\rho$')
# plt.xlabel(r"$U/t$")
# plt.plot(df_U["U_over_t"], df_U["rho"], label=r"$\rho$", color = 'b')
# plt.show()
# 
# plt.figure(figsize=(7,5))
# plt.ylabel(r'$\rho$')
# plt.xlabel(r"$U/t$")
# plt.plot(df_U["U_over_t"], df_U["rho_min"], label=r"$\rho_{min}$", color = 'r') 
# plt.show()
# =============================================================================

# =============================================================================
# #%% Plotting
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
# plt.xlim(0, Uvals[-1])
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