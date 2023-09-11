#!/usr/bin/env python3
import numpy as np
try:
    from opt_einsum import contract as use_einsum
except:
    use_einsum = np.einsum

# Set below to 0 to get all cases.
# If calculation is slow, set to 10 or 5.
SKIP = 0
    
def main():
    data = read_data()

    consistency_check(data)
    
    compute_orb_mag(data, ignore_imag = True)
    compute_orb_mag(data, ignore_imag = False)

def consistency_check(data):
    print("\n \n")
    print("="*75)
    print("  Consistency checks")
    for i in range(len(data) - SKIP):
        x = data[i]["x"]
        y = data[i]["y"]
        H = data[i]["H"]
        nx = data[i]["Nx"]
        ny = data[i]["Ny"]
        
        dist_from_i_to_j = pair_distances(x, y)

        dist_to_edge_ij = np.min(np.array([x - np.min(x), y - np.min(y),
                                           np.max(x) - x, np.max(y) - y]), axis = 0)
        dist_to_edge_ij = 0.5*(dist_to_edge_ij[:, None] + dist_to_edge_ij[None, :])

        print("   Nx =", str(nx).rjust(4), "   Ny =", str(ny).rjust(4))
        
        num_neigh_1 = np.sum(np.logical_and(dist_from_i_to_j[:, :] > 0.9999, dist_from_i_to_j[:, :] < 1.0001), axis = 0)

        if np.sum(num_neigh_1 == 2) != 4:
            print("There should be exactly four corners! Stopping.")
            exit()
        if np.sum(num_neigh_1 == 3) != 2*(nx - 2) + 2*(ny - 2):
            print("Incorrect number of orbitals on the edges. Stopping.")
            exit()
        if np.sum(num_neigh_1 == 4) != (nx - 2)*(ny - 2):
            print("Incorrect number of orbitals on the interior. Stopping.")
            exit()
        if np.max(num_neigh_1) > 4 or np.min(num_neigh_1) < 2:
            print("Wrong number of neighbors. Stopping.")
            exit()
        
        filt = (np.abs(np.imag(H)) > 1.0E-8)
        value = np.max(dist_to_edge_ij[filt])
        print("     H_ij with non-zero imag part is at most", value, " away from the edge.")

        value = np.max(np.abs(np.real(H)))
        print("     Max real part of H_ij", value)
        
        value = np.max(np.abs(np.imag(H)))
        print("     Max imag part of H_ij", value)

        filt = (np.abs(H) > 1.0E-8)
        value = np.max(dist_from_i_to_j[filt])
        print("     The furthest hopping element is ", value)

    print("="*75)

def pair_distances(x, y):
    return np.sqrt(np.power(x[:, None] - x[None, :], 2.0) + np.power(y[:, None] - y[None, :], 2.0))
    
def compute_orb_mag(data, ignore_imag):
    print("\n \n")
    print("="*75)
    print("  Computing orbital magnetization")
    if ignore_imag is True:
        print("  WITHOUT imaginary part of H")
    for i in range(len(data) - SKIP):
        x = data[i]["x"]
        y = data[i]["y"]
        H = np.copy(data[i]["H"])
        if ignore_imag is True:
            H = np.real(H)
        nx = data[i]["Nx"]
        ny = data[i]["Ny"]
        num_sites = nx*ny
        
        # diagonalize hamiltonian
        check_hermitian(H)
        e_n, psi_ni = np.linalg.eigh(H)
        psi_ni = psi_ni.T
        srt=np.argsort(e_n)
        e_n = e_n[srt]
        psi_ni = psi_ni[srt,:]

        # select occupied states
        occ = (e_n < -2.55)

        # compute L operator
        xy = x[:,None]*y[None,:]
        loper = (1.0j)*(xy*H - xy.T*H)
        check_hermitian(loper)

        # get orbital magnetization
        orb_mag = 0.5*np.real(use_einsum("bi, ij, bj ->",psi_ni.conjugate()[occ, :], loper, psi_ni[occ, :]))
        
        print("   Nx =", str(nx).rjust(4), "   Ny =", str(ny).rjust(4))
        print("      orbital mag  = " + "{:.8F}".format(orb_mag), " mu_B   occ/N^2 = " + "{:.8F}".format(np.sum(occ)/float(nx*ny)))
        
    print("="*75)

def check_hermitian(mat):
    if np.max(np.abs(mat - np.conjugate(mat).T)) > 1.0E-9:
        print("Matrix not hermitian!  Stopping.")
        exit()
        
def read_data():
    f = open("data.txt","r")
    ln = f.readlines()
    f.close()

    num_cases = int(ln[0].split()[0])
    print("*"*75)
    print("  Trying to read", num_cases, "hamiltonians and positions.")

    data = []
    
    j = 0
    for i in range(num_cases):
        j = j + 1
        nx = int(ln[j].split()[0])
        ny = int(ln[j].split()[1])
        num_pos = int(ln[j].split()[2])
        num_ham = int(ln[j].split()[3])

        pos_x = []
        pos_y = []
        for k in range(num_pos):
            j = j + 1
            pos_x.append(float(ln[j].split()[0]))
            pos_y.append(float(ln[j].split()[1]))
        pos_x = np.array(pos_x)
        pos_y = np.array(pos_y)

        ham = np.zeros((num_pos, num_pos), dtype = complex)
        for k in range(num_ham):
            j = j + 1
            ind_i = int(ln[j].split()[0])
            ind_j = int(ln[j].split()[1])
            ham[ind_i, ind_j] = float(ln[j].split()[2]) + 1.0j*float(ln[j].split()[3])

        data.append({"x": pos_x, "y": pos_y, "H": ham, "Nx": nx, "Ny": ny})
        print("     Read in case with   Nx =", str(nx).rjust(4), "   Ny =", str(ny).rjust(4))
        
    print("*"*75)
        
    return data
            
main()
