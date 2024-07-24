import numpy as np
from ase.io import read
from mendeleev import element
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


class CubeSTM:
    """This class is used to read the .cube file containing the partial charge density, and make the simulated STM plot of the .cube file.

    Note: The class only works for:
    1. periodic cube file (non-periodic .cube still generates a plot, but they don't look good)
    2. The x-y lattice vectors of the cube file should be the same as the original atomic structure.
    3. The x-y lattice vectors should be the following form:
    [[x1, x2, 0],
     [y1, y2, 0],
     [0, 0, z3]]
     
    Attributes:
        path (str): the path of the cube file.
        num_atoms (int): number of atoms in a unit cell.
        origin (np.ndarray): cartesian coordinates of the center of the cube file.
        cube_cell (np.ndarray): lattice vectors of the cube file.
        num_voxel (list): number of voxels along the 3 cube_cell vectors.
        voxel_basis (np.ndarray): basis vectors of the cube file. num_voxel * voxel_basis = cube_cell.
        atom_species (list): atomic species of all the atoms in the unit cell.
        atom_charges (list): atomic charges of all the atoms in the unit cell.
        atom_positions (np.ndarray): atomic positions of all the atoms in the unit cell.
        xyz_voxel (np.ndarray): volumetric data of the cube file.
        lattice (np.ndarray): the lattice vectors of the original structure file.
                              The x and y vectors are expected to be the same as cube_cell.
    """
    
    bohr_to_A = 0.52917721 # as in FHI-aims
    fig_dpi = 400
    scale_bar_len = 10
    scale_bar_name = "1 nm"
    
    def __init__(self, cube_path, geo_path=None):
        self.path = cube_path
        self.num_atoms = None
        self.origin = None
        self.cube_cell = None
        self.num_voxel = list() # [num_x, num_y, num_z]
        self.voxel_basis = list() # [[x_1, 0, 0], [y_1, y_2, 0], [0, 0, z_3]]
        self.atom_species = list()
        self.atom_charges = list()
        self.atom_positions = list()
        self.xyz_voxel = list()
        self.lattice = None
        if geo_path is not None:
            self.readGeo(geo_path)
            
        self.readCube(cube_path)
    
    
    def readGeo(self, path):
        self.lattice = read(path).get_cell()
    
    
    def readCube(self, path):
        self.path = path
        
        with open(path, 'r') as f:
            # skip comments
            f.readline()
            f.readline()
            
            origin_line = np.fromstring(f.readline().strip(), dtype=float, sep=' ')
            self.num_atoms = int(origin_line[0])
            self.origin = origin_line[-3:] * CubeSTM.bohr_to_A
            
            # read voxel info
            for i in range(3):
                basis_line = np.fromstring(f.readline().strip(), dtype=float, sep=' ')
                self.num_voxel.append(int(basis_line[0]))
                self.voxel_basis.append(basis_line[-3:])
            self.num_voxel = np.array(self.num_voxel)
            self.voxel_basis = np.array(self.voxel_basis) * CubeSTM.bohr_to_A
            
            # obtain info of the cube cell
            self.cube_cell = self.num_voxel * self.voxel_basis
            self.cube_center = self.cube_cell.sum(axis=0)/2
            
            # read atoms info
            for i in range(self.num_atoms):
                atoms_line = np.fromstring(f.readline().strip(), dtype=float, sep=' ')
                self.atom_species.append(int(atoms_line[0]))
                self.atom_charges.append(atoms_line[1])
                self.atom_positions.append(atoms_line[-3:])
            self.atom_species = np.array(self.atom_species)
            self.atom_charges = np.array(self.atom_charges)
            # Cartesian coordinates of the atoms
            self.atom_positions = np.array(self.atom_positions) * CubeSTM.bohr_to_A
            
            # read voxel info
            while len(self.xyz_voxel) != self.num_voxel[0]:
                yz_voxel = []
                while len(yz_voxel) != self.num_voxel[1]:
                    z_voxel = np.array([])
                    while len(z_voxel) != self.num_voxel[2]:
                        voxel_line = np.fromstring(f.readline().strip(), dtype=float, sep=' ')
                        z_voxel = np.concatenate((z_voxel, voxel_line))
                    yz_voxel.append(z_voxel)
                self.xyz_voxel.append(yz_voxel) 
            self.xyz_voxel = np.array(self.xyz_voxel)

            # sanity check
            for i in range(3):
                if self.xyz_voxel.shape[i] != self.num_voxel[i]:
                    print(f"Dimension {i}: number of voxel does not match! Expected {self.num_voxel[i]}, got {self.xyz_voxel.shape[i]}.")
        
    
    def plot(self, 
             height: float = 0.8, 
             canvas_width: float = 40,
             canvas_height: float = 35,
             rotation: float = 0,
             neighbourhood: float = 0.5,
             xy_decay: float = 1.,
             z_decay: float = 1.,
             vmin: float = 1e-3,
             vmax: float = 1.,
             cnorm: str = 'log', 
             show_atoms: list = None,
             inset_size: float = 0.5,
             atoms_size: float = 1.2,
             atoms_edge: float = 1.2,
             cmap = None):
        """This function makes a canvas plot of the simulated STM image.
           A large neighbourhood with a strong xy_decay is close to a small neighbourhood with a weak xy_decay.

        Args:
            height (float, optional): percentage height in z direction of the cube file. Defaults to 0.8.
            canvas_width (float, optional): size of canvas width in Angstrom. Defaults to 40.
            canvas_height (float, optional): size of canvas height in Angstrom. Defaults to 35.
            rotation (float, optional): degree of rotation. Defaults to 0.
            neighbourhood (float, optional): size of the xy neighbourhood of the electron density which could contribute to the tunneling current. Unit is Angstrom. Defaults to 0.5.
            xy_decay (float, optional): strength of decay of electron density contribution in xy direction. Defaults to 1.
            z_decay (float, optional): strength of decay of electron density contribution in z direction. Defaults to 1.
            vmin (float, optional): minimum value to show in the plot. Defaults to 1e-3.
            vmax (float, optional): maximum value to show in the plot. Defaults to 1.
            cnorm (str, optional): normalization scheme for the plot. Could be either 'log' or 'normal'. Defaults to 'log'.
            show_atoms (list, optional): species of atoms to show, e.g. ['C', 'N']. Defaults to None.
            inset_size (float, optional): size of the inset to show the atoms. Defaults to 0.35.
            atoms_size (float, optional): size of the atoms. Defaults to 1.
            atoms_edge (float, optional): size of the edges of the atoms. Defaults to 1.
            cmap: colormap of the figure. Defaults to navy-white colormap.
        """
        
        if cnorm.lower() == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif cnorm.lower() == 'normal':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            print(f"Normalization method {cnorm} not supported, defaulting to LogNorm...")
            norm = LogNorm(vmin=vmin, vmax=vmax)
        
        assert height > 0. and height < 1., "height must be in range (0,1)"
        height = int(height*self.num_voxel[2])
        
        diagonal = np.sqrt(canvas_height**2+canvas_width**2)
        
        # apply rotation to origin and cube cell
        o = self._apply_rot(rotation, self.origin[:2])
        basis_a = self.cube_cell[0, :2]
        basis_b = self.cube_cell[1, :2]
        basis_a = self._apply_rot(rotation, basis_a)
        basis_b = self._apply_rot(rotation, basis_b)
        
        # build supercell
        i, j = 1, 1
        i_pass, j_pass = False, False
        while True:
            vec_a = basis_a * i
            vec_b = basis_b * j
            end_c = vec_a + vec_b
            
            oa = self._line_from_points(np.array([0,0]), vec_a)
            ob = self._line_from_points(np.array([0,0]), vec_b)
            ac = self._line_from_points(vec_a, end_c)
            bc = self._line_from_points(vec_b, end_c)

            # sanity check
            tol = 1e-8
            assert abs(oa[0] - bc[0]) < tol
            assert abs(oa[1] - bc[1]) < tol
            assert abs(ob[0] - ac[0]) < tol
            assert abs(ob[1] - ac[1]) < tol
            
            if self._parallel_distance(oa[0], oa[1], oa[2], bc[2]) < diagonal:
                j += 1
            else:
                j_pass = True
            if self._parallel_distance(ob[0], ob[1], ob[2], ac[2]) < diagonal:
                i += 1
            else:
                i_pass = True
            if i_pass and j_pass:
                break
        
        supercell = [i, j]
        
        # construct X, Y and Z
        x = np.arange(self.num_voxel[0]*supercell[0])
        y = np.arange(self.num_voxel[1]*supercell[1])
        _X, _Y = np.meshgrid(x,y)
        rotated_basis = self._apply_rot(rotation, self.voxel_basis[:2, :2].T).T
        X = self.origin[0] + rotated_basis[0][0] * _X + rotated_basis[1][0] * _Y
        Y = self.origin[1] + rotated_basis[0][1] * _X + rotated_basis[1][1] * _Y
        Z = self._tunnel(height, neighbourhood, xy_decay, z_decay)
        Z = np.block(self._make_supercell(Z, supercell))
        
        # coordinates of the center of the plot
        x_center = (X[0,0]+X[0,-1]+X[-1,0]+X[-1,-1])/4
        y_center = (Y[0,0]+Y[0,-1]+Y[-1,0]+Y[-1,-1])/4

        # color map
        if cmap:
            colormap = cmap
        else:
            colormap = colors.LinearSegmentedColormap.from_list("custom", ["navy", "white"])
        
        # plot stm image
        fig, ax = plt.subplots(dpi=CubeSTM.fig_dpi)
        mesh = ax.pcolormesh(X, Y, Z, 
                             shading='gouraud', 
                             cmap=colormap, 
                             norm=norm,
                             rasterized=True,
                             antialiased=True)
        ax.set_xlim(x_center - canvas_width/2, x_center + canvas_width/2)
        ax.set_ylim(y_center - canvas_height/2, y_center + canvas_height/2)
        
        # add scale bar
        fontprops = fm.FontProperties(size=16, weight='bold')
        scalebar = AnchoredSizeBar(ax.transData,
                                CubeSTM.scale_bar_len,
                                CubeSTM.scale_bar_name,
                                'lower left',
                                pad=0.5,
                                color='black',
                                frameon=False,
                                size_vertical=0.3,
                                fontproperties=fontprops)
        ax.add_artist(scalebar)
        
        # plot atoms
        if show_atoms:
            # sanity check
            if self.lattice is None:
                print(f"WARNING: Lattice vectors are obtained from cube file. Please make sure they are the same as the original structure.")
                lattice_xy = self.cube_cell[:2]
            else:
                mismatch = self.lattice[:2] - self.cube_cell[:2]
                if np.max(mismatch) > 1e-3:
                    print(f"WARNING: Mismatch between the original lattice and the cube lattice detected. Defaulting to cube lattice.")
                    lattice_xy = self.cube_cell[:2]
                else:
                    lattice_xy = self.lattice[:2]
                  
            # plot atoms inset
            axins = ax.inset_axes([1-inset_size, 1-inset_size, inset_size, inset_size])
            
            axins.pcolormesh(X, Y, Z,
                             shading='gouraud',
                             cmap=colormap, 
                             norm=norm,
                             rasterized=True,
                             antialiased=True)
            
            x_length = ax.get_xlim()[1]-ax.get_xlim()[0]
            y_length = ax.get_ylim()[1]-ax.get_ylim()[0]
            inset_left = ax.get_xlim()[0] + x_length*(1-inset_size)
            inset_bottom = ax.get_ylim()[0] + y_length*(1-inset_size)
            inset_width = x_length*inset_size
            inset_height = y_length*inset_size
            
            axins.set_xlim(inset_left, inset_left+inset_width)
            axins.set_ylim(inset_bottom, inset_bottom+inset_height)
            axins.set_xticks([])
            axins.set_yticks([])
            
            # apply rotation to lattice and atoms
            lattice_xy = self._apply_rot_3d(rotation, lattice_xy)
            rotated_atoms = self._apply_rot_3d(rotation, self.atom_positions)
            
            for name in show_atoms:
                species = element(name)
                id = species.atomic_number
                # plot options
                rad = self._data_units_to_points(species.vdw_radius/1500, ax, fig) * atoms_size
                color = species.jmol_color
                # select species
                indx = np.where(self.atom_species == id)[0]
                # create supercell
                for i in range(-1, supercell[0]+1):
                    for j in range(-1, supercell[1]+1):
                        if i == -1 and j == -1:
                            atoms = rotated_atoms[indx]
                        else:
                            atoms = np.vstack((atoms, rotated_atoms[indx] + i*lattice_xy[0] + j*lattice_xy[1]))
                # plot based on height
                atoms_x = atoms[:, 0]
                atoms_y = atoms[:, 1]
                atoms_z = atoms[:, 2]
                order = np.argsort(atoms_z)
                # plot inset
                axins.scatter(atoms_x[order], atoms_y[order], label=f"{species.name}", color=color, s=rad, edgecolors='black', linewidths=atoms_edge)
                ax.scatter([],[], label=f"{species.name}", color=color, s=rad, edgecolors='black', linewidths=atoms_edge)
            ax.legend(loc='upper left', framealpha=1., prop={'size':16}, edgecolor='black')
                
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.tight_layout()
        plt.show()
    

    def plot_cell(self, 
                  height: float = 0.8,
                  supercell: list = [1,1],
                  neighbourhood: float = 0.5,
                  xy_decay: float = 1., 
                  z_decay: float = 1.,
                  vmin: float = 1e-3, 
                  vmax: float = 1., 
                  cnorm: str = 'log', 
                  show_atoms: list = None,
                  atoms_size: float = 1,
                  atoms_edge: float = 2,
                  cmap = None,
                  show_cbar: bool = False):
        """This function makes a plot of the simulated STM image based on cell sizes.
           A large neighbourhood with a strong xy_decay is close to a small neighbourhood with a weak xy_decay.
           Note that the show_atoms option in this function does not perform as good as plot_canvas.
           It is recommended to use plot_canvas if atomic positions are plotted as well.

        Args:
            height (float, optional): percentage height in z direction of the cube file. Defaults to 0.8.
            supercell (list, optional): the size of the supercell to plot, e.g. [2,2]. Defaults to [1,1].
            neighbourhood (float, optional): size of the xy neighbourhood of the electron density which could contribute to the tunneling current. Unit is Angstrom. Defaults to 0.5.
            xy_decay (float, optional): strength of decay of electron density contribution in xy direction. Defaults to 1.
            z_decay (float, optional): strength of decay of electron density contribution in z direction. Defaults to 1.
            vmin (float, optional): minimum value to show in the plot. Defaults to 1e-3.
            vmax (float, optional): maximum value to show in the plot. Defaults to 1.
            cnorm (str, optional): normalization scheme for the plot. Could be either 'log' or 'normal'. Defaults to 'log'.
            show_atoms (list, optional): species of atoms to show, e.g. ['C', 'N']. Defaults to None.
            atoms_size (float, optional): size of the atoms. Defaults to 1.
            atoms_edge (float, optional): size of the edges of the atoms. Defaults to 1.
            cmap: colormap of the figure. Defaults to navy-white colormap.
            show_cbar (boolean, optional): whether to show the color bar. Defaults to False.
        """
        
        if cnorm.lower() == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif cnorm.lower() == 'normal':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            print(f"Normalization method {cnorm} not supported, defaulting to LogNorm...")
            norm = LogNorm(vmin=vmin, vmax=vmax)
        
        assert height > 0. and height < 1., "height must be in range (0,1)"
        height = int(height*self.num_voxel[2])
        
        x = np.arange(self.num_voxel[0]*supercell[0])
        y = np.arange(self.num_voxel[1]*supercell[1])
        _X, _Y = np.meshgrid(x,y)
        X = self.origin[0] + self.voxel_basis[0][0] * _X + self.voxel_basis[1][0] * _Y
        Y = self.origin[1] + self.voxel_basis[0][1] * _X + self.voxel_basis[1][1] * _Y   
        Z = self._tunnel(height, neighbourhood, xy_decay, z_decay)
        Z = np.block(self._make_supercell(Z, supercell))
        
        if cmap:
            colormap = cmap
        else:
            colormap = colors.LinearSegmentedColormap.from_list("custom", ["navy", "white"])
        
        fig, ax = plt.subplots(dpi=CubeSTM.fig_dpi)
        
        mesh = ax.pcolormesh(X, Y, Z, 
                             shading='gouraud', 
                             cmap=colormap, 
                             norm=norm,
                             rasterized=True,
                             antialiased=True)
        
        # plot atoms - in dev
        if show_atoms:
            # draw unit cell border line and only plot atoms inside the unit cell
            if supercell[0] > 1 or supercell[1] > 1:
                cell_vertices = self._get_cell_vertices(X, Y)
                x_verts, y_verts = zip(*cell_vertices)
                ax.plot(x_verts, y_verts, '--', c='black')
            
            for name in show_atoms:
                species = element(name)
                id = species.atomic_number
                rad = self._data_units_to_points(species.vdw_radius/1500, ax, fig) * atoms_size
                color = species.jmol_color
                indx = np.where(self.atom_species == id)[0]
                atoms = self.atom_positions[indx]
                # plot based on height
                atoms_x = atoms[:, 0]
                atoms_y = atoms[:, 1]
                atoms_z = atoms[:, 2]
                order = np.argsort(atoms_z)
                # plot
                ax.scatter(atoms_x[order], atoms_y[order], label=f"{species.name}", color=color, s=rad, edgecolors='black', linewidths=atoms_edge)
                ax.legend(loc='best', framealpha=1., prop={'size':16}, edgecolor='black')
                
        if show_cbar:
            cbar = fig.colorbar(mesh)
            
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.show()


    def xyz_map(self, x_id, y_id, z_id):
        # the Cartesian coordinates of the voxel, given indices of the voxel in self.xyz_voxel
        ids = np.array([[x_id], [y_id], [z_id]])
        return self.origin + (self.voxel_basis * ids).sum(axis=0)
    
    
    def _tunnel(self, height: int, neighbourhood: float, xy_decay: float, z_decay: float):
        """This function performs the summation of local electron densities scaled by their distance to the tip.

        Args:
            height (float): z index of the tip in the reference frame of the cube file
            neighbourhood (float): It determines how large the xy region is where the electrons could tunnel to the tip. Unit is Angstrom.
            xy_decay (float): The decaying factor of the electron density contribution in xy direction.
                              The larger the xy_decay, the less important the neighbouring electron densities are.
                              A large neighbourhood with a strong xy_decay is close to a small neighbourhood with a weak xy_decay.
            z_decay (float): The decaying factor of the electron density contribution in z direction.
                             The larger the z_decay, the less important the below electron densities are.
                             A very small z_cay indicates a uniform summation of all local electron densities below the tip.
            
        Returns:
            np.ndarray: The integrated electron density projected along z direction.
        """

        j_cut = int(neighbourhood//np.linalg.norm(self.voxel_basis[0]))
        k_cut = int(neighbourhood//np.linalg.norm(self.voxel_basis[1]))
        
        Z = np.zeros((self.num_voxel[0], self.num_voxel[1]))
        for i in range(height+1):
            dist_z = (height - i) * np.linalg.norm(self.voxel_basis[2])
            for j in range(-j_cut, j_cut+1):
                x_shifted = np.roll(self.xyz_voxel[:, :, i], j, axis=0)
                for k in range(-k_cut, k_cut+1):
                    xy_shifted = np.roll(x_shifted, k, axis=1)
                    dist_xy = np.linalg.norm(k*self.voxel_basis[1] + j*self.voxel_basis[0])
                    if dist_xy <= neighbourhood:
                        Z += np.exp(-z_decay * dist_z) * np.exp(-xy_decay * dist_xy) * xy_shifted
                
        Z = self._normalize_matrix(Z)
        
        # without the transpose, the dimension is wrong, and the view point is from bottom
        return Z.T
    
    
    def _normalize_matrix(self, matrix):
        return matrix / np.max(matrix)

    
    def _make_supercell(self, Z, supercell):
        Z_block = []
        _temp = []
        for i in range(supercell[0]):
            _temp.append(Z)
        for j in range(supercell[1]):
            Z_block.append(_temp)
        return Z_block
    
    
    def _get_cell_vertices(self, X, Y):
        # we add the fifth point to form a closed parallelogram
        return [(X[0,0], Y[0,0]),
                (X[0, self.num_voxel[0]-1], Y[0, self.num_voxel[0]-1]),
                (X[self.num_voxel[1]-1, self.num_voxel[0]-1], Y[self.num_voxel[1]-1, self.num_voxel[0]-1]),
                (X[self.num_voxel[1]-1,0], Y[self.num_voxel[1]-1,0]),
                (X[0,0], Y[0,0])]
    
    
    def _data_units_to_points(self, radius, ax, fig):
        # convert radius in data units to points
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        fig_width, fig_height = fig.get_size_inches()
        
        x_pixels_per_unit = fig.dpi * fig_width / x_range
        y_pixels_per_unit = fig.dpi * fig_height / y_range
        
        pixels_per_unit = min(x_pixels_per_unit, y_pixels_per_unit)
        radius_in_points = radius * pixels_per_unit
        
        return np.pi * radius_in_points ** 2
    
    
    def _line_from_points(self, point_1, point_2):
        tol = 1e-8
        x1, y1 = point_1[0], point_1[1]
        x2, y2 = point_2[0], point_2[1]
        if np.abs(x1 - x2) < tol:
            A = 1
            B = 0
            C = -x1
        elif np.abs(y1 - y2) < tol:
            A = 0
            B = 1
            C = -y1
        else:
            A = -(y2-y1)/(x2-x1)
            B = 1
            C = (x1*y2-y1*x2)/(x2-x1)
        return A, B, C
    
    
    def _parallel_distance(self, A, B, C1, C2):
        return np.abs(C1-C2)/np.sqrt(A**2+B**2)


    def _apply_rot(self, degree, vector):
        # apply counter clockwise rotation to a 2d vector
        rot_rad = degree*np.pi/180.0
        rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                            [np.sin(rot_rad), np.cos(rot_rad)]])
        return np.matmul(rot_mat, vector)
    
    
    def _apply_rot_3d(self, degree, atom_positions):
        # apply counter clockwise rotation in x-y plane to atomic positions
        rot_rad = degree*np.pi/180.0
        rot_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0],
                            [np.sin(rot_rad), np.cos(rot_rad), 0],
                            [0, 0, 1]])
        return np.matmul(rot_mat, atom_positions.T).T