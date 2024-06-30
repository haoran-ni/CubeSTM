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
    3. The x-y lattice vectors are already transformed to a lower triangular matrix. e.g.
    [[x1, 0, 0],
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
    

    def plot_cell(self, 
                  height: float = 0.8,
                  supercell: list = [1,1],
                  neighbourhood: int = 5,
                  xy_decay: float = 1., 
                  z_decay: float = 1.,
                  vmin: float = 1e-3, 
                  vmax: float = 1., 
                  cnorm: str = 'log', 
                  show_atoms: list = None,
                  atoms_size: float = 1,
                  atoms_edge: float = 2,
                  show_cbar: bool = False):
        """This function makes a plot of the simulated STM image based on cell sizes.
           A large neighbourhood with a strong xy_decay is close to a small neighbourhood with a weak xy_decay.
           Note that the show_atoms option in this function does not perform as good as plot_canvas.
           It is recommended to use plot_canvas if atomic positions are plotted as well.

        Args:
            height (float, optional): percentage height in z direction of the cube file. Defaults to 0.8.
            supercell (list, optional): the size of the supercell to plot, e.g. [2,2]. Defaults to [1,1].
            neighbourhood (int, optional): size of the xy neighbourhood of the electron density which could contribute to the tunneling current. Defaults to 5.
            xy_decay (float, optional): strength of decay of electron density contribution in xy direction. Defaults to 1.
            z_decay (float, optional): strength of decay of electron density contribution in z direction. Defaults to 1.
            vmin (float, optional): minimum value to show in the plot. Defaults to 1e-3.
            vmax (float, optional): maximum value to show in the plot. Defaults to 1.
            cnorm (str, optional): normalization scheme for the plot. Could be either 'log' or 'normal'. Defaults to 'log'.
            show_atoms (list, optional): species of atoms to show, e.g. ['C', 'N']. Defaults to None.
            atoms_size (float, optional): size of the atoms. Defaults to 1.
            atoms_edge (float, optional): size of the edges of the atoms. Defaults to 1.
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
        
        # customize this in the future
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
        plt.show()
        
    
    def plot_canvas(self, 
                  height: float = 0.8, 
                  canvas_width: float = 40,
                  canvas_height: float = 35,
                  neighbourhood: int = 5,
                  xy_decay: float = 1.,
                  z_decay: float = 1.,
                  vmin: float = 1e-3,
                  vmax: float = 1.,
                  cnorm: str = 'log', 
                  show_atoms: list = None,
                  inset_size: float = 0.35,
                  atoms_size: float = 1.,
                  atoms_edge: float = 1.):
        """This function makes a canvas plot of the simulated STM image.
           A large neighbourhood with a strong xy_decay is close to a small neighbourhood with a weak xy_decay.

        Args:
            height (float, optional): percentage height in z direction of the cube file. Defaults to 0.8.
            canvas_width (float, optional): size of canvas width in Angstrom. Defaults to 40.
            canvas_height (float, optional): size of canvas height in Angstrom. Defaults to 35.
            neighbourhood (int, optional): size of the xy neighbourhood of the electron density which could contribute to the tunneling current. Defaults to 5.
            xy_decay (float, optional): strength of decay of electron density contribution in xy direction. Defaults to 1.
            z_decay (float, optional): strength of decay of electron density contribution in z direction. Defaults to 1.
            vmin (float, optional): minimum value to show in the plot. Defaults to 1e-3.
            vmax (float, optional): maximum value to show in the plot. Defaults to 1.
            cnorm (str, optional): normalization scheme for the plot. Could be either 'log' or 'normal'. Defaults to 'log'.
            show_atoms (list, optional): species of atoms to show, e.g. ['C', 'N']. Defaults to None.
            inset_size (float, optional): size of the inset to show the atoms. Defaults to 0.35.
            atoms_size (float, optional): size of the atoms. Defaults to 1.
            atoms_edge (float, optional): size of the edges of the atoms. Defaults to 1.
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
            
        # determine supercell and range
        cell_ll = self.origin[:2]
        cell_lr = cell_ll + self.cube_cell[0, :2]
        cell_ul = cell_ll + self.cube_cell[1, :2]
        cell_ur = cell_lr + cell_ul
        j = 1
        while cell_ul[1] - cell_ll[1] < canvas_height:
            j += 1
            cell_ul += self.cube_cell[1, :2]
            cell_ur = cell_lr + cell_ul
        i = 1
        while self._para_rect_width(cell_ll, cell_ul, cell_lr, cell_ur) < canvas_width:
            i += 1
            cell_lr += self.cube_cell[0, :2]
            cell_ur = cell_lr + cell_ul
        supercell = [i, j]
        
        xlim_min = self._get_larger_x(cell_ll, cell_ul)
        ylim_min = cell_ll[1]
            
        x = np.arange(self.num_voxel[0]*supercell[0])
        y = np.arange(self.num_voxel[1]*supercell[1])
        _X, _Y = np.meshgrid(x,y)
        X = self.origin[0] + self.voxel_basis[0][0] * _X + self.voxel_basis[1][0] * _Y
        Y = self.origin[1] + self.voxel_basis[0][1] * _X + self.voxel_basis[1][1] * _Y
        Z = self._tunnel(height, neighbourhood, xy_decay, z_decay)
        Z = np.block(self._make_supercell(Z, supercell))
        
        # customize this in the future
        colormap = colors.LinearSegmentedColormap.from_list("custom", ["navy", "white"])
        
        fig, ax = plt.subplots(dpi=CubeSTM.fig_dpi)
        mesh = ax.pcolormesh(X, Y, Z, 
                             shading='gouraud', 
                             cmap=colormap, 
                             norm=norm,
                             rasterized=True,
                             antialiased=True)
        
        ax.set_xlim(xlim_min, xlim_min + canvas_width)
        ax.set_ylim(ylim_min, ylim_min + canvas_height)
        
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
                  
            # set atoms inset
            axins = ax.inset_axes([1-inset_size, 1-inset_size, inset_size, inset_size])
            
            axins.pcolormesh(X, Y, Z,
                             shading='gouraud',
                             cmap=colormap, 
                             norm=norm,
                             rasterized=True,
                             antialiased=True)
            
            inset_left = sum(ax.get_xlim())*(1-inset_size)
            inset_bottom = sum(ax.get_ylim())*(1-inset_size)
            inset_width = (ax.get_xlim()[1]-ax.get_xlim()[0])*inset_size
            inset_height = (ax.get_ylim()[1]-ax.get_ylim()[0])*inset_size
            
            axins.set_xlim(inset_left, inset_left+inset_width)
            axins.set_ylim(inset_bottom, inset_bottom+inset_height)
            axins.set_xticks([])
            axins.set_yticks([])
            
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
                            atoms = self.atom_positions[indx]
                        else:
                            atoms = np.vstack((atoms, self.atom_positions[indx] + i*lattice_xy[0] + j*lattice_xy[1]))
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
        # remove frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.show()
    
    
    def xyz_map(self, x_id, y_id, z_id):
        # the Cartesian coordinates of the voxel, given indices of the voxel in self.xyz_voxel
        ids = np.array([[x_id], [y_id], [z_id]])
        return self.origin + (self.voxel_basis * ids).sum(axis=0)
    
    
    def _tunnel(self, height: int, neighbourhood: int, xy_decay: float, z_decay: float):
        """This function performs the summation of local electron densities scaled by their distance to the tip.

        Args:
            height (float): z index of the tip in the reference frame of the cube file
            neighbourhood (int): It determines how large the xy region is where the electrons could tunnel to the tip.
            xy_decay (float): The decaying factor of the electron density contribution in xy direction.
                              The larger the xy_decay, the less important the neighbouring electron densities are.
                              A large neighbourhood with a strong xy_decay is close to a small neighbourhood with a weak xy_decay.
            z_decay (float): The decaying factor of the electron density contribution in z direction.
                             The larger the z_decay, the less important the below electron densities are.
                             A very small z_cay indicates a uniform summation of all local electron densities below the tip.
            
        Returns:
            np.ndarray: The integrated electron density projected along z direction.
        """

        Z = np.zeros((self.num_voxel[0], self.num_voxel[1]))
        for i in range(height+1):
            dist_z = (height - i) * np.linalg.norm(self.voxel_basis[2])
            for j in range(-neighbourhood, neighbourhood+1):
                x_shifted = np.roll(self.xyz_voxel[:, :, i], j, axis=0)
                for k in range(-neighbourhood, neighbourhood+1):
                    xy_shifted = np.roll(x_shifted, k, axis=1)
                    dist_xy = np.linalg.norm(k*self.voxel_basis[1] - j*self.voxel_basis[0])
                    Z += np.exp(-z_decay * dist_z) * np.exp(-xy_decay * dist_xy) * xy_shifted
                
        Z = self._normalize_matrix(Z)
        
        # without the transpose, the dimension is wrong, and the view point is from bottom
        return Z.T
    
    
    def _normalize_matrix(self, matrix):
        return matrix / np.max(matrix)
    
    
    def _get_larger_x(self, cell_vertice_1, cell_vertice_2):
        return max(cell_vertice_1[0], cell_vertice_2[0])
    
    
    def _get_smaller_x(self, cell_vertice_1, cell_vertice_2):
        return min(cell_vertice_1[0], cell_vertice_2[0])
    
    
    def _para_rect_width(self, cell_ll, cell_ul, cell_lr, cell_ur):
        return self._get_smaller_x(cell_lr, cell_ur) - self._get_larger_x(cell_ll, cell_ul)
    
    
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