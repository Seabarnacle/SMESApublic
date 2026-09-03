import os
from enum import Enum
from itertools import product

import numpy as np
from qcelemental.util import which_import

from psi4 import core

from .. import psifiles as psif


class RemovalPolicy(Enum):
    LargestError = 1
    OldestAdded = 2

class StoragePolicy(Enum):
    InCore = 1
    OnDisk = 2

def axpy(y, alpha, x):
    if isinstance(y, (core.Matrix, core.Vector)):
        y.axpy(alpha, x)
    elif isinstance(y, (core.dpdbuf4, core.dpdfile2)):
        y.axpy_matrix(x, alpha)
    elif which_import("ambit", return_bool=True):
        import ambit
        if isinstance(x, ambit.BlockedTensor):
            y.axpy(alpha, x)
        else:
            raise TypeError("Unrecognized object type for DIIS.")
    else:
        raise TypeError("Unrecognized object type for DIIS.")

def template_helper(*args):
    """
    Store the _kind of object_ we want to I/O and _its dimensions_.
    But in practice, given one piece of information, we can deduce the other.
    If there's one dimension, it must be a Vector.
    If there are two dimensions, it must be written to disk as a Matrix.
    If it's a BlockedTensor, we can just read the dimensions.
    """
    template = []
    for arg in args:
        if isinstance(arg, core.Vector):
            template.append([arg.dimpi()])
        elif isinstance(arg, (core.Matrix, core.dpdfile2, core.dpdbuf4)):
            template.append([arg.rowdim(), arg.coldim()])
        elif isinstance(arg, float):
            template.append(float(0))
        elif which_import("ambit", return_bool=True):
            import ambit
            if isinstance(arg, ambit.BlockedTensor):
                template.append(ambit.BlockedTensor)
            else:
                raise TypeError("Unrecognized object type for DIIS.")
        else:
            raise TypeError("Unrecognized object type for DIIS.")

    return template

class DIIS:

    def __init__(self, max_vecs: int, name: str, removal_policy = RemovalPolicy.LargestError, storage_policy = StoragePolicy.OnDisk, 
                 closed_shell = True, engines = {"diis"}, X=None, S=None):
        # We don't have a good sense for how this class may need to expand, so the current structure is amorphous.
        # Currently supported storage types: ambit.BlockedTensor, Psi.Vector, Psi.Matrix, Psi.dpdfile2, Psi.dpdbuf4, float

        # LargestError is only _defined_ for the case of one engine and not theoretically sound for adiis/ediis:
        # those methods want to traverse a wide range of solution space. As such:

        self.Xmetric = X
        self.Smetric = S 

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!! NOTE! - Change this back!
    
        if engines != {"diis"}:
            self.removal_policy = RemovalPolicy.OldestAdded
        elif not isinstance(removal_policy, RemovalPolicy):
            raise TypeError(f"removal_policy must be a RemovalPolicy, not a {type(removal_policy)}")
        else:
            self.removal_policy = removal_policy

        if not isinstance(storage_policy, StoragePolicy):
            raise TypeError(f"storage_policy must be a StoragePolicy, not a {type(storage_policy)}")

        if not which_import("scipy", return_bool=True) and ("ediis" in engines or "adiis" in engines):
            raise ModuleNotFoundError("Python module scipy not found. Solve by\n" +
                                      "    (1) installing it: `conda install scipy` or `pip install scipy`, or" +
                                      "    (2) de-activating a/ediis with option: `set scf scf_initial_accelerator none`")
        self.max_vecs = max_vecs
        self.name = name
        self.storage_policy = storage_policy
        # The template matches each entry key to the expected dimensions of each of its items.
        # For the simple DIIS case, there are functions to populate this. (Useful C-side.)
        # For all other cases, this is set automatically the first time an entry is added.
        self.template = {}
        self.reset_subspace()

        # Resource Acquired: Open PSIO file.
        self.opened_libdiis = False
        if self.storage_policy == StoragePolicy.OnDisk:
            psio = core.IO.shared_object()
            if not psio.open_check(psif.PSIF_LIBDIIS):
                psio.open(psif.PSIF_LIBDIIS, 1) # 1 = PSIO_OPEN_OLD
                self.opened_libdiis = True
        self.created_files = set()

        self.closed_shell = closed_shell # Only needed for A/EDIIS, which doesn't allow ROHF anyways. #????
        self.engines = engines

    def __del__(self):
        # RAII the PSIO file away.
        if self.opened_libdiis:
            psio = core.IO.shared_object()
            if psio.open_check(psif.PSIF_LIBDIIS):
                psio.close(psif.PSIF_LIBDIIS, 1) # 1 = KEEP

    def reset_subspace(self):
        """ Wipe all data from previous iterations. """
        self.stored_vectors = [] # elt. i is entry i
        self.iter_num = -1
        # At present, we only cache for DIIS, not EDIIS or ADIIS. In principle, we could, but
        # their quantities are N^2, so we assume the savings are negligible.
        self.cached_dot_products = dict()

    def copier(self, x, new_name: str):
        """ Copy the object x and give it a new_name. Save it to disk if needed. """
        if isinstance(x, (core.Matrix, core.Vector)):
            copy = x.clone()
        elif isinstance(x, (core.dpdbuf4, core.dpdfile2)):
            copy = core.Matrix(x)
        elif isinstance(x, float):
            # Never cache a _number_.
            return x
        elif which_import("ambit", return_bool=True):
            import ambit
            if isinstance(x, ambit.BlockedTensor):
                copy = x.clone()
            else:
                raise TypeError("Unrecognized object type for DIIS.")
        else:
            raise TypeError("Unrecognized object type for DIIS.")
        copy.name = new_name

        if self.storage_policy == StoragePolicy.OnDisk:
            psio = core.IO.shared_object()
            if isinstance(copy, core.Vector):
                copy.save(psio, psif.PSIF_LIBDIIS)
            elif isinstance(copy, core.Matrix):
                copy.save(psio, psif.PSIF_LIBDIIS, core.SaveType.SubBlocks)
            elif isinstance(copy, ambit.BlockedTensor):
                filename = f"{core.IOManager.shared_object().get_default_path()}/libdiis.{copy.name}"
                copy.save(filename)
                self.created_files.add(filename)
            else:
                raise TypeError("Unrecognized object type for DIIS. This shouldn't be possible.")
            copy = None

        return copy

    def get_name(self, name, entry_num, item_num):
        """ This is what we'll save an object to disk with."""
        return f"{self.name}: {name} Entry {entry_num}, Item {item_num}"

    def load_quantity(self, name, entry_num, item_num, force_new = True):
        """ Load quantity from wherever it's stored, constructing a new object if needed. """
        template_object = self.template[name][item_num]
        if isinstance(template_object, float) or self.storage_policy == StoragePolicy.InCore:
            quantity = self.stored_vectors[entry_num][name][item_num]
            try:
                quantity = quantity.clone()
            except AttributeError:
                # The quantity must have been a float. No need to clone.
                pass
        elif self.storage_policy == StoragePolicy.OnDisk:
            full_name = self.get_name(name, entry_num, item_num)
            psio = core.IO.shared_object()
            if hasattr(template_object, "__len__"):
                # Looks like we have dimensions.
                if len(template_object) == 2:
                    quantity = core.Matrix(full_name, *template_object)
                    quantity.load(psio, psif.PSIF_LIBDIIS, core.SaveType.SubBlocks)
                elif len(template_object) == 1:
                    quantity = core.Vector(full_name, *template_object)
                    quantity.load(psio, psif.PSIF_LIBDIIS)
            elif which_import("ambit", return_bool=True):
                import ambit
                if template_object == ambit.BlockedTensor:
                    quantity = ambit.BlockedTensor.load_and_build(f"{core.IOManager.shared_object().get_default_path()}/libdiis.{full_name}")
        else:
            raise Exception(f"StoragePolicy {self.storage_policy} not recognized. This is a bug: contact developers.")

        return quantity


    def get_list_dot(self, i: int, k: int, first_element: str, second_element: str):
        """Get LIST dot product, where i is the first element, k is second element"""
        dot_product = 0

        # for item_num in range(len(self.template[first_element])):
        #     d_out = self.load_quantity(first_element, i, item_num)
        #     f_diff = self.load_quantity(second_element, k, item_num)

        #     # w/ transform included
        #     #d_out  = self.form_XMX(d_out)
        #     #f_diff = self.form_XMX(f_diff)

        #     dot_product += d_out.vector_dot(f_diff)

        X = self.Xmetric     # may be None
        S = self.Smetric
        
        use_metric = X is not None and S is not None
        #core.print_out(X)
        #core.print_out(S)

        
        for item in range(len(self.template[first_element])):

            left  = self.load_quantity(first_element,  i, item).clone()
            right = self.load_quantity(second_element, k, item).clone()
            raw_left  = self.load_quantity(first_element,  i, item).clone()
            raw_right = self.load_quantity(second_element, k, item).clone()

            if use_metric:
                left_is_density  = ("densit" in first_element) or ("d diff" in first_element)
                right_is_density = ("densit" in second_element) or ("d diff" in second_element)

                if left_is_density:
                    left = core.triplet(S, left, S)      # S D S
                if right_is_density:
                    right = core.triplet(S, right, S)    # S D S

                # similarity transform with X:
                left.transform(X)
                right.transform(X)

                
                #left.print_out()   
                #right.print_out() 
                #          
                #core.print_out(f"Raw  dot = {raw_left.vector_dot(raw_right):.6e}\n")
                #core.print_out(f"Trans dot= {left.vector_dot(right):.6e}\n")

                # add othagonal check here

                #self.Smetric.print_out()
                #self.Xmetric.print_out()

    
            dot_product += left.vector_dot(right)

        return dot_product


    def get_dot_product(self, i: int, j: int, use_fdiis: bool = False):
        """ Get a DIIS dot product. i and j represent entry numbers. """
        key = frozenset([i, j])
        error_type = "f diff" if use_fdiis else "error"
        try:
            if error_type not in self.cached_dot_products.keys() and error_type not in self.template.keys():
                error_type = "error"
            return self.cached_dot_products[key]
        except KeyError:
            dot_product = 0
            for item_num in range(len(self.template[error_type])):
                Rix = self.load_quantity(error_type, i, item_num)
                Rjx = self.load_quantity(error_type, j, item_num)
                dot_product += Rix.vector_dot(Rjx)

            self.cached_dot_products[key] = dot_product
            return dot_product


    def set_error_vector_size(self, *args):
        """ Set the template for the DIIS error. Kept mainly for backwards compatibility. """
        self.template["error"] = template_helper(*args)

    def set_vector_size(self, *args):
        """ Set the template for the extrapolation target. Kept mainly for backwards compatibility. """
        self.template["target"] = template_helper(*args)

    def build_entry(self, entry, target_index):
        return {key: [self.copier(elt, self.get_name(key, target_index, i)) for i, elt in enumerate(val)] for key, val in entry.items()}

    def add_entry(self, *args):
        if self.max_vecs == 0:
            return False

        # Convert from "raw list of args" syntax to a proper entry.
        # While "entry" format is more general, "raw list of args" won't break C-side code, which doesn't need the generality.
        if not (len(args) == 1 and isinstance(args[0], dict)):
            R_len = len(self.template.get("error", []))
            T_len = len(self.template.get("target", []))
            if R_len + T_len != len(args):
                raise Exception(f"Cannot build {R_len} residuals and {T_len} amplitudes from {len(args)} items.")
            entry = {"error": args[:R_len], "target": args[R_len:]}
        else:
            entry = args[0]
            self.template = {key: template_helper(*val) for key, val in entry.items()}

        self.iter_num += 1

        if len(self.stored_vectors) >= self.max_vecs:
            if self.removal_policy == RemovalPolicy.OldestAdded:
                target_index = self.iter_num % self.max_vecs
            elif self.removal_policy == RemovalPolicy.LargestError:
                target_index = np.argmax([self.get_dot_product(i, i) for i in range(len(self.stored_vectors))])
            else:
                raise Exception(f"RemovalPolicy {self.removal_policy} not recognized. This is a bug: contact developers.")
            # Purge imminently-outdated values from cache.
            self.cached_dot_products = {key: val for key, val in self.cached_dot_products.items() if target_index not in key}
            # Set the new entry.
            self.stored_vectors[target_index] = self.build_entry(entry, target_index)
        else:
            self.stored_vectors.append(self.build_entry(entry, self.iter_num))

        return True

    def get_list3_dot(self):
        dot_product = 0
        # for item_num in range(len(self.template[first_element])):
        #     d_out = self.load_quantity(first_element, i, item_num)
        #     f_diff = self.load_quantity(second_element, k, item_num)
        #     dot_product += d_out.vector_dot(f_diff)
        return dot_product

    def list3_coefficients(self):
        dim = len(self.stored_vectors) + 1
        B = np.empty((dim, dim))
        B[0, :] = -1
        B[:, 0] = -1
        B[0, 0] = 0

        for i in range(len(self.stored_vectors)):
            for k in range(len(self.stored_vectors)):
                B[i + 1, k + 1] = self.get_list3_dot(i, k)

        resid = np.zeros(dim)
        resid[0] = -1
        return np.linalg.lstsq(B, resid, rcond=None)[0][1:]

    def list_coefficients(self, list_option):
        first_element = "densities"
        second_element = "f diff"
        if list_option == "LISTF":
            first_element = "f in"
            second_element = "f diff"
        elif list_option == "LISTB":
            first_element = "densities"
            second_element = "f diff"
        elif list_option == "LISTR":
            first_element = "d diff"
            second_element = "v diff"
        elif list_option == "FDIIS":
            first_element = "f diff"
            second_element = "f diff"


        dim = len(self.stored_vectors) + 1
        B = np.empty((dim, dim))
        B[0, :] = -1
        B[:, 0] = -1
        B[0, 0] = 0

        for i in range(len(self.stored_vectors)):
            for k in range(len(self.stored_vectors)):
                B[i + 1, k + 1] = self.get_list_dot(i, k, first_element, second_element)

        resid = np.zeros(dim)
        resid[0] = -1
        return np.linalg.lstsq(B, resid, rcond=None)[0][1:]

    def diis_coefficients(self, use_fdiis: bool = False):
        dim = len(self.stored_vectors) + 1
        B = np.zeros((dim, dim))
        for i in range(len(self.stored_vectors)):
            for j in range(len(self.stored_vectors)):
                B[i, j] = self.get_dot_product(i, j, use_fdiis)
        B[-1, :-1] = B[:-1, -1] = -1

        rhs = np.zeros((dim))
        rhs[-1] = -1

        # Trick to improve numerical conditioning.
        # Instead of solving B c = r, we solve D B D^-1 D c = D r, using
        # D r = r. D is the diagonals ^ -1/2 matrix.
        # This improves the conditioning of the problem.
        diagonals = B.diagonal().copy()
        diagonals[-1] = 1
        if np.all(diagonals > 0):
            diagonals = diagonals ** (- 0.5)
            B = np.einsum("i,ij,j -> ij", diagonals, B, diagonals)
            return np.linalg.lstsq(B, rhs, rcond=None)[0][:-1] * diagonals[:-1]
        else:
            return np.linalg.lstsq(B, rhs, rcond=None)[0][:-1]

    def adiis_energy(self, x):
        return np.dot(self.adiis_linear, x) + np.einsum("i,ij,j->", x, self.adiis_quadratic, x) / 2

    def adiis_gradient(self, x):
        return self.adiis_linear + np.einsum("i,ij->j", x, self.adiis_quadratic)

    def adiis_coefficients(self):
        from scipy.optimize import minimize
        self.adiis_populate()
        result = minimize(self.adiis_energy, np.ones(len(self.stored_vectors)), method="SLSQP",
                          bounds = tuple((0, 1) for i in self.stored_vectors),
                          constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1, "jac": lambda x: np.ones_like(x)}],
                          jac=self.adiis_gradient, tol=5e-6, options={"maxiter": 200})

        if not result.success:
            raise Exception("ADIIS minimization failed. File a bug, and include your entire input and output files.")

        return result.x

    def adiis_populate(self):
        """ Fills linear and quadratic coefficients in ADIIS energy estimate. """
        # We are currently assuming that all of dD and dF fit in-core.
        # These quantities are N^2, so this should be fine in most cases.

        num_entries = len(self.stored_vectors)
        dD = [[] for x in range(num_entries)]
        dF = [[] for x in range(num_entries)]
        for name, array in zip(["densities", "target"], [dD, dF]):
            for item_num in range(len(self.template[name])):
                latest_entry = self.load_quantity(name, len(self.stored_vectors) - 1, item_num)
                for entry_num in range(num_entries):
                    temp = self.load_quantity(name, entry_num, item_num, force_new=True)
                    temp.subtract(latest_entry)
                    array[entry_num].append(temp)

        self.adiis_linear = np.zeros((num_entries))
        latest_fock = []
        for item_num in range(len(self.template["target"])):
            latest_fock.append(self.load_quantity("target", len(self.stored_vectors) - 1, item_num))
        for i in range(num_entries):
            self.adiis_linear[i] = sum(d.vector_dot(f) for d, f in zip(dD[i], latest_fock))

        self.adiis_quadratic = np.zeros((num_entries, num_entries))
        for i, j in product(range(num_entries), repeat = 2):
            self.adiis_quadratic[i][j] = sum(d.vector_dot(f) for d, f in zip(dD[i], dF[j]))

        if self.closed_shell:
            self.adiis_linear *= 2
            self.adiis_quadratic *= 2

    def ediis_energy(self, x):
        ediis_linear = np.array([entry["energy"][0] for entry in self.stored_vectors])
        return np.dot(ediis_linear, x) + np.einsum("i,ij,j->", x, self.ediis_quadratic, x) / 2

    def ediis_gradient(self, x):
        """ Gradient of energy estimate w.r.t. input coefficient """
        ediis_linear = np.array([entry["energy"][0] for entry in self.stored_vectors])
        return ediis_linear + np.einsum("i,ij->j", x, self.ediis_quadratic)

    def ediis_coefficients(self):
        from scipy.optimize import minimize
        self.ediis_populate()
        result = minimize(self.ediis_energy, np.ones(len(self.stored_vectors)), method="SLSQP",
                          bounds = tuple((0, 1) for i in self.stored_vectors),
                          constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1, "jac": lambda x: np.ones_like(x)}],
                          jac=self.ediis_gradient, tol=5e-6, options={"maxiter": 200})

        if not result.success:
            raise Exception("EDIIS minimization failed. File a bug, and include your entire input and output files.")

        return result.x

    def ediis_populate(self):
        """ Fills quadratic coefficients in ADIIS energy estimate. """
        num_entries = len(self.stored_vectors)

        self.ediis_quadratic = np.zeros((num_entries, num_entries))
        for i in range(num_entries):
            for item_num in range(len(self.template["densities"])):
                d = self.load_quantity("densities", i, item_num)
                for j in range(num_entries):
                    f = self.load_quantity("target", j, item_num)
                    self.ediis_quadratic[i][j] += d.vector_dot(f)

        diag = np.diag(self.ediis_quadratic)
        # D_i F_i + D_j F_j - D_i F_j - D_j F_i; First two terms use broadcasting tricks
        self.ediis_quadratic = diag[:, None] + diag - self.ediis_quadratic - self.ediis_quadratic.T

        self.ediis_quadratic *= -1/2

        if self.closed_shell:
            self.ediis_quadratic *= 2

    def extrapolate(self, *args, Dnorm=None, component=0):


        """ Perform extrapolation. Must be passed in an error metric to decide how to handle hybrid algorithms. """
        # def mesa_method_check(meth):
        #     mesa = core.get_option('SCF', "MESA") and args[0] == "mesa"
        #     scf_conv_method: str = args[1]
        #     if mesa:
        #         return meth in scf_conv_method.lower() and mesa
        #     return False

        is_mesa = False
        conv_method = ""
        if args:
            if args[0] == "mesa":
                is_mesa = True
                conv_method = args[1]

        diis = is_mesa and "DIIS" == conv_method
        adiis = is_mesa and "ADIIS" == conv_method
        ediis = is_mesa and "EDIIS" == conv_method
        fdiis = is_mesa and "FDIIS" == conv_method
        listb = is_mesa and "LISTB" == conv_method
        listr = is_mesa and "LISTR" == conv_method
        listf = is_mesa and "LISTF" == conv_method
        list3 = is_mesa and "LIST3" == conv_method

        if args[0] == 'mesa' and core.get_option('SCF', "MESA"):
           args = args[2:]

        if {"adiis", "ediis"}.intersection(self.engines) and Dnorm is None:
            raise ValidationError("An extrapolation engine insists you specify the error metric.")

        performed = set()
        if is_mesa:
            if diis:
                coeffs = self.diis_coefficients(use_fdiis=False)
                performed.add("DIIS")
            elif adiis:
                coeffs = self.adiis_coefficients()
                performed.add("ADIIS")
            elif ediis:
                coeffs = self.ediis_coefficients()
                performed.add("EDIIS")
            elif fdiis:
                coeffs = self.list_coefficients("FDIIS")
                performed.add("FDIIS")
            elif listb:
                coeffs = self.list_coefficients("LISTB")
                performed.add("LISTB")
            elif listf:
                coeffs = self.list_coefficients("LISTF")
                performed.add("LISTF")
            elif listr:
                coeffs = self.list_coefficients("LISTR")
                performed.add("LISTR")
            elif list3:
                coeffs = self.list3_coefficients()
                performed.add("LIST3")
        else:
            if self.engines == {"diis"}:
                coeffs = self.diis_coefficients()
                performed.add("DIIS")
            elif self.engines == {"list"}:
                list_varient = core.get_option('SCF', 'LIST')
                coeffs = self.list_coefficients(list_varient)
                performed.add(list_varient)
            elif len(self.engines) == 1:
                blend_stop = core.get_option("SCF", "SCF_INITIAL_FINISH_DIIS_TRANSITION")
                if Dnorm <= blend_stop:
                    return performed
                elif self.engines == {"ediis"}:
                    coeffs = self.ediis_coefficients()
                    performed.add("EDIIS")
                elif self.engines == {"adiis"}:
                    coeffs = self.adiis_coefficients()
                    performed.add("ADIIS")
                else:
                    raise Exception(f"DIIS engine not recognized: {self.engines[0]}.")
            elif self.engines == {"diis", "adiis"} or self.engines == {"diis", "ediis"}:
                blend_start = core.get_option("SCF", "SCF_INITIAL_START_DIIS_TRANSITION")
                blend_stop = core.get_option("SCF", "SCF_INITIAL_FINISH_DIIS_TRANSITION")
                if "adiis" in self.engines:
                    initial_coefficient_function = self.adiis_coefficients
                    initial_name = "ADIIS"
                else:
                    initial_coefficient_function = self.ediis_coefficients
                    initial_name = "EDIIS"

                if Dnorm >= blend_start:
                    coeffs = initial_coefficient_function()
                    performed.add(initial_name)
                elif Dnorm <= blend_stop:
                    coeffs = self.diis_coefficients()
                    performed.add("DIIS")
                else:
                    m = 1 - (Dnorm - blend_start) / (blend_stop - blend_start)
                    coeffs = m * initial_coefficient_function() + (1 - m) * self.diis_coefficients()
                    performed.add("DIIS")
                    performed.add(initial_name)
            else:
                raise Exception(f"DIIS engine combination not recognized: {self.engines}")

        # if self.engines == {"list"} or is_mesa:
        #     matrices = ["target", "densities", "potential"]
        #     for j, Tj in enumerate(args):
        #         Tj.zero()
        #         key = matrices[j]
        #         for i, ci in enumerate(coeffs):
        #             Tij = self.load_quantity(key, i, 0)
        #             axpy(Tj, ci, Tij)

        # else:
        #     # TODO fix only return F_in given both F and D in arg
        #     for j, Tj in enumerate([args[0]]):
        #         Tj.zero()
        #         for i, ci in enumerate(coeffs):
        #             Tij = self.load_quantity("target", i, j)
        #             axpy(Tj, ci, Tij)

        if self.engines == {"list"} or is_mesa:
            matrices = ["target", "densities", "potential"]
            for j, Tj in enumerate(args):
                Tj.zero()
                key = matrices[j]
                for i, ci in enumerate(coeffs):
                    # NO! hardcoding 0
                    Tij = self.load_quantity(key, i, component)
                    axpy(Tj, ci, Tij) # Tj = Tj + ci * Tij

        else:
            #DIIS
            for j, Tj in enumerate([args[0]]):
                Tj.zero()
                for i, ci in enumerate(coeffs):
                    
                    Tij = self.load_quantity("target", i, component)
                    axpy(Tj, ci, Tij)

        return performed

    def delete_diis_file(self):
        """ Purge all data in the DIIS file. """
        # libpsio deletion
        psio = core.IO.shared_object()
        if not psio.open_check(psif.PSIF_LIBDIIS):
            psio.open(psif.PSIF_LIBDIIS, 1) # 1 = PSIO_OPEN_OLD
        psio.close(psif.PSIF_LIBDIIS, 0) # 0 = DELETE
        # ambit deletion
        for filename in self.created_files:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

