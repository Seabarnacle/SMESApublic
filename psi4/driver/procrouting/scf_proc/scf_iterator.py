
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2024 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
"""
The SCF iteration functions
"""
import json

import numpy as np

from psi4 import core

from ... import p4util
from ...constants import constants
from ...p4util.exceptions import SCFConvergenceError, ValidationError
from ..solvent.efp import get_qm_atoms_opts, modify_Fock_induced, modify_Fock_permanent
import scipy.optimize as opt

#import logging
#logger = logging.getLogger("scf.scf_iterator")
#logger.setLevel(logging.DEBUG)

# Q: I expect more local settings of options for part of SCF.
#    For convcrit, do we want:
#   (A) easy to grep
#    with p4util.OptionsStateCM(['SCF', 'E_CONVERGENCE'], ['SCF', 'D_CONVERGENCE']):
#        core.set_local_option('SCF', 'E_CONVERGENCE', 1.e-5)
#        core.set_local_option('SCF', 'D_CONVERGENCE', 1.e-4)
#        self.iterations()
#
#   or (B) functional. options never touched
#    self.iterations(e_conv=1.e-5, d_conv=1.e-4)


# zmeta (smesa) helpers 

def zmeta_optimize_method(error_matrices, scale_factors, is_unrestricted):
    """
    Solves the LIST-like matrix equation using Psi4 matrices, scaled
    globally relative to the iteration to preserve physical magnitudes.
    """
    n = len(error_matrices)
    
    def get_dot(m1, m2):
        if is_unrestricted:
            return m1[0].vector_dot(m2[0]) + m1[1].vector_dot(m2[1])
        else:
            return m1.vector_dot(m2)

    # Build the Pulay Matrix
    B = np.zeros((n + 1, n + 1))
    B[0, 1:] = -1.0
    B[1:, 0] = -1.0
    
    E_k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            raw_dot = get_dot(error_matrices[i], error_matrices[j])
            # Divide the dot product by the pre-calculated Global Scale Factors.
            # This balances the metric magnitudes without destroying method comparisons!
            E_k[i, j] = raw_dot / (scale_factors[i] * scale_factors[j])
                
    B[1:, 1:] = E_k
    
    rhs = np.zeros(n + 1)
    rhs[0] = -1.0
    
    try:
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(B).dot(rhs)
        
    lambda_k_sq = sol[0]
    c_weights = sol[1:]
    
    return lambda_k_sq, c_weights


def zmeta_step(zmeta_vector_errors, all_combination_errors, is_unrestricted):
    """
    Executes the True Vector ZMETA logic with Global Iteration Scaling.
    """
    metrics = list(zmeta_vector_errors.keys())
    methods = list(zmeta_vector_errors[metrics[0]].keys())
    
    # 1. Calculate the Global Scale Factors for this iteration
    scale_factors = []
    for metric in metrics:
        # Find the max RMS error for this metric across all methods
        max_val = max(all_combination_errors[metric].values())
        # Prevent division by zero if a metric evaluates perfectly to 0.0
        scale_factors.append(max(max_val, 1e-100))
    
    min_lambda_sq = float('inf')
    best_method_idx = -1
    lambda_sq_scores = {}
    
    for k, method in enumerate(methods):
        error_matrices_for_method_k = [zmeta_vector_errors[metric][method] for metric in metrics]
        
        # Pass the global scale_factors into the optimizer
        lambda_k_sq, c_weights = zmeta_optimize_method(
            error_matrices_for_method_k, 
            scale_factors, 
            is_unrestricted
        )
        
        score = abs(lambda_k_sq)
        lambda_sq_scores[method] = score
        
        if score < min_lambda_sq:
            min_lambda_sq = score
            best_method_idx = k
            
    return methods[best_method_idx], lambda_sq_scores

def ymeta_optimize(error_matrices, is_unrestricted):
    """
    Solves for weights that optimally blend MULTIPLE METHODS for a SINGLE METRIC,
    using quadratic programming.
    """
    n = len(error_matrices)
    
    def get_dot_meta(m1, m2):
        if is_unrestricted:
            return m1[0].vector_dot(m2[0]) + m1[1].vector_dot(m2[1])
        else:
            return m1.vector_dot(m2)

    # 1. Build and Normalize the Gram matrix
    E_raw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            E_raw[i, j] = get_dot_meta(error_matrices[i], error_matrices[j])
            
    max_E = np.max(np.abs(E_raw))
    if max_E < 1e-100:
        max_E = 1.0 
    E_norm = E_raw / max_E
    
    # 2. Define the Objective Function to minimize: f(c) = c^T * E * c
    def objective_meta(c):
        return c.dot(E_norm).dot(c)

    # 3. Define the Constraints and Bounds
    # Constraint: The sum of all weights must exactly equal 1.0
    cons = ({'type': 'eq', 'fun': lambda c: np.sum(c) - 1.0})
    
    # Every weight must be between 0.0 (0%) and 1.0 (100%).
    bounds = [(0.0, 1.0) for _ in range(n)]

    # 4. Initial guess (start by averaging them all equally)
    c0 = np.ones(n) / n

    # 5. Execute the cheap, fast SLSQP solver
    res = opt.minimize(objective_meta, c0, method='SLSQP', bounds=bounds, constraints=cons)
    
    weights = res.x
    lambda_sq = res.fun * max_E  # Un-scale the final error
    
    # Clean up microscopic bound violations.
    weights = np.clip(weights, 0.0, 1.0)
    weights /= np.sum(weights) # Ensure perfect sum to 1.0
    
    return lambda_sq, weights

def ymeta_step(ymeta_vector_errors, target_metric, is_unrestricted):
    """
    Executes the YMETA logic for a chosen target metric.
    """
    methods = list(ymeta_vector_errors[target_metric].keys())
    
    # Extract the error matrices across ALL methods for the SINGLE chosen metric
    error_matrices_for_metric_l = [ymeta_vector_errors[target_metric][method] for method in methods]
    
    lambda_sq, c_weights = ymeta_optimize(error_matrices_for_metric_l, is_unrestricted)
    
    # Map the resulting weights back to the method names
    weights_dict = {methods[i]: c_weights[i] for i in range(len(methods))}
    
    return lambda_sq, weights_dict

def zymeta_optimize(zymeta_vector_errors, is_unrestricted):
    """
    ZYMETA: Blends MULTIPLE methods across MULTIPLE metrics simultaneously.
    Uses SLSQP optimization.
    """
    metrics = list(zymeta_vector_errors.keys())
    methods = list(zymeta_vector_errors[metrics[0]].keys())
    n = len(methods)
    
    def get_dot_meta(m1, m2):
        if is_unrestricted:
            return m1[0].vector_dot(m2[0]) + m1[1].vector_dot(m2[1])
        else:
            return m1.vector_dot(m2)

    B_grand = np.zeros((n, n))
    
    # 1. Build the Grand Gram Matrix across all metrics
    for metric in metrics:
        local_Gram = np.zeros((n, n))
        for i, method_a in enumerate(methods):
            for j, method_b in enumerate(methods):
                err_a = zymeta_vector_errors[metric][method_a]
                err_b = zymeta_vector_errors[metric][method_b]
                local_Gram[i, j] = get_dot_meta(err_a, err_b)
        
        # ZMETA Global Scaling: Find the max self-overlap (diagonal) for THIS metric
        s_i_sq = np.max(np.diag(local_Gram))
        if s_i_sq < 1e-100:
            s_i_sq = 1.0 # Prevent division by zero
            
        # Divide the metric space by its own max, and add to the Grand Matrix
        B_grand += (local_Gram / s_i_sq)

    # 2. Condition-Number Normalization for the QP Solver
    max_B = np.max(np.abs(B_grand))
    if max_B < 1e-100:
        max_B = 1.0
    B_norm = B_grand / max_B
    
    # 3. SLSQP optimization
    def objective_meta(c):
        return c.dot(B_norm).dot(c)

    cons = ({'type': 'eq', 'fun': lambda c: np.sum(c) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(n)]
    c0 = np.ones(n) / n  # Initial guess: evenly averaged

    res = opt.minimize(objective_meta, c0, method='SLSQP', bounds=bounds, constraints=cons)
    
    weights = res.x
    lambda_sq = res.fun * max_B # Un-scale to physical magnitude
    
    # Clean up microscopic bound violations.
    weights = np.clip(weights, 0.0, 1.0)
    weights /= np.sum(weights)
    
    weights_dict = {methods[i]: weights[i] for i in range(len(methods))}
    
    return lambda_sq, weights_dict

def scf_compute_energy(self):
    """Base class Wavefunction requires this function. Here it is
    simply a wrapper around initialize(), iterations(), finalize_energy(). It
    returns the SCF energy computed by finalize_energy().

    """
    if core.get_option('SCF', 'DF_SCF_GUESS') and (core.get_global_option('SCF_TYPE') == 'DIRECT'):
        # speed up DIRECT algorithm (recomputes full (non-DF) integrals
        #   each iter) by first converging via fast DF iterations, then
        #   fully converging in fewer slow DIRECT iterations. aka Andy trick 2.0
        core.print_out("  Starting with a DF guess...\n\n")
        with p4util.OptionsStateCM(['SCF_TYPE']):
            core.set_global_option('SCF_TYPE', 'DF')
            self.initialize()
            try:
                self.iterations()
            except SCFConvergenceError:
                self.finalize()
                raise SCFConvergenceError("""SCF DF preiterations""", self.iteration_, self, 0, 0)
        core.print_out("\n  DF guess converged.\n\n")

        # reset the DIIS & JK objects in prep for DIRECT
        if self.initialized_diis_manager_:
            self.diis_manager_.reset_subspace()
        self.initialize_jk(self.memory_jk_)
    else:
        self.initialize()
    self.iteration_energies = []

    try:
        self.iterations()
    except SCFConvergenceError as e:
        if core.get_option("SCF", "FAIL_ON_MAXITER"):
            core.print_out("  Failed to converge.\n")
            # energy = 0.0
            # A P::e fn to either throw or protest upon nonconvergence
            # die_if_not_converged()
            raise e
        else:
            core.print_out("  Energy and/or wave function did not converge, but proceeding anyway.\n\n")
    else:
        core.print_out("  Energy and wave function converged.\n\n")

    scf_energy = self.finalize_energy()
    return scf_energy


def _build_jk(wfn, memory):
    jk = core.JK.build(wfn.get_basisset("ORBITAL"),
                       aux=wfn.get_basisset("DF_BASIS_SCF"),
                       do_wK=wfn.functional().is_x_lrc(),
                       memory=memory)
    return jk


def initialize_jk(self, memory, jk=None):

    functional = self.functional()
    if jk is None:
        jk = _build_jk(self, memory)

    self.set_jk(jk)

    jk.set_print(self.get_print())
    jk.set_memory(memory)
    jk.set_do_K(functional.is_x_hybrid())
    jk.set_do_wK(functional.is_x_lrc())
    jk.set_omega(functional.x_omega())

    jk.set_omega_alpha(functional.x_alpha())
    jk.set_omega_beta(functional.x_beta())

    jk.initialize()
    jk.print_header()


def scf_initialize(self):
    """Specialized initialization, compute integrals and does everything to prepare for iterations"""

    # Figure out memory distributions

    # Get memory in terms of doubles
    total_memory = (core.get_memory() / 8) * core.get_global_option("SCF_MEM_SAFETY_FACTOR")

    # Figure out how large the DFT collocation matrices are
    vbase = self.V_potential()
    if vbase:
        collocation_size = vbase.grid().collocation_size()
        if vbase.functional().ansatz() == 1:
            collocation_size *= 4  # First derivs
        elif vbase.functional().ansatz() == 2:
            collocation_size *= 10  # Second derivs
    else:
        collocation_size = 0

    # Change allocation for collocation matrices based on DFT type
    jk = _build_jk(self, total_memory)
    jk_size = jk.memory_estimate()

    # Give remaining to collocation
    if total_memory > jk_size:
        collocation_memory = total_memory - jk_size
    # Give up to 10% to collocation
    elif (total_memory * 0.1) > collocation_size:
        collocation_memory = collocation_size
    else:
        collocation_memory = total_memory * 0.1

    if collocation_memory > collocation_size:
        collocation_memory = collocation_size

    # Set constants
    self.iteration_ = 0
    self.memory_jk_ = int(total_memory - collocation_memory)
    self.memory_collocation_ = int(collocation_memory)

    if self.get_print():
        core.print_out("  ==> Integral Setup <==\n\n")

    # Initialize EFP
    efp_enabled = hasattr(self.molecule(), 'EFP')
    if efp_enabled:
        # EFP: Set QM system, options, and callback. Display efp geom in [A]
        efpobj = self.molecule().EFP
        core.print_out(efpobj.banner())
        core.print_out(efpobj.geometry_summary(units_to_bohr=constants.bohr2angstroms))

        efpptc, efpcoords, efpopts = get_qm_atoms_opts(self.molecule())
        efpobj.set_point_charges(efpptc, efpcoords)
        efpobj.set_opts(efpopts, label='psi', append='psi')

        efpobj.set_electron_density_field_fn(efp_field_fn)

    # Initialize all integrals and perform the first guess
    if self.attempt_number_ == 1:
        mints = core.MintsHelper(self.basisset())

        self.initialize_jk(self.memory_jk_, jk=jk)
        if self.V_potential():
            self.V_potential().build_collocation_cache(self.memory_collocation_)
        core.timer_on("HF: Form core H")
        self.form_H()
        core.timer_off("HF: Form core H")

        if efp_enabled:
            # EFP: Add in permanent moment contribution and cache
            core.timer_on("HF: Form Vefp")
            verbose = core.get_option('SCF', "PRINT")
            Vefp = modify_Fock_permanent(self.molecule(), mints, verbose=verbose - 1)
            Vefp = core.Matrix.from_array(Vefp)
            self.H().add(Vefp)
            Horig = self.H().clone()
            self.Horig = Horig
            core.print_out("  QM/EFP: iterating Total Energy including QM/EFP Induction\n")
            core.timer_off("HF: Form Vefp")

        core.timer_on("HF: Form S/X")
        self.form_Shalf()
        core.timer_off("HF: Form S/X")

        core.print_out("\n  ==> Pre-Iterations <==\n\n")

        # force SCF_SUBTYPE to AUTO during SCF guess
        optstash = p4util.OptionsState(["SCF", "SCF_SUBTYPE"])
        core.set_local_option("SCF", "SCF_SUBTYPE", "AUTO")

        core.timer_on("HF: Guess")
        self.guess()
        core.timer_off("HF: Guess")

        optstash.restore()

        # Print out initial docc/socc/etc data
        if self.get_print():
            lack_occupancy = core.get_local_option('SCF', 'GUESS') in ['SAD']
            if core.get_global_option('GUESS') in ['SAD']:
                lack_occupancy = core.get_local_option('SCF', 'GUESS') in ['AUTO']
                self.print_preiterations(small=lack_occupancy)
            else:
                self.print_preiterations(small=lack_occupancy)

    else:
        # We're reading the orbitals from the previous set of iterations.
        self.form_D()
        self.set_energies("Total Energy", self.compute_initial_E())

    # turn off VV10 for iterations
    if core.get_option('SCF', "DFT_VV10_POSTSCF") and self.functional().vv10_b() > 0.0:
        core.print_out("  VV10: post-SCF option active \n \n")
        self.functional().set_lock(False)
        self.functional().set_do_vv10(False)
        self.functional().set_lock(True)

    # Print iteration header
    is_dfjk = core.get_global_option('SCF_TYPE').endswith('DF')
    diis_rms = core.get_option('SCF', 'DIIS_RMS_ERROR')
    core.print_out("  ==> Iterations <==\n\n")
    core.print_out("%s                        Total Energy        Delta E     %s |[F,P]|\n\n" %
                   ("   " if is_dfjk else "", "RMS" if diis_rms else "MAX"))


def scf_iterate(self, e_conv=None, d_conv=None):

    is_dfjk = core.get_global_option('SCF_TYPE').endswith('DF')
    verbose = core.get_option('SCF', "PRINT")
    reference = core.get_option('SCF', "REFERENCE")

    # f_in matrices for LIST, TODO: move to Cpp side later
    f_ins = []
    d_ins = []
    v_ins = []
    current_mesa_scf_conv_method = "DIIS"
    self.methods = []
    self.rms = []
    self.energies = []
    self.best_metrics = []
    lice_1 = 0
    lice_2 = 1

    # self.member_data_ signals are non-local, used internally by c-side fns
    self.diis_enabled_ = self.validate_diis()
    self.list_enabled = self.validate_list()
    self.mesa_enabled = self.validate_mesa()


    self.MOM_excited_ = _validate_MOM()
    self.diis_start_ = core.get_option('SCF', 'DIIS_START')
    damping_enabled = _validate_damping()
    soscf_enabled = _validate_soscf()
    frac_enabled = _validate_frac()
    efp_enabled = hasattr(self.molecule(), 'EFP')
    cosx_enabled = "COSX" in core.get_option('SCF', 'SCF_TYPE')

    # does the JK algorithm use severe screening approximations for early SCF iterations?
    early_screening = False
    if cosx_enabled:
        early_screening = True
        self.jk().set_COSX_grid("Initial")

    # maximum number of scf iterations to run after early screening is disabled
    scf_maxiter_post_screening = core.get_option('SCF', 'COSX_MAXITER_FINAL')

    if scf_maxiter_post_screening < -1:
        raise ValidationError('COSX_MAXITER_FINAL ({}) must be -1 or above. If you wish to attempt full SCF converge on the final COSX grid, set COSX_MAXITER_FINAL to -1.'.format(scf_maxiter_post_screening))

    # has early_screening changed from True to False?
    early_screening_disabled = False

    # SCF iterations!
    count = {}
    for method in core.get_option('SCF', 'MESA_SCF_METHODS'):
        count[method] = 1

    SCFE_old = 0.0
    Dnorm = 0.0
    scf_iter_post_screening = 0
    while True:
        self.iteration_ += 1

        diis_performed = False
        soscf_performed = False
        self.frac_performed_ = False
        #self.MOM_performed_ = False  # redundant from common_init()

        self.save_density_and_energy()

        if efp_enabled:
            # EFP: Add efp contribution to Fock matrix
            self.H().copy(self.Horig)
            global mints_psi4_yo
            mints_psi4_yo = core.MintsHelper(self.basisset())
            Vefp = modify_Fock_induced(self.molecule().EFP, mints_psi4_yo, verbose=verbose - 1)
            Vefp = core.Matrix.from_array(Vefp)
            self.H().add(Vefp)

        SCFE = 0.0
        self.clear_external_potentials()

        # Two-electron contribution to Fock matrix from self.jk()
        core.timer_on("HF: Form G")
        self.form_G()
        core.timer_off("HF: Form G")

        # Check if special J/K construction algorithms were used
        incfock_performed = hasattr(self.jk(), "do_incfock_iter") and self.jk().do_incfock_iter()
        upcm = 0.0
        if core.get_option('SCF', 'PCM'):
            calc_type = core.PCM.CalcType.Total
            if core.get_option("PCM", "PCM_SCF_TYPE") == "SEPARATE":
                calc_type = core.PCM.CalcType.NucAndEle
            Dt = self.Da().clone()
            Dt.add(self.Db())
            upcm, Vpcm = self.get_PCM().compute_PCM_terms(Dt, calc_type)
            SCFE += upcm
            self.push_back_external_potential(Vpcm)
        self.set_variable("PCM POLARIZATION ENERGY", upcm)  # P::e PCM
        self.set_energies("PCM Polarization", upcm)

        uddx = 0.0
        if core.get_option('SCF', 'DDX'):
            Dt = self.Da().clone()
            Dt.add(self.Db())
            uddx, Vddx, self.ddx_state = self.ddx.get_solvation_contributions(Dt, self.ddx_state)
            SCFE += uddx
            self.push_back_external_potential(Vddx)
        self.set_variable("DD SOLVATION ENERGY", uddx)  # P::e DDX
        self.set_energies("DD Solvation Energy", uddx)

        upe = 0.0
        if core.get_option('SCF', 'PE'):
            Dt = self.Da().clone()
            Dt.add(self.Db())
            upe, Vpe = self.pe_state.get_pe_contribution(
                Dt, elec_only=False
            )
            SCFE += upe
            self.push_back_external_potential(Vpe)
        self.set_variable("PE ENERGY", upe)  # P::e PE
        self.set_energies("PE Energy", upe)

        core.timer_on("HF: Form F")
        # SAD: since we don't have orbitals yet, we might not be able
        # to form the real Fock matrix. Instead, build an initial one
        if (self.iteration_ == 0) and self.sad_:
            self.form_initial_F()
        else:
            self.form_F()
        core.timer_off("HF: Form F")

        if verbose > 3:
            self.Fa().print_out()
            self.Fb().print_out()

        SCFE += self.compute_E()
        if efp_enabled:
            global efp_Dt_psi4_yo

            # EFP: Add efp contribution to energy
            efp_Dt_psi4_yo = self.Da().clone()
            efp_Dt_psi4_yo.add(self.Db())
            SCFE += self.molecule().EFP.get_wavefunction_dependent_energy()

        self.set_energies("Total Energy", SCFE)
        core.set_variable("SCF ITERATION ENERGY", SCFE)
        self.iteration_energies.append(SCFE)

        Ediff = SCFE - SCFE_old
        SCFE_old = SCFE

        status = []

        # Check if we are doing SOSCF
        if (soscf_enabled and (self.iteration_ >= 3) and (Dnorm < core.get_option('SCF', 'SOSCF_START_CONVERGENCE'))):
            Dnorm = self.compute_orbital_gradient(False, core.get_option('SCF', 'DIIS_MAX_VECS'))
            diis_performed = False
            if self.functional().needs_xc():
                base_name = "SOKS, nmicro="
            else:
                base_name = "SOSCF, nmicro="

            if not _converged(Ediff, Dnorm, e_conv=e_conv, d_conv=d_conv):
                nmicro = self.soscf_update(core.get_option('SCF', 'SOSCF_CONV'),
                                           core.get_option('SCF', 'SOSCF_MIN_ITER'),
                                           core.get_option('SCF', 'SOSCF_MAX_ITER'),
                                           core.get_option('SCF', 'SOSCF_PRINT'))
                # if zero, the soscf call bounced for some reason
                soscf_performed = (nmicro > 0)

                if soscf_performed:
                    self.find_occupation()
                    status.append(base_name + str(nmicro))
                else:
                    if verbose > 0:
                        core.print_out("Did not take a SOSCF step, using normal convergence methods\n")

            else:
                # need to ensure orthogonal orbitals and set epsilon
                status.append(base_name + "conv")
                core.timer_on("HF: Form C")
                self.form_C()
                core.timer_off("HF: Form C")
                soscf_performed = True  # Stops DIIS

        if not soscf_performed:
            # Normal convergence procedures if we do not do SOSCF

            # SAD: form initial orbitals from the initial Fock matrix, and
            # reset the occupations. The reset is necessary because SAD
            # nalpha_ and nbeta_ are not guaranteed physical.
            # From here on, the density matrices are correct.
            if (self.iteration_ == 0) and self.sad_:
                self.form_initial_C()
                self.reset_occupation()
                self.find_occupation()

            else:
                # Run DIIS
                core.timer_on("HF: DIIS")
                Da_prev_out = self.Da().clone()
                Fa_out_prev = self.Fa().clone()
                if not self.same_a_b_orbs():
                    Db_prev_out = self.Db().clone()
                    Fb_out_prev = self.Fb().clone()

                diis_performed = False
                add_to_diis_subspace =  self.mesa_enabled or (self.diis_enabled_ and self.iteration_ >= self.diis_start_) or self.list_enabled

                Dnorm = self.compute_orbital_gradient(add_to_diis_subspace,
                                                        core.get_option('SCF', 'DIIS_MAX_VECS'),
                                                        f_ins,
                                                        d_ins,
                                                        v_ins)
                
                if core.get_option('SCF', 'MESA_ERROR') == "META" or core.get_option('SCF', 'MESA_ERROR') == "EMETA":
                    scf_conv_methods = core.get_option('SCF', 'MESA_SCF_METHODS')
                    scf_conv_method_errors = {}
                    all_berrors ={}
                    all_combination_errors = {}
                    normalized_reciprocal_errors = {}
                    column_sums = {}

                    metrics = ["INTRA_D", "COMMUTATOR"]

                    #metrics = ["INTRA_D", "COMMUTATOR","INTRA_D", "COMMUTATOR","INTER_F"]
                    scf_error_metrics = [
                        "INTRA_D",
                        #"INTER_F",
                        #"INTER_D",
                        "COMMUTATOR",
                        #"D_F"
                    ]

                    for metric in scf_error_metrics:
                        all_berrors[metric] = {'error': float('inf'), 'method': None}
                    for metric in scf_error_metrics:
                        all_combination_errors[metric] = {}
                    for metric in scf_error_metrics:
                        normalized_reciprocal_errors[metric] = {}
                    for method in scf_conv_methods:
                        column_sums[method] = 0 

                    best_error_val = 1.0e20
                    best_method = None
                    best_metric = None

                    # for split later 
                    best_Fa = None
                    best_Fb = None
                    best_Da = None
                    best_Db = None

                    #save a copy of the *current* Fa/Fb/Da/Db for multiple tries ;P
                    Fa_before = self.Fa().clone()
                    Fb_before = self.Fb().clone()
                    Da_before = self.Da().clone()
                    Db_before = self.Db().clone()

                    if len(self.rms) >= 2:
                        lice_2 = self.rms[-2]
                        lice_1 = self.rms[-1]

                    for scf_conv_method in scf_conv_methods:
                        if not self.same_a_b_orbs():
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja())

                        D_in_a = self.Da().clone()
                        if not self.same_a_b_orbs():
                            D_in_b = self.Db().clone()

                        self.form_C()
                        self.form_D()

                        for metric in scf_error_metrics:

                            error = None
                            if not self.same_a_b_orbs():
                                error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())

                                if metric == "INTRA_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())
                                elif metric == "INTER_F":
                                    error_a = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                    error_b = self.form_FoutmFin(self.Fb().clone(), Fb_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    #error_a = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    #error_b = self.form_FDSmSDF(self.Fb().clone(), self.Db().clone())
                                    error_a = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                    error_b = self.form_FDSmSDF(self.Fb().clone(), D_in_b.clone())
                                elif metric == "D_F":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_add_a = self.form_MaddM(self.Fa().clone(), f_prev_a)
                                        f_add_b = self.form_MaddM(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_add_a = self.Fa().clone()
                                        f_add_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_add_a)
                                    error_b = d_diff_b.chain_dot(f_add_b)                                   
                                elif metric == "D_F-":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_diff_a = self.form_FoutmFin(self.Fa().clone(), f_prev_a)
                                        f_diff_b = self.form_FoutmFin(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_diff_a = self.Fa().clone()
                                        f_diff_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_diff_a)
                                    error_b = d_diff_b.chain_dot(f_diff_b)

                                error = np.sqrt(0.5 * (error_a.rms()**2 + error_b.rms()**2))
                                #error = 0.5 * (error_a.rms() + error_b.rms())

                            else:
                                if metric == "INTRA_D":
                                    error = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                elif metric == "INTER_F":
                                    error = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                elif metric == "INTER_D":
                                    error = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    # error = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    error = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                elif metric == "D_F":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    f_add = self.form_MaddM(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()
                                    error = d_diff.chain_dot(f_add)
                                elif metric == "D_F-":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())                                    
                                    f_diff = self.form_FoutmFin(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()                                    
                                    error = d_diff.chain_dot(f_diff)

                                error = error.rms()    
                            

                            # compare vs. best
                            # here done with various (1) "aggregation" methods:


                            #This part is deprecated (FOR testing )
                            # if metric == "INTRA_D":
                            #     error = 0.07*error
                            # elif metric == "INTER_F":
                            #     if self.iteration_ <= 4:
                            #         error = 2.0e14*error
                            #     else:
                            #         error = 1.0e14*error
                            # elif metric == "INTER_D":
                            #     error = 0.5*error
                            # elif metric == "COMMUTATOR":
                            #     error = 1.0e14*error
                            # elif metric == "D_F":
                            #     error = 0.2*error

                            all_combination_errors[metric][scf_conv_method] = error


                            if error < all_berrors[metric]['error']:
                                all_berrors[metric]['error'] = error
                                all_berrors[metric]['method'] = scf_conv_method

                            previous_best_metric = None
                            previous_method = None 
                            if len(self.best_metrics) > 0:
                                previous_best_metric = self.best_metrics[-1]
                                previous_method = self.methods[-1]

                            # if (metric == previous_best_metric and (lice_1 > lice_2 or np.abs(np.log(lice_1) - np.log(lice_2)) < 0.06)):
                            #     error = 2*count[scf_conv_method]*error
                            #     count[scf_conv_method] += 1

                            if core.get_option('SCF', 'MESA_ERROR') == "META":
                                if error < best_error_val:
                                    if (lice_1 > lice_2 or np.abs(np.log(lice_1) - np.log(lice_2)) < 0.06) and previous_best_metric == metric:
                                        pass
                                    best_error_val = error
                                    best_method = scf_conv_method
                                    best_metric = metric
                                    scf_conv_method_errors[best_method] = error


                        # revert Fa/Fb/Da/Db
                        self.Fa().copy(Fa_before)
                        self.Fb().copy(Fb_before)
                        self.Da().copy(Da_before)
                        self.Db().copy(Db_before)

                    # try ensemble
                    if core.get_option('SCF', 'MESA_ERROR') == "EMETA":
                        # metric_min_errors = {}
                        # for metric in all_combination_errors:
                        #     min_error = float('inf')
                        #     for method in all_combination_errors[metric]:
                        #         if all_combination_errors[metric][method] < min_error:
                        #             min_error = all_combination_errors[metric][method]
                        #     metric_min_errors[metric] = min_error

                        # sorted_metrics = sorted(metric_min_errors.items(), key=lambda x: x[1])

                        # best_metric = sorted_metrics[0][0]
                        # second_best_metric = sorted_metrics[1][0]
                        # third_best_metric = sorted_metrics[2][0]

                        # method_combined_errors = {}
                        # metric_avg_errors = {}

                        # for metric in all_combination_errors:
                        #     error_values = list(all_combination_errors[metric].values())
                            
                        #     if error_values:
                        #         avg_error = sum(error_values) / len(error_values)
                        #         metric_avg_errors[metric] = avg_error

                        # for method in all_combination_errors[best_metric]:

                        #     error1 = all_combination_errors[best_metric].get(method, float('inf'))
                        #     error2 = all_combination_errors[second_best_metric].get(method, float('inf'))
                        #     error3 = all_combination_errors[third_best_metric].get(method, float('inf'))

                        #     method_combined_errors[method] = 10*error1*metric_avg_errors[best_metric] 
                        #     +error2*metric_avg_errors[second_best_metric]
                        #     +0.1*error3*metric_avg_errors[third_best_metric]

                        for method in all_combination_errors["INTRA_D"]:
                            for metric in metrics:
                                if all_combination_errors[metric][method] == 0:
                                    normalized_reciprocal_errors[metric][method] = 1.0
                                else:
                                    normalized_reciprocal_errors[metric][method] = min(all_combination_errors[metric].values())/all_combination_errors[metric][method]
                                column_sums[method] += normalized_reciprocal_errors[metric][method]

                        best_method = max(column_sums, key=column_sums.get)

                        scf_conv_method_errors[best_method] = column_sums[best_method]

                    if best_method is not None:
                        if not self.same_a_b_orbs():
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fa(), self.Da(), self.Ja())

                        self.form_C()
                        self.form_D()

                    all_errors_formatted = {metric: f"{data['error']} ({data['method']})" for metric, data in all_berrors.items()}
                    current_mesa_scf_conv_method = list(scf_conv_method_errors.keys())[np.argmin(list(scf_conv_method_errors.values()))]

                    if core.get_option('SCF', 'MESA_ERROR') == "EMETA":
                        # core.print_out(f" Best combined method: {best_method} with sum of errors {best_error_val}\n")
                        # core.print_out(f" Error for {best_metric}: {all_combination_errors[best_metric][best_method]}\n")
                        # core.print_out(f" Error for {second_best_metric}: {all_combination_errors[second_best_metric][best_method]}\n")
                        # core.print_out(f" Error for {third_best_metric}: {all_combination_errors[third_best_metric][best_method]}\n")
                        core.print_out(f"\n Most votes {best_method}: {column_sums[best_method]}")
                        methods = list(column_sums.keys())
                        methods.sort(key=lambda x: column_sums[x], reverse=True)
                        col_width = 14
                        output_str = "\n"
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        output_str += "  {:<{width}}".format("Method", width=col_width)
                        for metric in metrics:
                            output_str += " | {:<{width}}".format(metric, width=col_width-2)
                        output_str += " | {:<{width}}\n".format("Total", width=col_width-2)
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        for method in methods:
                            output_str += "  {:<{width}}".format(method, width=col_width)
                            for metric in metrics:
                                if metric in normalized_reciprocal_errors and method in normalized_reciprocal_errors[metric]:
                                    value = normalized_reciprocal_errors[metric][method]
                                    output_str += " | {:<{width}.4f}".format(value, width=col_width-2)
                                else:
                                    output_str += " | {:<{width}}".format("N/A", width=col_width-2)
                            output_str += " | {:<{width}.4f}\n".format(column_sums[method], width=col_width-2)
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        output_str += f"  Best method: {best_method} with total score {column_sums[best_method]:.4f}\n"
                        core.print_out(output_str)
                    else:
                        core.print_out(f"[META-MESA] Best method this iteration is {best_method}, w/ metric {best_metric}, META-MESA error = {best_error_val}\n, all: {all_errors_formatted}\n")                    
                    self.best_metrics.append(best_metric)

                if core.get_option('SCF', 'MESA_ERROR') == "ZMETA":
                    scf_conv_methods = core.get_option('SCF', 'MESA_SCF_METHODS')
                    scf_conv_method_errors = {}
                    all_berrors = {}
                    all_combination_errors = {}
                    zmeta_vector_errors = {}  # Stores raw Psi4 Matrices
                    
                    normalized_reciprocal_errors = {}
                    column_sums = {}

                    metrics = [
                        "INTRA_D",
                        "INTER_F",       
                        "COMMUTATOR",
                        #"D_F"
                        #,"INTER_D",
                    ]
                    scf_error_metrics = metrics


                    if Dnorm > 1.0e-3:
                        scf_error_metrics = ["INTRA_D", "INTER_F", "COMMUTATOR"
                                             #, "D_F-", "INTER_D"
                                             ]
                    else:
                        scf_error_metrics = ["INTRA_D", "COMMUTATOR"]


                    for metric in scf_error_metrics:
                        all_berrors[metric] = {'error': float('inf'), 'method': None}
                        all_combination_errors[metric] = {}
                        zmeta_vector_errors[metric] = {}
                        normalized_reciprocal_errors[metric] = {}
                        
                    for method in scf_conv_methods:
                        column_sums[method] = 0 

                    best_error_val = 1.0e20
                    best_method = None
                    best_metric = None

                    Fa_before = self.Fa().clone()
                    Fb_before = self.Fb().clone()
                    Da_before = self.Da().clone()
                    Db_before = self.Db().clone()

                    if len(self.rms) >= 2:
                        lice_2 = self.rms[-2]
                        lice_1 = self.rms[-1]
                        
                    is_unrestricted = not self.same_a_b_orbs()

                    for scf_conv_method in scf_conv_methods:
                        if is_unrestricted:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja())

                        D_in_a = self.Da().clone()
                        if is_unrestricted:
                            D_in_b = self.Db().clone()

                        self.form_C()
                        self.form_D()

                        for metric in scf_error_metrics:

                            error_val = None

                            if is_unrestricted:
                                if metric == "INTRA_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())
                                elif metric == "INTER_F":
                                    error_a = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                    error_b = self.form_FoutmFin(self.Fb().clone(), Fb_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    #error_a = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    #error_b = self.form_FDSmSDF(self.Fb().clone(), self.Db().clone())
                                    error_a = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                    error_b = self.form_FDSmSDF(self.Fb().clone(), D_in_b.clone())
                                elif metric == "D_F":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_add_a = self.form_MaddM(self.Fa().clone(), f_prev_a)
                                        f_add_b = self.form_MaddM(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_add_a = self.Fa().clone()
                                        f_add_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_add_a)
                                    error_b = d_diff_b.chain_dot(f_add_b)                                   
                                elif metric == "D_F-":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_diff_a = self.form_FoutmFin(self.Fa().clone(), f_prev_a)
                                        f_diff_b = self.form_FoutmFin(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_diff_a = self.Fa().clone()
                                        f_diff_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_diff_a)
                                    error_b = d_diff_b.chain_dot(f_diff_b)

                                # no numpy no!
                                zmeta_vector_errors[metric][scf_conv_method] = (error_a, error_b)
                                error_val = np.sqrt(0.5 * (error_a.rms()**2 + error_b.rms()**2))

                            else:
                                if metric == "INTRA_D":
                                    error_mat = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                elif metric == "INTER_F":
                                    error_mat = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_mat = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    # error = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    error_mat = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                elif metric == "D_F":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    f_add = self.form_MaddM(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()
                                    error_mat = d_diff.chain_dot(f_add)
                                elif metric == "D_F-":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())                                    
                                    f_diff = self.form_FoutmFin(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()                                    
                                    error_mat = d_diff.chain_dot(f_diff)

                                #No no numpy!
                                zmeta_vector_errors[metric][scf_conv_method] = error_mat
                                error_val = error_mat.rms()                               

                            all_combination_errors[metric][scf_conv_method] = error_val

                            if error_val < all_berrors[metric]['error']:
                                all_berrors[metric]['error'] = error_val
                                all_berrors[metric]['method'] = scf_conv_method

                        self.Fa().copy(Fa_before)
                        self.Fb().copy(Fb_before)
                        self.Da().copy(Da_before)
                        self.Db().copy(Db_before)

                    if core.get_option('SCF', 'MESA_ERROR') == "ZMETA":
                        # Pass the matrices and the restriction flag
                        best_method, lambda_sq_scores = zmeta_step(zmeta_vector_errors, all_combination_errors, is_unrestricted)
                        scf_conv_method_errors[best_method] = lambda_sq_scores[best_method]

                    if best_method is not None:
                        if is_unrestricted:
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fa(), self.Da(), self.Ja())

                        self.form_C()
                        self.form_D()

                    all_errors_formatted = {metric: f"{data['error']} ({data['method']})" for metric, data in all_berrors.items()}
                    current_mesa_scf_conv_method = list(scf_conv_method_errors.keys())[np.argmin(list(scf_conv_method_errors.values()))]
                    
                    if core.get_option('SCF', 'MESA_ERROR') == "ZMETA":
                        core.print_out(f"\n Lowest composite error {best_method}: {lambda_sq_scores[best_method]:.4e}")
                        
                        methods = list(lambda_sq_scores.keys())
                        methods.sort(key=lambda x: lambda_sq_scores[x])
                        
                        col_width = 14
                        output_str = "\n"
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        output_str += "  {:<{width}}".format("Method", width=col_width)
                        for metric in scf_error_metrics:
                            output_str += " | {:<{width}}".format(metric, width=col_width-2)
                        output_str += " | {:<{width}}\n".format("Total (Lam^2)", width=col_width-2)
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        
                        for method in methods:
                            output_str += "  {:<{width}}".format(method, width=col_width)
                            for metric in scf_error_metrics:
                                value = all_combination_errors[metric][method]
                                output_str += " | {:<{width}.4e}".format(value, width=col_width-2)
                            output_str += " | {:<{width}.4e}\n".format(lambda_sq_scores[method], width=col_width-2)
                            
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        output_str += f"  Best method: {best_method} with composite score {lambda_sq_scores[best_method]:.4e}\n"
                        core.print_out(output_str)

                if core.get_option('SCF', 'MESA_ERROR') == "YMETA":
                    scf_conv_methods = core.get_option('SCF', 'MESA_SCF_METHODS')
                    ymeta_vector_errors = {}
                    Fa_candidates = {}  # store generated input Fock matrices
                    Fb_candidates = {}
                    Da_candidates = {}  # store matching generated input density matrices
                    Db_candidates = {}
                    Ja_candidates = {}  # store matching generated input J/Hartree matrices when available
                    Jb_candidates = {}

                    # Define the metric to optimizing against. 
                    target_ymeta_metric = "COMMUTATOR"
                    
                    scf_error_metrics = [target_ymeta_metric]
                    for metric in scf_error_metrics:
                        ymeta_vector_errors[metric] = {}

                    Fa_before = self.Fa().clone()
                    Fb_before = self.Fb().clone()
                    Da_before = self.Da().clone()
                    Db_before = self.Db().clone()
                    Ja_before = self.Ja().clone() if self.Ja() else None
                    Jb_before = self.Jb().clone() if self.Jb() else None
                        
                    is_unrestricted = not self.same_a_b_orbs()

                    for scf_conv_method in scf_conv_methods:
                        # 1. EXTRAPOLATE (Modifies self.Fa() in place)
                        if is_unrestricted:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja())

                        # 2. SAVE THE CANDIDATE INPUT MATRICES
                        # The hybrid update is propagated through F_in, but LIST-style
                        # history also uses the matching D_in and J/V_H input histories.
                        Fa_candidates[scf_conv_method] = self.Fa().clone()
                        D_in_a = self.Da().clone()
                        Da_candidates[scf_conv_method] = D_in_a.clone()
                        Ja_candidates[scf_conv_method] = self.Ja().clone() if self.Ja() else None

                        if is_unrestricted:
                            Fb_candidates[scf_conv_method] = self.Fb().clone()
                            D_in_b = self.Db().clone()
                            Db_candidates[scf_conv_method] = D_in_b.clone()
                            Jb_candidates[scf_conv_method] = self.Jb().clone() if self.Jb() else None

                        # Note: We must form C and D based on the current candidate Fock 
                        # to properly calculate the Commutator error matrix for this candidate
                        self.form_C()
                        self.form_D()

                        # 3. CALCULATE THE ERROR METRIC
                        for metric in scf_error_metrics:
                            if is_unrestricted:
                                if metric == "INTRA_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())
                                elif metric == "INTER_F":
                                    error_a = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                    error_b = self.form_FoutmFin(self.Fb().clone(), Fb_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    #error_a = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    #error_b = self.form_FDSmSDF(self.Fb().clone(), self.Db().clone())
                                    error_a = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                    error_b = self.form_FDSmSDF(self.Fb().clone(), D_in_b.clone())
                                elif metric == "D_F":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_add_a = self.form_MaddM(self.Fa().clone(), f_prev_a)
                                        f_add_b = self.form_MaddM(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_add_a = self.Fa().clone()
                                        f_add_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_add_a)
                                    error_b = d_diff_b.chain_dot(f_add_b)                                   
                                elif metric == "D_F-":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_diff_a = self.form_FoutmFin(self.Fa().clone(), f_prev_a)
                                        f_diff_b = self.form_FoutmFin(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_diff_a = self.Fa().clone()
                                        f_diff_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_diff_a)
                                    error_b = d_diff_b.chain_dot(f_diff_b)

                                # no numpy no!
                                ymeta_vector_errors[metric][scf_conv_method] = (error_a, error_b)
                                error_val = np.sqrt(0.5 * (error_a.rms()**2 + error_b.rms()**2))

                            else:
                                if metric == "INTRA_D":
                                    error_mat = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                elif metric == "INTER_F":
                                    error_mat = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_mat = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    # error = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    error_mat = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                elif metric == "D_F":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    f_add = self.form_MaddM(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()
                                    error_mat = d_diff.chain_dot(f_add)
                                elif metric == "D_F-":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())                                    
                                    f_diff = self.form_FoutmFin(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()                                    
                                    error_mat = d_diff.chain_dot(f_diff)

                                #No no numpy!
                                ymeta_vector_errors[metric][scf_conv_method] = error_mat
                                error_val = error_mat.rms()

                        # Revert the base states for the next candidate method
                        self.Fa().copy(Fa_before)
                        self.Fb().copy(Fb_before)
                        self.Da().copy(Da_before)
                        self.Db().copy(Db_before)
                        if Ja_before is not None and self.Ja():
                            self.Ja().copy(Ja_before)
                        if Jb_before is not None and self.Jb():
                            self.Jb().copy(Jb_before)

                    # 4. PERFORM YMETA OPTIMIZATION
                    if core.get_option('SCF', 'MESA_ERROR') == "YMETA":
                        hybrid_lambda_sq, method_weights = ymeta_step(ymeta_vector_errors, target_ymeta_metric, is_unrestricted)
                        
                        # 5. CONSTRUCT THE HYBRID INPUT MATRICES
                        # Create empty matrices that perfectly retain point-group symmetry.
                        F_hybrid_a = self.Fa().clone()
                        F_hybrid_a.zero()
                        D_hybrid_a = self.Da().clone()
                        D_hybrid_a.zero()
                        J_hybrid_a = self.Ja().clone() if self.Ja() and all(Ja_candidates.get(method) is not None for method in method_weights) else None
                        if J_hybrid_a is not None:
                            J_hybrid_a.zero()

                        if is_unrestricted:
                            F_hybrid_b = self.Fb().clone()
                            F_hybrid_b.zero()
                            D_hybrid_b = self.Db().clone()
                            D_hybrid_b.zero()
                            J_hybrid_b = self.Jb().clone() if self.Jb() and all(Jb_candidates.get(method) is not None for method in method_weights) else None
                            if J_hybrid_b is not None:
                                J_hybrid_b.zero()

                        # Linearly combine them using pure Psi4 C++ methods. The Fock
                        # matrix is propagated; the matching D/J histories are restored
                        # before f_ins/d_ins/v_ins are appended below.
                        for method, weight in method_weights.items():
                            temp_Fa = Fa_candidates[method].clone()
                            temp_Fa.scale(weight)
                            F_hybrid_a.add(temp_Fa)

                            temp_Da = Da_candidates[method].clone()
                            temp_Da.scale(weight)
                            D_hybrid_a.add(temp_Da)

                            if J_hybrid_a is not None:
                                temp_Ja = Ja_candidates[method].clone()
                                temp_Ja.scale(weight)
                                J_hybrid_a.add(temp_Ja)
                            
                            if is_unrestricted:
                                temp_Fb = Fb_candidates[method].clone()
                                temp_Fb.scale(weight)
                                F_hybrid_b.add(temp_Fb)

                                temp_Db = Db_candidates[method].clone()
                                temp_Db.scale(weight)
                                D_hybrid_b.add(temp_Db)

                                if J_hybrid_b is not None:
                                    temp_Jb = Jb_candidates[method].clone()
                                    temp_Jb.scale(weight)
                                    J_hybrid_b.add(temp_Jb)
                        
                        # 6. INJECT HYBRID MATRICES BACK INTO SCF
                        self.Fa().copy(F_hybrid_a)
                        self.Da().copy(D_hybrid_a)
                        if J_hybrid_a is not None and self.Ja():
                            self.Ja().copy(J_hybrid_a)
                        if is_unrestricted:
                            self.Fb().copy(F_hybrid_b)
                            self.Db().copy(D_hybrid_b)
                            if J_hybrid_b is not None and self.Jb():
                                self.Jb().copy(J_hybrid_b)

                        # Form the final C and D matrices from our brand new hybrid Fock matrix
                        self.form_C()
                        self.form_D()

                        self._saved_hybrid_Fa = F_hybrid_a.clone()
                        self._saved_hybrid_Da = D_hybrid_a.clone()
                        self._saved_hybrid_Ja = J_hybrid_a.clone() if J_hybrid_a is not None else None
                        if is_unrestricted:
                            self._saved_hybrid_Fb = F_hybrid_b.clone()
                            self._saved_hybrid_Db = D_hybrid_b.clone()
                            self._saved_hybrid_Jb = J_hybrid_b.clone() if J_hybrid_b is not None else None

                        # --- YMETA PRINT BLOCK ---
                        core.print_out(f"\n [{core.get_option('SCF', 'MESA_ERROR')}] Optimized Hybrid Fock Matrix for metric: {target_ymeta_metric}\n")
                        core.print_out(f"  Hybrid Minimized Error (Lam^2): {hybrid_lambda_sq:.4e}\n")
                        core.print_out("  ----------------------------------------\n")
                        core.print_out("  {:<15} | {:<20}\n".format("Method", "Hybrid Weight (d_j)"))
                        core.print_out("  ----------------------------------------\n")
                        
                        # Sort by highest weight contribution
                        sorted_methods = sorted(method_weights.keys(), key=lambda k: method_weights[k], reverse=True)
                        for method in sorted_methods:
                            core.print_out("  {:<15} | {:>15.4f} %\n".format(method, method_weights[method] * 100))
                        core.print_out("  ----------------------------------------\n")

                if core.get_option('SCF', 'MESA_ERROR') == "ZYMETA":
                    scf_conv_methods = core.get_option('SCF', 'MESA_SCF_METHODS')
                    
                    zymeta_vector_errors = {}
                    Fa_candidates = {}
                    Fb_candidates = {}
                    Da_candidates = {}
                    Db_candidates = {}
                    Ja_candidates = {}
                    Jb_candidates = {}

                    # Define the multiple metrics you want to simultaneously satisfy
                    #target_zymeta_metrics = ["INTRA_D", "INTER_D", "D_F", "COMMUTATOR", "INTER_F"]

                    #target_zymeta_metrics = ["D_F"]

                    if Dnorm > 1.0e-3:
                        target_zymeta_metrics = ["INTRA_D", "COMMUTATOR", "INTER_F" 
                                                 #"D_F-", "INTER_D"
                                                 ]
                    else:
                        target_zymeta_metrics = ["INTRA_D", "COMMUTATOR"
                                                 #, "INTER_F"
                                                 ]
                    

                    #target_zymeta_metrics = ["D_F"]

                    for metric in target_zymeta_metrics:
                        zymeta_vector_errors[metric] = {}

                    Fa_before = self.Fa().clone()
                    Fb_before = self.Fb().clone()
                    Da_before = self.Da().clone()
                    Db_before = self.Db().clone()
                    Ja_before = self.Ja().clone() if self.Ja() else None
                    Jb_before = self.Jb().clone() if self.Jb() else None
                        
                    is_unrestricted = not self.same_a_b_orbs()

                    for scf_conv_method in scf_conv_methods:
                        # 1. EXTRAPOLATE
                        if is_unrestricted:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja())

                        # 2. SAVE THE CANDIDATE INPUT MATRICES
                        # The hybrid update is propagated through F_in, but LIST-style
                        # history also uses the matching D_in and J/V_H input histories.
                        Fa_candidates[scf_conv_method] = self.Fa().clone()
                        D_in_a = self.Da().clone()
                        Da_candidates[scf_conv_method] = D_in_a.clone()
                        Ja_candidates[scf_conv_method] = self.Ja().clone() if self.Ja() else None

                        if is_unrestricted:
                            Fb_candidates[scf_conv_method] = self.Fb().clone()
                            D_in_b = self.Db().clone()
                            Db_candidates[scf_conv_method] = D_in_b.clone()
                            Jb_candidates[scf_conv_method] = self.Jb().clone() if self.Jb() else None

                        self.form_C()
                        self.form_D()

                        # 3. CALCULATE ALL ERROR METRICS                                
                        for metric in target_zymeta_metrics:
                            if is_unrestricted:
                                if metric == "INTRA_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())
                                elif metric == "INTER_F":
                                    error_a = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                    error_b = self.form_FoutmFin(self.Fb().clone(), Fb_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    #error_a = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    #error_b = self.form_FDSmSDF(self.Fb().clone(), self.Db().clone())
                                    error_a = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                    error_b = self.form_FDSmSDF(self.Fb().clone(), D_in_b.clone())
                                elif metric == "D_F":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_add_a = self.form_MaddM(self.Fa().clone(), f_prev_a)
                                        f_add_b = self.form_MaddM(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_add_a = self.Fa().clone()
                                        f_add_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_add_a)
                                    error_b = d_diff_b.chain_dot(f_add_b)                                   
                                elif metric == "D_F-":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    if len(f_ins) > 0:
                                        f_prev_a = f_ins[-1][0].clone()
                                        f_prev_b = f_ins[-1][1].clone()
                                        f_diff_a = self.form_FoutmFin(self.Fa().clone(), f_prev_a)
                                        f_diff_b = self.form_FoutmFin(self.Fb().clone(), f_prev_b)
                                    else:
                                        f_diff_a = self.Fa().clone()
                                        f_diff_b = self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_diff_a)
                                    error_b = d_diff_b.chain_dot(f_diff_b)

                                # no numpy no!
                                zymeta_vector_errors[metric][scf_conv_method] = (error_a, error_b)
                                error_val = np.sqrt(0.5 * (error_a.rms()**2 + error_b.rms()**2))

                            else:
                                if metric == "INTRA_D":
                                    error_mat = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                elif metric == "INTER_F":
                                    error_mat = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                elif metric == "INTER_D":
                                    error_mat = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                elif metric == "COMMUTATOR":
                                    # error = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                    error_mat = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                elif metric == "D_F":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    f_add = self.form_MaddM(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()
                                    error_mat = d_diff.chain_dot(f_add)
                                elif metric == "D_F-":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())                                    
                                    f_diff = self.form_FoutmFin(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()                                    
                                    error_mat = d_diff.chain_dot(f_diff)

                                zymeta_vector_errors[metric][scf_conv_method] = error_mat
                                error_val = error_mat.rms()

                        # Revert the base states
                        self.Fa().copy(Fa_before)
                        self.Fb().copy(Fb_before)
                        self.Da().copy(Da_before)
                        self.Db().copy(Db_before)
                        if Ja_before is not None and self.Ja():
                            self.Ja().copy(Ja_before)
                        if Jb_before is not None and self.Jb():
                            self.Jb().copy(Jb_before)

                    # 4. PERFORM ZYMETA OPTIMIZATION
                    if core.get_option('SCF', 'MESA_ERROR') == "ZYMETA":
                        hybrid_lambda_sq, method_weights = zymeta_optimize(zymeta_vector_errors, is_unrestricted)
                        
                        # 5. CONSTRUCT THE HYBRID INPUT MATRICES
                        F_hybrid_a = self.Fa().clone()
                        F_hybrid_a.zero()
                        D_hybrid_a = self.Da().clone()
                        D_hybrid_a.zero()
                        J_hybrid_a = self.Ja().clone() if self.Ja() and all(Ja_candidates.get(method) is not None for method in method_weights) else None
                        if J_hybrid_a is not None:
                            J_hybrid_a.zero()

                        if is_unrestricted:
                            F_hybrid_b = self.Fb().clone()
                            F_hybrid_b.zero()
                            D_hybrid_b = self.Db().clone()
                            D_hybrid_b.zero()
                            J_hybrid_b = self.Jb().clone() if self.Jb() and all(Jb_candidates.get(method) is not None for method in method_weights) else None
                            if J_hybrid_b is not None:
                                J_hybrid_b.zero()

                        for method, weight in method_weights.items():
                            temp_Fa = Fa_candidates[method].clone()
                            temp_Fa.scale(weight)
                            F_hybrid_a.add(temp_Fa)

                            temp_Da = Da_candidates[method].clone()
                            temp_Da.scale(weight)
                            D_hybrid_a.add(temp_Da)

                            if J_hybrid_a is not None:
                                temp_Ja = Ja_candidates[method].clone()
                                temp_Ja.scale(weight)
                                J_hybrid_a.add(temp_Ja)
                            
                            if is_unrestricted:
                                temp_Fb = Fb_candidates[method].clone()
                                temp_Fb.scale(weight)
                                F_hybrid_b.add(temp_Fb)

                                temp_Db = Db_candidates[method].clone()
                                temp_Db.scale(weight)
                                D_hybrid_b.add(temp_Db)

                                if J_hybrid_b is not None:
                                    temp_Jb = Jb_candidates[method].clone()
                                    temp_Jb.scale(weight)
                                    J_hybrid_b.add(temp_Jb)
                        
                        # 6. INJECT HYBRID MATRICES
                        self.Fa().copy(F_hybrid_a)
                        self.Da().copy(D_hybrid_a)
                        if J_hybrid_a is not None and self.Ja():
                            self.Ja().copy(J_hybrid_a)
                        if is_unrestricted:
                            self.Fb().copy(F_hybrid_b)
                            self.Db().copy(D_hybrid_b)
                            if J_hybrid_b is not None and self.Jb():
                                self.Jb().copy(J_hybrid_b)

                        self.form_C()
                        self.form_D()

                        self._saved_hybrid_Fa = F_hybrid_a.clone()
                        self._saved_hybrid_Da = D_hybrid_a.clone()
                        self._saved_hybrid_Ja = J_hybrid_a.clone() if J_hybrid_a is not None else None
                        if is_unrestricted:
                            self._saved_hybrid_Fb = F_hybrid_b.clone()
                            self._saved_hybrid_Db = D_hybrid_b.clone()
                            self._saved_hybrid_Jb = J_hybrid_b.clone() if J_hybrid_b is not None else None

                        # --- ZYMETA PRINT BLOCK ---
                        core.print_out(f"\n [{core.get_option('SCF', 'MESA_ERROR')}] Multi-Metric Hybrid (Metrics: {len(target_zymeta_metrics)+1})\n")
                        core.print_out(f"  Hybrid Minimized Error (Lam^2): {hybrid_lambda_sq:.4e}\n")
                        core.print_out("  ----------------------------------------\n")
                        
                        sorted_methods = sorted(method_weights.keys(), key=lambda k: method_weights[k], reverse=True)
                        for method in sorted_methods:
                            core.print_out("  {:<15} | {:>15.4f} %\n".format(method, method_weights[method] * 100))
                        core.print_out("  ----------------------------------------\n")

                if core.get_option('SCF', 'MESA_ERROR') == "LMETA":
                    scf_conv_methods = core.get_option('SCF', 'MESA_SCF_METHODS')
                    scf_conv_method_errors = {}
                    all_berrors ={}
                    all_combination_errors = {}
                    normalized_reciprocal_errors = {}
                    column_sums = {}

                    metrics = ["INTRA_D", "COMMUTATOR"]
                    scf_error_metrics = [
                        "INTRA_D",
                        "COMMUTATOR"
                        #,"D_F"
                        #,"INTER_F"
                        #,"INTER_D"
                    ]


                    X = self.diis_manager_.Xmetric
                    S = self.diis_manager_.Smetric

                    def ortho(M, density):
                        """Return Xᵀ M X  or  Xᵀ S M S X."""
                        M = M.clone()
                        if density:
                            M = core.triplet(S, M, S)
                        M.transform(X)
                        return M
                    

                    for metric in scf_error_metrics:
                        all_berrors[metric] = {'error': float('inf'), 'method': None}
                    for metric in scf_error_metrics:
                        all_combination_errors[metric] = {}
                    for metric in scf_error_metrics:
                        normalized_reciprocal_errors[metric] = {}
                    for method in scf_conv_methods:
                        column_sums[method] = 0 

                    best_error_val = 1.0e20
                    best_method = None
                    best_metric = None

                    # for split later 
                    best_Fa = None
                    best_Fb = None
                    best_Da = None
                    best_Db = None

                    #save a copy of the *current* Fa/Fb/Da/Db for multiple tries ;P
                    Fa_before = self.Fa().clone()
                    Fb_before = self.Fb().clone()
                    Da_before = self.Da().clone()
                    Db_before = self.Db().clone()

                    if len(self.rms) >= 2:
                        lice_2 = self.rms[-2]
                        lice_1 = self.rms[-1]

                    for scf_conv_method in scf_conv_methods:
                        if not self.same_a_b_orbs():
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method, self.Fa(), self.Da(), self.Ja())

                        D_in_a = self.Da().clone()
                        if not self.same_a_b_orbs():
                            D_in_b = self.Db().clone()

                        self.form_C()
                        self.form_D()

                        for metric in scf_error_metrics:

                            error = None
                            if not self.same_a_b_orbs():
                                # I want to note that I've only changed the RHF version here to add the ortho. This won't work for UHF right now
                                error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())

                                if metric == "INTRA_D":
                                    error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                    error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())
                                elif metric == "INTER_F":
                                    error_a = self.form_FoutmFin(Fa_out_prev.clone(), self.Fa().clone())
                                    error_b = self.form_FoutmFin(Fb_out_prev.clone(), self.Fb().clone())
                                elif metric == "INTER_D":
                                    error_a = self.form_FoutmFin(Da_prev_out.clone(), self.Da().clone())
                                    error_b = self.form_FoutmFin(Db_prev_out.clone(), self.Db().clone())

                                elif metric == "D_F":
                                    d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                    f_add_a = self.form_MaddM(self.Fa().clone(), Fa_out_prev.clone()) if len(f_ins) > 0 else self.Fa().clone()
                                    f_add_b = self.form_MaddM(self.Fb().clone(), Fb_out_prev.clone()) if len(f_ins) > 0 else self.Fb().clone()
                                    error_a = d_diff_a.chain_dot(f_add_a)
                                    error_b = d_diff_b.chain_dot(f_add_b)
                                error = np.sqrt(0.5 * (error_a.rms()**2 + error_b.rms()**2))
                                #error = 0.5 * (error_a.rms() + error_b.rms())

                            else:
                                #error = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                if metric == "INTRA_D":
                                    error = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                    error = ortho(error, density=True)
                                elif metric == "INTER_F":
                                    error = self.form_FoutmFin(Fa_out_prev.clone(), self.Fa().clone())
                                    error = ortho(error, density=False)
                                elif metric == "INTER_D":
                                    error = self.form_FoutmFin(Da_prev_out.clone(), self.Da().clone())
                                    error = ortho(error, density=True)
                                elif metric == "COMMUTATOR":
                                    error = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                    error = ortho(error, density=False)
                                elif metric == "D_F":
                                    d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                    f_add = self.form_MaddM(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()
                                    error = d_diff.chain_dot(f_add)
                                    error = ortho(error, density=False)  

                            error = (error.rms())
                            all_combination_errors[metric][scf_conv_method] = error

                        
                            if error < all_berrors[metric]['error']:
                                all_berrors[metric]['error'] = error
                                all_berrors[metric]['method'] = scf_conv_method

                            previous_best_metric = None
                            previous_method = None 
                            if len(self.best_metrics) > 0:
                                previous_best_metric = self.best_metrics[-1]
                                previous_method = self.methods[-1]

                        # revert Fa/Fb/Da/Db
                        self.Fa().copy(Fa_before)
                        self.Fb().copy(Fb_before)
                        self.Da().copy(Da_before)
                        self.Db().copy(Db_before)

                    # try ensemble
                    if core.get_option('SCF', 'MESA_ERROR') == "LMETA":

                        for method in all_combination_errors["INTRA_D"]:
                            for metric in metrics:
                                if all_combination_errors[metric][method] == 0:
                                    normalized_reciprocal_errors[metric][method] = 1.0
                                else:
                                    recip = 1.0 / all_combination_errors[metric][method]
                                    row_sum  = sum(1.0 / v for v in all_combination_errors[metric].values())
                                    normalized_reciprocal_errors[metric][method] = recip / row_sum
                                    
                                column_sums[method] += normalized_reciprocal_errors[metric][method]

                        best_method = max(column_sums, key=column_sums.get)

                        scf_conv_method_errors[best_method] = column_sums[best_method]

                    if best_method is not None:
                        if not self.same_a_b_orbs():
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fa(), self.Da(), self.Ja(), Dnorm=Dnorm, component=0)
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fb(), self.Db(), self.Jb(), Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", best_method, self.Fa(), self.Da(), self.Ja())

                        self.form_C()
                        self.form_D()

                    all_errors_formatted = {metric: f"{data['error']} ({data['method']})" for metric, data in all_berrors.items()}
                    current_mesa_scf_conv_method = list(scf_conv_method_errors.keys())[np.argmin(list(scf_conv_method_errors.values()))]

                    if core.get_option('SCF', 'MESA_ERROR') == "LMETA":
                        # core.print_out(f" Best combined method: {best_method} with sum of errors {best_error_val}\n")
                        # core.print_out(f" Error for {best_metric}: {all_combination_errors[best_metric][best_method]}\n")
                        # core.print_out(f" Error for {second_best_metric}: {all_combination_errors[second_best_metric][best_method]}\n")
                        # core.print_out(f" Error for {third_best_metric}: {all_combination_errors[third_best_metric][best_method]}\n")
                        core.print_out(f"\n Most votes {best_method}: {column_sums[best_method]}")
                        methods = list(column_sums.keys())
                        methods.sort(key=lambda x: column_sums[x], reverse=True)
                        col_width = 14
                        output_str = "\n"
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        output_str += "  {:<{width}}".format("Method", width=col_width)
                        for metric in metrics:
                            output_str += " | {:<{width}}".format(metric, width=col_width-2)
                        output_str += " | {:<{width}}\n".format("Total", width=col_width-2)
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        for method in methods:
                            output_str += "  {:<{width}}".format(method, width=col_width)
                            for metric in metrics:
                                if metric in normalized_reciprocal_errors and method in normalized_reciprocal_errors[metric]:
                                    value = normalized_reciprocal_errors[metric][method]
                                    output_str += " | {:<{width}.4f}".format(value, width=col_width-2)
                                else:
                                    output_str += " | {:<{width}}".format("N/A", width=col_width-2)
                            output_str += " | {:<{width}.4f}\n".format(column_sums[method], width=col_width-2)
                        output_str += "  "+"-"*((col_width+2)*(len(metrics)+1))+"\n"
                        output_str += f"  Best method: {best_method} with total score {column_sums[best_method]:.4f}\n"
                        core.print_out(output_str)
                    else:
                        core.print_out(f"[LMETA-MESA] Best method this iteration is {best_method}, w/ metric {best_metric}, LMETA-MESA error = {best_error_val}\n, all: {all_errors_formatted}\n")                    
                    self.best_metrics.append(best_metric)


                elif self.mesa_enabled and not (core.get_option('SCF', 'MESA_ERROR') in ["EMETA","META","LMETA","ZMETA","YMETA","ZYMETA", "YZMETA"]):
                    scf_conv_methods = core.get_option('SCF', 'MESA_SCF_METHODS')
                    scf_conv_method_errors = {}
                    error_type = core.get_option('SCF', 'MESA_ERROR')

                    for scf_conv_method in scf_conv_methods:
                        if not self.same_a_b_orbs():
                            self.diis_manager_.extrapolate("mesa", scf_conv_method,
                                                        self.Fa(), self.Da(), self.Ja(),
                                                        Dnorm=Dnorm, component=0)
                            
                            self.diis_manager_.extrapolate("mesa", scf_conv_method,
                                                        self.Fb(), self.Db(), self.Jb(),
                                                        Dnorm=Dnorm, component=1)
                        else:
                            self.diis_manager_.extrapolate("mesa", scf_conv_method,
                                                        self.Fa(),
                                                        self.Da(),
                                                        self.Ja())

                        D_in_a = self.Da().clone()
                        if not self.same_a_b_orbs():
                            D_in_b = self.Db().clone()

                        self.form_C()
                        
                        self.form_D()

                        if not self.same_a_b_orbs():
                            error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                            error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())

                            if error_type == "INTRA_D":
                                error_a = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                                error_b = self.form_FoutmFin(self.Db().clone(), D_in_b.clone())
                            elif error_type == "INTER_F":
                                error_a = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                                error_b = self.form_FoutmFin(self.Fb().clone(), Fb_out_prev.clone())
                            elif error_type == "INTER_D":
                                error_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                error_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                            elif error_type == "COMMUTATOR":
                                #error_a = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                #error_b = self.form_FDSmSDF(self.Fb().clone(), self.Db().clone())
                                error_a = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                                error_b = self.form_FDSmSDF(self.Fb().clone(), D_in_b.clone())
                            elif error_type == "D_F":
                                d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                if len(f_ins) > 0:
                                    f_prev_a = f_ins[-1][0].clone()
                                    f_prev_b = f_ins[-1][1].clone()
                                    f_add_a = self.form_MaddM(self.Fa().clone(), f_prev_a)
                                    f_add_b = self.form_MaddM(self.Fb().clone(), f_prev_b)
                                else:
                                    f_add_a = self.Fa().clone()
                                    f_add_b = self.Fb().clone()
                                error_a = d_diff_a.chain_dot(f_add_a)
                                error_b = d_diff_b.chain_dot(f_add_b)                                   
                            elif error_type == "D_F-":
                                d_diff_a = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                d_diff_b = self.form_FoutmFin(self.Db().clone(), Db_prev_out.clone())
                                if len(f_ins) > 0:
                                    f_prev_a = f_ins[-1][0].clone()
                                    f_prev_b = f_ins[-1][1].clone()
                                    f_diff_a = self.form_FoutmFin(self.Fa().clone(), f_prev_a)
                                    f_diff_b = self.form_FoutmFin(self.Fb().clone(), f_prev_b)
                                else:
                                    f_diff_a = self.Fa().clone()
                                    f_diff_b = self.Fb().clone()
                                error_a = d_diff_a.chain_dot(f_diff_a)
                                error_b = d_diff_b.chain_dot(f_diff_b)

                            error = np.sqrt(0.5 * (error_a.rms()**2 + error_b.rms()**2))
                            #error = 0.5 * (error_a.rms() + error_b.rms())


                            scf_conv_method_errors[scf_conv_method] = error

                        else:
                            error = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                            if error_type == "INTRA_D":
                                error = self.form_FoutmFin(self.Da().clone(), D_in_a.clone())
                            elif error_type == "INTER_F":
                                error = self.form_FoutmFin(self.Fa().clone(), Fa_out_prev.clone())
                            elif error_type == "INTER_D":
                                error = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                            elif error_type == "COMMUTATOR":
                                # error = self.form_FDSmSDF(self.Fa().clone(), self.Da().clone())
                                error = self.form_FDSmSDF(self.Fa().clone(), D_in_a.clone())
                            elif error_type == "D_F":
                                d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())
                                f_add = self.form_MaddM(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()
                                error = d_diff.chain_dot(f_add)
                            elif error_type == "D_F-":
                                d_diff = self.form_FoutmFin(self.Da().clone(), Da_prev_out.clone())                                    
                                f_diff = self.form_FoutmFin(self.Fa().clone(), f_ins[-1].clone()) if len(f_ins) > 0 else self.Fa().clone()                                    
                                error = d_diff.chain_dot(f_diff)  

                            scf_conv_method_errors[scf_conv_method] = error.rms()
                        

                    current_mesa_scf_conv_method = list(scf_conv_method_errors.keys())[np.argmin(list(scf_conv_method_errors.values()))]

                if add_to_diis_subspace:
                    if self.mesa_enabled:
                        for engine_used in self.diis(Dnorm, "mesa", current_mesa_scf_conv_method):
                            status.append(engine_used)
                    else:
                        for engine_used in self.diis(Dnorm):
                            status.append(engine_used)

                core.timer_off("HF: DIIS")

                if hasattr(self, '_saved_hybrid_Fa') and self._saved_hybrid_Fa is not None:
                    self.Fa().copy(self._saved_hybrid_Fa)
                    if getattr(self, '_saved_hybrid_Da', None) is not None:
                        self.Da().copy(self._saved_hybrid_Da)
                    if getattr(self, '_saved_hybrid_Ja', None) is not None and self.Ja():
                        self.Ja().copy(self._saved_hybrid_Ja)

                    if not self.same_a_b_orbs() and getattr(self, '_saved_hybrid_Fb', None) is not None:
                        self.Fb().copy(self._saved_hybrid_Fb)
                        if getattr(self, '_saved_hybrid_Db', None) is not None:
                            self.Db().copy(self._saved_hybrid_Db)
                        if getattr(self, '_saved_hybrid_Jb', None) is not None and self.Jb():
                            self.Jb().copy(self._saved_hybrid_Jb)
                    
                    # Overwrite the log so it says "YMETA" instead of "DIIS"
                    status[-1] = core.get_option('SCF', 'MESA_ERROR')
                    
                    # Clean up
                    self._saved_hybrid_Fa = None
                    self._saved_hybrid_Fb = None
                    self._saved_hybrid_Da = None
                    self._saved_hybrid_Db = None
                    self._saved_hybrid_Ja = None
                    self._saved_hybrid_Jb = None

                if verbose > 4 and diis_performed:
                    core.print_out("  After DIIS:\n")
                    self.Fa().print_out()
                    self.Fb().print_out()

                if not self.same_a_b_orbs():
                    # UHF: store alpha and beta as tuples
                    f_ins.append((self.Fa().clone(), self.Fb().clone()))
                    d_ins.append((self.Da().clone(), self.Db().clone()))

                    if self.Ja() and self.Jb():
                        v_ins.append((self.Ja().clone(), self.Jb().clone()))

                    else:
                        v_ins.append((self.Ja().clone(), None))
    
                else:
                    f_ins.append(self.Fa().clone())
                    d_ins.append(self.Da().clone())
                    if self.Ja():
                        v_ins.append(self.Ja().clone())

                # frac, MOM invoked here from Wfn::HF::find_occupation
                core.timer_on("HF: Form C")
                level_shift = core.get_option("SCF", "LEVEL_SHIFT")
                if level_shift > 0 and Dnorm > core.get_option('SCF', 'LEVEL_SHIFT_CUTOFF'):
                    status.append("SHIFT")
                    self.form_C(level_shift)
                else:
                    self.form_C()
                core.timer_off("HF: Form C")

                if self.MOM_performed_:
                    status.append("MOM")

                if self.frac_performed_:
                    status.append("FRAC")

                if incfock_performed:
                    status.append("INCFOCK")

                # Reset occupations if necessary
                if (self.iteration_ == 0) and self.reset_occ_:
                    self.reset_occupation()
                    self.find_occupation()

        # Form new density matrix
        core.timer_on("HF: Form D")
        self.form_D()
        core.timer_off("HF: Form D")

        self.set_variable("SCF ITERATION ENERGY", SCFE)
        core.set_variable("SCF D NORM", Dnorm)

        # After we've built the new D, damp the update
        if (damping_enabled and self.iteration_ > 1 and Dnorm > core.get_option('SCF', 'DAMPING_CONVERGENCE')):
            damping_percentage = core.get_option('SCF', "DAMPING_PERCENTAGE")
            self.damping_update(damping_percentage * 0.01)
            status.append("DAMP={}%".format(round(damping_percentage)))

        if core.has_option_changed("SCF", "ORBITALS_WRITE"):
            filename = core.get_option("SCF", "ORBITALS_WRITE")
            self.to_file(filename)

        if verbose > 3:
            self.Ca().print_out()
            self.Cb().print_out()
            self.Da().print_out()
            self.Db().print_out()

        self.energies.append(SCFE)
        self.rms.append(Dnorm)
        self.methods.append(status)
        # Print out the iteration

        core.print_out(
            "   @%s%s iter %3s: %20.14f   %12.5e   %-11.5e %s\n" %
            ("DF-" if is_dfjk else "", reference, "SAD" if
            ((self.iteration_ == 0) and self.sad_) else self.iteration_, SCFE, Ediff, Dnorm, '/'.join(status)))
            
        # if a an excited MOM is requested but not started, don't stop yet
        # Note that MOM_performed_ just checks initialization, and our convergence measures used the pre-MOM orbitals
        if self.MOM_excited_ and ((not self.MOM_performed_) or self.iteration_ == core.get_option('SCF', "MOM_START")):
            continue

        # if a fractional occupation is requested but not started, don't stop yet
        if frac_enabled and not self.frac_performed_:
            continue

        # have we completed our post-early screening SCF iterations?
        if early_screening_disabled:
            scf_iter_post_screening += 1
            if scf_iter_post_screening >= scf_maxiter_post_screening and scf_maxiter_post_screening > 0:
                break

        # Call any postiteration callbacks
        if not ((self.iteration_ == 0) and self.sad_) and _converged(Ediff, Dnorm, e_conv=e_conv, d_conv=d_conv):

            method_str = "DIIS"
            if core.get_option('SCF', 'MESA'):
                method_str = "MESA"
                method_str += core.get_option('SCF', 'MESA_ERROR')
            elif core.get_option('SCF', "DIIS") and core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR') != "NONE":
                method_str = "DIIS_" + core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR')
            elif core.get_option('SCF', 'DIIS'):
                method_str = "DIIS"
            elif core.get_option('SCF', 'LIST') != "NONE":
                method_str = core.get_option('SCF', 'LIST')
            elif core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR') != "NONE":
                method_str = core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR')

            with open(method_str.lower() + '.json', 'w') as f:
                json.dump({"RMS": self.rms, "METHODS": self.methods, "ENERGIES": self.energies}, f)

            if early_screening:

                # we've reached convergence with early screning enabled; disable it
                early_screening = False

                # make note of the change to early screening; next SCF iteration(s) will be the last
                early_screening_disabled = True

                # cosx uses the largest grid for its final SCF iteration(s)
                if cosx_enabled:
                    self.jk().set_COSX_grid("Final")

                # clear any cached matrices associated with incremental fock construction
                # the change in the screening spoils the linearity in the density matrix
                if hasattr(self.jk(), 'clear_D_prev'):
                    self.jk().clear_D_prev()

                if scf_maxiter_post_screening == 0:
                    break
                else:
                    core.print_out("  Energy and wave function converged with early screening.\n")
                    core.print_out("  Continuing SCF iterations with tighter screening.\n\n")
            else:
                break

        if self.iteration_ >= core.get_option('SCF', 'MAXITER'):
            method_str = "DIIS"
            if core.get_option('SCF', 'MESA'):
                method_str = "MESA"
                method_str += core.get_option('SCF', 'MESA_ERROR')
            elif core.get_option('SCF', "DIIS") and core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR') != "NONE":
                method_str = "DIIS_" + core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR')
            elif core.get_option('SCF', 'DIIS'):
                method_str = "DIIS"
            elif core.get_option('SCF', 'LIST') != "NONE":
                method_str = core.get_option('SCF', 'LIST')
            elif core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR') != "NONE":
                method_str = core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR')

            with open(method_str.lower() + '.json', 'w') as f:
                json.dump({"RMS": self.rms, "METHODS": self.methods, "ENERGIES": self.energies}, f)
            raise SCFConvergenceError("""SCF iterations""", self.iteration_, self, Ediff, Dnorm)


def scf_finalize_energy(self):
    """Performs stability analysis and calls back SCF with new guess
    if needed, Returns the SCF energy. This function should be called
    once orbitals are ready for energy/property computations, usually
    after iterations() is called.

    """

    # post-scf vv10 correlation
    if core.get_option('SCF', "DFT_VV10_POSTSCF") and self.functional().vv10_b() > 0.0:
        self.functional().set_lock(False)
        self.functional().set_do_vv10(True)
        self.functional().set_lock(True)
        core.print_out("  ==> Computing Non-Self-Consistent VV10 Energy Correction <==\n\n")
        SCFE = 0.0
        self.form_V()
        SCFE += self.compute_E()
        self.set_energies("Total Energy", SCFE)

    # Perform wavefunction stability analysis before doing
    # anything on a wavefunction that may not be truly converged.
    if core.get_option('SCF', 'STABILITY_ANALYSIS') != "NONE":

        # We need the integral file, make sure it is written and
        # compute it if needed
        if core.get_option('SCF', 'REFERENCE') not in {"UHF", "UKS"}:
            # Don't bother computing needed integrals if we can't do anything with them.
            if self.functional().needs_xc():
                raise ValidationError("Stability analysis not yet supported for XC functionals.")

            #psio = core.IO.shared_object()
            #psio.open(constants.PSIF_SO_TEI, 1)  # PSIO_OPEN_OLD
            #try:
            #    psio.tocscan(constants.PSIF_SO_TEI, "IWL Buffers")
            #except TypeError:
            #    # "IWL Buffers" actually found but psio_tocentry can't be returned to Py
            #    psio.close(constants.PSIF_SO_TEI, 1)
            #else:
            #    # tocscan returned None
            #    psio.close(constants.PSIF_SO_TEI, 1)

            # logic above foiled by psio_tocentry not returning None<--nullptr in pb11 2.2.1
            #   so forcibly recomputing for now until stability revamp
            core.print_out("    SO Integrals not on disk. Computing...")
            mints = core.MintsHelper(self.basisset())

            mints.integrals()
            core.print_out("done.\n")

            # Q: Not worth exporting all the layers of psio, right?

        follow = self.stability_analysis()

        while follow and self.attempt_number_ <= core.get_option('SCF', 'MAX_ATTEMPTS'):
            self.attempt_number_ += 1
            core.print_out("    Running SCF again with the rotated orbitals.\n")

            if self.initialized_diis_manager_:
                self.diis_manager_.reset_subspace()
            # reading the rotated orbitals in before starting iterations
            self.form_D()
            self.set_energies("Total Energy", self.compute_initial_E())
            self.iterations()
            follow = self.stability_analysis()

        if follow and self.attempt_number_ > core.get_option('SCF', 'MAX_ATTEMPTS'):
            core.print_out("    There's still a negative eigenvalue. Try modifying FOLLOW_STEP_SCALE\n")
            core.print_out("    or increasing MAX_ATTEMPTS (not available for PK integrals).\n")

    # At this point, we are not doing any more SCF cycles
    #   and we can compute and print final quantities.

    if hasattr(self.molecule(), 'EFP'):
        efpobj = self.molecule().EFP

        efpobj.compute()  # do_gradient=do_gradient)
        efpene = efpobj.get_energy(label='psi')
        efp_wfn_independent_energy = efpene['total'] - efpene['ind']
        self.set_energies("EFP", efpene['total'])

        SCFE = self.get_energies("Total Energy")
        SCFE += efp_wfn_independent_energy
        self.set_energies("Total Energy", SCFE)
        core.print_out(efpobj.energy_summary(scfefp=SCFE, label='psi'))

        self.set_variable("EFP ELST ENERGY", efpene['electrostatic'] + efpene['charge_penetration'] + efpene['electrostatic_point_charges'])  # P::e EFP
        self.set_variable("EFP IND ENERGY", efpene['polarization'])  # P::e EFP
        self.set_variable("EFP DISP ENERGY", efpene['dispersion'])  # P::e EFP
        self.set_variable("EFP EXCH ENERGY", efpene['exchange_repulsion'])  # P::e EFP
        self.set_variable("EFP TOTAL ENERGY", efpene['total'])  # P::e EFP
        self.set_variable("CURRENT ENERGY", efpene['total'])  # P::e EFP

    core.print_out("\n  ==> Post-Iterations <==\n\n")

    if self.V_potential():
        quad = self.V_potential().quadrature_values()
        rho_a = quad['RHO_A']/2 if self.same_a_b_dens() else quad['RHO_A']
        rho_b = quad['RHO_B']/2 if self.same_a_b_dens() else quad['RHO_B']
        rho_ab = (rho_a + rho_b)
        self.set_variable("GRID ELECTRONS TOTAL",rho_ab)  # P::e SCF
        self.set_variable("GRID ELECTRONS ALPHA",rho_a)  # P::e SCF
        self.set_variable("GRID ELECTRONS BETA",rho_b)  # P::e SCF
        dev_a = rho_a - self.nalpha()
        dev_b = rho_b - self.nbeta()
        core.print_out(f"   Electrons on quadrature grid:\n")
        if self.same_a_b_dens():
            core.print_out(f"      Ntotal   = {rho_ab:15.10f} ; deviation = {dev_b+dev_a:.3e} \n\n")
        else:
            core.print_out(f"      Nalpha   = {rho_a:15.10f} ; deviation = {dev_a:.3e}\n")
            core.print_out(f"      Nbeta    = {rho_b:15.10f} ; deviation = {dev_b:.3e}\n")
            core.print_out(f"      Ntotal   = {rho_ab:15.10f} ; deviation = {dev_b+dev_a:.3e} \n\n")
        if ((dev_b+dev_a) > 0.1):
            core.print_out("   WARNING: large deviation in the electron count on grid detected. Check grid size!")
    self.check_phases()
    self.compute_spin_contamination()
    self.frac_renormalize()
    reference = core.get_option("SCF", "REFERENCE")

    energy = self.get_energies("Total Energy")

    #    fail_on_maxiter = core.get_option("SCF", "FAIL_ON_MAXITER")
    #    if converged or not fail_on_maxiter:
    #
    #        if print_lvl > 0:
    #            self.print_orbitals()
    #
    #        if converged:
    #            core.print_out("  Energy converged.\n\n")
    #        else:
    #            core.print_out("  Energy did not converge, but proceeding anyway.\n\n")

    if core.get_option('SCF', 'PRINT') > 0:
        self.print_orbitals()

    is_dfjk = core.get_global_option('SCF_TYPE').endswith('DF')
    core.print_out("  @%s%s Final Energy: %20.14f" % ('DF-' if is_dfjk else '', reference, energy))
    # if (perturb_h_) {
    #     core.print_out(" with %f %f %f perturbation" %
    #                    (dipole_field_strength_[0], dipole_field_strength_[1], dipole_field_strength_[2]))
    # }
    core.print_out("\n\n")
    self.print_energies()

    # force list into Matrix for storage
    iteration_energies = np.array(self.iteration_energies).reshape(-1, 1)
    iteration_energies = core.Matrix.from_array(iteration_energies)
    core.set_variable("SCF TOTAL ENERGIES", core.Matrix.from_array(iteration_energies))
    self.set_variable("SCF TOTAL ENERGIES", core.Matrix.from_array(iteration_energies))





    self.clear_external_potentials()
    if core.get_option('SCF', 'PCM'):
        calc_type = core.PCM.CalcType.Total
        if core.get_option("PCM", "PCM_SCF_TYPE") == "SEPARATE":
            calc_type = core.PCM.CalcType.NucAndEle
        Dt = self.Da().clone()
        Dt.add(self.Db())
        _, Vpcm = self.get_PCM().compute_PCM_terms(Dt, calc_type)
        self.push_back_external_potential(Vpcm)
        # Set callback function for CPSCF
        self.set_external_cpscf_perturbation("PCM", lambda pert_dm : self.get_PCM().compute_V(pert_dm))

    if core.get_option('SCF', 'PE'):
        Dt = self.Da().clone()
        Dt.add(self.Db())
        _, Vpe = self.pe_state.get_pe_contribution(
            Dt, elec_only=False
        )
        self.push_back_external_potential(Vpe)
        # Set callback function for CPSCF
        self.set_external_cpscf_perturbation("PE", lambda pert_dm : self.pe_state.get_pe_contribution(pert_dm, elec_only=True)[1])

    if core.get_option('SCF', 'DDX'):
        Dt = self.Da().clone()
        Dt.add(self.Db())
        Vddx = self.ddx.get_solvation_contributions(Dt)[1]
        self.push_back_external_potential(Vddx)
        # Set callback function for CPSCF
        self.set_external_cpscf_perturbation(
            "DDX", lambda pert_dm : self.ddx.get_solvation_contributions(pert_dm, elec_only=True, nonequilibrium=True)[1])

    # Orbitals are always saved, in case an MO guess is requested later
    # save_orbitals()

    # Shove variables into global space
    for k, v in self.variables().items():
        core.set_variable(k, v)

    # TODO re-enable
    self.finalize()
    if self.V_potential():
        self.V_potential().clear_collocation_cache()

    core.print_out("\nComputation Completed\n")
    core.del_variable("SCF D NORM")

    return energy


def scf_print_energies(self):
    enuc = self.get_energies('Nuclear')
    e1 = self.get_energies('One-Electron')
    e2 = self.get_energies('Two-Electron')
    exc = self.get_energies('XC')
    ed = self.get_energies('-D')
    self.del_variable('-D Energy')
    evv10 = self.get_energies('VV10')
    eefp = self.get_energies('EFP')
    epcm = self.get_energies('PCM Polarization')
    edd = self.get_energies('DD Solvation Energy')
    epe = self.get_energies('PE Energy')
    ke = self.get_energies('Kinetic')

    hf_energy = enuc + e1 + e2
    dft_energy = hf_energy + exc + ed + evv10
    total_energy = dft_energy + eefp + epcm + edd + epe
    full_qm = (not core.get_option('SCF', 'PCM') and not core.get_option('SCF', 'DDX') and not core.get_option('SCF', 'PE')
               and not hasattr(self.molecule(), 'EFP'))

    core.print_out("   => Energetics <=\n\n")
    core.print_out("    Nuclear Repulsion Energy =        {:24.16f}\n".format(enuc))
    core.print_out("    One-Electron Energy =             {:24.16f}\n".format(e1))
    core.print_out("    Two-Electron Energy =             {:24.16f}\n".format(e2))
    if self.functional().needs_xc():
        core.print_out("    DFT Exchange-Correlation Energy = {:24.16f}\n".format(exc))
        core.print_out("    Empirical Dispersion Energy =     {:24.16f}\n".format(ed))
        core.print_out("    VV10 Nonlocal Energy =            {:24.16f}\n".format(evv10))
    if core.get_option('SCF', 'PCM'):
        core.print_out("    PCM Polarization Energy =         {:24.16f}\n".format(epcm))
    if core.get_option('SCF', 'DDX'):
        core.print_out("    DD Solvation Energy =            {:24.16f}\n".format(edd))
    if core.get_option('SCF', 'PE'):
        core.print_out("    PE Energy =                       {:24.16f}\n".format(epe))
    if hasattr(self.molecule(), 'EFP'):
        core.print_out("    EFP Energy =                      {:24.16f}\n".format(eefp))
    core.print_out("    Total Energy =                    {:24.16f}\n".format(total_energy))

    if core.get_option('SCF', 'PE'):
        core.print_out(self.pe_state.cppe_state.summary_string)

    self.set_variable("NUCLEAR REPULSION ENERGY", enuc)  # P::e SCF
    self.set_variable("ONE-ELECTRON ENERGY", e1)  # P::e SCF
    self.set_variable("TWO-ELECTRON ENERGY", e2)  # P::e SCF
    if self.functional().needs_xc():
        self.set_variable("DFT XC ENERGY", exc)  # P::e SCF
        self.set_variable("DFT VV10 ENERGY", evv10)  # P::e SCF
        self.set_variable("DFT FUNCTIONAL TOTAL ENERGY", hf_energy + exc + evv10)  # P::e SCF
        #self.set_variable(self.functional().name() + ' FUNCTIONAL TOTAL ENERGY', hf_energy + exc + evv10)
        self.set_variable("DFT TOTAL ENERGY", dft_energy)  # overwritten later for DH  # P::e SCF
    else:
        potential = total_energy - ke
        self.set_variable("HF KINETIC ENERGY", ke)  # P::e SCF
        self.set_variable("HF POTENTIAL ENERGY", potential)  # P::e SCF
        if full_qm:
            self.set_variable("HF VIRIAL RATIO", - potential / ke)  # P::e SCF
        self.set_variable("HF TOTAL ENERGY", hf_energy)  # P::e SCF
    if hasattr(self, "_disp_functor"):
        self.set_variable("DISPERSION CORRECTION ENERGY", ed)  # P::e SCF
    #if abs(ed) > 1.0e-14:
    #    for pv, pvv in self.variables().items():
    #        if abs(pvv - ed) < 1.0e-14:
    #            if pv.endswith('DISPERSION CORRECTION ENERGY') and pv.startswith(self.functional().name()):
    #                fctl_plus_disp_name = pv.split()[0]
    #                self.set_variable(fctl_plus_disp_name + ' TOTAL ENERGY', dft_energy)  # overwritten later for DH
    #else:
    #    self.set_variable(self.functional().name() + ' TOTAL ENERGY', dft_energy)  # overwritten later for DH

    self.set_variable("SCF ITERATIONS", self.iteration_)  # P::e SCF


def scf_print_preiterations(self,small=False):
    # small version does not print Nalpha,Nbeta,Ndocc,Nsocc, e.g. for SAD guess where they are not
    # available
    ct = self.molecule().point_group().char_table()

    if not small:
        core.print_out("   -------------------------------------------------------\n")
        core.print_out("    Irrep   Nso     Nmo     Nalpha   Nbeta   Ndocc  Nsocc\n")
        core.print_out("   -------------------------------------------------------\n")

        for h in range(self.nirrep()):
            core.print_out(
                f"     {ct.gamma(h).symbol():<3s}   {self.nsopi()[h]:6d}  {self.nmopi()[h]:6d}  {self.nalphapi()[h]:6d}  {self.nbetapi()[h]:6d}  {self.doccpi()[h]:6d}  {self.soccpi()[h]:6d}\n"
            )

        core.print_out("   -------------------------------------------------------\n")
        core.print_out(
            f"    Total  {self.nso():6d}  {self.nmo():6d}  {self.nalpha():6d}  {self.nbeta():6d}  {self.nbeta():6d}  {self.nalpha() - self.nbeta():6d}\n"
        )
        core.print_out("   -------------------------------------------------------\n\n")
    else:
        core.print_out("   -------------------------\n")
        core.print_out("    Irrep   Nso     Nmo    \n")
        core.print_out("   -------------------------\n")

        for h in range(self.nirrep()):
            core.print_out(
                f"     {ct.gamma(h).symbol():<3s}   {self.nsopi()[h]:6d}  {self.nmopi()[h]:6d} \n"
            )

        core.print_out("   -------------------------\n")
        core.print_out(
            f"    Total  {self.nso():6d}  {self.nmo():6d}\n"
        )
        core.print_out("   -------------------------\n\n")

def get_best_metrics_history(self):
    """
    Returns the history of best error metrics used in each META-MESA iteration.

    list
    list of strings representing the best error metric (eg., "INTRA_D", "INTER_F", so forth.) 
        used in each META-MESA iteration.
    """
    return self.best_metrics


# Bind functions to core.HF class
core.HF.initialize = scf_initialize
core.HF.initialize_jk = initialize_jk
core.HF.iterations = scf_iterate
core.HF.compute_energy = scf_compute_energy
core.HF.finalize_energy = scf_finalize_energy
core.HF.print_energies = scf_print_energies
core.HF.print_preiterations = scf_print_preiterations
core.HF.get_best_metrics_history = get_best_metrics_history
core.HF.iteration_energies = []


def _converged(e_delta, d_rms, e_conv=None, d_conv=None):
    if e_conv is None:
        e_conv = core.get_option("SCF", "E_CONVERGENCE")
    if d_conv is None:
        d_conv = core.get_option("SCF", "D_CONVERGENCE")

    return (abs(e_delta) < e_conv and d_rms < d_conv)


def _validate_damping():
    """Sanity-checks DAMPING control options

    Raises
    ------
    ValidationError
        If any of |scf__damping_percentage|, |scf__damping_convergence|
        don't play well together.

    Returns
    -------
    bool
        Whether DAMPING is enabled during scf.

    """
    # Q: I changed the enabled criterion get_option <-- has_option_changed
    enabled = (core.get_option('SCF', 'DAMPING_PERCENTAGE') > 0.0)
    if enabled:
        parameter = core.get_option('SCF', "DAMPING_PERCENTAGE")
        if parameter < 0.0 or parameter > 100.0:
            raise ValidationError('SCF DAMPING_PERCENTAGE ({}) must be between 0 and 100'.format(parameter))

        stop = core.get_option('SCF', 'DAMPING_CONVERGENCE')
        if stop < 0.0:
            raise ValidationError('SCF DAMPING_CONVERGENCE ({}) must be > 0'.format(stop))

    return enabled

def _validate_metamesa(self):
        # TODO
    return core.get_option('SCF', 'MESA')

def _validate_mesa(self):
    """Sanity checks for MESA, checking DIIS options and LIST options are able to run in isolation.

    Raises
    ------
    psi4.driver.p4util.exceptions.ValidationError
        If any DIIS or LIST enabled things are not compatible.

    Returns
    -------
    bool
        Whether MESA enabled during SCF.
    """
    # TODO
    return core.get_option('SCF', 'MESA')

def _validate_list(self):
    """Sanity-checks LIST control options

    Raises
    ------
    psi4.driver.p4util.exceptions.ValidationError
        If DIIS or any DIIS varients are turned on, then can't run LIST.

    Returns
    -------
    bool
        Whether LIST enabled during SCF.
    """
    enabled = core.get_option('SCF', 'LIST') != "NONE"
    restricted_open = self.same_a_b_orbs() and not self.same_a_b_dens()
    aediis_active = enabled and core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR') != "NONE" and not restricted_open

    if aediis_active:
        raise ValidationError('SCF_INITIAL_ACCELERATOR is not used for LIST.')

    diis_and_list_enabled = enabled and bool(core.get_option('SCF', 'DIIS'))
    if diis_and_list_enabled:
        raise ValidationError("DIIS cannot be True when LIST is True.")

    return enabled


def _validate_diis(self):
    """Sanity-checks DIIS control options

    Raises
    ------
    psi4.driver.p4util.exceptions.ValidationError
        If any of DIIS options don't play well together.

    Returns
    -------
    bool
        Whether some form of DIIS is enabled during SCF.

    """

    restricted_open = self.same_a_b_orbs() and not self.same_a_b_dens()
    aediis_active = core.get_option('SCF', 'SCF_INITIAL_ACCELERATOR') != "NONE" and not restricted_open

    if aediis_active:
        start = core.get_option('SCF', 'SCF_INITIAL_START_DIIS_TRANSITION')
        stop = core.get_option('SCF', 'SCF_INITIAL_FINISH_DIIS_TRANSITION')
        if start < stop:
            raise ValidationError('SCF_INITIAL_START_DIIS_TRANSITION error magnitude cannot be less than SCF_INITIAL_FINISH_DIIS_TRANSITION.')
        elif start < 0:
            raise ValidationError('SCF_INITIAL_START_DIIS_TRANSITION cannot be negative.')
        elif stop < 0:
            raise ValidationError('SCF_INITIAL_FINISH_DIIS_TRANSITION cannot be negative.')

    enabled = bool(core.get_option('SCF', 'DIIS')) or aediis_active
    if enabled:
        start = core.get_option('SCF', 'DIIS_START')
        if start < 1:
            raise ValidationError('SCF DIIS_START ({}) must be at least 1'.format(start))

    list_and_diis_enabled = enabled and (core.get_option('SCF', 'LIST') != "NONE")
    if list_and_diis_enabled:
        raise ValidationError("Please choose to use DIIS or LIST as main SCF convergence method.")

    return enabled


def _validate_frac():
    """Sanity-checks FRAC control options

    Raises
    ------
    ValidationError
        If any of |scf__frac_start| don't play well together.

    Returns
    -------
    bool
        Whether FRAC is enabled during scf.

    """
    enabled = (core.get_option('SCF', 'FRAC_START') != 0)
    if enabled:
        if enabled < 0:
            raise ValidationError('SCF FRAC_START ({}) must be at least 1'.format(enabled))

    return enabled


def _validate_MOM():
    """Sanity-checks MOM control options

    Raises
    ------
    ValidationError
        If any of |scf__mom_start|, |scf__mom_occ| don't play well together.

    Returns
    -------
    bool
        Whether excited-state MOM (not just the plain stabilizing MOM) is enabled during scf.

    """
    enabled = (core.get_option('SCF', "MOM_START") != 0 and len(core.get_option('SCF', "MOM_OCC")) > 0)
    if enabled:
        start = core.get_option('SCF', "MOM_START")
        if enabled < 0:
            raise ValidationError('SCF MOM_START ({}) must be at least 1'.format(start))

    return enabled


def _validate_soscf():
    """Sanity-checks SOSCF control options

    Raises
    ------
    ValidationError
        If any of |scf__soscf|, |scf__soscf_start_convergence|,
        |scf__soscf_min_iter|, |scf__soscf_max_iter| don't play well together.

    Returns
    -------
    bool
        Whether SOSCF is enabled during scf.

    """
    enabled = core.get_option('SCF', 'SOSCF')
    if enabled:
        start = core.get_option('SCF', 'SOSCF_START_CONVERGENCE')
        if start < 0.0:
            raise ValidationError('SCF SOSCF_START_CONVERGENCE ({}) must be positive'.format(start))

        miniter = core.get_option('SCF', 'SOSCF_MIN_ITER')
        if miniter < 1:
            raise ValidationError('SCF SOSCF_MIN_ITER ({}) must be at least 1'.format(miniter))

        maxiter = core.get_option('SCF', 'SOSCF_MAX_ITER')
        if maxiter < miniter:
            raise ValidationError('SCF SOSCF_MAX_ITER ({}) must be at least SOSCF_MIN_ITER ({})'.format(
                maxiter, miniter))

        conv = core.get_option('SCF', 'SOSCF_CONV')
        if conv < 1.e-10:
            raise ValidationError('SCF SOSCF_CONV ({}) must be achievable'.format(conv))

    return enabled


core.HF.validate_diis = _validate_diis
core.HF.validate_list = _validate_list
core.HF.validate_mesa = _validate_mesa


def efp_field_fn(xyz):
    """Callback function for PylibEFP to compute electric field from electrons
    in ab initio part for libefp polarization calculation.

    Parameters
    ----------
    xyz : list
        (3 * npt, ) flat array of points at which to compute electric field

    Returns
    -------
    list
        (3 * npt, ) flat array of electric field at points in `xyz`.

    Notes
    -----
    Function signature defined by libefp, so function uses number of
    basis functions and integrals factory `mints_psi4_yo` and total density
    matrix `efp_Dt_psi4_yo` from global namespace.

    """
    points = core.Matrix.from_array(np.array(xyz).reshape(-1, 3))
    field = mints_psi4_yo.electric_field_value(points, efp_Dt_psi4_yo).np.flatten()
    return field
