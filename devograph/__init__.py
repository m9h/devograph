"""devograph: Developmental neurobiology extensions for hgx.

Dynamics, geometric convolutions, optimal transport, information geometry,
and C. elegans tooling — built on top of the hgx hypergraph library.
"""

from devograph._conv import (
    LorentzHypergraphConv as LorentzHypergraphConv,
    PoincareHypergraphConv as PoincareHypergraphConv,
    ProductHypergraphConv as ProductHypergraphConv,
    ProductManifold as ProductManifold,
    ProductManifoldConv as ProductManifoldConv,
    ProductManifoldMLP as ProductManifoldMLP,
    ProductSpaceConv as ProductSpaceConv,
    ProductSpaceEmbedding as ProductSpaceEmbedding,
)
from devograph._info_geometry import (
    fisher_rao_distance as fisher_rao_distance,
    fisher_rao_metric as fisher_rao_metric,
    free_energy as free_energy,
    info_belief_update as info_belief_update,
    InfoGeometricDynamics as InfoGeometricDynamics,
    js_divergence as js_divergence,
    kl_divergence as kl_divergence,
    natural_gradient as natural_gradient,
    natural_gradient_descent as natural_gradient_descent,
    symmetrized_kl as symmetrized_kl,
    wasserstein_on_simplex as wasserstein_on_simplex,
)
from devograph._ndp import (
    CellProgram as CellProgram,
    develop_trajectory as develop_trajectory,
    HypergraphNDP as HypergraphNDP,
)
from devograph._ot import (
    feature_cost_matrix as feature_cost_matrix,
    gromov_wasserstein as gromov_wasserstein,
    hypergraph_gromov_wasserstein as hypergraph_gromov_wasserstein,
    hypergraph_wasserstein as hypergraph_wasserstein,
    ot_hyperedge_aggregation as ot_hyperedge_aggregation,
    ot_hypergraph_alignment as ot_hypergraph_alignment,
    OTConv as OTConv,
    OTLayer as OTLayer,
    sinkhorn as sinkhorn,
    structural_cost_matrix as structural_cost_matrix,
    unbalanced_sinkhorn as unbalanced_sinkhorn,
    wasserstein_barycenter as wasserstein_barycenter,
    wasserstein_distance as wasserstein_distance,
)
from devograph._perturbation import (
    in_silico_knockout as in_silico_knockout,
    perturbation_screen as perturbation_screen,
    PerturbationEncoder as PerturbationEncoder,
    PerturbationPredictor as PerturbationPredictor,
    train_perturbation_predictor as train_perturbation_predictor,
)
from devograph._temporal import (
    align_topologies as align_topologies,
    fit_neural_ode as fit_neural_ode,
    from_snapshots as from_snapshots,
    interpolate as interpolate,
    sliding_window as sliding_window,
    temporal_smoothness_loss as temporal_smoothness_loss,
    TemporalHypergraph as TemporalHypergraph,
)
from devograph._topology import (
    hodge_laplacians as hodge_laplacians,
    TopologicalLayer as TopologicalLayer,
)
from devograph._wavelets import (
    cheeger_constant_bound as cheeger_constant_bound,
    hypergraph_scattering as hypergraph_scattering,
    hypergraph_wavelet_transform as hypergraph_wavelet_transform,
    HypergraphWaveletLayer as HypergraphWaveletLayer,
    spectral_features as spectral_features,
)


try:
    from devograph._topology import (
        compute_persistence as compute_persistence,
        persistence_features as persistence_features,
        persistence_image as persistence_image,
        persistence_landscape as persistence_landscape,
    )
except ImportError:
    pass

try:
    from devograph._dynamics import (
        evolve as evolve,
        HypergraphNeuralCDE as HypergraphNeuralCDE,
        HypergraphNeuralODE as HypergraphNeuralODE,
        HypergraphNeuralSDE as HypergraphNeuralSDE,
        trajectory as trajectory,
    )
    from devograph._geometric_dynamics import (
        AbstractManifold as AbstractManifold,
        EuclideanManifold as EuclideanManifold,
        PoincareBall as PoincareBall,
        riemannian_trajectory as riemannian_trajectory,
        RiemannianHypergraphODE as RiemannianHypergraphODE,
    )
    from devograph._info_geometry import (
        FisherRaoDrift as FisherRaoDrift,
        FreeEnergyDrift as FreeEnergyDrift,
        InfoGeometricODE as InfoGeometricODE,
    )
    from devograph._latent import (
        LatentHypergraphODE as LatentHypergraphODE,
        LatentHypergraphSDE as LatentHypergraphSDE,
    )
except ImportError:
    pass

try:
    from devograph._conv._se3 import SE3HypergraphConv as SE3HypergraphConv
except ImportError:
    pass

try:
    from devograph._grn import (
        grn_to_temporal_hypergraphs as grn_to_temporal_hypergraphs,
        load_grn_from_anndata as load_grn_from_anndata,
        load_grn_from_csv as load_grn_from_csv,
        load_grn_from_edge_list as load_grn_from_edge_list,
        load_pando_modules as load_pando_modules,
    )
except ImportError:
    pass

try:
    from devograph._data import (
        load_cell_lineage as load_cell_lineage,
        load_connectome as load_connectome,
        load_devograph as load_devograph,
        load_synthetic_karate as load_synthetic_karate,
    )
except ImportError:
    pass

try:
    from devograph._pgmax import (
        ActiveInferenceStep as ActiveInferenceStep,
        hypergraph_to_factor_graph as hypergraph_to_factor_graph,
        learn_potentials as learn_potentials,
        run_cell_fate_inference as run_cell_fate_inference,
    )
except ImportError:
    pass
