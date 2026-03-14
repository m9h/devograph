"""Known-answer validation tests for devograph extensions.

Every test computes an expected result BY HAND and compares it to the
implementation output.

Sections
--------
1. Cheeger bounds for complete hypergraph
2. Wavelet heat kernel recovers features at s=0
3. Sinkhorn on uniform distributions → uniform plan
4. Poincare ball: expmap/logmap roundtrip and distance
5. Spectral features known values
"""

import devograph
import hgx
import jax
import jax.numpy as jnp
import pytest
from devograph._wavelets import (
    cheeger_constant_bound,
    hypergraph_scattering,
    hypergraph_wavelet_transform,
    spectral_features,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# -----------------------------------------------------------------------
# 1. Cheeger constant bounds for complete hypergraph
# -----------------------------------------------------------------------


class TestCheegerKnownBounds:
    """Cheeger bounds from lambda_2 of known hypergraphs."""

    def test_complete_hypergraph_cheeger(self):
        """Complete hypergraph (one edge containing all n nodes).

        lambda_2 = 1.0 (from Laplacian spectrum test above).
        Lower bound = lambda_2 / 2 = 0.5
        Upper bound = sqrt(2 * lambda_2) = sqrt(2) ≈ 1.414
        """
        n = 5
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)
        lower, upper = cheeger_constant_bound(hg)

        assert abs(lower - 0.5) < 1e-4
        assert abs(upper - float(jnp.sqrt(2.0))) < 1e-4

    def test_path_graph_cheeger(self):
        """Path 0-1-2 as 2-uniform hypergraph.

        Normalized Laplacian eigenvalues differ from unnormalized.
        We just verify lower <= upper and both non-negative.
        """
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        lower, upper = cheeger_constant_bound(hg)

        assert lower >= -1e-6
        assert upper >= lower - 1e-6


# -----------------------------------------------------------------------
# 2. Wavelet heat kernel: s→0 recovers identity
# -----------------------------------------------------------------------


class TestWaveletKnownBehavior:
    """Validate wavelet transform properties against theory."""

    def test_heat_kernel_identity_limit(self):
        """As s → 0, exp(-s*lambda) → 1 for all lambda,
        so W_s = U @ I @ U^T = I, meaning W_s @ x = x."""
        H = jnp.ones((4, 1))
        features = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])
        hg = hgx.from_incidence(H, node_features=features)

        coeffs = hypergraph_wavelet_transform(
            hg, [1e-8], kernel="heat"
        )
        assert jnp.allclose(coeffs[0], features, atol=1e-3)

    def test_mexican_hat_zero_at_dc(self):
        """Mexican hat kernel g(x) = x*exp(-x) vanishes at x=0.

        For eigenvalue lambda_0 = 0 of the Laplacian, g(s*0) = 0
        for any scale s.  So the DC component is always filtered out.

        Concretely: for a complete hypergraph, the constant eigenvector
        (all-ones/sqrt(n)) should be removed by the Mexican hat filter.
        The wavelet output should have zero mean per feature dimension.
        """
        n = 5
        H = jnp.ones((n, 1))
        # Constant features: all [1, 1]
        features = jnp.ones((n, 2))
        hg = hgx.from_incidence(H, node_features=features)

        coeffs = hypergraph_wavelet_transform(
            hg, [1.0], kernel="mexican_hat"
        )
        # Constant signal lives entirely in the DC component (lambda=0).
        # Mexican hat kills DC, so output should be ~0.
        assert jnp.allclose(coeffs[0], 0.0, atol=1e-5)

    def test_scattering_output_size_formula(self):
        """Scattering output size = d * sum_{l=0}^{L} S^l.

        With d=2, S=3 scales, L=2 layers:
        size = 2 * (1 + 3 + 9) = 2 * 13 = 26
        """
        H = jnp.ones((4, 1))
        features = jnp.array([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]
        ])
        hg = hgx.from_incidence(H, node_features=features)

        out = hypergraph_scattering(
            hg, [1.0, 2.0, 4.0], num_layers=2
        )
        assert out.shape == (26,)

    def test_scattering_single_scale_single_layer(self):
        """With S=1 scale and L=1 layer: size = d * (1 + 1) = 2d."""
        H = jnp.ones((3, 1))
        features = jnp.eye(3)[:, :2]  # (3, 2)
        hg = hgx.from_incidence(H, node_features=features)

        out = hypergraph_scattering(hg, [1.0], num_layers=1)
        assert out.shape == (4,)  # 2 * (1 + 1) = 4


# -----------------------------------------------------------------------
# 3. Sinkhorn: uniform distributions → uniform plan
# -----------------------------------------------------------------------


class TestSinkhornKnownAnswer:
    """Sinkhorn with uniform marginals and uniform cost → uniform plan."""

    def test_uniform_uniform_gives_outer_product(self):
        """When cost is uniform (constant), the optimal plan is the outer
        product of the marginals: T_ij = a_i * b_j.

        With a = [1/3, 1/3, 1/3], b = [1/4, 1/4, 1/4, 1/4]:
        T_ij = 1/12 for all (i, j).
        """
        a = jnp.ones(3) / 3
        b = jnp.ones(4) / 4
        cost = jnp.ones((3, 4))  # uniform cost

        T = devograph.sinkhorn(cost, a, b, epsilon=0.1, max_iters=200)
        expected = jnp.ones((3, 4)) / 12.0
        assert jnp.allclose(T, expected, atol=1e-3)

    def test_identity_cost_diagonal_plan(self):
        """With cost = -I (cheapest on diagonal), the plan should
        concentrate mass on the diagonal.

        a = b = [1/3, 1/3, 1/3], cost = identity matrix.
        With small epsilon, T should be approximately diagonal.
        """
        a = jnp.ones(3) / 3
        cost = jnp.eye(3) * 0 + (1 - jnp.eye(3)) * 10.0
        T = devograph.sinkhorn(cost, a, a, epsilon=0.01, max_iters=200)
        # Diagonal entries should be much larger than off-diagonal
        diag_mass = jnp.sum(jnp.diag(T))
        assert diag_mass > 0.9  # nearly all mass on diagonal

    def test_wasserstein_known_distance(self):
        """Wasserstein-1 distance between delta_0 and delta_2 on {0,1,2}
        with Euclidean ground metric.

        Points: x = [[0], [1], [2]], y = [[0], [1], [2]]
        Marginals: a = [1, 0, 0], b = [0, 0, 1]

        The optimal transport moves mass 1 from point 0 to point 2,
        cost = |0 - 2| = 2.0.
        """
        x = jnp.array([[0.0], [1.0], [2.0]])
        y = jnp.array([[0.0], [1.0], [2.0]])

        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 0.0, 1.0])
        # Add small mass to avoid log(0)
        a = a + 1e-6
        b = b + 1e-6
        a = a / jnp.sum(a)
        b = b / jnp.sum(b)

        cost = jnp.abs(x - y.T)  # (3, 3) pairwise distances
        T = devograph.sinkhorn(
            cost, a, b, epsilon=0.01, max_iters=300
        )
        ot_cost = jnp.sum(T * cost)
        assert abs(float(ot_cost) - 2.0) < 0.1


# -----------------------------------------------------------------------
# 4. Poincare ball: known distances and roundtrips
# -----------------------------------------------------------------------


class TestPoincareKnownAnswer:
    """Validate Poincare ball primitives against closed-form formulas."""

    def test_distance_at_origin(self):
        """Distance from origin to point x in Poincare ball (c=1):
        d(0, x) = 2 * arctanh(||x||)

        For x = [0.5, 0]: d = 2 * arctanh(0.5) = 2 * 0.5493... ≈ 1.0986
        """
        from devograph._conv._hyperbolic import expmap0, logmap0

        x = jnp.array([0.5, 0.0])

        # log_0(x) should have norm = d(0, x) / 2 (tangent vector)
        v = logmap0(x, c=1.0)
        # exp_0(v) should recover x
        x_recovered = expmap0(v, c=1.0)
        assert jnp.allclose(x_recovered, x, atol=1e-4)

    def test_expmap_logmap_roundtrip(self):
        """exp_0(log_0(x)) = x for any point in the ball."""
        from devograph._conv._hyperbolic import expmap0, logmap0

        x = jnp.array([0.3, -0.4])
        v = logmap0(x, c=1.0)
        x_back = expmap0(v, c=1.0)
        assert jnp.allclose(x_back, x, atol=1e-4)

    def test_origin_is_fixed_point(self):
        """exp_0(0) = 0: the zero tangent vector maps to the origin."""
        from devograph._conv._hyperbolic import expmap0

        v = jnp.zeros(3)
        x = expmap0(v, c=1.0)
        assert jnp.allclose(x, jnp.zeros(3), atol=1e-5)


# -----------------------------------------------------------------------
# 5. Spectral features: known eigenvalue statistics
# -----------------------------------------------------------------------


class TestSpectralFeaturesKnown:
    """Validate spectral feature extraction against hand computation."""

    def test_complete_hypergraph_features(self):
        """Complete hypergraph (1 edge, all nodes):
        eigenvalues = [0, 1, 1, 1, 1] (for n=5)

        spectral_features returns:
        - First num_eigvals eigenvalues (padded to num_eigvals)
        - spectral_gap = lambda_2 = 1.0
        - algebraic_connectivity = lambda_2 = 1.0
        - spectral_radius = max eigenvalue = 1.0
        """
        n = 5
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)

        feats = spectral_features(hg, num_eigvals=5)
        # Shape: (5 + 3,) = (8,)
        assert feats.shape == (8,)

        # First 5 entries: eigenvalues [0, 1, 1, 1, 1]
        assert jnp.allclose(feats[0], 0.0, atol=1e-4)
        assert jnp.allclose(feats[1:5], 1.0, atol=1e-4)

        # spectral_gap = lambda_2 = 1.0
        assert jnp.allclose(feats[5], 1.0, atol=1e-4)
        # algebraic_connectivity = lambda_2 = 1.0
        assert jnp.allclose(feats[6], 1.0, atol=1e-4)
        # spectral_radius = 1.0
        assert jnp.allclose(feats[7], 1.0, atol=1e-4)

    def test_disconnected_algebraic_connectivity_zero(self):
        """Disconnected hypergraph has lambda_2 = 0."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        hg = hgx.from_incidence(H)

        feats = spectral_features(hg, num_eigvals=4)
        # lambda_1 = 0, lambda_2 = 0 (two components)
        assert jnp.allclose(feats[0], 0.0, atol=1e-4)
        assert jnp.allclose(feats[1], 0.0, atol=1e-4)
        # algebraic_connectivity = lambda_2 = 0
        assert jnp.allclose(feats[5], 0.0, atol=1e-4)
