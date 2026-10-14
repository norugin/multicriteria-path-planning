import math
import unittest

from path_planning.config import ExperimentConfig
from path_planning.geometry import Box, WorldMap
from path_planning.planner import AStarPlanner
from path_planning.stats import pairwise_deltas, wilcoxon_greater
from path_planning.postprocess import configuration_edge_valid, configuration_path_metrics, postprocess_configurations


class MultiCriteriaExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(
            bounds_m=(30.0, 30.0, 10.0),
            grid_step_m=5.0,
            heading_count=16,
            heading_change_length_m=15.0,
            safety_clearance_m=0.0,
            minimum_pair_distance_m=10.0,
            max_expansions=50000,
        )

    def test_edge_cost(self):
        world = WorldMap("empty", self.config.bounds_m, [], 0.0)
        planner = AStarPlanner(world, self.config)
        edge = (5.0, 0.0, 0.0)
        self.assertAlmostEqual(planner.edge_cost(edge, "Astar-L"), 1.0)
        self.assertAlmostEqual(planner.edge_cost(edge, "Astar-L-Psi-K"), 0.5)

    def test_astar_finds_path_on_empty_grid(self):
        world = WorldMap("empty", self.config.bounds_m, [], 0.0)
        planner = AStarPlanner(world, self.config)
        result = planner.plan((0, 0, 0, 0), (4, 0, 1, 0), "Astar-L-Psi-K")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.states[0], (0, 0, 0, 0))
        self.assertEqual(result.states[-1], (4, 0, 1, 0))

    def test_obstacle_blocks_segment(self):
        world = WorldMap("blocked", self.config.bounds_m, [Box(8, 12, 0, 10, 0, 10)], 0.0)
        self.assertFalse(world.segment_free((0, 5, 5), (20, 5, 5)))

    def test_segment_cannot_tunnel_between_collision_samples(self):
        world = WorldMap(
            "thin-wall",
            self.config.bounds_m,
            [Box(9.9, 10.1, 0, 10, 0, 10)],
            0.0,
        )
        self.assertFalse(world.segment_free((0, 5, 5), (20, 5, 5)))
        self.assertTrue(world.segment_free((0, 20, 5), (20, 20, 5)))


    def test_wilcoxon_direction(self):
        result = wilcoxon_greater([1, 2, 3, 4, 5])
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["p_method"], "exact")
        self.assertAlmostEqual(result["p_one_sided"], 1 / 32)
        self.assertGreater(result["z"], 0)
        self.assertLess(result["p_one_sided"], 0.1)

    def test_pairwise_delta_sign_favors_lower_candidate_value(self):
        common = {
            "map": "test",
            "pair_id": 0,
            "status": "ok",
            "course_change_rad": 2.0,
            "heading_penalty_inv_m": 2.0,
            "normalized_criterion": 2.0,
            "runtime_s": 2.0,
            "expanded": 20,
        }
        rows = [
            {**common, "method": "Astar-L-K", "path_sha256": "a", "length_m": 10.0},
            {**common, "method": "Astar-L-Psi-K", "path_sha256": "b", "length_m": 8.0},
        ]
        result = pairwise_deltas(rows, baseline="Astar-L-K")[0]
        self.assertEqual(result["improvement_length_m"], 2.0)
        self.assertEqual(result["improvement_length_m_percent"], 20.0)
        self.assertFalse(result["paths_identical"])

    def test_postprocessing_preserves_boundary_courses_and_admissibility(self):
        world = WorldMap("empty", self.config.bounds_m, [], 0.0)
        planner = AStarPlanner(world, self.config)
        result = planner.plan((0, 0, 0, 0), (5, 5, 1, 4), "Astar-L")
        self.assertEqual(result.status, "ok")
        path = [(planner.point(s), planner.heading(s)) for s in result.states]
        processed = postprocess_configurations(path, world, self.config)
        self.assertEqual(processed.path[0], path[0])
        self.assertEqual(processed.path[-1], path[-1])
        self.assertTrue(all(configuration_edge_valid(a, b, world, self.config)
                            for a, b in zip(processed.path, processed.path[1:])))
        self.assertGreaterEqual(configuration_path_metrics(processed.path, self.config).course_change_rad,
                                math.pi / 2 - 1e-10)

    def test_postprocessing_rejects_invalid_input_instead_of_hiding_it(self):
        world = WorldMap("blocked", self.config.bounds_m, [Box(8, 12, 0, 10, 0, 10)], 0.0)
        with self.assertRaises(ValueError):
            postprocess_configurations([((0., 5., 5.), 0.), ((20., 5., 5.), 0.)], world, self.config)

    def test_signed_rank_normal_variance_with_equal_ranks(self):
        n = 30
        result = wilcoxon_greater([1.] * n)
        rank = (n + 1) / 2
        expected_z = (n * rank / 2 - .5) / math.sqrt(n * rank * rank / 4)
        self.assertAlmostEqual(result["z"], expected_z)


if __name__ == "__main__":
    unittest.main()
