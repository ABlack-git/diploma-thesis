import pytest
import torch
import im2gps.core.nn.functional as f
from im2gps.core.nn.enum import NNEnum


class TestHaversine:

    @pytest.fixture
    def prediction(self):
        return torch.tensor([[14.430575, 50.079149], [14.414050, 50.075491], [14.427924, 50.087222],
                             [14.444172, 50.098335], [14.401520, 50.090994]])

    @pytest.fixture
    def g_true(self):
        return torch.tensor([[14.431575, 50.079349], [14.414250, 50.078491], [14.423924, 50.087252],
                             [14.454172, 50.099335], [14.401920, 50.092994]])

    @pytest.fixture
    def true_distance(self):
        return torch.tensor([0.07471776, 0.33371788, 0.28533468, 0.7219815, 0.22397016])

    def test_haversine_loss_should_be_zero(self, prediction):
        result = f.haversine_loss(prediction, prediction)
        assert result == 0.0

    def test_haversine_distance_non_zero(self, prediction, g_true, true_distance):
        result = f.haversine_distance(prediction, g_true)
        assert torch.equal(result, true_distance)

    def test_haversine_loss_non_zero(self, prediction, g_true, true_distance):
        result = f.haversine_loss(prediction, g_true)
        assert result == torch.mean(true_distance)

    def test_haversine_loss_1_input(self, prediction, g_true, true_distance):
        p = prediction[0]
        gt = g_true[0]
        result = f.haversine_loss(p, gt)
        assert result == true_distance[0]


class TestDistance:
    @pytest.fixture
    def queries(self):
        """
        Batch size = 2
        Dimension = 5
        :return: 2x5
        """
        return torch.tensor([[-0.0802184269, 0.4425332844, -0.1906075180, -0.3068793416,
                              0.5817509294],
                             [-0.4381977916, -1.7224467993, 1.9919376373, -0.9736493826,
                              -1.4822139740]])

    @pytest.fixture
    def neighbours(self):
        """
        batch size = 2
        number of neighbours = 2
        dimension = 5
        :return: 2x2x5
        """
        return torch.tensor([[[-0.2095449716, 2.6645848751, 1.4389493465, 0.0472722761,
                               0.7187049985],
                              [1.5262542963, 0.2260963023, 0.5487589240, 0.9781837463,
                               -1.0428373814]],

                             [[-0.3364185691, 0.3000151813, -0.7504291534, 0.8402043581,
                               0.9733895659],
                              [0.1385173798, 0.3221608400, 0.6292650700, 0.0852660239,
                               0.0886382610]]])

    @pytest.fixture
    def true_l2_dist(self):
        """
        batch_size = 2
        number of neighbours =2
        2x2
        """
        return torch.tensor([[2.7845778465, 2.7322034836],
                             [4.5761709213, 3.1557528973]])

    @pytest.fixture
    def true_cos_dist(self):
        return torch.tensor([[0.5185089111, -0.5804580450],
                             [-0.8274030685, 0.1807945222]])

    def test_distance_from_query_l2_case_1(self, queries, neighbours, true_l2_dist):
        res = f.distance_from_query(queries, neighbours, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == [2, 2]
        assert torch.allclose(true_l2_dist, res, rtol=0)

    def test_distance_from_query_l2_case_2(self, queries, neighbours, true_l2_dist):
        n = neighbours[:, 0]
        true_dist = true_l2_dist[:, 0].unsqueeze(1)
        res = f.distance_from_query(queries, n, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == [2, 1]
        assert torch.allclose(true_dist, res, rtol=0)

    def test_distance_from_query_l2_case_3(self, queries, neighbours, true_l2_dist):
        q = queries[0]
        n = neighbours[0]
        true_dist = true_l2_dist[0]
        res = f.distance_from_query(q, n, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == [2]
        assert torch.allclose(true_dist, res, rtol=0)

    def test_distance_from_query_l2_case_4(self, queries, neighbours, true_l2_dist):
        q = queries[0]
        n = neighbours[0, 0]
        true_dist = true_l2_dist[0, 0]
        res = f.distance_from_query(q, n, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == []
        assert torch.allclose(true_dist, res, rtol=0)

    def test_distance_from_query_cos_case_1(self, queries, neighbours, true_cos_dist):
        res = f.distance_from_query(queries, neighbours, dist_type=NNEnum.COS_DIST)
        assert list(res.shape) == [2, 2]
        assert torch.allclose(true_cos_dist, res)

    def test_distance_from_query_cos_case_2(self, queries, neighbours, true_cos_dist):
        n = neighbours[:, 0]
        true_dist = true_cos_dist[:, 0].unsqueeze(1)
        res = f.distance_from_query(queries, n, dist_type=NNEnum.COS_DIST)
        assert list(res.shape) == [2, 1]
        assert torch.allclose(true_dist, res, rtol=0)

    def test_distance_from_query_cos_case_3(self, queries, neighbours, true_cos_dist):
        q = queries[0]
        n = neighbours[0]
        true_dist = true_cos_dist[0]
        res = f.distance_from_query(q, n, dist_type=NNEnum.COS_DIST)
        assert list(res.shape) == [2]
        assert torch.allclose(true_dist, res, rtol=0)

    def test_distance_from_query_cos_case_4(self, queries, neighbours, true_cos_dist):
        q = queries[0]
        n = neighbours[0, 0]
        true_dist = true_cos_dist[0, 0]
        res = f.distance_from_query(q, n, dist_type=NNEnum.COS_DIST)
        assert list(res.shape) == []
        assert torch.allclose(true_dist, res, rtol=0)

    def test_distance_from_query_should_raise_error(self):
        # Raise when q is 3D
        q = torch.randn(2, 2, 5)
        n = torch.randn(2, 2, 5)

        with pytest.raises(ValueError):
            f.distance_from_query(q, n)

        with pytest.raises(ValueError):
            f.distance_from_query(q, n, dist_type=NNEnum.COS_DIST)

        # Raise when q is ok, but n is 1d
        q = torch.randn(2, 5)
        n = torch.randn(5)

        with pytest.raises(ValueError):
            f.distance_from_query(q, n)

        with pytest.raises(ValueError):
            f.distance_from_query(q, n, dist_type=NNEnum.COS_DIST)

        # raise when q is 1d and n is 3d
        q = torch.randn(5)
        n = torch.randn(2, 2, 5)

        with pytest.raises(ValueError):
            f.distance_from_query(q, n)

        with pytest.raises(ValueError):
            f.distance_from_query(q, n, dist_type=NNEnum.COS_DIST)


class TestDescriptors2Weights:
    @pytest.fixture
    def distances(self):
        return torch.tensor([[0.3020126224, 0.0478253961, 11., 0.6419240832, 0.3358792663],
                             [1.25, 0.4331991673, 0.5436970592, 0.2418629527, 0.5687664151],
                             [0., 0.0095019341, 0.9684504271, 0.8812267780, 0.5172470212],
                             [0.0001, 20., 0.6162103415, 0.2013697028, 0.9906123281]])

    @pytest.fixture
    def true_weights_l2_no_m(self):
        return torch.tensor([[3.3111197948e+00, 2.0909387589e+01, 9.0909093618e-02, 1.5578166246e+00,
                              2.9772603512e+00],
                             [8.0000001192e-01, 2.3084070683e+00, 1.8392595053e+00, 4.1345725060e+00,
                              1.7581909895e+00],
                             [1.0000000000e+08, 1.0524161530e+02, 1.0325773954e+00, 1.1347817183e+00,
                              1.9333122969e+00],
                             [9.9990009766e+03, 5.0000000745e-02, 1.6228225231e+00, 4.9659900665e+00,
                              1.0094766617e+00]])

    def test_dist2weights_l2_m_1(self, distances, true_weights_l2_no_m):
        res = f.dist2weights(distances, m=1, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == [4, 5]
        assert torch.allclose(res, true_weights_l2_no_m), \
            f"actual = {res}\n expected={true_weights_l2_no_m}\n " \
            f"close = {torch.isclose(res, true_weights_l2_no_m)}"

    def test_dist2weights_l2_m_4(self, distances, true_weights_l2_no_m):
        m = 4
        res = f.dist2weights(distances, m=m, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == [4, 5]
        # assert all_is_finite(res), f"actual = {res}\n isfinite = {torch.isfinite(res)}"
        assert torch.allclose(res, true_weights_l2_no_m ** m), \
            f"actual = {res}\n expected={true_weights_l2_no_m ** m}\n " \
            f"close = {torch.isclose(res, true_weights_l2_no_m ** m)}"

    def test_dist2weights_l2_m_20(self, distances, true_weights_l2_no_m):
        m = 20
        res = f.dist2weights(distances, m=m, dist_type=NNEnum.L2_DIST)
        assert list(res.shape) == [4, 5]
        # assert all_is_finite(res), f"actual = {res}\n isfinite = {torch.isfinite(res)}"
        assert torch.allclose(res, true_weights_l2_no_m ** m,
                              rtol=0), f"actual = {res}\n expected={true_weights_l2_no_m ** m}\n " \
                                       f"close = {torch.isclose(res, true_weights_l2_no_m ** m, rtol=0)}"


class TestMultivariateNormalPDF:
    @pytest.fixture
    def means(self):
        return torch.tensor([[[14.453039, 50.104427], [14.414444, 50.074670], [14.400455, 50.090763]],
                             [[14.400455, 50.090763], [14.414444, 50.074670], [14.453039, 50.104427]]])

    @pytest.fixture
    def points(self):
        return torch.tensor([[[14.44308635, 50.04943377],
                              [14.48446979, 50.11236874],
                              [14.45012951, 50.09965328],
                              [14.49049402, 50.1095411],
                              [14.42802713, 50.06363684]],

                             [[14.46309713, 50.0860999],
                              [14.40795061, 50.09153721],
                              [14.41553589, 50.03612124],
                              [14.35454744, 50.10319183],
                              [14.37444846, 50.09177377]]
                             ])

    @pytest.fixture
    def pdfs(self):
        return torch.tensor([[[33.3888816833, 76.8036727905, 27.3060855865],
                              [94.1035079956, 6.7360591888, 3.6959283352],
                              [156.6871948242, 61.6250381470, 44.5486221313],
                              [77.8950119019, 4.8070263863, 2.3164992332],
                              [50.6613273621, 136.5597381592, 75.3280639648]],

                             [[22.1311855316, 45.6495323181, 127.9134597778],
                              [154.6997985840, 135.1715545654, 53.0016708374],
                              [31.9221649170, 75.6621932983, 7.6430606842],
                              [51.3614692688, 17.6252384186, 1.2446093559],
                              [113.4312973022, 61.7928161621, 6.6965308189]]])

    def test_multivariate_normal(self, points, means, pdfs):
        res = f.multivariate_normal_pdf(points, means, 0.001)
        assert torch.allclose(res, pdfs, atol=1e-02, rtol=0), f"res = {res}\npdfs={pdfs}"

    def test_multivariate_normal_no_batch(self, points, means, pdfs):
        p = points[0]
        mu = means[0]
        pdf = pdfs[0]

        res = f.multivariate_normal_pdf(p, mu, 0.001)
        assert list(res.shape) == [5, 3]
        assert torch.allclose(res, pdf, atol=1e-02, rtol=0), f"res = {res}\npdfs={pdfs}"

    def test_multivariate_normal_should_raise(self, points, means, pdfs):
        p = points[0]
        with pytest.raises(AssertionError):
            f.multivariate_normal_pdf(p, means, 0.001)


class TestKDE:
    pass


def all_is_finite(x):
    return all(torch.isfinite(torch.flatten(x)).tolist())
