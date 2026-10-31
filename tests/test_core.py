import unittest
from inno_medical_drug_trace import core


class TestDrugTrace(unittest.TestCase):
    def test_transfers_conserve_and_respect_surplus(self):
        s = {"A": 50, "B": 30, "C": 0}; d = {"X": 60, "Y": 40}
        mv = core.suggest_transfers(s, d)
        for c in s:
            self.assertLessEqual(sum(m["units"] for m in mv if m["from"] == c), s[c])
        for c in d:
            self.assertLessEqual(sum(m["units"] for m in mv if m["to"] == c), d[c])
        self.assertEqual(sum(m["units"] for m in mv), 80)

    def test_no_transfer_without_surplus_or_deficit(self):
        self.assertEqual(core.suggest_transfers({}, {"X": 10}), [])
        self.assertEqual(core.suggest_transfers({"A": 10}, {}), [])

    def test_forecast_nonnegative_and_shape(self):
        _, y = core.simulate(3)
        f = core.forecast(y[:, :, :100])
        self.assertEqual(f.shape, (core.ITEMS, core.CENTERS, core.HORIZON))
        self.assertTrue((f >= 0).all())

    def test_unsold_fraction_bounds(self):
        self.assertEqual(core.unsold_fraction(100, 150), 0)
        self.assertAlmostEqual(core.unsold_fraction(100, 40), 0.6)
