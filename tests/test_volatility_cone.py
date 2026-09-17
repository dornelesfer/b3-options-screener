import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from volatility_cone import cone, implied_vol
from streamlit.testing.v1 import AppTest


class ConeTests(unittest.TestCase):
    def test_no_future_data_or_current_reference(self):
        rng = np.random.default_rng(8)
        s = pd.Series(np.exp(rng.normal(0, .01, 900).cumsum())*100,
                      index=pd.bdate_range('2020-01-01', periods=900))
        day = s.index[700]
        a = cone(s, day, 504)
        s.iloc[701:] *= 20
        pd.testing.assert_frame_equal(a, cone(s, day, 504))
        s.iloc[700] *= 2
        b = cone(s, day, 504)
        np.testing.assert_allclose(a['median'], b['median'])
        self.assertFalse(np.allclose(a.current, b.current))

    def test_invalid_price_and_quote_order(self):
        self.assertTrue(np.isnan(implied_vol(200, 100, 100, .1, .1, True)))
        self.assertLess(implied_vol(4, 100, 100, .1, .1, True),
                        implied_vol(6, 100, 100, .1, .1, True))

    def test_controls(self):
        root = Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root/'app_options_screener.py'), default_timeout=120).run()
        self.assertFalse(at.exception)
        at.selectbox(key='cone_type').select('Puts').run()
        self.assertFalse(at.exception)
        next(b for b in at.button if b.label == 'Latest quoted IV').click().run()
        self.assertFalse(at.exception)
        next(b for b in at.button if b.label == 'March 2020').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(str(at.date_input(key='cone_date').value), '2020-03-23')


if __name__ == '__main__':
    unittest.main()
