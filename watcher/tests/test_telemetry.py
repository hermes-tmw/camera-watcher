import unittest

from watcher.model import StealthcamTelemetry
from watcher.tests.utils import TransactionalTestCase


class TestStealthcamTelemetry(TransactionalTestCase, unittest.TestCase):
    def test_telemetry_roundtrip(self):
        t = StealthcamTelemetry(
            device_id='402_867490078575292',
            battery_pct=52,
            battery_volt=9.98,
            sd_card_free_pct=80,
            rssi=67,
            signal_strength='Excellent',
            firmware_version='26799.313.066',
            on_demand_state='OnDemandEnabled',
            errors=['NetworkRegistrationError'],
            raw={'batteryLevel': 52},
        )
        self.session.add(t)
        self.session.commit()

        loaded = self.session.get(StealthcamTelemetry, t.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.battery_pct, 52)
        self.assertEqual(loaded.sd_card_free_pct, 80)
        self.assertEqual(loaded.signal_strength, 'Excellent')
        self.assertEqual(loaded.errors, ['NetworkRegistrationError'])
        self.assertEqual(loaded.raw['batteryLevel'], 52)
        self.assertIsNotNone(loaded.captured_at)

    def test_telemetry_nulls_allowed(self):
        t = StealthcamTelemetry()  # all optional fields None
        self.session.add(t)
        self.session.commit()
        loaded = self.session.get(StealthcamTelemetry, t.id)
        self.assertIsNotNone(loaded)
        self.assertIsNone(loaded.battery_pct)
        self.assertIsNone(loaded.signal_strength)


if __name__ == '__main__':
    unittest.main()
