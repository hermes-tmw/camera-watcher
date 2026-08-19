import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from watcher.model import WatcherBase
from watcher.tests.utils import test_db_url


class TestThumbnailRoute(unittest.TestCase):
    """Exercise the lazy /thumb route.

    The route generates a downscaled JPEG on first request, caches it on disk
    under LOCAL_DATA_DIR/thumb/, and serves it with a long Cache-Control.
    LOCAL_DATA_DIR is overridden via the WATCHER_SYSTEM_LOCAL_DATA_DIR env var
    (the same override the running services use) so the test never touches the
    real /data/video tree.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._data_dir = Path(cls._tmp.name)
        os.environ['WATCHER_SYSTEM_LOCAL_DATA_DIR'] = str(cls._data_dir)

        from api import create_app, db
        cls.app = create_app(db_url=test_db_url(), testing=True)
        cls.app.config['TESTING'] = True
        with cls.app.app_context():
            WatcherBase.metadata.create_all(db.engine)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        from api import db
        with cls.app.app_context():
            WatcherBase.metadata.drop_all(db.engine)
        os.environ.pop('WATCHER_SYSTEM_LOCAL_DATA_DIR', None)
        cls._tmp.cleanup()

    def _make_source(self, relpath, size=(1024, 576)):
        src = self._data_dir / relpath
        src.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', size, (120, 40, 200)).save(src, 'JPEG', quality=90)
        return src

    def test_generates_and_serves_thumbnail(self):
        relpath = 'stealthcam/test-guid_1.JPG'
        self._make_source(relpath)

        r = self.client.get(f'/thumb/{relpath}')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, 'image/jpeg')
        self.assertIn('immutable', r.headers.get('Cache-Control', ''))

        # Thumbnail was written to disk and is smaller than the source.
        thumb = self._data_dir / 'thumb' / 'stealthcam' / 'test-guid_1.jpg'
        self.assertTrue(thumb.is_file())
        src = self._data_dir / relpath
        self.assertLess(thumb.stat().st_size, src.stat().st_size)

        # Downscaled to <= THUMB_MAX_WIDTH.
        with Image.open(thumb) as im:
            self.assertLessEqual(im.width, 320)

    def test_second_request_serves_cached_file(self):
        relpath = 'stealthcam/test-guid-cached_1.JPG'
        self._make_source(relpath)

        r1 = self.client.get(f'/thumb/{relpath}')
        self.assertEqual(r1.status_code, 200)
        thumb = self._data_dir / 'thumb' / 'stealthcam' / 'test-guid-cached_1.jpg'
        mtime = thumb.stat().st_mtime

        r2 = self.client.get(f'/thumb/{relpath}')
        self.assertEqual(r2.status_code, 200)
        # Cached file is not regenerated (mtime unchanged).
        self.assertEqual(thumb.stat().st_mtime, mtime)

    def test_missing_source_returns_404(self):
        r = self.client.get('/thumb/stealthcam/does-not-exist_1.JPG')
        self.assertEqual(r.status_code, 404)

    def test_path_traversal_rejected(self):
        r = self.client.get('/thumb/../../etc/passwd')
        self.assertEqual(r.status_code, 404)


if __name__ == '__main__':
    unittest.main()
