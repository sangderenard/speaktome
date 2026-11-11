"""The two handlers added to the shared REST contract: pictures (show, with a camera) and a map as pages."""
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import speaktome_man as manual  # noqa: E402
from speaktome_api import routes, site  # noqa: E402
from test_speaktome_site import GraphView  # noqa: E402  the real indexer when the turing checkout is beside this one


class PictureError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Handler:
    """The parts of the host's request handler the shared routes use."""

    def __init__(self, *, path, caps=("photo_submit", "view_site"), account={"id": 7}, body=None, viewer=False,
                 content_type="application/json"):
        self.path, self.caps, self.account, self.body, self.viewer = path, set(caps), account, body, viewer
        self.headers = {"Content-Type": content_type}
        self.sent, self.status, self.out_headers, self.wfile = None, None, [], io.BytesIO()
        self.close_connection = False

    def _is_viewer(self):
        return self.viewer

    def _refuse_viewer(self):
        return self._json({"error": "viewers may not"}, 403)

    def _require(self, cap):
        if cap not in self.caps:
            self._json({"error": f"this needs {cap}"}, 403)
            return False
        return True

    def _account(self):
        return self.account

    def _body_json(self, limit):
        return self.body

    def _json(self, obj, status=200):
        self.sent, self.status = obj, int(status)

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.out_headers.append((name, value))

    def end_headers(self):
        pass


class Service:
    def __init__(self):
        self.submitted = []

    def submit(self, account_id, owners, recheck, body):
        if body.get("request_id") == "taken":
            raise PictureError("request_id already used with different content", 409)
        self.submitted.append((account_id, tuple(owners), recheck(), body))
        return {"picture_id": "a" * 32, "state": "queued", **({"replayed": True} if body.get("request_id") == "again" else {})}

    def status(self, account_id, picture_id):
        if account_id != 7:
            raise PictureError("picture not found", 404)
        return {"picture_id": picture_id, "state": "committed", "job": {"digest": "d" * 64}}


def picture(handler, method, service="default"):
    service = Service() if service == "default" else service
    routes.picture_route(handler, handler.path.partition("?")[0], method, service=service, owners=["account:7"],
                         recheck=lambda: True, PictureError=PictureError, MAX_BODY=1024)
    return handler, service


def test_a_picture_is_asked_for_and_its_receipt_read():
    handler, service = picture(Handler(path="/api/show/pictures", body={"request_id": "one", "photo": {}}), "POST")
    assert handler.status == 202 and handler.sent["state"] == "queued" and handler.close_connection
    assert service.submitted == [(7, ("account:7",), True, {"request_id": "one", "photo": {}})]
    handler, _ = picture(Handler(path="/api/show/pictures", body={"request_id": "again"}), "POST")
    assert handler.status == 200 and handler.sent["replayed"] is True
    handler, _ = picture(Handler(path="/api/show/pictures/" + "a" * 32), "GET")
    assert handler.status == 200 and handler.sent["job"]["digest"] == "d" * 64


@pytest.mark.parametrize("handler, method, status", [
    (Handler(path="/api/show/pictures", caps=(), body={}), "POST", 403),
    (Handler(path="/api/show/pictures", account=None, body={}), "POST", 401),
    (Handler(path="/api/show/pictures", viewer=True, body={}), "POST", 403),
    (Handler(path="/api/show/pictures", body={}, content_type="text/plain"), "POST", 415),
    (Handler(path="/api/show/pictures?x=1", body={}), "POST", 400),
    (Handler(path="/api/show/pictures/extra", body={}), "POST", 404),
    (Handler(path="/api/show/pictures", body={"request_id": "taken"}), "POST", 409),
    (Handler(path="/api/show/pictures/not-an-id"), "GET", 404),
    (Handler(path="/api/show/pictures/" + "a" * 32, account={"id": 8}), "GET", 404),
])
def test_picture_refusals(handler, method, status):
    assert picture(handler, method)[0].status == status


def test_no_studio_destination_is_a_plain_503():
    assert picture(Handler(path="/api/show/pictures", body={}), "POST", service=None)[0].status == 503


@pytest.fixture
def maps(tmp_path):
    made = site.Site(tmp_path, GraphView)
    return made, {"manual": {"source": manual.as_map, "capability": "view_site", "ttl": 60.0},
                  "system": {"source": lambda: (_ for _ in ()).throw(RuntimeError("down")), "capability": "dev_ui"}}


def page(handler, maps):
    routes.site_route(handler, handler.path, site=maps[0], maps=maps[1])
    return handler


def test_a_map_is_served_as_pages_and_kept(maps, tmp_path):
    root = page(Handler(path="/site/manual/"), maps)
    assert root.status == 200 and b"SpeakToMe Manual" in root.wfile.getvalue()
    headers = dict(root.out_headers)
    assert headers["Content-Type"].startswith("text/html") and "default-src 'none'" in headers["Content-Security-Policy"]
    address = site.address(manual.place(manual.page("go")))
    go = page(Handler(path=f"/site/manual/{address}.html"), maps)
    assert go.status == 200 and b"follow one declared edge" in go.wfile.getvalue()
    kept = sorted(p.name for p in tmp_path.glob("manual/*/*.html"))
    assert kept == sorted(["index.html", address + ".html"]), "only the places somebody looked at were written"


@pytest.mark.parametrize("path, caps, status", [
    ("/site/manual", ("view_site",), 301),
    ("/site/manual/", (), 403),
    ("/site/unknown/", ("view_site",), 404),
    ("/site/manual/nothing-here.html", ("view_site",), 404),
    ("/site/manual/../x.html", ("view_site",), 404),
    ("/site/system/", ("view_site",), 403),
    ("/site/system/", ("dev_ui",), 503),
])
def test_site_refusals(maps, path, caps, status):
    assert page(Handler(path=path, caps=caps), maps).status == status
