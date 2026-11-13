"""Machine-readable catalog of the SpeakToMe service's REST API.

This is pure data. It is declared BY HAND from the real handlers: ``turing/examples/meowdoku/server.py``
(``do_GET`` / ``do_POST`` and the helper methods they call), ``speaktome_api/routes.py`` and ``reports.py``,
and the modules those delegate to. A tool generator reads ``ENDPOINTS`` to learn what each call accepts.
Nothing here is imported by the server, so nothing here can change what the server does.

How to keep it true: when a handler gains, loses or renames a path, a query key, a body field or a
capability, change the matching entry in the same commit. ``tests/test_speaktome_catalog.py`` checks the
shape of every entry and that each path's literal prefix still occurs in the handler sources; it cannot
prove the fields, so read the handler when you edit.

Entry keys: name, group, method (GET | POST), path ({name} marks a variable part), summary, capability
(None, one name, or a tuple meaning "any of"; with capability_all True the tuple means "all of"), account (a
signed-in account is required), headers (fixed request headers the handler insists on), request (json, or
binary for an upload whose body is not a JSON object and which a generated tool therefore cannot send), changes
(the call changes server state), path_params, query, body (None, or {"fields": {...}, "open": bool};
open means more fields pass through to a deeper layer that is not enumerated), returns, tool (False means
never offer it to an automated agent as a generated tool), kiosk (True: a kiosk account, the read-only intern
guest agent, may use it; the host answers a kiosk account on these entries only, see ``kiosk_endpoint``), guest (True:
a visitor with no account keeps it beside the kiosk entries: the pages' and the game table's calls), credential (None, or
the name of the credential the handler itself checks, such as admin_token: a caller with no account reaches the entry
and the handler's check is the authority; see ``guest_endpoint``).
"""

CATALOG_VERSION = 1

GROUPS = {
    "identity": "Who you are: sign-in state, accounts, sessions and role grants.",
    "messages": "User-to-user messaging: contacts, threads, images and the moderation view.",
    "living_data": "Navigating and querying Living Data graph maps, static DM worlds and the stored world maps.",
    "photo": "Marble photo jobs and requests for pictures of the Studio scene.",
    "studio": "Studio assets: loading animations, OBJ assets, marble objects and the camera housing.",
    "nodus": "Inspect a running Nodus canvas and send authorized UI actions.",
    "control": "Operating the service: instance restarts, server route choice, the log ring and lifecycle.",
    "profile": "The profile page data: your account summary and, for operators, instances and leases.",
    "reports": "Experience reports, account-owned issue tickets and legacy tickets.",
    "reviews": "Changes put up for peer review before they reach a default branch.",
    "git": "Git in the core, regulated: declared read operations, recorded in SQL. Changes go through reviews.",
    "relay": "Datagrams and packages, address to address, in and out, for any signed-in account.",
    "files": "The flat index of every file in the core, kept by the cleanup account.",
    "compile": "Compile-side jobs: the source divider, the math lab and the lowering leases.",
    "site": "Maps rendered as read-only web pages.",
    "play": "The game table tied to a browser session, sharing and spectating, and page support calls.",
    "markets": "Alpaca paper market observation, strategy seats and attributed paper orders.",
    "foyer": "The one shared demo world anyone may walk and photograph; nothing in it is yours.",
}


def _fields(spec):
    return {k: dict(type=v[0], required=v[1], description=v[2]) if v[1] else dict(type=v[0], description=v[2])
            for k, v in spec.items()}


def _body(spec, open_=False):
    return {"fields": _fields(spec), "open": open_}


def _named(spec):
    return {k: {"type": v[0], "description": v[1]} for k, v in spec.items()}


def _ep(name, group, method, path, summary, capability, account, changes, *, path_params=None, query=None,
        body=None, returns="json", tool=True, headers=None, capability_all=False, request="json", kiosk=False,
        guest=False, credential=None):
    return {
        "name": name, "group": group, "method": method, "path": path, "summary": summary,
        "capability": capability, "capability_all": capability_all, "account": account, "changes": changes,
        "headers": dict(headers or {}), "request": request,
        "path_params": _named(path_params or {}), "query": _named(query or {}), "body": body,
        "returns": returns, "tool": tool, "kiosk": kiosk, "guest": guest, "credential": credential,
    }


_ID32 = {"id": ("string", "The 32 hex character identifier.")}
_ADMIN_OR_LIFECYCLE = ("account_manage", "lifecycle_admin")

ENDPOINTS = (
    # ---- identity
    _ep("identity_login", "identity", "POST", "/api/account/login", "Sign in with a username and password.",
        None, False, True, tool=False, kiosk=True,
        body=_body({"username": ("string", True, "The account name."), "password": ("string", True, "The password.")})),
    _ep("identity_signup", "identity", "POST", "/api/account/signup", "Create an account and sign it in.",
        None, False, True, tool=False,
        body=_body({"username": ("string", True, "The new account name."), "password": ("string", True, "The password.")}), guest=True),
    _ep("identity_logout", "identity", "POST", "/api/account/logout", "End this session.",
        None, False, True, tool=False, kiosk=True),
    _ep("identity_me", "identity", "GET", "/api/account/me", "Show the signed-in account, its role and its instance, or no account.",
        None, False, False, kiosk=True),
    _ep("identity_assets", "identity", "GET", "/api/account/assets", "List the marble objects, photos, OBJ assets and profile this account owns.",
        None, True, False),
    _ep("identity_index", "identity", "GET", "/api/index",
        "List every REST endpoint with what it accepts and whether you may use it.",
        None, False, False, query={"group": ("string", "Only this group's endpoints.")}, kiosk=True),
    _ep("identity_whoami", "identity", "GET", "/api/whoami", "Show this request's tier, role and the capabilities it holds.",
        None, False, False, kiosk=True),
    _ep("identity_admin_list", "identity", "GET", "/api/account/admin/list", "List the admin accounts and all users.",
        "account_manage", True, False, tool=False),
    _ep("identity_admin_grant", "identity", "POST", "/api/account/admin/grant", "Grant the admin role to an account.",
        "account_manage", True, True, tool=False,
        body=_body({"username": ("string", True, "The account to promote.")})),
    _ep("identity_admin_revoke", "identity", "POST", "/api/account/admin/revoke", "Revoke the admin role from an account.",
        "account_manage", True, True, tool=False,
        body=_body({"username": ("string", True, "The account to demote.")})),
    _ep("identity_operator_grant", "identity", "POST", "/api/admin/account/grant",
        "Bootstrap the first admin with the operator's loopback token.",
        None, False, True, tool=False,
        body=_body({"username": ("string", True, "The account to promote.")}), credential="admin_token"),

    # ---- foyer: the shared demo world (owner, 2026-11-13); kiosk, so guests, guest agents and interns all reach it
    _ep("foyer_status", "foyer", "GET", "/api/foyer", "Show the foyer greeting, the demo world's identity and its start camera.",
        "view_site", False, False, kiosk=True),
    _ep("foyer_command", "foyer", "POST", "/api/foyer/commands",
        "Ask the shared demo host for one inspect or one picture (render-phong) at a camera; one picture at a time.",
        "view_site", False, True, kiosk=True,
        body=_body({"request_id": ("string", True, "A fresh id of letters, digits, _ or -, up to 64."),
                    "operation": ("string", True, "inspect or render-phong."),
                    "at": ("string", False, "inspect: the object identity; the root when omitted."),
                    "radius": ("integer", False, "inspect: graph hops."),
                    "position": ("array", False, "render-phong: the camera position [x, y, z] in metres."),
                    "facing": ("array", False, "render-phong: the camera direction [x, y, z]."),
                    "width": ("integer", False, "render-phong: 64..640 pixels."),
                    "height": ("integer", False, "render-phong: 64..640 pixels.")})),
    _ep("foyer_receipt", "foyer", "GET", "/api/foyer/commands/{request_id}",
        "Read a foyer command's receipt; a finished picture carries its PNG.",
        "view_site", False, False, kiosk=True, path_params={"request_id": ("string", "Your request id.")}),

    # ---- messages
    _ep("messages_conversations", "messages", "GET", "/api/messages/conversations", "List your conversations with their handles.",
        "message", True, False),
    _ep("messages_contacts", "messages", "GET", "/api/messages/contacts", "List your contacts and pending contact requests.",
        "message", True, False),
    _ep("messages_contacts_update", "messages", "POST", "/api/messages/contacts",
        "Request, accept, decline, block or remove a contact.",
        "message", True, True,
        body=_body({"action": ("string", True, "One of request, accept, decline, block or remove."),
                    "username": ("string", True, "The other account's username.")})),
    _ep("messages_images_status", "messages", "GET", "/api/messages/images", "Show whether images are allowed in a chat with a peer.",
        "message", True, False, query={"peer": ("string", "The peer's username.")}),
    _ep("messages_images_set", "messages", "POST", "/api/messages/images", "Allow or refuse images in one conversation.",
        "message", True, True,
        body=_body({"conversation_handle": ("string", True, "The conversation handle."),
                    "enabled": ("boolean", True, "True to allow images.")})),
    _ep("messages_image", "messages", "GET", "/api/messages/image", "Fetch one message image as a PNG.",
        "message", True, False, query={"id": ("string", "The image id from a message.")}, returns="png"),
    _ep("messages_thread", "messages", "GET", "/api/messages/thread", "Read a thread with a peer, newest page first.",
        "message", True, False,
        query={"peer": ("string", "The peer's username."), "before": ("integer", "Only messages with a smaller id."),
               "subject_kind": ("string", "Restrict to a subject kind."), "subject_ref": ("string", "Restrict to a subject reference.")}),
    _ep("messages_since", "messages", "GET", "/api/messages/since", "Wait for messages newer than an id.",
        "message", True, False,
        query={"after": ("integer", "Return messages with a larger id."), "wait": ("number", "Seconds to wait for news."),
               "subject_kind": ("string", "Restrict to a subject kind."), "subject_ref": ("string", "Restrict to a subject reference.")}),
    _ep("messages_inbox", "messages", "GET", "/api/messages/inbox", "List the messages waiting past your acknowledged cursor.",
        "message", True, False),
    _ep("messages_send", "messages", "POST", "/api/messages/send", "Send a message to a user or into a conversation.",
        "message", True, True,
        body=_body({"to": ("string", False, "The recipient's username; omit when conversation_handle is given."),
                    "body": ("string", False, "The message text."),
                    "subject": ("string", False, "An optional subject."),
                    "type": ("string", False, "The message type."),
                    "conversation_handle": ("string", False, "Send into this conversation instead of naming a recipient."),
                    "idempotency_key": ("string", False, "A key that makes a retry safe."),
                    "image": ("object", False, "A bounded PNG image attachment.")})),
    _ep("messages_ack", "messages", "POST", "/api/messages/ack", "Move your acknowledged cursor forward.",
        "message", True, True,
        body=_body({"up_to": ("integer", True, "Acknowledge every message up to this id.")})),
    _ep("messages_read", "messages", "POST", "/api/messages/read", "Mark a peer's messages read up to an id.",
        "message", True, True,
        body=_body({"peer": ("string", True, "The peer's username."), "up_to": ("integer", False, "Mark read up to this id.")})),
    _ep("messages_admin_thread", "messages", "GET", "/api/messages/admin/thread", "Read the audited thread between two users.",
        "account_manage", True, False,
        query={"a": ("string", "The first username."), "b": ("string", "The second username."),
               "before": ("integer", "Only messages with a smaller id.")}),
    _ep("messages_admin_remove", "messages", "POST", "/api/messages/admin/remove", "Remove one message as a moderator.",
        "account_manage", True, True, body=_body({"id": ("integer", True, "The message id.")})),
    _ep("messages_admin_suspend", "messages", "POST", "/api/messages/admin/suspend", "Suspend or restore a user's messaging.",
        "account_manage", True, True,
        body=_body({"username": ("string", True, "The account."), "suspended": ("boolean", True, "True to suspend.")})),

    # ---- living_data
    _ep("living_data_session_create", "living_data", "POST", "/api/living-data/sessions",
        "Open a graph navigation session over a Living Data map.",
        "dev_ui", True, True,
        body=_body({"map": ("object", True, "The canonical Living Data map."),
                    "at": ("string", True, "The identity of the node to start at."),
                    "layer": ("string", False, "The layer to navigate; layout when omitted."),
                    "request_id": ("string", True, "A caller id of at most 128 characters, for safe retries.")})),
    _ep("living_data_session_status", "living_data", "GET", "/api/living-data/sessions/{id}",
        "Show where a graph session stands and the exits from there.",
        "dev_ui", True, False, path_params=_ID32, query={"offset": ("integer", "Page offset into the exits.")}),
    _ep("living_data_session_move", "living_data", "POST", "/api/living-data/sessions/{id}/move",
        "Move a graph session along one declared exit.",
        "dev_ui", True, True, path_params=_ID32,
        body=_body({"to": ("string", True, "The identity of the node to move to."),
                    "relationship": ("string", True, "The relationship of the exit."),
                    "direction": ("string", False, "outgoing or incoming; outgoing when omitted."),
                    "expected_revision": ("integer", True, "The session revision you last read."),
                    "request_id": ("string", True, "A caller id for safe retries.")})),
    _ep("living_data_session_close", "living_data", "POST", "/api/living-data/sessions/{id}/close", "Close a graph session.",
        "dev_ui", True, True, path_params=_ID32,
        body=_body({"expected_revision": ("integer", True, "The session revision you last read.")})),
    _ep("living_data_query", "living_data", "POST", "/api/living-data/query", "Read a neighbourhood of a supplied Living Data map.",
        "dev_ui", True, False,
        body=_body({"map": ("object", False, "The canonical Living Data map."),
                    "at": ("string", False, "The node identity to centre on."),
                    "radius": ("integer", False, "Neighbourhood radius; 1 when omitted."),
                    "limit": ("integer", False, "Nodes per page; 64 when omitted."),
                    "offset": ("integer", False, "Node page offset."),
                    "layer": ("string", False, "The layer; layout when omitted."),
                    "edge_offset": ("integer", False, "Edge page offset.")})),
    _ep("living_data_dm_register", "living_data", "POST", "/api/living-data/dm/worlds", "Register a static world for DM commands.",
        "dev_ui", True, True,
        body=_body({"map": ("object", False, "The world map."), "source_sha256": ("string", False, "Digest of the source."),
                    "renderer_identity": ("string", False, "Identity of the renderer host."),
                    "request_id": ("string", False, "A caller id for safe retries.")})),
    _ep("living_data_dm_status", "living_data", "GET", "/api/living-data/dm/worlds/{id}", "Show a registered static world.",
        "dev_ui", True, False, path_params=_ID32),
    _ep("living_data_dm_command", "living_data", "POST", "/api/living-data/dm/worlds/{id}/commands",
        "Queue an inspect or render-phong command on a static world.",
        "dev_ui", True, True, path_params=_ID32,
        body=_body({"request_id": ("string", False, "A caller id for safe retries."),
                    "operation": ("string", False, "inspect or render-phong."),
                    "at": ("string", False, "Inspect: the node to centre on."), "layer": ("string", False, "Inspect: the layer."),
                    "radius": ("integer", False, "Inspect: radius."), "limit": ("integer", False, "Inspect: page size."),
                    "offset": ("integer", False, "Inspect: node page offset."), "edge_offset": ("integer", False, "Inspect: edge page offset."),
                    "position": ("array", False, "Render: camera position vector."), "facing": ("array", False, "Render: camera facing vector."),
                    "width": ("integer", False, "Render: image width, 64 to 1024."),
                    "height": ("integer", False, "Render: image height, 64 to 1024.")})),
    _ep("living_data_dm_receipt", "living_data", "GET", "/api/living-data/dm/worlds/{id}/commands/{command}",
        "Read the receipt of one static world command.",
        "dev_ui", True, False,
        path_params={"id": ("string", "The 32 hex character world id."), "command": ("string", "The command request id.")}),
    _ep("living_data_dm_host", "living_data", "POST", "/api/living-data/dm/worlds/{id}/host",
        "Let a renderer host poll for and complete commands.",
        "dev_ui", True, True, path_params=_ID32,
        body=_body({"host_token": ("string", False, "The host's token."), "action": ("string", False, "poll or complete."),
                    "request_id": ("string", False, "The command being completed."), "result": ("object", False, "The completed result.")})),
    _ep("living_data_dm_close", "living_data", "POST", "/api/living-data/dm/worlds/{id}/close", "Close a static world.",
        "dev_ui", True, True, path_params=_ID32, body=_body({})),
    _ep("living_data_worldmap_index", "living_data", "GET", "/api/worldmap/index", "List the stored world maps by repository and commit.",
        "dev_ui", False, False),
    _ep("living_data_worldmap_map", "living_data", "GET", "/api/worldmap/map", "Fetch one stored world map.",
        "dev_ui", False, False,
        query={"repo": ("string", "The repository name, owner__name."), "commit": ("string", "The 40 hex character commit.")}),

    # ---- photo
    _ep("photo_jobs", "photo", "GET", "/api/marble_photo/jobs", "List your marble photo jobs.", "photo_read", False, False,
        kiosk=True),
    _ep("photo_lenses", "photo", "GET", "/api/marble_photo/lenses", "List the available camera lenses.", "photo_read", False, False,
        kiosk=True),
    _ep("photo_status", "photo", "GET", "/api/marble_photo", "Show the status of a photo job.",
        "photo_read", False, False, query={"digest": ("string", "The job digest.")}, kiosk=True),
    _ep("photo_calibration_qr", "photo", "POST", "/api/marble_photo/calibration_qr",
        "Locate the authored exposure QR in an owned saved photo's PNG.",
        "photo_read", True, False,
        body=_body({"digest": ("string", True, "The owned photo digest."),
                    "name": ("string", True, "The PNG artifact name."),
                    "lane": ("number", False, "Sensor lane index; zero when omitted."),
                    "modules": ("array", True, "The 33 rows of authored QR module bits.")})),
    _ep("photo_image", "photo", "GET", "/api/marble_photo/image", "Fetch a finished photo or artifact as a PNG.",
        "photo_read", False, False,
        query={"digest": ("string", "The job digest."), "name": ("string", "The artifact name.")}, returns="png", kiosk=True),
    _ep("photo_hdr", "photo", "GET", "/api/marble_photo/hdr", "Download a finished photo's HDR file.",
        "photo_read", False, False,
        query={"digest": ("string", "The job digest."), "name": ("string", "The artifact name.")}, returns="binary"),
    _ep("photo_field", "photo", "GET", "/api/marble_photo/field", "Download a photo job's field file.",
        "photo_read", False, False,
        query={"digest": ("string", "The job digest."), "name": ("string", "The field name.")}, returns="binary"),
    _ep("photo_animation", "photo", "GET", "/api/marble_photo/animation", "Download a photo job's orbit animation.",
        "photo_read", False, False, query={"digest": ("string", "The job digest.")}, returns="binary"),
    _ep("photo_submit", "photo", "POST", "/api/marble_photo", "Start a photo job from an MRP1 binary upload.",
        "photo_submit", False, True, request="binary"),
    _ep("photo_cancel", "photo", "POST", "/api/marble_photo/cancel", "Cancel a photo job.",
        "photo_submit", False, True, body=_body({"digest": ("string", True, "The job digest.")})),
    _ep("photo_live", "photo", "POST", "/api/marble_photo/live", "Update or stop the live view of a photo job.",
        "share_live", False, True,
        body=_body({"digest": ("string", True, "The job digest."), "stop": ("boolean", False, "True to stop the live view.")}, True)),
    _ep("photo_retouch", "photo", "POST", "/api/marble_photo/retouch",
        "Re-develop a photo from its saved linear exposure: band projection (linear to RGB), linear corrections, then tone and "
        "transfer. The raw HDR is never changed; the result is the job's retouched_image (photo_status, then photo_image).",
        "photo_submit", False, True, body=_body({
            "digest": ("string", True, "The job digest."),
            "spectral_look": ("string", False, "Eight-band to RGB projection: visible_reference (visible bands as RGB), "
                              "compressed (false colour, the default), nir_red (near-infrared shown as red), uv_violet."),
            "exposure": ("number", False, "Linear exposure in stops, -6 to 6."),
            "red": ("number", False, "Linear red gain, 0 to 3."),
            "green": ("number", False, "Linear green gain, 0 to 3."),
            "blue": ("number", False, "Linear blue gain, 0 to 3."),
            "whitebalance": ("number", False, "Warm/cool balance, -3 to 3."),
            "tint": ("number", False, "Green/magenta tint, -3 to 3."),
            "normalization": ("string", False, "Display white: auto99 (default), percentile, manual or specular."),
            "normalization_percentile": ("number", False, "Luminance percentile used as white, 1 to 100."),
            "normalization_white": ("number", False, "Manual white in linear units, 1e-6 to 1e12."),
            "black_point": ("number", False, "Black point as a fraction of white, 0 to 0.5."),
            "tone_curve": ("string", False, "exponential (default) or linear."),
            "display_look": ("string", False, "reference (default) or color_priority."),
            "saturation": ("number", False, "Chroma scale, 0 to 2."),
            "vibrance": ("number", False, "Chroma boost weighted toward muted colours, -1 to 1."),
            "gamma": ("number", False, "Artistic gamma after the sRGB transfer, 0.2 to 3."),
            "intensity": ("number", False, "Artistic intensity after the sRGB transfer, 0 to 3."),
            "firefly_filter": ("boolean", False, "Suppress isolated hot pixels."),
            "firefly_threshold": ("number", False, "Firefly threshold over the brightest neighbour, 2 to 64."),
            "firefly_strength": ("number", False, "Firefly blend strength, 0 to 1."),
        }, True)),
    _ep("photo_film_preview", "photo", "POST", "/api/marble_photo/film_preview", "Preview a film look without rendering a photo.",
        "photo_submit", False, False, body=_body({}, True)),
    _ep("photo_picture_ask", "photo", "POST", "/api/show/pictures", "Ask the Studio for a picture of a scene.",
        "photo_submit", True, True,
        body=_body({"request_id": ("string", True, "A caller id; a repeat returns the same receipt."),
                    "settings": ("object", False, "Picture settings."), "photo": ("object", False, "Photo options including lightTimeline and the Spectral cameraTriggerGraph, cameraTriggerGenerators and cameraTriggerTiming for routed lights.")})),
    _ep("photo_light_program", "photo", "POST", "/api/marble_camera/trigger_grid", "Edit or resolve the signed-in Studio's Spectral light program through its original routing mixer.",
        "photo_submit", True, True,
        body=_body({"action": ("string", True, "open, configure, event, or pulse."),
                    "receivers": ("array", False, "Named Studio/shop/filter receiver IDs when opening."),
                    "buses": ("array", False, "Named lighting buses when opening."),
                    "graph": ("object", False, "Saved RoutingGraph when opening."),
                    "generators": ("array", False, "Spectral modulator settings when opening or configuring.")}, True)),
    _ep("photo_picture_receipt", "photo", "GET", "/api/show/pictures/{id}", "Read the receipt of a requested picture.",
        "photo_submit", True, False, path_params=_ID32),

    # ---- studio
    _ep("studio_loading_library", "studio", "GET", "/api/loading_library", "List the loading animation library.",
        None, False, False, tool=False, guest=True),
    _ep("studio_loading_anim", "studio", "GET", "/api/loading_anim", "Fetch the active loading animation, or a library entry.",
        None, False, False, query={"v": ("string", "A library digest; the active animation when omitted.")},
        returns="binary", tool=False, guest=True),
    _ep("studio_loading_anim_put", "studio", "POST", "/api/loading_anim", "Store a loading animation in the library.",
        "owner_tools", False, True, tool=False, request="binary"),
    _ep("studio_loading_candidate", "studio", "POST", "/api/loading_candidate", "Store a temporary candidate loading animation.",
        "play", False, True, tool=False, request="binary"),
    _ep("studio_loading_activate", "studio", "POST", "/api/loading_library/activate", "Make a library entry the active animation.",
        "library_manage", False, True, tool=False, body=_body({"digest": ("string", True, "The library entry digest.")})),
    _ep("studio_loading_graduate", "studio", "POST", "/api/loading_library/graduate", "Graduate a candidate into the library.",
        "library_manage", False, True, tool=False, body=_body({"digest": ("string", True, "The library entry digest.")})),
    _ep("studio_loading_delete", "studio", "POST", "/api/loading_library/delete", "Delete a library entry.",
        "library_manage", False, True, tool=False, body=_body({"digest": ("string", True, "The library entry digest.")})),
    _ep("studio_obj_assets", "studio", "GET", "/api/obj_assets", "List the OBJ assets you can see.", None, False, False, guest=True),
    _ep("studio_obj_asset", "studio", "GET", "/api/obj_assets/{id}", "Show one OBJ asset.",
        None, False, False, path_params={"id": ("string", "The asset id.")}, guest=True),
    _ep("studio_obj_asset_source", "studio", "GET", "/api/obj_assets/{id}/source", "Fetch an OBJ asset's source text.",
        None, False, False, path_params={"id": ("string", "The asset id.")}, returns="text", guest=True),
    _ep("studio_obj_asset_mtl", "studio", "GET", "/api/obj_assets/{id}/mtl", "Fetch an OBJ asset's material text.",
        None, False, False, path_params={"id": ("string", "The asset id.")}, returns="text", guest=True),
    _ep("studio_obj_asset_add", "studio", "POST", "/api/obj_assets", "Add an OBJ asset to your account.",
        "studio_write", True, True, body=_body({}, True)),
    _ep("studio_marble_objects", "studio", "GET", "/api/marble_objects", "List the marble objects you can see.", None, False, False, guest=True),
    _ep("studio_marble_object", "studio", "GET", "/api/marble_objects/{digest}", "Show one marble object.",
        None, False, False, path_params={"digest": ("string", "The object digest.")}, guest=True),
    _ep("studio_marble_object_add", "studio", "POST", "/api/marble_objects", "Save a marble object design.",
        "studio_write", False, True, body=_body({}, True), headers={"X-Meowdoku-Marble": "1"}),
    _ep("studio_marble_preview_spec", "studio", "GET", "/api/marble_preview_spec", "Fetch the Studio drive's compute spec for N marbles.",
        None, False, False, query={"count": ("integer", "Marbles to run, 1 to 16; 1 when omitted.")}, guest=True),
    _ep("studio_camera_housing", "studio", "POST", "/api/marble_camera/housing", "Compute a camera housing response.",
        "studio_write", False, False, body=_body({}, True)),

    # ---- nodus
    _ep("nodus_instances", "nodus", "GET", "/api/nodus/instances", "List the running Nodus instances.", ("dev_ui", "nodus_edit"), False, False),
    _ep("nodus_digest", "nodus", "GET", "/api/nodus/digest", "Show a Nodus instance's canvas digest.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_graph", "nodus", "GET", "/api/nodus/graph", "Show a Nodus instance's graph.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_tools", "nodus", "GET", "/api/nodus/tools", "List the tools a Nodus instance exposes.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_activity", "nodus", "GET", "/api/nodus/activity", "Show recent activity on a Nodus instance.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_abstract", "nodus", "GET", "/api/nodus/abstract", "Show a Nodus graph as generic nodes.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_map", "nodus", "GET", "/api/nodus/map", "Show a Nodus graph as a Living Data map.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_picture", "nodus", "GET", "/api/nodus/picture", "Show the Nodus graph's Living Data map picture and its view path.",
        ("dev_ui", "nodus_edit"), False, False, query={"instance": ("string", "The Nodus instance name; the default when omitted.")}),
    _ep("nodus_frame", "nodus", "GET", "/api/nodus/frame", "Look at a console device's full UI feed or a bounded region.",
        ("dev_ui", "nodus_edit"), False, False, returns="png",
        query={"instance": ("string", "The Nodus instance name; the default when omitted."),
               "x": ("integer", "Region left; supply with y, width and height."),
               "y": ("integer", "Region top; supply with x, width and height."),
               "width": ("integer", "Region width; supply with x, y and height."),
               "height": ("integer", "Region height; supply with x and y and width.")}),
    _ep("nodus_action", "nodus", "POST", "/api/nodus/action", "Apply a UI event to the managed Nodus canvas at an expected revision.",
        "nodus_edit", True, True,
        body=_body({"instance": ("string", True, "Managed instance name."),
                    "action": ("object", True, "Canvas UI action with op and expected_revision.")})),
    _ep("nodus_share", "nodus", "POST", "/api/nodus/share", "Send another user a note about a Nodus instance.",
        ("dev_ui", "message"), True, True, capability_all=True,
        body=_body({"instance": ("string", False, "The Nodus instance; the default when omitted."),
                    "to": ("string", True, "The recipient's username."), "note": ("string", False, "An optional note.")})),

    # ---- reviews (turing examples/meowdoku/branch_review.py)
    _ep("reviews_list", "reviews", "GET", "/api/reviews", "List the changes under review and the recently settled ones.",
        "dev_ui", True, False),
    _ep("reviews_submit", "reviews", "POST", "/api/reviews", "Put a pushed branch, at one commit, up for review.",
        "dev_ui", True, True,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."),
                    "branch": ("string", True, "The pushed branch."),
                    "head": ("string", True, "The full 40-character commit id the branch is at."),
                    "timeout_hours": ("number", False, "Hours until silence accepts it; the policy's default when omitted."),
                    "keep": ("boolean", False, "True to keep the branch after it lands.")})),
    _ep("reviews_identity", "reviews", "GET", "/api/reviews/identity",
        "Show the Git author identity this account declared for its own commits.", "dev_ui", True, False),
    _ep("reviews_identity_set", "reviews", "POST", "/api/reviews/identity",
        "Declare the Git author name and email this account's commits carry; landing names commits that are not by it.",
        "dev_ui", True, True,
        body=_body({"name": ("string", True, "The author name."), "email": ("string", True, "The author email.")})),
    _ep("reviews_checkout", "reviews", "POST", "/api/reviews/checkout",
        "Check a branch out from the server: register its name to yourself and get the steps to submit it.",
        "dev_ui", True, True,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."),
                    "branch": ("string", True, "The branch name you will work on.")})),
    _ep("reviews_remark", "reviews", "POST", "/api/reviews/remark", "Agree with or object to a change you did not write.",
        "dev_ui", True, True,
        body=_body({"id": ("string", True, "The review id."), "verdict": ("string", True, "agree or object."),
                    "note": ("string", False, "Why.")})),
    _ep("reviews_decide", "reviews", "POST", "/api/reviews/decide", "As an admin, approve or reject exactly the commit you read.",
        "dev_ui", True, True,
        body=_body({"id": ("string", True, "The review id."), "head": ("string", True, "The commit id you read."),
                    "decision": ("string", True, "approved or rejected."), "note": ("string", False, "Why.")})),
    _ep("reviews_withdraw", "reviews", "POST", "/api/reviews/withdraw", "Withdraw your own open change.",
        "dev_ui", True, True, body=_body({"id": ("string", True, "The review id.")})),
    _ep("reviews_removed", "reviews", "GET", "/api/reviews/removed",
        "List the notices of removal: what the cleanup account took off this computer, and at which commit.",
        "dev_ui", True, False),
    _ep("reviews_standing", "reviews", "GET", "/api/reviews/standing", "Say whether one commit has an accepted review, and how to get one.",
        None, False, False,
        query={"repository": ("string", "The repository's name in the branch policy."),
               "head": ("string", "The full 40-character commit id.")}, guest=True),

    # ---- git (turing examples/meowdoku/core_git.py)
    _ep("git_repositories", "git", "GET", "/api/git/repositories", "List the core repositories, their default branches and the Git operations offered.",
        "core_git", True, False),
    _ep("git_calls", "git", "GET", "/api/git/calls", "Read the record of Git operations asked for in the core.",
        "core_git", True, False),
    _ep("git_status", "git", "POST", "/api/git/status", "Show a core repository's branch and which files differ from it.",
        "core_git", True, False, body=_body({"repository": ("string", True, "The repository's name in the branch policy.")})),
    _ep("git_branches", "git", "POST", "/api/git/branches", "List a core repository's local and remote branches with their commits.",
        "core_git", True, False, body=_body({"repository": ("string", True, "The repository's name in the branch policy.")})),
    _ep("git_log", "git", "POST", "/api/git/log", "List recent commits of a branch, optionally only those touching a path.",
        "core_git", True, False,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."), "ref": ("string", False, "A branch, tag or commit; HEAD when omitted."),
                    "path": ("string", False, "Only commits touching this path."),
                    "count": ("integer", False, "How many, at most 200; 20 when omitted.")})),
    _ep("git_commit", "git", "POST", "/api/git/commit", "Show one commit: author, time, message and the files it changed.",
        "core_git", True, False,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."), "commit": ("string", True, "The commit, branch or tag.")})),
    _ep("git_diff", "git", "POST", "/api/git/diff", "Show the difference between two commits, optionally for one path.",
        "core_git", True, False,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."), "base": ("string", True, "The older commit, branch or tag."),
                    "head": ("string", True, "The newer commit, branch or tag."),
                    "path": ("string", False, "Only this path.")})),
    _ep("git_file", "git", "POST", "/api/git/file", "Read one file as it is at a commit.",
        "core_git", True, False,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."), "ref": ("string", False, "A branch, tag or commit; HEAD when omitted."),
                    "path": ("string", True, "The file's path inside the repository.")})),
    _ep("git_tree", "git", "POST", "/api/git/tree", "List the files under a path at a commit.",
        "core_git", True, False,
        body=_body({"repository": ("string", True, "The repository's name in the branch policy."), "ref": ("string", False, "A branch, tag or commit; HEAD when omitted."),
                    "path": ("string", False, "Only files under this folder.")})),

    # ---- relay (turing examples/meowdoku/relay.py)
    _ep("relay_addresses", "relay", "GET", "/api/relay/addresses", "List the relay addresses you hold open.", None, True, False),
    _ep("relay_open", "relay", "POST", "/api/relay/open",
        "Open an address: a UDP port of the service that anyone may send to and you may send from.",
        None, True, True, body=_body({})),
    _ep("relay_close", "relay", "POST", "/api/relay/close", "Close one of your addresses.",
        None, True, True, body=_body({"address": ("integer", True, "The address's number.")})),
    _ep("relay_send", "relay", "POST", "/api/relay/send",
        "Send from your address: a datagram to host:port anywhere, or a package to another address's number.",
        None, True, True,
        body=_body({"address": ("integer", True, "Your address's number."),
                    "to": ("string", True, "host:port for a datagram, or another address's number for a package."),
                    "text": ("string", False, "The body as text; give this or body_base64."),
                    "body_base64": ("string", False, "The body as base64 bytes; give this or text."),
                    "kind": ("string", False, "A short word naming a package's kind; datagram when omitted.")})),
    _ep("relay_receive", "relay", "GET", "/api/relay/receive", "Read what arrived at your address, oldest first, with who sent it.",
        None, True, False,
        query={"address": ("integer", "Your address's number."), "after": ("integer", "Only packages after this id."),
               "limit": ("integer", "At most this many, up to 500.")}),

    # ---- files (turing examples/meowdoku/file_index.py)
    _ep("files_search", "files", "GET", "/api/files/search", "Filter the flat index of the core's files by name, kind, size, age and state.",
        "dev_ui", True, False,
        query={"q": ("string", "Text the path contains."), "repository": ("string", "Only this repository."),
               "ext": ("string", "Only this file extension, without the dot."),
               "status": ("string", "tracked, modified, untracked or ignored."),
               "under": ("string", "Only files under this folder of the repository."),
               "min_bytes": ("integer", "At least this large."), "max_bytes": ("integer", "At most this large."),
               "newer_than": ("number", "Changed after this Unix time."), "older_than": ("number", "Changed before this Unix time."),
               "binary": ("boolean", "True for binary files only, false for text only."),
               "order": ("string", "path, name, ext, bytes, mtime, lines, commits, last_commit_at, first_commit_at or depth; - in front for descending."),
               "limit": ("integer", "Rows to return, at most 500; 100 when omitted."),
               "offset": ("integer", "Rows to skip.")}),
    _ep("files_summary", "files", "GET", "/api/files/summary", "Show when the file index was built and its totals by state and kind.",
        "dev_ui", True, False),

    # ---- control
    _ep("control_instances", "control", "GET", "/api/control/instances", "List the service instances, recent control requests and the running jobs a restart waits for.",
        _ADMIN_OR_LIFECYCLE, True, False),
    _ep("control_instances_request", "control", "POST", "/api/control/instances",
        "Ask the supervisor to restart, shut down or start an instance.",
        _ADMIN_OR_LIFECYCLE, True, True,
        body=_body({"instance": ("string", True, "The instance name."), "action": ("string", True, "restart, shutdown or start."),
                    "reason": ("string", False, "Why, for the record."),
                    "delay": ("number", False, "Seconds of countdown before it is performed."),
                    "after_jobs": ("boolean", False, "Wait until the server-wide job table shows no running job on the named "
                                   "instances (Pale Horse authorizes it then); true when omitted. false restarts at once.")})),
    _ep("control_instances_cancel", "control", "POST", "/api/control/instances/cancel", "Cancel a control request not yet performed.",
        _ADMIN_OR_LIFECYCLE, True, True, body=_body({"id": ("integer", True, "The request id.")})),
    _ep("control_route", "control", "GET", "/api/route", "Show the server routes and which one this browser is on.", None, False, False, guest=True),
    _ep("control_route_set", "control", "POST", "/api/route", "Choose the Stable or Development server route.",
        "dev_ui", False, True,
        body=_body({"route": ("string", False, "A route name from the routes list."),
                    "site": ("string", False, "stable or dev; used instead of route.")})),
    _ep("control_log", "control", "GET", "/api/log", "Read the server's text log ring.",
        "dev_ui", False, False,
        query={"since": ("integer", "Return entries after this sequence number."), "limit": ("integer", "Entries to return; 500 when omitted."),
               "src": ("string", "Comma separated sources to keep.")}),
    _ep("control_lifecycle", "control", "GET", "/api/lifecycle", "Show the shutdown countdown and this session's restorable copy.",
        None, False, False,
        query={"snap": ("string", "Set to include this session's snapshot."), "have": ("string", "The snapshot id you already hold.")},
        tool=False, guest=True),
    _ep("control_lifecycle_restore", "control", "POST", "/api/lifecycle/restore", "Re-seed this session from a saved copy.",
        None, False, True, body=_body({"dump": ("object", False, "The session copy.")}), tool=False, guest=True),
    _ep("control_lifecycle_admin", "control", "POST", "/api/admin/lifecycle", "Request a save, restart or exit with the loopback token.",
        None, False, True, body=_body({}, True), tool=False, credential="admin_token"),

    # ---- profile
    _ep("profile_me", "profile", "GET", "/api/profile/me", "Show your account summary and what it may do.", None, True, False),
    _ep("profile_instances", "profile", "GET", "/api/profile/instances", "List the declared service instances and their state.",
        _ADMIN_OR_LIFECYCLE, True, False),
    _ep("profile_leases", "profile", "GET", "/api/profile/leases", "List the open server leases.",
        _ADMIN_OR_LIFECYCLE, True, False),
    _ep("profile_diagnostics", "profile", "GET", "/api/profile/diagnostics", "Read recent diagnostics for an instance.",
        _ADMIN_OR_LIFECYCLE, True, False,
        query={"name": ("string", "The instance name."), "lines": ("integer", "Lines to return; 80 when omitted.")}),

    # ---- reports
    _ep("reports_list", "reports", "GET", "/api/reports", "List your reports, or all reports for an administrator.",
        None, True, False,
        query={"offset": ("integer", "Page offset."), "category": ("string", "DOC, TTICKET, AUDIT or LOG."),
               "scope": ("string", "mine (default), shared group spaces, public signed-off reports, or all for an administrator.")}),
    _ep("reports_map", "reports", "GET", "/api/reports/map", "Show your reports as a map, or all reports for an administrator.",
        None, True, False,
        query={"offset": ("integer", "Page offset."), "category": ("string", "DOC, TTICKET, AUDIT or LOG."),
               "scope": ("string", "mine (default), shared group spaces, public signed-off reports, or all for an administrator.")}),
    _ep("reports_get", "reports", "GET", "/api/reports/{id}", "Read an accessible experience report; administrators may read any report.",
        None, True, False, path_params=_ID32),
    _ep("reports_make", "reports", "POST", "/api/reports", "Write an experience report.",
        None, True, True,
        body=_body({"request_id": ("string", True, "A caller id; a repeat returns the same report."),
                    "category": ("string", True, "DOC, TTICKET, AUDIT or LOG."), "title": ("string", True, "Title, at most 200 characters."),
                    "body": ("string", True, "Body text."),
                    "prompt_history": ("array", True, "The prompts that shaped the work, as text."),
                    "next_steps": ("array", True, "Next steps, as text; may be empty.")})),
    _ep("reports_public_list", "reports", "GET", "/api/reports/public", "List Internet-public signed-off reports without exposing account IDs.",
        None, False, False, query={"offset": ("integer", "Page offset."), "category": ("string", "DOC, TTICKET, AUDIT or LOG.")},
        kiosk=True),
    _ep("reports_public_get", "reports", "GET", "/api/reports/public/{id}", "Read a public report, including its prompt history and next steps.",
        None, False, False, path_params=_ID32, kiosk=True),
    _ep("reports_sharing", "reports", "POST", "/api/reports/{id}/sharing", "Change your report's personal, group or Internet-public visibility with owner sign-off.",
        None, True, True, path_params=_ID32,
        body=_body({"visibility": ("string", True, "personal, group, or public."),
                    "group_id": ("string", False, "Group ID for group scope; omit to send null otherwise."),
                    "expected_revision": ("integer", True, "The sharing revision from a fresh report read."),
                    "sign_off": ("boolean", True, "True only for public: explicitly approve the whole report, prompts and next steps.")})),

    # ---- shared account groups: membership places, not the messaging graph
    _ep("groups_list", "identity", "GET", "/api/account/groups", "List your flat group memberships, not friendships.",None,True,False),
    _ep("groups_create", "identity", "POST", "/api/account/groups", "Create a group space owned by your current account.",None,True,True,
        body=_body({"name":("string",True,"Group name, at most 100 characters."),"request_id":("string",True,"Idempotency key.")})),
    _ep("groups_get", "identity", "GET", "/api/account/groups/{id}", "Read a group and its member identities only as an accepted member.",None,True,False,path_params=_ID32),
    _ep("groups_invite", "identity", "POST", "/api/account/groups/{id}/invite", "Request membership with a message; acceptance is required before any group content access.",None,True,True,path_params=_ID32,
        body=_body({"username":("string",True,"Known account name; response does not confirm existence."),
                    "request_message":("string",True,"Required invitation message, at most 2000 characters.")})),
    _ep("groups_invitations", "identity", "GET", "/api/account/groups/invitations", "Read your pending place invitations without exposing member identities.",None,True,False),
    _ep("groups_respond", "identity", "POST", "/api/account/groups/invitations/{id}", "Accept or decline your pending group invitation.",None,True,True,path_params=_ID32,
        body=_body({"accept":("boolean",True,"True to join the group, false to decline.")})),
    _ep("groups_member_role", "identity", "POST", "/api/account/groups/{id}/members", "Change or remove an existing member; only the group owner may change group administrators.",None,True,True,path_params=_ID32,
        body=_body({"username":("string",True,"An existing group member."),
                    "role":("string",False,"member or admin; omit to send null and remove membership."),
                    "expected_revision":("integer",True,"Current group revision.")})),
    _ep("groups_leave", "identity", "POST", "/api/account/groups/{id}/leave", "Leave a group and immediately lose its shared-content access.",None,True,True,path_params=_ID32,body=_body({})),

    _ep("authority_self", "identity", "GET", "/api/account/authority", "Show your direct permission groups, highest rank and implied lower groups; content sharing groups are separate.", None, True, False),
    _ep("authority_grant", "identity", "POST", "/api/account/authority/grant", "Administer a dev, operator or nodus permission-group grant for an account.", "account_manage", True, True,
        body=_body({"username":("string",True,"Account name."),
                    "group":("string",True,"dev, operator or nodus; admin uses the existing admin grant endpoint."),
                    "enabled":("boolean",True,"True to grant, false to revoke.")})),

    _ep("paper_account", "markets", "GET", "/api/paper/account", "Read the effective Alpaca paper account status and buying power without returning credentials.", None, True, False),
    _ep("paper_bots", "markets", "GET", "/api/paper/bots", "List your registered paper bots and their source references.", None, True, False),
    _ep("paper_bot_register", "markets", "POST", "/api/paper/bots", "Register a Python, symbolic, compiled, remote API or agent bot without executing its source.", None, True, True,
        body=_body({"request_id":("string",True,"Caller idempotency key."),
                    "name":("string",True,"Display name."),
                    "kind":("string",True,"python, symbolic, compiled, remote_api or agent."),
                    "source":("object",True,"Artifact ref for local code/math, or HTTPS url for a remote API.")})),
    _ep("paper_presence", "markets", "GET", "/api/paper/presence", "List bots active in the last five minutes with their latest interaction path; clients fade them by age.", None, True, False,
        query={"offset":("integer","Page offset; follow next_offset until null.")}),
    _ep("paper_qualification_live", "markets", "GET", "/api/paper/qualification/live", "Watch sanitized historical mock qualification progress from server training workers, including checkpoint validation and final admission status.", None, False, False),
    _ep("paper_trainer_events", "markets", "GET", "/api/paper/trainer-events", "Read sanitized paper interaction events for server/admin network trainers.", "paper_train", True, False,
        query={"after":("integer","Last event cursor, or zero for the first page.")}),
    _ep("paper_strategies", "markets", "GET", "/api/paper/strategies", "List your own strategy seats and risk limits.", None, True, False),
    _ep("paper_strategy_create", "markets", "POST", "/api/paper/strategies", "Create a named strategy seat with an idempotency key and risk tier.", None, True, True,
        body=_body({"request_id":("string",True,"Caller generated idempotency key."),
                    "name":("string",True,"Strategy name."),
                    "risk":("string",True,"cautious, standard or explore."),
                    "bot_id":("string",False,"Registered bot that owns this strategy, when applicable."),
                    "policy":("string",False,"manual, momentum, mean_reversion, tensor or external; manual by default."),
                    "outcomes":("array",False,"Any nonempty selection of fixed_horizon, closed_trade and fill_quality; all by default.")})),
    _ep("paper_orders", "markets", "GET", "/api/paper/orders", "List your attributed paper order intents and broker receipts.", None, True, False),
    _ep("paper_scoreboard", "markets", "GET", "/api/paper/scoreboard", "Mark each of your strategy's reconciled buy fills at latest available market bars; report return before fees.", None, True, False),
    _ep("paper_proposals", "markets", "GET", "/api/paper/proposals", "Compare dry-run proposals from modular strategy policies on one cached historical market snapshot; no order is placed.", None, True, False,
        query={"stock":("string","The wildcard stock: AAPL, MSFT, NVDA or TSLA."),
               "offset":("integer","Strategy page offset."), "limit":("integer","Page size, up to 200.")}),
    _ep("paper_map", "markets", "GET", "/api/paper/map", "Read your live paper arena as a Living Data world map of monitored markets, strategies and orders.", None, True, False),
    _ep("paper_map_query", "markets", "GET", "/api/paper/map/query", "Examine a node and its declared market, strategy and order relationships.", None, True, False,
        query={"at":("string","Node identity, or the root when omitted."), "radius":("integer","Graph radius 0-4.")}),
    _ep("paper_dm", "markets", "POST", "/api/paper/dm", "Paper Dungeon Master relay: inspect the paper arena, request the Nodus Living Data map picture, submit paper actions, or send authorized Nodus UI actions.", None, True, True,
        body=_body({"verb":("string",True,"look, examine, make or use."),
                    "target":("string",True,"Paper target or nodus_ui for an authorized canvas action."),
                    "input":("object",True,"Exact input for the selected command; empty object for look.")})),
    _ep("paper_order_submit", "markets", "POST", "/api/paper/orders", "Submit one capped buy order to Alpaca paper trading for your strategy; use a fresh request_id per decision.", None, True, True,
        body=_body({"request_id":("string",True,"Caller generated idempotency key; reuse only for an exact retry."),
                    "strategy_id":("string",True,"Your strategy seat ID."),
                    "symbol":("string",True,"One monitored ETF, wildcard stock or BTC/USD."),
                    "side":("string",True,"buy; sells require a reconciled strategy holdings ledger."),
                    "amount":("string",True,"USD notional decimal within the strategy risk limit.")})),
    _ep("paper_mock_orders", "markets", "GET", "/api/paper/mock/orders", "List your internal mock fills, separate from Alpaca paper receipts.", None, True, False),
    _ep("paper_mock_ledger", "markets", "GET", "/api/paper/mock/ledger", "Read a strategy's internal mock cash, realized P&L and holdings.", None, True, False,
        query={"strategy_id":("string","Your strategy seat ID.")}),
    _ep("paper_mock_order", "markets", "POST", "/api/paper/mock/orders", "Fill a small order immediately in the internal mock venue using cached marks and explicit spread, markup and fee assumptions.", None, True, True,
        body=_body({"request_id":("string",True,"Caller idempotency key."),
                    "strategy_id":("string",True,"Your strategy seat ID."),
                    "symbol":("string",True,"Monitored ETF, wildcard stock or BTC/USD."),
                    "side":("string",True,"buy or sell."),
                    "amount":("string",True,"USD notional decimal within the strategy risk limit.")})),
    _ep("paper_order_reconcile", "markets", "POST", "/api/paper/orders/{id}/reconcile", "Read an Alpaca order by client order ID after an uncertain submit; never submit again with a new key.", None, True, True,
        path_params=_ID32, body=_body({})),
    _ep("paper_shared_account", "markets", "GET", "/api/paper/shared-account", "Show whether the shared server/admin paper account is configured.", "account_manage", True, False),
    _ep("paper_shared_account_set", "markets", "POST", "/api/paper/shared-account", "Administer which account's existing Alpaca paper credentials back server/admin strategies.", "account_manage", True, True,
        body=_body({"username":("string",True,"Account holding the shared Alpaca paper credentials.")})),

    _ep("reports_issues_list", "reports", "GET", "/api/issues", "List issue tickets.",
        "view_site", True, False,
        query={"scope": ("string", "mine or another scope; mine when omitted."), "status": ("string", "A status, or all."),
               "offset": ("integer", "Page offset.")}),
    _ep("reports_issue_get", "reports", "GET", "/api/issues/{id}", "Read one issue ticket.",
        "view_site", True, False, path_params={"id": ("string", "The ticket id.")},
        query={"before_version": ("integer", "Read the ticket as of an earlier version.")}),
    _ep("reports_issue_create", "reports", "POST", "/api/issues", "Open an issue ticket.",
        "view_site", True, True, headers={"X-Meowdoku-Issue": "1"},
        body=_body({"title": ("string", False, "Title, at most 200 characters."), "description": ("string", False, "Description."),
                    "diagnostics": ("object", False, "Diagnostic data."), "client_key": ("string", False, "A caller key for safe retries.")})),
    _ep("reports_issue_status", "reports", "POST", "/api/issues/{id}/status", "Change an issue ticket's status.",
        "view_site", True, True, path_params={"id": ("string", "The ticket id.")}, headers={"X-Meowdoku-Issue": "1"},
        body=_body({"status": ("string", True, "The new status."), "note": ("string", True, "A note on the change."),
                    "version": ("integer", True, "The ticket version you last read.")})),
    _ep("reports_tickets_list", "reports", "GET", "/api/tickets", "List legacy tickets, or read one.",
        "dev_ui", False, False, query={"id": ("string", "A ticket id; all tickets when omitted.")}),
    _ep("reports_tickets_submit", "reports", "POST", "/api/tickets", "Submit a legacy ticket.",
        "play", False, True, body=_body({}, True), headers={"X-Meowdoku-Ticket": "1"}),
    _ep("reports_tickets_status", "reports", "POST", "/api/tickets/status", "Change a legacy ticket's status.",
        "owner_tools", False, True, headers={"X-Meowdoku-Ticket": "1"},
        body=_body({"id": ("string", True, "The ticket id."), "status": ("string", True, "The new status."),
                    "note": ("string", False, "A note on the change.")})),

    # ---- compile
    _ep("compile_divider_jobs", "compile", "GET", "/api/divider/jobs", "List your source divider jobs.",
        "compile_submit", True, False),
    _ep("compile_divider_job", "compile", "GET", "/api/divider/jobs/{id}", "Show one source divider job.",
        "compile_submit", True, False, path_params={"id": ("string", "The 12 hex character job id.")}),
    _ep("compile_divider_submit", "compile", "POST", "/api/divider/jobs", "Record a repository link as a divider job.",
        "compile_submit", True, True,
        body=_body({"repo": ("string", False, "The repository link."), "ref": ("string", False, "The ref to divide.")})),
    _ep("compile_divider_cancel", "compile", "POST", "/api/divider/jobs/{id}/cancel", "Cancel a divider job.",
        "compile_submit", True, True, path_params={"id": ("string", "The 12 hex character job id.")}),
    _ep("compile_mathlab_jobs", "compile", "GET", "/api/mathlab/jobs", "List your math lab jobs.", "compile_submit", True, False),
    _ep("compile_mathlab_job", "compile", "GET", "/api/mathlab/jobs/{id}", "Show one math lab job.",
        "compile_submit", True, False, path_params={"id": ("string", "The 16 hex character job id.")}),
    _ep("compile_mathlab_file", "compile", "GET", "/api/mathlab/jobs/{id}/file/{name}", "Fetch one product file of a math lab job.",
        "compile_submit", True, False,
        path_params={"id": ("string", "The 16 hex character job id."), "name": ("string", "A product file name the job lists.")},
        returns="text"),
    _ep("compile_mathlab_submit", "compile", "POST", "/api/mathlab/jobs", "Submit a law to the math lab.",
        "compile_submit", True, True,
        body=_body({"text": ("string", True, "The law source text."), "targets": ("array", False, "Target names to build.")})),
    _ep("compile_mathlab_profile", "compile", "POST", "/api/mathlab/jobs/{id}/browser_profile",
        "Record a browser timing for a completed math lab job.",
        "compile_submit", True, True, path_params={"id": ("string", "The 16 hex character job id.")},
        body=_body({"target": ("string", True, "wasm or wgsl."), "rows": ("integer", True, "Rows timed, 1 to 1e9."),
                    "ns_per_row": ("number", True, "Nanoseconds per row."),
                    "max_abs_err": ("number", False, "Largest absolute error seen."),
                    "user_agent_hint": ("string", False, "The browser, at most 200 characters.")})),
    _ep("compile_lowerings", "compile", "GET", "/api/lowerings", "List the lowering cores and their stored lowerings.",
        "view_site", False, False, guest=True),
    _ep("compile_lowering_blob", "compile", "GET", "/api/lowerings/blob/{sha}", "Fetch a leased lowering's bytes.",
        "studio_write", False, False, path_params={"sha": ("string", "The 64 hex character sha256 of the blob.")}, returns="binary"),
    _ep("compile_lowering_checkout", "compile", "POST", "/api/lowerings/checkout", "Lease a lowering that fits a purpose.",
        "studio_write", False, True,
        body=_body({"purpose": ("array", True, "Two strings: family and name."),
                    "capabilities": ("array", True, "Backend capability names."),
                    "prefer": ("array", False, "Preferred backend names."),
                    "seconds": ("number", False, "Lease length in seconds, up to 3600; 60 when omitted.")})),
    _ep("compile_lowering_checkin", "compile", "POST", "/api/lowerings/checkin", "Return a lowering lease.",
        "studio_write", False, True,
        body=_body({"lease_id": ("integer", True, "The lease id."), "generation": ("integer", True, "The lease generation.")})),

    # ---- site
    _ep("site_map_root", "site", "GET", "/site/{map}/", "Read a declared map's root place as an HTML page.",
        None, False, False, path_params={"map": ("string", "The map name.")}, returns="html", kiosk=True),
    _ep("site_place", "site", "GET", "/site/{map}/{address}.html", "Read one place of a declared map as an HTML page.",
        None, False, False,
        path_params={"map": ("string", "The map name."), "address": ("string", "The place address.")}, returns="html",
        kiosk=True),

    # ---- play
    _ep("play_state", "play", "GET", "/api/state", "Read this browser table's game state.", None, False, False, tool=False, guest=True),
    _ep("play_catalog", "play", "GET", "/api/catalog", "List the games, levels and boards.", None, False, False, tool=False, guest=True),
    _ep("play_new", "play", "POST", "/api/new", "Start a game on this browser table.", None, False, True,
        body=_body({"game": ("string", False, "The game name."), "id": ("string", False, "The board id.")}, True), tool=False, guest=True),
    _ep("play_action", "play", "POST", "/api/action", "Send one action to this browser table's game.",
        None, False, True, body=_body({"type": ("string", False, "The action type.")}, True), tool=False, guest=True),
    _ep("play_action_batch", "play", "POST", "/api/action/batch", "Send a batch of actions to this browser table's game.",
        None, False, True, body=_body({"actions": ("array", False, "The actions, up to 64 KiB in all.")}, True), tool=False, guest=True),
    _ep("play_profile", "play", "GET", "/api/profile", "Read this browser's player profile and the game's top scores.",
        None, False, False, tool=False, guest=True),
    _ep("play_profile_set", "play", "POST", "/api/profile", "Pick this browser's player profile.",
        None, False, True, body=_body({"name": ("string", False, "The profile name.")}, True), tool=False, guest=True),
    _ep("play_spectate_watch", "play", "GET", "/api/spectate/watch", "Poll a shared session as a viewer.",
        None, False, False,
        query={"vid": ("string", "The viewer id."), "rev": ("string", "The revision you hold."), "lrev": ("integer", "The live revision you hold."),
               "nostate": ("string", "Set to 1 when a socket carries the game."), "gk": ("string", "The game key you hold."),
               "kseq": ("integer", "The keyframe sequence you hold.")}, tool=False, guest=True),
    _ep("play_spectate", "play", "POST", "/api/spectate", "Start, stop or read the live spectating of your table.",
        None, False, True, body=_body({"op": ("string", False, "start, stop or status.")}), tool=False, guest=True),
    _ep("play_spectate_key", "play", "POST", "/api/spectate/key", "Push the owner's position keyframe to spectators.",
        None, False, True, body=_body({"key": ("string", False, "Base64 keyframe.")}), tool=False, guest=True),
    _ep("play_spectate_live", "play", "POST", "/api/spectate/live", "Push the owner's live state to spectators.",
        None, False, True, body=_body({"live": ("object", False, "The live state.")}), tool=False, guest=True),
    _ep("play_share_create", "play", "POST", "/api/share", "Store a view and get a share id.",
        None, False, True, body=_body({"view": ("object", False, "The view to share.")}), tool=False, guest=True),
    _ep("play_share_get", "play", "GET", "/api/share/{id}", "Read a shared view.",
        None, False, False, path_params={"id": ("string", "The share id.")}, tool=False, guest=True),
    _ep("play_clientlog", "play", "POST", "/api/clientlog", "Report a page's uncaught error to the server log.",
        None, False, True,
        body=_body({"kind": ("string", False, "The error kind."), "msg": ("string", False, "The message."),
                    "where": ("string", False, "Where it happened."), "stack": ("string", False, "The stack."),
                    "ua": ("string", False, "The user agent.")}), tool=False, guest=True),
    _ep("play_ready", "play", "GET", "/api/ready", "Show per-product readiness and the loading animation.",
        None, False, False, tool=False, guest=True),
    _ep("play_radio", "play", "GET", "/api/radio", "Show the Nodus Radio snapshot the site navigation displays.",
        "view_site", False, False, guest=True),
    _ep("play_client_rays_join", "play", "POST", "/api/client_rays/join", "Join a photo job's client ray participation.",
        "share_live", False, True, body=_body({"job": ("string", False, "The photo job.")}), tool=False),
    _ep("play_client_rays_engine", "play", "POST", "/api/client_rays/engine/{action}",
        "Worker-only mailbox for a photo engine on loopback.",
        None, False, True, path_params={"action": ("string", "The engine action, for example scene.")},
        body=_body({}, True), tool=False, credential="engine_secret"),
)


def by_name() -> dict:
    """name -> endpoint."""
    return {e["name"]: e for e in ENDPOINTS}


def kiosk_endpoint(method, path):
    """The kiosk entry whose method and path template match a request path (query removed), or None.

    A host answers a kiosk account (the read-only intern guest agent) only when this finds an entry; ``{name}`` matches
    one non-empty path segment, or the part of one segment before a literal suffix. The handler still validates
    everything and still asks for the entry's capability.
    """
    for entry, pattern in _KIOSK_PATTERNS:
        if entry["method"] == method and pattern.fullmatch(path):
            return entry
    return None


def guest_endpoint(method, path):
    """The entry a caller with no account may reach (kiosk, guest or credential entries) for a request path, or None.

    A host answers a visitor with no account on these entries and on its pages (paths outside /api/ and /ws); everything
    else is refused before any handler runs. The handler still validates everything and still asks for its capability.
    """
    for entry, pattern in _GUEST_PATTERNS:
        if entry["method"] == method and pattern.fullmatch(path):
            return entry
    return None


def endpoint_for(method, path):
    """Any entry whose method and path template match a request path, the most literal template first, or None.

    A host that refuses a request before its handler uses it to answer as that handler's first check would: the entry's
    capability, then its account requirement.
    """
    for entry, pattern in _ALL_PATTERNS:
        if entry["method"] == method and pattern.fullmatch(path):
            return entry
    return None


def reachable_by_guest(endpoint) -> bool:
    """True for an entry a caller with no account may reach: kiosk, guest, or answered by its own credential."""
    return bool(endpoint.get("kiosk") or endpoint.get("guest") or endpoint.get("credential"))


def allowed(endpoint, capabilities, signed_in, kiosk=False, guest=False) -> bool:
    """True when this caller may use the endpoint: signed in if it needs an account, and holding its capability.

    A kiosk caller may use only the entries declared kiosk; a gated guest only the entries ``reachable_by_guest``.
    """
    if kiosk and not endpoint.get("kiosk"):
        return False
    if guest and not reachable_by_guest(endpoint):
        return False
    if endpoint["account"] and not signed_in:
        return False
    need = endpoint["capability"]
    if need is None:
        return True
    have = set(capabilities or ())
    if isinstance(need, (tuple, list)):
        return (all if endpoint.get("capability_all") else any)(c in have for c in need)
    return need in have


def public(endpoint, capabilities=None, signed_in=False, kiosk=False, guest=False) -> dict:
    """A JSON-serializable copy of an endpoint; with ``capabilities`` given it also carries ``allowed``."""
    out = _copy(endpoint)
    if capabilities is not None:
        out["allowed"] = allowed(endpoint, capabilities, signed_in, kiosk, guest)
    return out


def index(capabilities=None, signed_in=False, group=None, kiosk=False, guest=False) -> dict:
    """The whole catalog as one JSON document; with ``capabilities`` given every endpoint says whether it is allowed.

    ``kiosk`` is True for a kiosk caller: only the kiosk entries are allowed, and the document says so. ``guest`` is
    True for a gated caller with no account: only the entries ``reachable_by_guest`` are allowed.
    """
    rows = [public(e, capabilities, signed_in, kiosk, guest) for e in ENDPOINTS if group is None or e["group"] == group]
    return {"catalog_version": CATALOG_VERSION, "groups": dict(GROUPS), "kiosk": bool(kiosk), "guest": bool(guest),
            "endpoints": rows}


def _copy(value):
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(v) for v in value]
    return value


def _template(path):
    """A path template as a pattern: each {name} is one non-empty segment (or the part before a literal suffix)."""
    import re
    return re.compile("".join("[^/]+" if part.startswith("{") else re.escape(part)
                              for part in re.split(r"(\{[a-z_]+\})", path) if part))


_KIOSK_PATTERNS = tuple((e, _template(e["path"])) for e in ENDPOINTS if e["kiosk"])
_GUEST_PATTERNS = tuple((e, _template(e["path"])) for e in ENDPOINTS if reachable_by_guest(e))
_ALL_PATTERNS = tuple(sorted(((e, _template(e["path"])) for e in ENDPOINTS), key=lambda pair: pair[0]["path"].count("{")))
