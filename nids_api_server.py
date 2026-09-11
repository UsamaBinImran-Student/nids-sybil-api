"""
nids_api_server.py
======================================================
NIDS REST API Server for Unity 3D VR Integration
======================================================
Deploy this on any cloud server (Railway, Render, Google Cloud, AWS)

Endpoints:
  POST /api/register         - Register a new VR node
  POST /api/authenticate     - Authenticate a node + detect Sybil
  GET  /api/network          - Get full network graph for VR visualization
  GET  /api/node/<did>       - Get individual node status
  POST /api/simulate         - Add simulated traffic (for demo/testing)
  GET  /api/stats            - Get system-wide stats
  GET  /api/health           - Health check
  GET  /api/leaderboard      - Top suspicious nodes (for VR display)

Run locally:
  pip install flask flask-cors scikit-learn networkx python-louvain numpy
  python nids_api_server.py

Deploy to Render (free):
  1. Push to GitHub
  2. Connect to render.com → New Web Service
  3. Build: pip install -r requirements.txt
  4. Start: python nids_api_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib, json, secrets, time, random, threading
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# ── Import your NIDS modules ───────────────────────────────────
# (These are your phase1-4 files — ensure they are in the same directory)
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from phase1.identity_manager import IdentityManager
    from phase2.authenticator    import Authenticator, ZKPEngine, ZKP_G, ZKP_P
    from phase3.detection_engine import DetectionEngine
    from phase4.response_engine  import ResponseEngine
    FULL_NIDS = True
    print("[✅] Full NIDS modules loaded")
except ImportError:
    FULL_NIDS = False
    print("[⚠️] Running in standalone mode (NIDS modules not found)")
    print("     Place phase1-4 folders next to nids_api_server.py for full NIDS")

app = Flask(__name__)
CORS(app)  # Allow Unity WebGL and any origin

# ═══════════════════════════════════════════════════════════════
# NIDS STATE (shared across all API calls)
# ═══════════════════════════════════════════════════════════════

class NIDSState:
    """Thread-safe NIDS state for the API server."""
    def __init__(self):
        self.lock = threading.Lock()
        # Node registry
        self.nodes: Dict[str, dict] = {}
        # Network graph (for VR visualization)
        self.edges: List[dict] = []
        self.edge_set: Set[tuple] = set()
        # Detection results
        self.sybil_scores: Dict[str, float] = {}
        self.confirmed_sybil: Set[str] = set()
        self.revoked: Set[str] = set()
        # Auth events
        self.auth_events: deque = deque(maxlen=500)
        # Stats
        self.total_reg   = 0
        self.total_auth  = 0
        self.total_sybil = 0
        self.total_revoked = 0
        self.start_time  = time.time()
        # Full NIDS components (if available)
        if FULL_NIDS:
            self.im   = IdentityManager()
            self.auth = Authenticator(self.im)
            self.det  = DetectionEngine()
            self.resp = ResponseEngine(self.im, self.auth)
            # Train ML on synthetic data at startup
            self._train_ml()
        else:
            self.im = self.auth = self.det = self.resp = None

    def _train_ml(self):
        """Train ML classifier with synthetic data at startup."""
        try:
            rng = np.random.RandomState(42)
            X, y = [], []
            for _ in range(800):
                X.append([rng.uniform(.1,3), rng.uniform(0,.1), rng.uniform(0,.05),
                          rng.uniform(.1,2), rng.uniform(0,.05), rng.uniform(0,.2), 0,
                          rng.uniform(.7,1), rng.uniform(.9,1), rng.randint(0,2),
                          rng.uniform(2.5,4), rng.uniform(300,3600), rng.uniform(0,.5)])
                y.append(0)
            for _ in range(200):
                X.append([rng.uniform(15,60), rng.uniform(.7,1), rng.uniform(.5,1),
                          rng.uniform(10,40), rng.uniform(.3,.8), rng.uniform(.5,1),
                          rng.randint(3,15), rng.uniform(0,.3), rng.uniform(0,.3),
                          rng.randint(5,20), rng.uniform(0,.8), rng.uniform(1,30),
                          rng.uniform(2,5)])
                y.append(1)
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            self.det.train(X, y)
            print("[✅] ML classifier trained at startup")
        except Exception as e:
            print(f"[⚠️] ML training skipped: {e}")

state = NIDSState()

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def make_did(address: str) -> str:
    return f"did:meta:{hashlib.sha256(address.encode()).hexdigest()[:20]}"

def get_node_color(score: float, is_sybil: bool, revoked: bool) -> dict:
    """
    Returns RGB color for Unity VR visualization.
    Green=safe, Yellow=suspicious, Red=confirmed Sybil, Gray=revoked
    """
    if revoked:
        return {"r": 0.4, "g": 0.4, "b": 0.4, "hex": "#666666"}
    if is_sybil:
        return {"r": 0.9, "g": 0.1, "b": 0.1, "hex": "#E61A1A"}
    if score > 0.5:
        t = (score - 0.5) / 0.15
        return {"r": min(1.0, t), "g": max(0.0, 1.0-t), "b": 0.0,
                "hex": "#FF8800"}
    return {"r": 0.1, "g": 0.8, "b": 0.2, "hex": "#1ACC33"}

def get_node_size(score: float) -> float:
    """Node size in VR — bigger = more suspicious."""
    return 0.3 + score * 0.7

def compute_sybil_score_simple(node_data: dict) -> float:
    """
    Fast Sybil score computation without full NIDS modules.
    Used as fallback when NIDS modules are not loaded.
    """
    score = 0.0
    # Auth rate
    rate = node_data.get("auth_rate", 0)
    if rate > 15:  score += 0.3
    elif rate > 5: score += 0.1
    # ZKP failure
    zkp_fail = node_data.get("zkp_failure_rate", 0)
    score += zkp_fail * 0.3
    # Reputation
    rep = node_data.get("reputation", 80)
    score += max(0, (80 - rep) / 80) * 0.2
    # Non-member attempts
    nm = node_data.get("non_member_attempts", 0)
    if nm > 0: score += 0.2
    # IP clustering
    ip_cluster = node_data.get("ip_cluster_score", 0)
    score += ip_cluster * 0.15
    return min(1.0, round(score, 4))

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status":      "online",
        "nids_loaded": FULL_NIDS,
        "uptime_s":    round(time.time() - state.start_time, 1),
        "total_nodes": len(state.nodes),
        "version":     "1.0.0"
    })

# ── REGISTER ──────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register_node():
    """
    Register a new VR metaverse node.

    Unity sends:
    {
        "address": "0xABC123...",
        "node_type": "avatar" | "iot" | "agent",
        "display_name": "Player_001",
        "deposit": 0.05,
        "position": {"x": 1.2, "y": 0.0, "z": -3.4}
    }

    Returns:
    {
        "success": true,
        "did": "did:meta:...",
        "color": {"r": 0.1, "g": 0.8, "b": 0.2},
        "size": 0.3,
        "sybil_score": 0.0
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data"}), 400

    address   = data.get("address", secrets.token_hex(20))
    node_type = data.get("node_type", "avatar")
    name      = data.get("display_name", f"Node_{len(state.nodes)+1}")
    deposit   = float(data.get("deposit", 0.05))
    position  = data.get("position", {"x": 0, "y": 0, "z": 0})

    with state.lock:
        did = make_did(address)

        # Check duplicate
        if did in state.nodes:
            return jsonify({"success": False,
                            "error": "DID already registered",
                            "did": did}), 409

        # Register in NIDS
        if FULL_NIDS:
            r = state.im.register(address, hashlib.sha256(address.encode()).hexdigest(),
                                  deposit)
            if not r["success"]:
                return jsonify({"success": False, "error": r["error"]}), 400
            state.im.issue_vc(did, {"role": node_type, "name": name})
            state.resp.register_identity(did, address)

        # Store node
        node = {
            "did":          did,
            "address":      address,
            "display_name": name,
            "node_type":    node_type,
            "deposit":      deposit,
            "reputation":   80,
            "sybil_score":  0.0,
            "is_sybil":     False,
            "revoked":      False,
            "registered_at": time.time(),
            "position":     position,
            "auth_count":   0,
            "auth_rate":    0.0,
            "zkp_failure_rate": 0.0,
            "non_member_attempts": 0,
            "ip_cluster_score": 0.0,
            "peer_flag_count": 0,
            "last_seen":    time.time(),
        }
        state.nodes[did] = node
        state.sybil_scores[did] = 0.0
        state.total_reg += 1

    color = get_node_color(0.0, False, False)
    return jsonify({
        "success":     True,
        "did":         did,
        "color":       color,
        "size":        get_node_size(0.0),
        "sybil_score": 0.0,
        "message":     f"Node {name} registered successfully"
    })

# ── AUTHENTICATE ──────────────────────────────────────────────
@app.route('/api/authenticate', methods=['POST'])
def authenticate_node():
    """
    Authenticate a node and run Sybil detection.

    Unity sends:
    {
        "did": "did:meta:...",
        "ip_address": "192.168.1.5",
        "auth_rate": 2.3,
        "zkp_failure_rate": 0.02,
        "non_member_attempts": 0,
        "reputation": 78,
        "deposit_ratio": 0.95,
        "peer_flag_count": 0,
        "session_id": "abc123"
    }

    Returns:
    {
        "success": true,
        "sybil_score": 0.12,
        "is_sybil": false,
        "color": {"r":0.1,"g":0.8,"b":0.2,"hex":"#1ACC33"},
        "size": 0.38,
        "triggered_rules": [],
        "action": "allow"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data"}), 400

    did = data.get("did", "")
    with state.lock:
        if did not in state.nodes:
            return jsonify({"success": False,
                            "error": "Node not registered",
                            "sybil_flag": True,
                            "flag_reason": "Non-membership: DID absent"}), 404

        node = state.nodes[did]
        if node["revoked"]:
            return jsonify({"success": False,
                            "error": "Node revoked",
                            "sybil_flag": True}), 403

        # Update node behavioral features
        node["auth_count"]          += 1
        node["auth_rate"]            = float(data.get("auth_rate", node["auth_rate"]))
        node["zkp_failure_rate"]     = float(data.get("zkp_failure_rate", 0))
        node["non_member_attempts"]  = int(data.get("non_member_attempts", 0))
        node["reputation"]           = float(data.get("reputation", node["reputation"]))
        node["ip_cluster_score"]     = float(data.get("ip_cluster_score", 0))
        node["peer_flag_count"]      = int(data.get("peer_flag_count", 0))
        node["last_seen"]            = time.time()
        ip = data.get("ip_address", "10.0.0.1")

        # Compute Sybil score
        if FULL_NIDS:
            et = "auth_attempt"
            if node["zkp_failure_rate"] > 0.3: et = "zkp_fail"
            if node["non_member_attempts"] > 0: et = "non_member"
            ss = state.det.process(
                did, et, {"ip": ip},
                rep=node["reputation"],
                dep=data.get("deposit_ratio", 1.0))
            score   = ss.score
            rules   = ss.rules
            is_sybil= ss.is_sybil
        else:
            score    = compute_sybil_score_simple(node)
            rules    = []
            is_sybil = score > 0.65
            if score > 0.5:  rules.append("HIGH_SCORE")
            if node["non_member_attempts"] > 0: rules.append("NON_MEMBER")
            if node["zkp_failure_rate"] > 0.3:  rules.append("ZKP_FAIL")

        node["sybil_score"] = score
        node["is_sybil"]    = is_sybil
        state.sybil_scores[did] = score

        if is_sybil:
            state.confirmed_sybil.add(did)
            state.total_sybil += 1

        # Log event
        state.auth_events.append({
            "ts":    time.time(),
            "did":   did[:20],
            "score": score,
            "sybil": is_sybil,
            "ip":    ip
        })
        state.total_auth += 1

    # Determine action
    if score > 0.90:   action = "revoke"
    elif score > 0.65: action = "isolate"
    elif score > 0.40: action = "monitor"
    else:              action = "allow"

    color = get_node_color(score, is_sybil, node["revoked"])

    return jsonify({
        "success":        True,
        "did":            did,
        "sybil_score":    score,
        "is_sybil":       is_sybil,
        "color":          color,
        "size":           get_node_size(score),
        "triggered_rules": rules,
        "action":         action,
        "reputation":     node["reputation"],
    })

# ── NETWORK GRAPH ─────────────────────────────────────────────
@app.route('/api/network', methods=['GET'])
def get_network():
    """
    Returns full network graph for Unity VR 3D visualization.

    Unity polls this every second to update the VR scene.

    Returns:
    {
        "nodes": [...],   ← all nodes with position, color, size, score
        "edges": [...],   ← all connections between nodes
        "stats": {...}    ← system-wide statistics
    }
    """
    with state.lock:
        nodes_out = []
        for did, node in state.nodes.items():
            score = node["sybil_score"]
            color = get_node_color(score, node["is_sybil"], node["revoked"])
            nodes_out.append({
                "did":          did,
                "display_name": node["display_name"],
                "node_type":    node["node_type"],
                "position":     node["position"],
                "color":        color,
                "size":         get_node_size(score),
                "sybil_score":  score,
                "is_sybil":     node["is_sybil"],
                "revoked":      node["revoked"],
                "reputation":   node["reputation"],
                "auth_count":   node["auth_count"],
                "last_seen":    node["last_seen"],
                # Status label for VR display
                "status_label": (
                    "REVOKED"     if node["revoked"] else
                    "SYBIL"       if node["is_sybil"] else
                    "SUSPICIOUS"  if score > 0.4 else
                    "SAFE"
                ),
                "triggered_rules": [],
            })

        edges_out = []
        for edge in state.edges:
            edges_out.append({
                "from":   edge["from"],
                "to":     edge["to"],
                "weight": edge.get("weight", 1.0),
                # Edge color: red if either node is Sybil
                "color":  "#FF3333" if (
                    state.nodes.get(edge["from"], {}).get("is_sybil") or
                    state.nodes.get(edge["to"], {}).get("is_sybil")
                ) else "#44AAFF"
            })

    uptime = time.time() - state.start_time
    return jsonify({
        "nodes":     nodes_out,
        "edges":     edges_out,
        "timestamp": time.time(),
        "stats": {
            "total_nodes":     len(state.nodes),
            "safe_nodes":      sum(1 for n in state.nodes.values()
                                   if not n["is_sybil"] and not n["revoked"]),
            "suspicious_nodes":sum(1 for n in state.nodes.values()
                                   if n["sybil_score"] > 0.4 and not n["is_sybil"]),
            "sybil_nodes":     sum(1 for n in state.nodes.values()
                                   if n["is_sybil"]),
            "revoked_nodes":   len(state.revoked),
            "total_auth":      state.total_auth,
            "total_sybil":     state.total_sybil,
            "uptime_s":        round(uptime, 1),
            "detection_rate":  round(
                state.total_sybil / max(state.total_auth, 1), 4),
        },
        "recent_events": list(state.auth_events)[-10:]
    })

# ── SINGLE NODE ───────────────────────────────────────────────
@app.route('/api/node/<path:did>', methods=['GET'])
def get_node(did):
    """Get detailed info for a single node (click on node in VR)."""
    with state.lock:
        node = state.nodes.get(did)
    if not node:
        return jsonify({"error": "Node not found"}), 404

    score = node["sybil_score"]
    color = get_node_color(score, node["is_sybil"], node["revoked"])
    return jsonify({
        **node,
        "color":  color,
        "size":   get_node_size(score),
        "status": ("REVOKED"    if node["revoked"] else
                   "SYBIL"      if node["is_sybil"] else
                   "SUSPICIOUS" if score > 0.4 else "SAFE"),
    })

# ── ADD EDGE (peer interaction) ───────────────────────────────
@app.route('/api/interact', methods=['POST'])
def add_interaction():
    """
    Record a peer interaction — adds an edge in the network graph.

    Unity sends when two avatars communicate:
    {
        "from_did": "did:meta:...",
        "to_did":   "did:meta:...",
        "type":     "chat" | "trade" | "proximity"
    }
    """
    data = request.get_json()
    from_did = data.get("from_did", "")
    to_did   = data.get("to_did",   "")
    itype    = data.get("type", "proximity")

    with state.lock:
        edge_key = tuple(sorted([from_did, to_did]))
        if edge_key not in state.edge_set:
            state.edges.append({
                "from": from_did, "to": to_did,
                "type": itype, "weight": 1.0,
                "created_at": time.time()
            })
            state.edge_set.add(edge_key)
            if FULL_NIDS:
                state.det.add_interaction(from_did, to_did)
        else:
            # Increase weight on existing edge
            for e in state.edges:
                if e["from"] == from_did and e["to"] == to_did:
                    e["weight"] = min(5.0, e["weight"] + 0.5)
                    break

    return jsonify({"success": True, "edge": f"{from_did[:12]}↔{to_did[:12]}"})

# ── REVOKE (Phase 4) ──────────────────────────────────────────
@app.route('/api/revoke', methods=['POST'])
def revoke_node():
    """
    Revoke a confirmed Sybil node (Phase 4 pipeline).
    Unity calls this when admin confirms a Sybil node in VR.

    {  "did": "did:meta:...", "admin_token": "secret" }
    """
    data = request.get_json()
    did  = data.get("did", "")

    with state.lock:
        node = state.nodes.get(did)
        if not node:
            return jsonify({"error": "Node not found"}), 404

        score = node["sybil_score"]
        node["revoked"] = True
        node["is_sybil"] = True
        state.revoked.add(did)
        state.total_revoked += 1

        # Phase 4 pipeline
        slash = node["deposit"] * (1.0 if score > 0.9 else 0.5 if score > 0.8 else 0.25)
        sev   = "FULL" if score > 0.9 else "MODERATE" if score > 0.8 else "MILD"
        node["deposit"] = max(0, node["deposit"] - slash)
        node["reputation"] = 0

        if FULL_NIDS:
            state.im.revoke(did)

    return jsonify({
        "success":   True,
        "did":       did,
        "severity":  sev,
        "slashed":   round(slash, 4),
        "color":     get_node_color(1.0, True, True),
        "message":   f"Node revoked. {slash:.4f} ETH slashed ({sev})."
    })

# ── SIMULATE (inject test traffic) ────────────────────────────
@app.route('/api/simulate', methods=['POST'])
def simulate_traffic():
    """
    Inject simulated Sybil + legitimate traffic for VR demo.
    Call this to populate the VR scene with test nodes.

    { "n_legit": 10, "n_sybil": 5 }
    """
    data    = request.get_json() or {}
    n_legit = int(data.get("n_legit", 8))
    n_sybil = int(data.get("n_sybil", 4))
    import random; random.seed(int(time.time()))

    created = []

    # Register legitimate nodes
    for i in range(n_legit):
        addr = f"0x{hashlib.sha256(f'sim_legit_{i}_{time.time()}'.encode()).hexdigest()[:40]}"
        did  = make_did(addr)
        angle = random.uniform(0, 360)
        radius = random.uniform(3, 8)
        import math
        pos = {
            "x": radius * math.cos(math.radians(angle)),
            "y": random.uniform(0, 2),
            "z": radius * math.sin(math.radians(angle))
        }
        with state.lock:
            if did not in state.nodes:
                state.nodes[did] = {
                    "did": did, "address": addr,
                    "display_name": f"Avatar_{i+1:03d}",
                    "node_type": "avatar", "deposit": 0.1,
                    "reputation": random.randint(70, 100),
                    "sybil_score": random.uniform(0.0, 0.15),
                    "is_sybil": False, "revoked": False,
                    "registered_at": time.time(), "position": pos,
                    "auth_count": random.randint(1, 20),
                    "auth_rate": random.uniform(0.5, 3.0),
                    "zkp_failure_rate": random.uniform(0, 0.03),
                    "non_member_attempts": 0,
                    "ip_cluster_score": random.uniform(0, 0.08),
                    "peer_flag_count": 0, "last_seen": time.time()
                }
                state.total_reg += 1
                created.append(did)

    # Register Sybil nodes (clustered together)
    cluster_x = random.uniform(-5, 5)
    cluster_z = random.uniform(-5, 5)
    for i in range(n_sybil):
        addr = f"0x{hashlib.sha256(f'sim_sybil_{i}_{time.time()}'.encode()).hexdigest()[:40]}"
        did  = make_did(addr)
        score = random.uniform(0.68, 0.95)
        pos = {
            "x": cluster_x + random.uniform(-1, 1),
            "y": random.uniform(0, 1),
            "z": cluster_z + random.uniform(-1, 1)
        }
        with state.lock:
            if did not in state.nodes:
                state.nodes[did] = {
                    "did": did, "address": addr,
                    "display_name": f"Unknown_{i+1:03d}",
                    "node_type": "agent", "deposit": 0.01,
                    "reputation": random.randint(5, 30),
                    "sybil_score": score,
                    "is_sybil": score > 0.65, "revoked": False,
                    "registered_at": time.time(), "position": pos,
                    "auth_count": random.randint(50, 200),
                    "auth_rate": random.uniform(15, 60),
                    "zkp_failure_rate": random.uniform(0.3, 0.8),
                    "non_member_attempts": random.randint(1, 10),
                    "ip_cluster_score": random.uniform(0.6, 0.95),
                    "peer_flag_count": random.randint(3, 15),
                    "last_seen": time.time()
                }
                if score > 0.65:
                    state.confirmed_sybil.add(did)
                    state.total_sybil += 1
                state.total_reg += 1
                created.append(did)

    # Add edges between Sybil nodes (cluster)
    sybil_list = [d for d in created
                  if state.nodes.get(d, {}).get("is_sybil")]
    with state.lock:
        for i in range(len(sybil_list)):
            for j in range(i+1, len(sybil_list)):
                ek = tuple(sorted([sybil_list[i], sybil_list[j]]))
                if ek not in state.edge_set:
                    state.edges.append({"from": sybil_list[i],
                                        "to":   sybil_list[j],
                                        "type": "sybil_cluster",
                                        "weight": 3.0,
                                        "created_at": time.time()})
                    state.edge_set.add(ek)

    return jsonify({
        "success":       True,
        "created_legit": n_legit,
        "created_sybil": n_sybil,
        "total_nodes":   len(state.nodes),
        "message":       f"Simulated {n_legit} legit + {n_sybil} Sybil nodes"
    })

# ── STATS ─────────────────────────────────────────────────────
@app.route('/api/stats', methods=['GET'])
def get_stats():
    with state.lock:
        return jsonify({
            "total_nodes":    len(state.nodes),
            "total_auth":     state.total_auth,
            "total_sybil":    state.total_sybil,
            "total_revoked":  state.total_revoked,
            "total_reg":      state.total_reg,
            "sybil_rate":     round(state.total_sybil/max(state.total_auth,1),4),
            "uptime_s":       round(time.time()-state.start_time,1),
            "nids_loaded":    FULL_NIDS,
        })

# ── LEADERBOARD (top suspicious nodes) ───────────────────────
@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    """Top 10 most suspicious nodes — for VR HUD display."""
    with state.lock:
        ranked = sorted(state.nodes.values(),
                        key=lambda n: n["sybil_score"], reverse=True)[:10]
        return jsonify({
            "leaderboard": [{
                "rank":         i+1,
                "display_name": n["display_name"],
                "did":          n["did"],
                "sybil_score":  n["sybil_score"],
                "is_sybil":     n["is_sybil"],
                "color":        get_node_color(n["sybil_score"],
                                               n["is_sybil"], n["revoked"]),
            } for i, n in enumerate(ranked)]
        })

# ── CLEAR (reset for demo) ────────────────────────────────────
@app.route('/api/clear', methods=['POST'])
def clear_all():
    """Reset everything — useful for demo restart."""
    global state
    state = NIDSState()
    return jsonify({"success": True, "message": "State cleared"})

# ═══════════════════════════════════════════════════════════════
# START SERVER
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  NIDS API Server — Unity 3D VR Integration")
    print("="*55)
    print(f"  NIDS modules: {'Loaded ✅' if FULL_NIDS else 'Standalone mode ⚠️'}")
    print(f"  Endpoints:")
    print(f"    POST /api/register       — Register VR node")
    print(f"    POST /api/authenticate   — Auth + detect Sybil")
    print(f"    GET  /api/network        — Full graph for Unity")
    print(f"    POST /api/interact       — Add peer edge")
    print(f"    POST /api/revoke         — Phase 4 revocation")
    print(f"    POST /api/simulate       — Inject test nodes")
    print(f"    GET  /api/stats          — System stats")
    print(f"    GET  /api/leaderboard    — Top suspicious nodes")
    print(f"    GET  /api/health         — Health check")
    print(f"\n  Running on http://0.0.0.0:5000")
    print("="*55 + "\n")
    import os
port = int(os.environ.get("PORT", 8000))
app.run(host="0.0.0.0", port=port, debug=False)
