"""
SEFS Web Server
Flask + SocketIO server providing the interactive 2D UI and REST API.
"""

import os
import logging
import subprocess
import platform
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

logger = logging.getLogger("sefs.web_server")


def create_app(sync_manager, config):
    """Create and configure the Flask application."""

    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    logger.info(f"web_server __file__: {__file__}")
    logger.info(f"template_dir: {template_dir} (exists: {template_dir.exists()})")
    logger.info(f"static_dir: {static_dir} (exists: {static_dir.exists()})")

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config["SECRET_KEY"] = "sefs-secret-key-2025"
    
    logger.info(f"Flask static_folder: {app.static_folder}")
    logger.info(f"Flask static_url_path: {app.static_url_path}")

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    # Inject socketio into sync_manager
    sync_manager.socketio = socketio

    # ----- Routes -----

    @app.route("/")
    def index():
        """Serve the main UI."""
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        """Get system status."""
        return jsonify(sync_manager.get_status())

    @app.route("/api/graph")
    def api_graph():
        """Get current graph data for visualization."""
        return jsonify(sync_manager.get_graph())

    @app.route("/api/folders")
    def api_folders():
        """Get managed folder structure."""
        return jsonify(sync_manager.get_folders())

    @app.route("/api/config")
    def api_config():
        """Get current configuration."""
        return jsonify(config.to_dict())

    @app.route("/api/rescan", methods=["POST"])
    def api_rescan():
        """Trigger a manual rescan."""
        sync_manager.manual_rescan()
        return jsonify({"status": "rescan_triggered"})

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        """Reset semantic folder structure."""
        sync_manager.reset_structure()
        return jsonify({"status": "reset_complete"})

    @app.route("/api/open-file", methods=["POST"])
    def api_open_file():
        """Open a file with the system default application."""
        data = request.get_json()
        file_path = data.get("path", "")

        if not file_path or not Path(file_path).exists():
            return jsonify({"error": "File not found"}), 404

        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])

            return jsonify({"status": "opened", "path": file_path})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/open-folder", methods=["POST"])
    def api_open_folder():
        """Open the root folder in the system file explorer."""
        try:
            folder = str(config.ROOT_FOLDER)
            system = platform.system()
            if system == "Windows":
                os.startfile(folder)
            elif system == "Darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])

            return jsonify({"status": "opened", "path": folder})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/file-info")
    def api_file_info():
        """Get metadata for a specific file."""
        file_path = request.args.get("path", "")
        info = sync_manager.get_file_info(file_path)
        if info:
            return jsonify(info)
        return jsonify({"error": "File not found in registry"}), 404

    # ----- WebSocket Events -----

    @socketio.on("connect")
    def handle_connect():
        """Client connected - send current state."""
        logger.info("WebSocket client connected")
        status = sync_manager.get_status()
        graph = sync_manager.get_graph()
        folders = sync_manager.get_folders()
        clusters_data = sync_manager.get_clusters_serialized()
        logger.info(f"Sending initial data: {len(graph.get('nodes',[]))} nodes, {len(folders)} folders")
        # Use emit() to send only to the connecting client
        emit("update", {
            "graph": graph,
            "folders": folders,
            "clusters": clusters_data,
            "file_count": status.get("file_count", 0),
            "cluster_count": status.get("cluster_count", 0),
            "last_sync": status.get("last_sync"),
        })

    @socketio.on("disconnect")
    def handle_disconnect():
        logger.info("WebSocket client disconnected")

    @socketio.on("request_rescan")
    def handle_rescan():
        """Client requests a rescan."""
        sync_manager.manual_rescan()

    @socketio.on("request_reset")
    def handle_reset():
        """Client requests a reset."""
        sync_manager.reset_structure()

    return app, socketio
