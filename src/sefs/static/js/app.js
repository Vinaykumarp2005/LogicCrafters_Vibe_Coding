/**
 * SEFS - Semantic Entropy File System
 * Interactive 2D Visualization with D3.js Force-Directed Graph
 * Connects via WebSocket for live updates + REST API fallback
 */

(function () {
    'use strict';

    // ===== Configuration =====
    const CLUSTER_COLORS = [
        '#6C63FF', '#00D1B2', '#FF6B6B', '#FFD93D', '#4DA8FF',
        '#FB923C', '#F472B6', '#4ADE80', '#A78BFA', '#38BDF8'
    ];

    const EXT_ICONS = {
        '.pdf': '📄', '.txt': '📝', '.md': '📋', '.csv': '📊',
        '.json': '⚙️', '.rst': '📑', '.log': '📜'
    };

    // ===== State =====
    let graphData = { nodes: [], edges: [] };
    let simulation = null;
    let svg, g, linkGroup, nodeGroup, hullGroup, labelGroup;
    let zoom;
    let showLabels = true;
    let showEdges = true;
    let width, height;

    // ===== Socket.IO Connection =====
    const socket = io();

    socket.on('connect', () => {
        console.log('[SEFS] WebSocket connected');
        updateStatus('Connected', 'idle');
        // Also fetch via REST as fallback (in case WS initial data is delayed)
        setTimeout(fetchFullState, 500);
    });

    socket.on('disconnect', () => {
        console.log('[SEFS] WebSocket disconnected');
        updateStatus('Disconnected', 'error');
    });

    socket.on('update', (data) => {
        console.log('[SEFS] Received update:', Object.keys(data));
        applyUpdate(data);
    });

    socket.on('status', (data) => {
        updateStatus(data.message, data.processing ? 'processing' : 'idle');
    });

    socket.on('file_event', (data) => {
        if (data.events) {
            data.events.forEach(evt => {
                showToast(`File ${evt.type}: ${evt.file}`, 'info');
            });
        }
    });

    socket.on('reset', () => {
        graphData = { nodes: [], edges: [] };
        renderGraph();
        renderFolderTree([]);
        document.getElementById('file-count').textContent = '0';
        document.getElementById('cluster-count').textContent = '0';
        document.getElementById('folder-count').textContent = '0';
        showToast('Semantic structure reset', 'warning');
    });

    // ===== REST API Fallback =====
    function fetchFullState() {
        console.log('[SEFS] Fetching state via REST API...');

        Promise.all([
            fetch('/api/graph').then(r => r.json()),
            fetch('/api/folders').then(r => r.json()),
            fetch('/api/status').then(r => r.json())
        ]).then(([graph, folders, status]) => {
            console.log('[SEFS] REST data:', graph.nodes?.length, 'nodes,', folders.length, 'folders');

            // Only apply if we don't already have data from WebSocket
            if (!graphData.nodes || graphData.nodes.length === 0) {
                applyUpdate({
                    graph: graph,
                    folders: folders,
                    file_count: status.file_count,
                    cluster_count: status.cluster_count,
                    last_sync: status.last_sync,
                });
            }

            // Update status
            if (status.processing) {
                updateStatus('Processing...', 'processing');
            } else if (status.file_count > 0) {
                updateStatus(`${status.file_count} files organized`, 'idle');
            } else {
                updateStatus('Ready - add files to begin', 'idle');
            }
        }).catch(err => {
            console.error('[SEFS] REST fetch error:', err);
        });
    }

    function applyUpdate(data) {
        if (data.graph && data.graph.nodes && data.graph.nodes.length > 0) {
            graphData = data.graph;
            renderGraph();
        }
        if (data.folders && Array.isArray(data.folders)) {
            renderFolderTree(data.folders);
            document.getElementById('folder-count').textContent = data.folders.length;
        }
        if (data.file_count !== undefined) {
            document.getElementById('file-count').textContent = data.file_count;
        }
        if (data.cluster_count !== undefined) {
            document.getElementById('cluster-count').textContent = data.cluster_count;
        }
        if (data.last_sync) {
            const time = new Date(data.last_sync).toLocaleTimeString();
            document.getElementById('last-sync').textContent = `Last sync: ${time}`;
        }
    }

    // ===== UI Helper Functions =====
    function updateStatus(message, state) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');
        indicator.className = `status-dot ${state}`;
        text.textContent = message;
    }

    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 3500);
    }

    function formatSize(bytes) {
        if (!bytes) return '0 B';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function clusterColor(clusterId) {
        return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
    }

    // ===== Folder Tree =====
    function renderFolderTree(folders) {
        const tree = document.getElementById('folder-tree');

        if (!folders || folders.length === 0) {
            tree.innerHTML = `
                <div class="empty-state">
                    <p>No semantic folders yet.</p>
                    <p class="hint">Add files to the monitored folder to begin.</p>
                </div>`;
            return;
        }

        let html = '';
        folders.forEach((folder, idx) => {
            const color = clusterColor(idx);
            const escapedName = folder.name.replace(/'/g, "\\'").replace(/"/g, '&quot;');
            html += `
                <div class="folder-item">
                    <div class="folder-header" data-folder="${escapedName}">
                        <span class="folder-color-dot" style="background: ${color}"></span>
                        <span class="folder-name" title="${folder.name}">${folder.name}</span>
                        <span class="folder-count">${folder.file_count}</span>
                        <span class="folder-chevron">▶</span>
                    </div>
                    <div class="folder-files">`;

            if (folder.files) {
                folder.files.forEach(file => {
                    const ext = '.' + file.name.split('.').pop().toLowerCase();
                    const icon = EXT_ICONS[ext] || '📄';
                    const escapedPath = file.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
                    html += `
                        <div class="file-item" data-path="${escapedPath}">
                            <span class="file-icon">${icon}</span>
                            <span class="file-name" title="${file.name}">${file.name}</span>
                        </div>`;
                });
            }

            html += `</div></div>`;
        });

        tree.innerHTML = html;

        // Attach event listeners
        tree.querySelectorAll('.folder-header').forEach(header => {
            header.addEventListener('click', () => {
                const files = header.nextElementSibling;
                const chevron = header.querySelector('.folder-chevron');
                files.classList.toggle('open');
                chevron.classList.toggle('open');
            });
        });

        tree.querySelectorAll('.file-item').forEach(item => {
            item.addEventListener('click', () => {
                const path = item.getAttribute('data-path');
                openFile(path);
            });
        });
    }

    function openFile(path) {
        fetch('/api/open-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path })
        }).then(r => r.json()).then(data => {
            if (data.error) showToast(`Error: ${data.error}`, 'error');
        }).catch(() => showToast('Error opening file', 'error'));
    }

    // ===== D3.js Graph Visualization =====
    function initGraph() {
        const container = document.getElementById('graph-canvas');
        width = container.clientWidth;
        height = container.clientHeight;

        svg = d3.select('#graph-canvas')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        zoom = d3.zoom()
            .scaleExtent([0.1, 5])
            .on('zoom', (event) => g.attr('transform', event.transform));

        svg.call(zoom);

        g = svg.append('g');
        hullGroup = g.append('g').attr('class', 'hulls');
        linkGroup = g.append('g').attr('class', 'links');
        nodeGroup = g.append('g').attr('class', 'nodes');
        labelGroup = g.append('g').attr('class', 'labels');

        window.addEventListener('resize', () => {
            width = container.clientWidth;
            height = container.clientHeight;
            svg.attr('width', width).attr('height', height);
            if (simulation) {
                simulation.force('center', d3.forceCenter(width / 2, height / 2));
                simulation.alpha(0.3).restart();
            }
        });
    }

    function renderGraph() {
        const emptyState = document.getElementById('graph-empty');

        if (!graphData.nodes || graphData.nodes.length === 0) {
            emptyState.style.display = 'block';
            hullGroup.selectAll('*').remove();
            linkGroup.selectAll('*').remove();
            nodeGroup.selectAll('*').remove();
            labelGroup.selectAll('*').remove();
            return;
        }

        emptyState.style.display = 'none';
        console.log('[SEFS] Rendering graph:', graphData.nodes.length, 'nodes,', graphData.edges.length, 'edges');

        // --- Prepare data ---
        const nodes = graphData.nodes.map(n => ({
            ...n,
            x: n.x || width / 2 + (Math.random() - 0.5) * 300,
            y: n.y || height / 2 + (Math.random() - 0.5) * 300,
            radius: Math.max(10, Math.min(24, 8 + Math.sqrt((n.size || 1000) / 800)))
        }));

        const nodeMap = new Map(nodes.map(n => [n.id, n]));

        const edges = (graphData.edges || []).filter(e => {
            const sid = e.source?.id || e.source;
            const tid = e.target?.id || e.target;
            return nodeMap.has(sid) && nodeMap.has(tid);
        }).map(e => ({ ...e }));

        // --- Stop old simulation ---
        if (simulation) simulation.stop();

        // --- Create simulation ---
        simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges)
                .id(d => d.id)
                .distance(d => 140 * (1 - (d.weight || 0)))
                .strength(d => (d.weight || 0.1) * 0.4))
            .force('charge', d3.forceManyBody()
                .strength(-250)
                .distanceMax(500))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide()
                .radius(d => d.radius + 12))
            .force('x', d3.forceX(width / 2).strength(0.04))
            .force('y', d3.forceY(height / 2).strength(0.04))
            .alphaDecay(0.02)
            .velocityDecay(0.4);

        // --- EDGES ---
        linkGroup.selectAll('.edge').remove();

        const links = linkGroup.selectAll('.edge')
            .data(edges)
            .enter()
            .append('line')
            .attr('class', 'edge')
            .attr('stroke', '#2D2F42')
            .attr('stroke-width', d => Math.max(0.5, (d.weight || 0.1) * 3))
            .attr('stroke-opacity', 0.3)
            .attr('visibility', showEdges ? 'visible' : 'hidden');

        // --- NODES ---
        nodeGroup.selectAll('.node-group').remove();

        const nodeGroups = nodeGroup.selectAll('.node-group')
            .data(nodes, d => d.id)
            .enter()
            .append('g')
            .attr('class', 'node-group')
            .call(d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded))
            .on('mouseover', handleMouseOver)
            .on('mousemove', handleMouseMove)
            .on('mouseout', handleMouseOut)
            .on('click', handleClick);

        // Circle
        nodeGroups.append('circle')
            .attr('class', 'node-circle')
            .attr('r', d => d.radius)
            .attr('fill', d => clusterColor(d.cluster))
            .attr('stroke', d => {
                const c = d3.color(clusterColor(d.cluster));
                return c ? c.brighter(0.5).toString() : '#fff';
            })
            .attr('stroke-width', 2)
            .attr('fill-opacity', 0.85);

        // Extension text inside node
        nodeGroups.append('text')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('font-size', d => Math.max(7, d.radius * 0.55))
            .attr('fill', 'white')
            .attr('font-weight', '600')
            .attr('font-family', 'Inter, sans-serif')
            .attr('pointer-events', 'none')
            .text(d => {
                const ext = (d.extension || '').replace('.', '').toUpperCase();
                return ext.substring(0, 3) || 'TXT';
            });

        // --- LABELS ---
        labelGroup.selectAll('.node-label').remove();

        const labels = labelGroup.selectAll('.node-label')
            .data(nodes, d => d.id)
            .enter()
            .append('text')
            .attr('class', 'node-label')
            .text(d => d.name.length > 20 ? d.name.substring(0, 18) + '…' : d.name)
            .attr('visibility', showLabels ? 'visible' : 'hidden');

        // --- SIMULATION TICK ---
        simulation.on('tick', () => {
            links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeGroups.attr('transform', d => `translate(${d.x},${d.y})`);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y + d.radius + 15);

            updateHulls(nodes);
        });
    }

    function updateHulls(nodes) {
        const clusterGroups = {};
        nodes.forEach(n => {
            if (n.x === undefined || n.y === undefined) return;
            if (!clusterGroups[n.cluster]) clusterGroups[n.cluster] = [];
            clusterGroups[n.cluster].push(n);
        });

        const hullData = [];
        Object.entries(clusterGroups).forEach(([clusterId, clusterNodes]) => {
            if (clusterNodes.length >= 3) {
                const points = clusterNodes.map(n => [n.x, n.y]);
                try {
                    const hull = d3.polygonHull(points);
                    if (hull) {
                        const centroid = d3.polygonCentroid(hull);
                        if (centroid && isFinite(centroid[0]) && isFinite(centroid[1])) {
                            hullData.push({
                                id: clusterId,
                                path: hull,
                                color: clusterColor(parseInt(clusterId)),
                                label: clusterNodes[0].cluster_label || `Cluster ${clusterId}`,
                                centroid: centroid
                            });
                        }
                    }
                } catch (e) { /* skip degenerate hulls */ }
            } else if (clusterNodes.length === 2) {
                // Two nodes - draw a rounded rect between them
                const cx = (clusterNodes[0].x + clusterNodes[1].x) / 2;
                const cy = (clusterNodes[0].y + clusterNodes[1].y) / 2;
                hullData.push({
                    id: clusterId,
                    path: null,
                    twoNodes: clusterNodes,
                    color: clusterColor(parseInt(clusterId)),
                    label: clusterNodes[0].cluster_label || `Cluster ${clusterId}`,
                    centroid: [cx, cy]
                });
            }
        });

        // Hulls
        hullGroup.selectAll('.cluster-hull').remove();
        hullGroup.selectAll('.cluster-label').remove();

        hullData.forEach(d => {
            if (d.path) {
                const centroid = d.centroid;
                const expandedPath = d.path.map(p => {
                    const dx = p[0] - centroid[0];
                    const dy = p[1] - centroid[1];
                    const len = Math.sqrt(dx * dx + dy * dy) || 1;
                    const expand = 35;
                    return [p[0] + (dx / len) * expand, p[1] + (dy / len) * expand];
                });

                hullGroup.append('path')
                    .attr('class', 'cluster-hull')
                    .attr('d', `M${expandedPath.join('L')}Z`)
                    .attr('fill', d.color)
                    .attr('fill-opacity', 0.04)
                    .attr('stroke', d.color)
                    .attr('stroke-opacity', 0.15)
                    .attr('stroke-width', 1.5)
                    .attr('stroke-dasharray', '6 3');
            } else if (d.twoNodes) {
                // Ellipse around two nodes
                const n1 = d.twoNodes[0], n2 = d.twoNodes[1];
                const rx = Math.abs(n1.x - n2.x) / 2 + 50;
                const ry = Math.abs(n1.y - n2.y) / 2 + 40;

                hullGroup.append('ellipse')
                    .attr('class', 'cluster-hull')
                    .attr('cx', d.centroid[0])
                    .attr('cy', d.centroid[1])
                    .attr('rx', rx)
                    .attr('ry', ry)
                    .attr('fill', d.color)
                    .attr('fill-opacity', 0.04)
                    .attr('stroke', d.color)
                    .attr('stroke-opacity', 0.15)
                    .attr('stroke-width', 1.5)
                    .attr('stroke-dasharray', '6 3');
            }

            // Cluster label
            hullGroup.append('text')
                .attr('class', 'cluster-label')
                .attr('x', d.centroid[0])
                .attr('y', d.centroid[1] - 50)
                .attr('fill', d.color)
                .attr('fill-opacity', 0.5)
                .attr('font-size', '12px')
                .attr('font-weight', '600')
                .attr('font-family', 'Inter, sans-serif')
                .attr('text-anchor', 'middle')
                .text(d.label);
        });
    }

    // ===== Drag Handlers =====
    function dragStarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragEnded(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // ===== Tooltip Handlers =====
    function handleMouseOver(event, d) {
        const tooltip = document.getElementById('tooltip');
        const ext = (d.extension || '.txt').replace('.', '').toUpperCase();

        document.getElementById('tooltip-icon').textContent = ext;
        document.getElementById('tooltip-icon').style.background = clusterColor(d.cluster);
        document.getElementById('tooltip-name').textContent = d.name;
        document.getElementById('tooltip-cluster').textContent = d.cluster_label || `Cluster ${d.cluster}`;
        document.getElementById('tooltip-size').textContent = formatSize(d.size);
        document.getElementById('tooltip-modified').textContent = d.modified
            ? new Date(d.modified).toLocaleString() : 'N/A';
        document.getElementById('tooltip-ext').textContent = ext;

        tooltip.classList.add('visible');

        // Highlight connected edges
        linkGroup.selectAll('.edge')
            .attr('stroke-opacity', l => {
                const sid = l.source?.id || l.source;
                const tid = l.target?.id || l.target;
                return (sid === d.id || tid === d.id) ? 0.7 : 0.08;
            })
            .attr('stroke', l => {
                const sid = l.source?.id || l.source;
                const tid = l.target?.id || l.target;
                return (sid === d.id || tid === d.id) ? clusterColor(d.cluster) : '#2D2F42';
            });

        // Dim other nodes
        nodeGroup.selectAll('.node-circle')
            .attr('fill-opacity', n => n.id === d.id ? 1 : 0.25);
    }

    function handleMouseMove(event) {
        const tooltip = document.getElementById('tooltip');
        let x = event.pageX + 16;
        let y = event.pageY - 10;
        const maxX = window.innerWidth - tooltip.offsetWidth - 20;
        const maxY = window.innerHeight - tooltip.offsetHeight - 20;
        tooltip.style.left = Math.min(x, maxX) + 'px';
        tooltip.style.top = Math.min(y, maxY) + 'px';
    }

    function handleMouseOut() {
        document.getElementById('tooltip').classList.remove('visible');
        linkGroup.selectAll('.edge')
            .attr('stroke-opacity', 0.3).attr('stroke', '#2D2F42');
        nodeGroup.selectAll('.node-circle')
            .attr('fill-opacity', 0.85);
    }

    function handleClick(event, d) {
        openFile(d.id);
    }

    // ===== Button Handlers =====
    document.getElementById('btn-rescan').addEventListener('click', () => {
        socket.emit('request_rescan');
        updateStatus('Rescanning...', 'processing');
        showToast('Rescan triggered...', 'info');
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        if (confirm('Reset all semantic folders? Files will be moved back to root.')) {
            socket.emit('request_reset');
        }
    });

    document.getElementById('btn-open-folder').addEventListener('click', () => {
        fetch('/api/open-folder', { method: 'POST' });
    });

    document.getElementById('btn-zoom-in').addEventListener('click', () => {
        svg.transition().duration(300).call(zoom.scaleBy, 1.3);
    });

    document.getElementById('btn-zoom-out').addEventListener('click', () => {
        svg.transition().duration(300).call(zoom.scaleBy, 0.7);
    });

    document.getElementById('btn-zoom-reset').addEventListener('click', () => {
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    });

    document.getElementById('btn-toggle-labels').addEventListener('click', function () {
        showLabels = !showLabels;
        this.classList.toggle('active', showLabels);
        labelGroup.selectAll('.node-label')
            .attr('visibility', showLabels ? 'visible' : 'hidden');
    });

    document.getElementById('btn-toggle-edges').addEventListener('click', function () {
        showEdges = !showEdges;
        this.classList.toggle('active', showEdges);
        linkGroup.selectAll('.edge')
            .attr('visibility', showEdges ? 'visible' : 'hidden');
    });

    // ===== Auto-refresh =====
    // Periodically check for updates in case WebSocket misses events
    setInterval(() => {
        if (graphData.nodes.length === 0) {
            fetchFullState();
        }
    }, 5000);

    // ===== Initialize =====
    initGraph();
    updateStatus('Connecting...', 'processing');

})();
