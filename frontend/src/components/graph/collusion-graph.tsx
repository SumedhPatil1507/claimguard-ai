"use client";

import React, { useCallback, useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  BackgroundVariant,
} from "reactflow";
import "reactflow/dist/style.css";
import type { CollusionRing } from "@/types";
import { getRiskColor } from "@/lib/utils";

interface Props {
  rings: CollusionRing[];
}

const RING_PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#3b82f6", "#ec4899", "#8b5cf6"];

function buildGraph(rings: CollusionRing[]) {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const seenNodes = new Set<string>();

  rings.forEach((ring, ri) => {
    const ringColor = RING_PALETTE[ri % RING_PALETTE.length];
    const gnnColor  = getRiskColor(ring.severity);
    const cx = (ri % 3) * 420 + 80;
    const cy = Math.floor(ri / 3) * 340 + 60;

    // Claimant nodes
    ring.claimant_ids.forEach((cid, ci) => {
      const id = `claimant:${cid}`;
      if (!seenNodes.has(id)) {
        const angle = (ci / ring.claimant_ids.length) * 2 * Math.PI;
        const r     = 110;
        const gnnScore = ring.gnn_scores?.[cid] ?? 0;
        nodes.push({
          id,
          type: "default",
          position: { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) },
          data: { label: cid },
          style: {
            background: `${ringColor}22`,
            border: `2px solid ${gnnScore > 0.5 ? "#ef4444" : ringColor}`,
            borderRadius: "50%",
            width: 72,
            height: 72,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 9,
            color: "#e2e8f0",
            fontWeight: 600,
            cursor: "pointer",
          },
        });
        seenNodes.add(id);
      }
    });

    // Shared entity nodes
    ring.shared_entities.forEach((eid, ei) => {
      const id = `entity:${eid}`;
      if (!seenNodes.has(id)) {
        nodes.push({
          id,
          type: "default",
          position: { x: cx + 10 + ei * 30, y: cy },
          data: { label: eid.replace(/^(shop|provider|witness):/, "") },
          style: {
            background: "#f59e0b22",
            border: "2px solid #f59e0b",
            borderRadius: 8,
            width: 80,
            height: 36,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 9,
            color: "#fbbf24",
          },
        });
        seenNodes.add(id);
      }

      // Edges claimant → entity
      ring.claimant_ids.forEach((cid) => {
        const edgeId = `${cid}-${eid}`;
        edges.push({
          id: edgeId,
          source: `claimant:${cid}`,
          target: `entity:${eid}`,
          style: { stroke: `${gnnColor}80`, strokeWidth: 1.5 },
          animated: ring.severity === "high",
        });
      });
    });
  });

  return { nodes, edges };
}

export function CollusionGraph({ rings }: Props) {
  const { nodes: initNodes, edges: initEdges } = useMemo(() => buildGraph(rings), [rings]);
  const [nodes, , onNodesChange] = useNodesState(initNodes);
  const [edges, , onEdgesChange] = useEdgesState(initEdges);

  return (
    <div className="w-full h-[560px] rounded-xl overflow-hidden border border-border bg-[hsl(222,47%,7%)]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={2}
        attributionPosition="bottom-right"
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="hsl(217 32% 20%)" />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={(n) => (n.style?.border as string)?.split(" ")[2] ?? "#888"}
          maskColor="hsl(222 47% 7% / 0.8)"
          style={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 32% 17%)" }}
        />
      </ReactFlow>
    </div>
  );
}
