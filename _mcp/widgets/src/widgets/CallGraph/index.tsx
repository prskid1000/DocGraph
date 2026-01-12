import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard } from "../../entity-utils";
import { NestedObject, formatKey, formatValue } from "../../entity-utils";

interface CallNode {
  name: string;
  type: string;
  file_path?: string;
  line_number?: number;
  children?: CallNode[];
}

interface CallGraphResponse {
  success?: boolean;
  data?: CallNode;
  error?: string;
  isError?: boolean;
  function_name?: string;
  codebase_id?: string;
  name?: string;
  type?: string;
  file_path?: string;
  line_number?: number;
  children?: CallNode[];
  [key: string]: any;
}

function CallGraphWidget() {
  const data = useWidgetProps<CallGraphResponse>({});
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  // Check for isError flag
  const isError = data.isError === true;

  // Check for errors
  const error = data.error;

  // Extract call graph data
  const callGraph = data.data || (data.name ? data as CallNode : null);
  const functionName = data.function_name || callGraph?.name || "";

  const toggleNode = (nodePath: string) => {
    setExpanded((prev) => ({ ...prev, [nodePath]: !prev[nodePath] }));
  };

  const isNodeExpanded = (nodePath: string) => expanded[nodePath];

  const renderNode = (node: CallNode, level = 0, path = ""): React.JSX.Element => {
    const nodePath = `${path}/${node.name}`;
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = isNodeExpanded(nodePath);

    return (
      <div key={nodePath} className="ml-0" style={{ marginLeft: level > 0 ? '1.5rem' : '0' }}>
        <div className="flex items-start gap-2 py-2 px-3 hover:bg-teal-50 rounded border-l-2 border-teal-300">
          {hasChildren && (
            <button
              onClick={() => toggleNode(nodePath)}
              className="text-gray-500 hover:text-gray-700 mt-0.5"
            >
              <svg 
                className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          )}
          {!hasChildren && <div className="w-4"></div>}
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-teal-100 text-teal-800 border border-teal-200">
                {node.type}
              </span>
              <span className="font-semibold text-gray-900 font-mono text-sm">{node.name}</span>
            </div>
            {node.file_path && (
              <div className="text-xs text-gray-600 font-mono mt-1">
                {node.file_path}:{node.line_number}
              </div>
            )}
          </div>
        </div>

        {hasChildren && isExpanded && (
          <div className="ml-2">
            {node.children!.map((child, idx) => renderNode(child, level + 1, nodePath))}
          </div>
        )}
      </div>
    );
  };

  if (isError) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Call Graph Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800">An error occurred while generating call graph.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Call Graph Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Loading state: no data available yet
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || !callGraph) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-16 sm:py-24">
          <div className="relative">
            <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
          </div>
          <p className="mt-6 text-lg sm:text-xl font-semibold text-gray-700">Loading Call Graph...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">
          Call Graph {functionName && <span className="text-gray-600">for {functionName}</span>}
        </h2>
        <p className="text-sm text-gray-600">
          Click nodes with arrows to expand/collapse call hierarchy
        </p>
      </div>

      {/* Call Tree */}
      <div className="bg-white rounded-lg border border-teal-200 p-4">
        {Array.isArray(callGraph?.children) && callGraph.children.length > 0 ? (
          <div className="space-y-4">
            {callGraph.children.map((call: any, idx: number) => (
              <ExpandableCard
                key={call.id || idx}
                expanded={!!expanded[call.id || idx]}
                onToggle={() => setExpanded((prev) => ({ ...prev, [call.id || idx]: !prev[call.id || idx] }))}
                header={
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900 truncate" title={call.name || call.id}>
                      {call.name || call.id}
                    </h3>
                    <span className="text-sm text-gray-600 font-mono">
                      {call.file_path}:{call.line_number}
                    </span>
                  </div>
                }
              >
                <div className="mt-2">
                  <NestedObject data={call} excludeKeys={["id", "name", "file_path", "line_number"]} />
                </div>
              </ExpandableCard>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="text-gray-600 text-lg mb-2">No call graph data found</div>
            <p className="text-gray-600 text-sm">
              No call hierarchy available for this function.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("call-graph-root")!).render(
  <CallGraphWidget />
);
