import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard, NestedObject, formatKey, formatValue } from "../../entity-utils";

interface Definition {
  name: string;
  type: string;
  file_path: string;
  line_number: number;
  line_end?: number;
  source?: string;
  docstring?: string;
  signature?: string;
  parameters?: Record<string, unknown>;
  id?: string;
  [key: string]: any;
}

interface GetDefinitionResponse {
  success?: boolean;
  data?: Definition | Definition[];
  error?: string;
  isError?: boolean;
  entity_name?: string;
  codebase_id?: string;
  name?: string;
  [key: string]: any;
}

function GetDefinitionWidget() {
  const data = useWidgetProps<GetDefinitionResponse>({});
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  // Check for isError flag (highest priority)
  const isError = data.isError === true;

  // Check for errors
  const error = data.error;

  // Check if we have definition data
  const definition = data.data && !Array.isArray(data.data)
    ? data.data
    : (data.name ? data as unknown as Definition : null);

  // Results rendering
  const hasResults = Array.isArray(data?.data) && data.data.length > 0;
  const results: Definition[] = hasResults ? data.data as Definition[] : [];

  if (isError) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Definition Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800">An error occurred while getting definition.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Definition Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Loading state: no data available yet
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || (!definition && !hasResults)) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-16 sm:py-24">
          <div className="relative">
            <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
          </div>
          <p className="mt-6 text-lg sm:text-xl font-semibold text-gray-700">Loading Definition...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      {/* Single Definition View */}
      {definition && (
        <>
          {/* Header */}
          <div className="mb-4 sm:mb-6">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-xl sm:text-2xl font-bold text-gray-900">{definition.name}</h2>
              <span className="px-2 py-1 rounded text-xs font-medium bg-teal-100 text-teal-800 border border-teal-200">
                {definition.type}
              </span>
            </div>
            <div className="text-sm text-gray-600 font-mono">
              {definition.file_path}:{definition.line_number}
              {definition.line_end && <span>-{definition.line_end}</span>}
            </div>
          </div>

          {/* Signature */}
          {definition.signature && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Signature</h3>
              <div className="bg-gray-50 rounded border border-gray-200 p-3">
                <pre className="text-sm font-mono text-gray-900 overflow-x-auto">{definition.signature}</pre>
              </div>
            </div>
          )}

          {/* Documentation */}
          {definition.docstring && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Documentation</h3>
              <div className="bg-blue-50 rounded border border-blue-200 p-3">
                <pre className="text-sm text-gray-800 whitespace-pre-wrap">{definition.docstring}</pre>
              </div>
            </div>
          )}

          {/* Parameters */}
          {definition.parameters && Object.keys(definition.parameters).length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Parameters</h3>
              <div className="bg-white rounded border border-gray-200">
                {Object.entries(definition.parameters).map(([key, value], idx) => (
                  <div key={idx} className="p-2 border-b border-gray-100 last:border-b-0">
                    <span className="font-mono text-sm text-teal-700 font-semibold">{formatKey(key)}</span>
                    <span className="text-sm text-gray-600 ml-2">
                      {typeof value === 'object' ? <NestedObject data={value} /> : formatValue(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Source Code */}
          {definition.source && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Source Code</h3>
              <div className="bg-gray-900 rounded border border-gray-700 p-3">
                <pre className="text-sm font-mono text-green-400 overflow-x-auto">{definition.source}</pre>
              </div>
            </div>
          )}
        </>
      )}

      {/* Multiple Results View */}
      {hasResults && (
        <div className="space-y-4">
          {!definition && (
            <div className="mb-4">
              <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">Definitions</h2>
              <p className="text-sm text-gray-600">Found {results.length} definition{results.length !== 1 ? 's' : ''}</p>
            </div>
          )}
          {results.map((def, idx) => (
            <ExpandableCard
              key={def.id || idx}
              expanded={!!expanded[def.id || idx]}
              onToggle={() => setExpanded((prev) => ({ ...prev, [def.id || idx]: !prev[def.id || idx] }))}
              header={
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900 truncate" title={def.name || def.id}>
                    {def.name || def.id}
                  </h3>
                  <span className="text-sm text-gray-600 font-mono">
                    {def.file_path}:{def.line_number}
                  </span>
                </div>
              }
            >
              <div className="mt-2">
                <NestedObject data={def} excludeKeys={["id", "name", "file_path", "line_number"]} />
              </div>
            </ExpandableCard>
          ))}
        </div>
      )}

      {/* No Results */}
      {!definition && !hasResults && (
        <div className="text-center py-12">
          <div className="text-gray-600 text-lg mb-2">No definitions found</div>
          <p className="text-gray-600 text-sm">No definitions available.</p>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("get-definition-root")!).render(
  <GetDefinitionWidget />
);