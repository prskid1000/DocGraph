import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard, NestedObject, formatKey, formatValue } from "../../entity-utils";

interface RelatedEntity {
  name: string;
  type: string;
  distance: number;
}

interface ContextData {
  file_path: string;
  line_number: number;
  code: string;
  related_entities?: RelatedEntity[];
  id?: string;
  name?: string;
}

interface CodeContextResponse {
  success?: boolean;
  data?: ContextData | ContextData[];
  error?: string;
  isError?: boolean;
  codebase_id?: string;
  file_path?: string;
  line_number?: number;
  code?: string;
  related_entities?: RelatedEntity[];
  [key: string]: any;
}

function CodeContextWidget() {
  const data = useWidgetProps<CodeContextResponse>({});
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  // Check for isError flag
  const isError = data.isError === true;

  // Check for errors
  const error = data.error;

  // Extract context data
  const context = data.data 
    ? (Array.isArray(data.data) ? data.data[0] : data.data)
    : (data.file_path ? data as unknown as ContextData : null);

  // Check if data has results array
  const hasResults = Array.isArray(data?.data) && data.data.length > 1;
  const results: ContextData[] = Array.isArray(data?.data) ? data.data : [];

  if (isError) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Code Context Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800">An error occurred while getting code context.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Code Context Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Loading state: no data available yet
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || !context) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-16 sm:py-24">
          <div className="relative">
            <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
          </div>
          <p className="mt-6 text-lg sm:text-xl font-semibold text-gray-700">Loading Context...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">Code Context</h2>
        <div className="text-sm text-gray-600 font-mono">
          {context.file_path}:{context.line_number}
        </div>
      </div>

      {/* Code Display */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Source Code</h3>
        <div className="bg-gray-900 rounded border border-gray-700 p-4">
          <pre className="text-sm font-mono text-green-400 overflow-x-auto">{context.code}</pre>
        </div>
      </div>

      {/* Related Entities */}
      {context.related_entities && context.related_entities.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Related Entities ({context.related_entities.length})
          </h3>
          <div className="space-y-2">
            {context.related_entities.map((entity, idx) => (
              <div key={idx} className="border border-teal-200 rounded-lg bg-white shadow-sm p-3">
                <div className="flex items-center gap-3">
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-teal-100 text-teal-800 border border-teal-200">
                    {entity.type}
                  </span>
                  <span className="font-semibold text-gray-900 font-mono text-sm">{entity.name}</span>
                  <span className="text-xs text-gray-600">Distance: {entity.distance}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Multiple Results Section */}
      {hasResults && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Additional Contexts ({results.length})
          </h3>
          {results.map((ctx, idx) => (
            <ExpandableCard
              key={ctx.id || idx}
              expanded={!!expanded[ctx.id || idx]}
              onToggle={() => setExpanded((prev) => ({ ...prev, [ctx.id || idx]: !prev[ctx.id || idx] }))}
              header={
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900 truncate" title={ctx.name || ctx.id}>
                    {ctx.name || ctx.id}
                  </h3>
                  <span className="text-sm text-gray-600 font-mono">
                    {ctx.file_path}:{ctx.line_number}
                  </span>
                </div>
              }
            >
              <div className="mt-2">
                <NestedObject data={ctx} excludeKeys={["id", "name", "file_path", "line_number"]} />
              </div>
            </ExpandableCard>
          ))}
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("code-context-root")!).render(
  <CodeContextWidget />
);