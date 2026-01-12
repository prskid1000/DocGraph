import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard, NestedObject, formatKey, formatValue } from "../../entity-utils";

interface Reference {
  id: string;
  entity_name: string;
  file_path: string;
  line_number: number;
  context?: string;
  type?: string;
  name?: string;
  [key: string]: any;
}

interface FindReferencesResponse {
  success?: boolean;
  data?: Reference[];
  error?: string;
  isError?: boolean;
  entity_name?: string;
  total_results?: number;
  codebase_id?: string;
  [key: string]: any;
}

function FindReferencesWidget() {
  const data = useWidgetProps<FindReferencesResponse>({ data: [], entity_name: "" });
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  // Extract results from API response structure
  const hasResults = Array.isArray(data?.data) && data.data.length > 0;
  const results: Reference[] = Array.isArray(data?.data) ? (data.data as Reference[]) : [];
  const entityName = data.entity_name || "";
  const totalResults = data.total_results || results.length;
  const success = data.success !== false;

  // Check for isError flag (highest priority)
  const isError = data.isError === true;

  // Check for errors
  const error = data.error || 
                (data.data && typeof data.data === 'object' && !Array.isArray(data.data) && (data.data as any).error);

  if (isError) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Find References Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800">An error occurred while finding references.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Find References Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Loading state: no data available yet
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || (!data.data && !entityName)) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-16 sm:py-24">
          <div className="relative">
            <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
          </div>
          <p className="mt-6 text-lg sm:text-xl font-semibold text-gray-700">Finding References...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">
          References {entityName && <span className="text-gray-600">for {entityName}</span>}
        </h2>
        {totalResults > 0 && (
          <p className="text-sm text-gray-600 mb-2">
            Found {results.length} reference{results.length !== 1 ? 's' : ''}
          </p>
        )}
      </div>

      {/* Results */}
      {hasResults ? (
        <div className="space-y-4">
          {results.map((ref, idx) => (
            <ExpandableCard
              key={ref.id || idx}
              expanded={!!expanded[ref.id || idx]}
              onToggle={() => setExpanded((prev) => ({ ...prev, [ref.id || idx]: !prev[ref.id || idx] }))}
              header={
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900 truncate" title={ref.name || ref.entity_name || ref.id}>
                    {ref.name || ref.entity_name || ref.id}
                  </h3>
                  <span className="text-sm text-gray-600 font-mono">
                    {ref.file_path}:{ref.line_number}
                  </span>
                </div>
              }
            >
              <div className="mt-2">
                <NestedObject data={ref} excludeKeys={["id", "name", "entity_name", "file_path", "line_number"]} />
              </div>
            </ExpandableCard>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-gray-600 text-lg mb-2">No references found</div>
          <p className="text-gray-600 text-sm">No references available.</p>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("find-references-root")!).render(
  <FindReferencesWidget />
);