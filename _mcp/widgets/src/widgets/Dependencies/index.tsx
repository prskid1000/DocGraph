import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard, NestedObject, formatKey, formatValue } from "../../entity-utils";

interface Dependency {
  name: string;
  type: string;
  file_path?: string;
  line_number?: number;
  id?: string;
  [key: string]: any;
}

interface DependenciesResponse {
  success?: boolean;
  data?: Dependency[];
  error?: string;
  isError?: boolean;
  file_path?: string;
  total_results?: number;
  codebase_id?: string;
  [key: string]: any;
}

function DependenciesWidget() {
  const data = useWidgetProps<DependenciesResponse>({ data: [], file_path: "" });
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  // Results rendering
  const hasResults = Array.isArray(data?.data) && data.data.length > 0;
  const results: Dependency[] = Array.isArray(data?.data) ? (data.data as Dependency[]) : [];

  // Extract results from API response structure
  const filePath = data.file_path || "";
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
            Dependencies Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800">An error occurred while getting dependencies.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Dependencies Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Loading state: no data available yet
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || (!data.data && !filePath)) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-16 sm:py-24">
          <div className="relative">
            <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
          </div>
          <p className="mt-6 text-lg sm:text-xl font-semibold text-gray-700">Loading Dependencies...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">
          Dependencies
        </h2>
        {filePath && (
          <p className="text-sm text-gray-700 mb-2">
            File: <span className="font-mono bg-teal-50 px-2 py-1 rounded border border-teal-200 text-gray-900">{filePath}</span>
          </p>
        )}
        {totalResults > 0 && (
          <p className="text-sm text-gray-600 mb-2">
            Found {results.length} dependenc{results.length !== 1 ? 'ies' : 'y'}
          </p>
        )}
      </div>

      {/* Results */}
      {hasResults ? (
        <div className="space-y-4">
          {results.map((dep, idx) => (
            <ExpandableCard
              key={dep.id || idx}
              expanded={!!expanded[dep.id || idx]}
              onToggle={() => setExpanded((prev) => ({ ...prev, [dep.id || idx]: !prev[dep.id || idx] }))}
              header={
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900 truncate" title={dep.name || dep.id}>
                    {dep.name || dep.id}
                  </h3>
                  {dep.file_path && dep.line_number && (
                    <span className="text-sm text-gray-600 font-mono">
                      {dep.file_path}:{dep.line_number}
                    </span>
                  )}
                </div>
              }
            >
              <div className="mt-2">
                <NestedObject data={dep} excludeKeys={["id", "name", "file_path", "line_number"]} />
              </div>
            </ExpandableCard>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-gray-600 text-lg mb-2">No dependencies found</div>
          <p className="text-gray-600 text-sm">No dependencies available.</p>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("dependencies-root")!).render(
  <DependenciesWidget />
);