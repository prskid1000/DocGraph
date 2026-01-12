import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard, NestedObject, formatKey, formatValue } from "../../entity-utils";

interface SearchEntity {
  id: string;
  name: string;
  type: string;
  file_path: string;
  line_number: number;
  description?: string;
  score?: number;
  metadata?: {
    entity_type?: string;
    name?: string;
    file_path?: string;
    start_line?: number;
  };
  [key: string]: any;
}

interface SearchEntitiesResponse {
  success?: boolean;
  data?: SearchEntity[];
  error?: string;
  isError?: boolean;
  query?: string;
  entity_type?: string;
  total_results?: number;
  codebase_id?: string;
  [key: string]: any;
}

function SearchEntitiesWidget() {
  const data = useWidgetProps<SearchEntitiesResponse>({ data: [], query: "", entity_type: "" });
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  // Results rendering
  const hasResults = Array.isArray(data?.data) && data.data.length > 0;
  const results: SearchEntity[] = Array.isArray(data?.data) ? (data.data as SearchEntity[]) : [];

  const query = data.query || "";
  const entityType = data.entity_type || "";
  const totalResults = data.total_results || results.length;
  const success = data.success !== false;

  // Check for isError flag (highest priority - must be checked first)
  const isError = data.isError === true;

  // Check for errors in multiple locations (must be checked before loading state)
  const error = data.error || 
                (data.data && typeof data.data === 'object' && !Array.isArray(data.data) && (data.data as any).error);

  // Check for isError flag - must be checked before other error checks
  if (isError) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Search Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800">An error occurred while searching entities.</p>
        </div>
      </div>
    );
  }

  // Check for errors - must be checked before loading state
  if (error) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Search Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Loading state: no data available yet
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || (!data.data && !query)) {
    return (
      <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-16 sm:py-24">
          <div className="relative">
            <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin"></div>
          </div>
          <p className="mt-6 text-lg sm:text-xl font-semibold text-gray-700">Searching...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">
          Search Results {entityType && <span className="text-gray-600">for {entityType}</span>}
        </h2>
        {query && (
          <p className="text-sm text-gray-700 mb-2">
            Query: <span className="font-mono bg-teal-50 px-2 py-1 rounded border border-teal-200 text-gray-900">{query}</span>
          </p>
        )}
        {totalResults > 0 && (
          <p className="text-sm text-gray-600">
            Showing {results.length} result{results.length !== 1 ? 's' : ''}
          </p>
        )}
      </div>

      {/* Results */}
      {hasResults ? (
        <div className="space-y-4">
          {results.map((entity, idx) => (
            <ExpandableCard
              key={entity.id || idx}
              expanded={!!expanded[entity.id || idx]}
              onToggle={() => setExpanded((prev) => ({ ...prev, [entity.id || idx]: !prev[entity.id || idx] }))}
              header={
                <div className="flex items-center gap-3 mb-2">
                  <span className="px-2 py-1 rounded text-xs font-medium bg-teal-100 text-teal-800 border border-teal-200">
                    {entity.metadata?.entity_type || <span className="text-gray-400">(no type)</span>}
                  </span>
                  <h3 className="text-lg font-semibold text-gray-900 truncate" title={entity.metadata?.name}>
                    {entity.metadata?.name || <span className="text-gray-400">(no name)</span>}
                  </h3>
                  {entity.distance !== undefined && (
                    <span className="text-sm text-gray-600">
                      {(entity.distance * 100).toFixed(0)}% match
                    </span>
                  )}
                </div>
              }
            >
              <div className="mt-2">
                {entity.description && (
                  <p className="text-sm text-gray-700 mb-2">{entity.description}</p>
                )}
                <NestedObject data={entity} excludeKeys={["id", "metadata", "description", "score"]} />
              </div>
            </ExpandableCard>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-gray-600 text-lg mb-2">No results found</div>
          <p className="text-gray-600 text-sm">
            {query ? `No entities match your search query "${query}"` : "No entities available"}
          </p>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("search-entities-root")!).render(
  <SearchEntitiesWidget />
);