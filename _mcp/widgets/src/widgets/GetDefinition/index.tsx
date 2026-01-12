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
  const definition = data.data && typeof data.data === 'object' && !Array.isArray(data.data)
    ? data.data as Definition
    : null;

  // Show all fields in a NestedObject for full visibility (remove custom sections)
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
  if (!data || typeof data !== 'object' || (Object.keys(data).length === 0) || !definition) {
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
      <div className="mb-4">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">{definition.name}</h2>
      </div>
      <NestedObject data={definition} />
    </div>
  );
}

createRoot(document.getElementById("get-definition-root")!).render(
  <GetDefinitionWidget />
);