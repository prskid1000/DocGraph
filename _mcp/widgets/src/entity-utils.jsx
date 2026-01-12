import React, { useState } from "react";

// Helper function to detect primary identifier field
export function getPrimaryIdentifier(item) {
  if (!item || typeof item !== 'object') return null;
  const identifierFields = ['id', 'name', 'title', 'code', 'key', 'identifier', 'uuid'];
  for (const field of identifierFields) {
    if (item[field] !== undefined && item[field] !== null) {
      return { field, value: item[field] };
    }
  }
  return null;
}

// Helper function to format keys to human-readable format
// Converts camelCase, PascalCase, snake_case, UPPER_CASE, etc. to "Human Readable"
export function formatKey(key) {
  if (!key || typeof key !== 'string') return key;
  
  // Handle snake_case and UPPER_SNAKE_CASE
  if (key.includes('_')) {
    return key
      .split('_')
      .filter(word => word.length > 0)
      .map(word => {
        // If word is all uppercase, keep it as acronym (e.g., "ID", "UOM")
        if (word === word.toUpperCase() && word.length <= 4) {
          return word;
        }
        // Otherwise capitalize first letter, lowercase rest
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      })
      .join(' ');
  }
  
  // Handle camelCase and PascalCase
  // Insert space before uppercase letters (but not at the start)
  const withSpaces = key.replace(/([a-z])([A-Z])/g, '$1 $2');
  
  // Handle sequences of uppercase letters (like "ID" or "UOM")
  const fixedAcronyms = withSpaces.replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2');
  
  // Split and format each word
  return fixedAcronyms
    .split(' ')
    .filter(word => word.length > 0)
    .map(word => {
      // If word is all uppercase and short (likely an acronym), keep it uppercase
      if (word === word.toUpperCase() && word.length <= 4 && word.length > 1) {
        return word;
      }
      // Otherwise capitalize first letter, lowercase rest
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(' ');
}

// Helper function to format values with truncation for long strings
function TruncatedString({ value, maxLength = 100 }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = value.length > maxLength;
  
  if (!isLong || expanded) {
    return (
      <span className="break-words">
        {value}
        {isLong && (
          <button
            onClick={() => setExpanded(false)}
            className="ml-2 text-xs text-teal-600 hover:underline"
          >
            [less]
          </button>
        )}
      </span>
    );
  }
  
  return (
    <span className="break-words">
      {value.substring(0, maxLength)}...
      <button
        onClick={() => setExpanded(true)}
        className="ml-1 text-xs text-teal-600 hover:underline"
      >
        [show more]
      </button>
    </span>
  );
}

// Helper function to format values
export function formatValue(value) {
  if (value === null || value === undefined) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-white text-gray-500 italic border border-teal-200">
        null
      </span>
    );
  }
  if (typeof value === 'boolean') {
    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
        value 
          ? 'bg-green-100 text-green-800 border border-green-200' 
          : 'bg-red-100 text-red-800 border border-red-200'
      }`}>
        {value ? '✓ Yes' : '✗ No'}
      </span>
    );
  }
  if (typeof value === 'number') {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-semibold bg-teal-50 text-teal-700 border border-teal-200">
        {value.toLocaleString()}
      </span>
    );
  }
  if (typeof value === 'string') {
    // Check if it looks like a date
    if (/^\d{4}-\d{2}-\d{2}/.test(value) || /^\d{4}-\d{2}-\d{2}T/.test(value)) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono bg-purple-50 text-purple-700 border border-purple-200">
          📅 {value}
        </span>
      );
    }
    // Check if it's a URL
    if (value.startsWith('http://') || value.startsWith('https://')) {
      return (
        <a 
          href={value} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-teal-50 text-teal-700 border border-teal-200 hover:bg-teal-100 transition-colors break-all"
        >
          🔗 {value.length > 50 ? value.substring(0, 50) + '...' : value}
        </a>
      );
    }
    // Long strings get truncation
    if (value.length > 100) {
      return <TruncatedString value={value} />;
    }
    return <span className="break-words text-gray-900">{value}</span>;
  }
  return value;
}

// Component to render nested objects with enhanced visual styling
export function NestedObject({ data, level = 0, maxLevel = 3 }) {
  const [expanded, setExpanded] = useState(false);

  // Color scheme based on nesting level - using teal/green theme with white backgrounds
  const levelColors = [
    { border: 'border-teal-300', bg: 'bg-white' },
    { border: 'border-emerald-300', bg: 'bg-white' },
    { border: 'border-teal-300', bg: 'bg-white' },
    { border: 'border-emerald-300', bg: 'bg-white' },
  ];
  const colors = levelColors[Math.min(level, levelColors.length - 1)];

  if (level > maxLevel) {
    return (
      <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-white text-gray-500 italic border border-teal-200">
        [Deeply nested object]
      </span>
    );
  }

  if (Array.isArray(data)) {
    if (data.length === 0) {
      return (
        <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-white text-gray-500 italic border border-teal-200">
          [Empty array]
        </span>
      );
    }
    if (data.length <= 3) {
      return (
        <div className="space-y-1.5">
          {data.map((item, idx) => (
            <div 
              key={idx} 
              className={`pl-3 border-l-4 ${colors.border} bg-white py-1 rounded-r hover:bg-teal-50/50 transition-colors`}
            >
              {typeof item === 'object' && item !== null ? (
                <NestedObject data={item} level={level + 1} maxLevel={maxLevel} />
              ) : (
                <div className="text-sm">{formatValue(item)}</div>
              )}
            </div>
          ))}
        </div>
      );
    }
    return (
      <div className="rounded-lg border border-teal-200 overflow-hidden bg-white">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full text-left px-3 py-2 text-sm font-medium text-teal-600 hover:bg-teal-50 transition-colors flex items-center justify-between group"
        >
          <span className="flex items-center gap-2">
            <span className="text-base">{expanded ? '▼' : '▶'}</span>
            <span>Array ({data.length} items)</span>
            <span className="text-xs font-normal text-gray-500">[{typeof data[0]}]</span>
          </span>
        </button>
        {expanded && (
          <div className={`space-y-1.5 p-2 ${colors.bg} border-t border-teal-200`}>
            {data.slice(0, 5).map((item, idx) => (
              <div 
                key={idx}
                className={`pl-3 border-l-2 ${colors.border} py-1.5 rounded-r hover:bg-teal-50/50 transition-colors bg-white`}
              >
                {typeof item === 'object' && item !== null ? (
                  <NestedObject data={item} level={level + 1} maxLevel={maxLevel} />
                ) : (
                  <div className="text-sm">{formatValue(item)}</div>
                )}
              </div>
            ))}
            {data.length > 5 && (
              <div className="text-xs text-gray-500 italic px-3 py-1 bg-white border border-teal-200 rounded">
                ... and {data.length - 5} more items
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  const entries = Object.entries(data || {});
  if (entries.length === 0) {
    return (
      <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-white text-gray-400 italic border border-teal-200">
        [Empty object]
      </span>
    );
  }

  // For nested objects, render fields inline with expandable field names
  return (
    <div className={`space-y-1 bg-white p-2 rounded-lg border ${colors.border}`}>
      {entries.map(([key, value]) => {
        const isNestedObject = typeof value === 'object' && value !== null && !Array.isArray(value);
        const isNestedArray = Array.isArray(value);
        
        if (isNestedObject || isNestedArray) {
          // Use a separate state for each nested object
          return (
            <NestedField 
              key={key} 
              fieldName={formatKey(key)} 
              value={value} 
              level={level} 
              maxLevel={maxLevel}
              colors={colors}
            />
          );
        }
        
        // Simple value - render inline
        return (
          <div 
            key={key}
            className={`pl-3 border-l-2 ${colors.border} py-1.5 rounded-r hover:bg-teal-50/50 transition-colors bg-white`}
          >
            <div className="flex items-start gap-2 text-sm">
              <span className="font-semibold text-gray-700 min-w-[120px] flex-shrink-0">
                {formatKey(key)}:
              </span>
              <div className="flex-1 min-w-0">
                {formatValue(value)}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Component for nested fields with inline expand/collapse
function NestedField({ fieldName, value, level, maxLevel, colors }) {
  const [expanded, setExpanded] = useState(false);
  const isArray = Array.isArray(value);
  const itemCount = isArray ? value.length : Object.keys(value || {}).length;
  
  return (
    <div className="rounded border border-teal-200 overflow-hidden bg-white">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-3 py-1.5 text-sm font-semibold text-gray-700 hover:bg-teal-50 transition-colors flex items-center gap-2"
      >
        <span className="text-base text-teal-600">{expanded ? '▼' : '▶'}</span>
        <span>{fieldName}</span>
        <span className="text-xs font-normal text-gray-500">
          ({itemCount} {isArray ? 'items' : 'fields'})
        </span>
      </button>
      {expanded && (
        <div className="border-t border-teal-200">
          <NestedObject data={value} level={level + 1} maxLevel={maxLevel} />
        </div>
      )}
    </div>
  );
}

// Component to render a single entity (for details, create, update widgets)
export function EntityView({ item, title, variant = 'default' }) {
  const [expanded, setExpanded] = useState(false);
  const primaryId = getPrimaryIdentifier(item);
  
  if (!item || typeof item !== 'object') {
    return (
      <div className="p-4 text-center text-gray-600">
        No data available
      </div>
    );
  }

  // Separate fields into primary (important) and secondary
  const primaryFields = ['id', 'name', 'title', 'code', 'status', 'type', 'description'];
  const allFields = Object.keys(item);
  const primaryEntries = [];
  const secondaryEntries = [];
  
  allFields.forEach(key => {
    const value = item[key];
    // Skip metadata fields that are usually in the wrapper
    if (['success', 'error', 'tenant', 'resource', 'environment', 'resource_id', 'message'].includes(key)) {
      return;
    }
    if (primaryFields.includes(key.toLowerCase())) {
      primaryEntries.push([key, value]);
    } else {
      secondaryEntries.push([key, value]);
    }
  });

  // Sort secondary entries by type (simple values first)
  secondaryEntries.sort((a, b) => {
    const aIsSimple = typeof a[1] !== 'object' || a[1] === null;
    const bIsSimple = typeof b[1] !== 'object' || b[1] === null;
    if (aIsSimple && !bIsSimple) return -1;
    if (!aIsSimple && bIsSimple) return 1;
    return 0;
  });

  // Variant styling - white backgrounds with teal/green borders
  const variantStyles = {
    default: {
      border: 'border-teal-200',
      bg: 'bg-white',
      headerBg: 'bg-white',
    },
    success: {
      border: 'border-emerald-200',
      bg: 'bg-white',
      headerBg: 'bg-white',
    },
    update: {
      border: 'border-teal-200',
      bg: 'bg-white',
      headerBg: 'bg-white',
    },
    delete: {
      border: 'border-red-200', // Keep red for errors
      bg: 'bg-white',
      headerBg: 'bg-white',
    },
  };

  const styles = variantStyles[variant] || variantStyles.default;

  return (
    <div className={`border rounded-lg ${styles.bg} ${styles.border} shadow-sm`}>
      {/* Header */}
      <div className={`p-4 border-b ${styles.border} ${styles.headerBg}`}>
        {title && (
          <h3 className="text-lg font-semibold text-gray-900 mb-3">{title}</h3>
        )}
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            {primaryId ? (
              <div className="flex items-center gap-3">
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  {formatKey(primaryId.field)}
                </span>
                <h4 className="text-xl font-bold text-gray-900 truncate flex-1" title={String(primaryId.value)}>
                  {formatValue(primaryId.value)}
                </h4>
              </div>
            ) : (
               <h4 className="text-xl font-bold text-gray-900">Entity Details</h4>
            )}
            {primaryEntries.length > 1 && (
              <div className="mt-3 flex flex-wrap gap-3">
                {primaryEntries
                  .filter(([key]) => key !== primaryId?.field)
                  .slice(0, 3)
                  .map(([key, value]) => (
                    <div 
                      key={key} 
                      className="px-3 py-1.5 rounded-md bg-white border border-teal-200 text-sm hover:bg-teal-50 transition-colors"
                    >
                      <span className="text-gray-500 font-medium">{formatKey(key)}:</span>{' '}
                      <span className="font-semibold text-gray-700">
                        {typeof value === 'object' && value !== null ? (
                          <span className="text-gray-500 italic text-xs">[Complex]</span>
                        ) : (
                          formatValue(value)
                        )}
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </div>
          {secondaryEntries.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="ml-4 px-4 py-2 text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 rounded-md transition-colors shadow-sm"
            >
              {expanded ? '▲ Less' : '▼ More'}
            </button>
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && secondaryEntries.length > 0 && (
        <div className="p-4 bg-white border-t border-teal-200">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {secondaryEntries.map(([key, value]) => (
              <div 
                key={key} 
                className="min-w-0 p-3 rounded-lg bg-white border border-teal-200 hover:border-teal-300 hover:shadow-sm transition-all"
              >
                <div className="text-xs font-semibold text-gray-600 mb-2 pb-1 border-b border-teal-200">
                  {formatKey(key)}
                </div>
                <div className="text-sm text-gray-700 break-words">
                  {typeof value === 'object' && value !== null ? (
                    <NestedObject data={value} />
                  ) : (
                    formatValue(value)
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Simple ExpandableCard component for collapsible sections
export function ExpandableCard({ expanded, onToggle, header, children }) {
  return (
    <div className="border rounded-lg bg-white shadow-sm">
      <div className="flex items-center justify-between cursor-pointer select-none px-4 py-2 border-b border-teal-200 bg-teal-50 hover:bg-teal-100 transition-colors" onClick={onToggle}>
        <span className="mr-2 text-teal-600">{expanded ? '▼' : '▶'}</span>
        <div className="flex-1 min-w-0">{header}</div>
      </div>
      {expanded && (
        <div className="p-4">{children}</div>
      )}
    </div>
  );
}
