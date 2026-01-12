import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useWidgetProps } from "../../use-widget-props";
import { ExpandableCard, NestedObject, formatKey, formatValue } from "../../entity-utils";

/**
 * Get the appropriate result component for a task type
 * @param {string} taskType - The task type
 * @param {object} result - The task result data
 * @returns {React.Component|null} - The component to render, or null if no specific component
 */
function getTaskResultComponent(taskType: string, result: any) {
  if (!taskType || !result) {
    return null;
  }

  const normalizedType = taskType.toUpperCase();

  switch (normalizedType) {
    case 'INDEX_CODEBASE':
      return <IndexCodebaseResult result={result} />;

    default:
      return null;
  }
}

// Task-specific result components
function IndexCodebaseResult({ result }: { result: any }) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <span className="text-teal-700 font-medium">Files Indexed:</span>
          <span className="ml-2 font-mono text-gray-900">{result.files_indexed || 0}</span>
        </div>
        <div>
          <span className="text-teal-700 font-medium">Entities Found:</span>
          <span className="ml-2 font-mono text-gray-900">{result.entities_found || 0}</span>
        </div>
        <div>
          <span className="text-teal-700 font-medium">References:</span>
          <span className="ml-2 font-mono text-gray-900">{result.references_found || 0}</span>
        </div>
        <div>
          <span className="text-teal-700 font-medium">Duration:</span>
          <span className="ml-2 font-mono text-gray-900">{result.duration_seconds || 0}s</span>
        </div>
      </div>
      {result.languages && result.languages.length > 0 && (
        <div className="mt-3">
          <span className="text-teal-700 font-medium text-sm">Languages:</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {result.languages.map((lang: string, idx: number) => (
              <span key={idx} className="px-2 py-1 bg-teal-100 text-teal-800 rounded text-xs border border-teal-200">
                {lang}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface TaskData {
  success?: boolean;
  isError?: boolean;
  pending?: any[];
  running?: any[];
  completed?: any[];
  summary?: any;
  error?: string | null;
  data?: any;
  [key: string]: any;
}

function TaskResultWidget() {
  const data = useWidgetProps<TaskData>({ 
    success: false, 
    pending: [], 
    running: [], 
    completed: [], 
    summary: {},
    error: null 
  });

  const [cancellingTasks, setCancellingTasks] = useState(new Set<string>());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [expanded, setExpanded] = useState<{ [id: string]: boolean }>({});

  const handleRefresh = async (silent = false) => {
    if (!silent && isRefreshing) {
      return;
    }

    if (!silent) {
      setIsRefreshing(true);
    }

    try {
      if (window.openai?.callTool) {
        const codebaseId = (data && (data.codebase_id || (data.codebaseId as any))) || (window.openai && window.openai.toolOutput && window.openai.toolOutput.codebase_id);
        if (!codebaseId) {
          console.warn("No codebase_id available for task-result refresh");
          if (!silent) alert("Unable to refresh tasks: missing codebase_id");
          return;
        }
        const response: any = await window.openai.callTool("task-result", { codebase_id: codebaseId });
        
        let result;
        // Prefer structuredContent when provided by the server
        if (response.structuredContent) {
          result = response.structuredContent;
        } else if (typeof response.result === 'string') {
          try {
            result = JSON.parse(response.result);
          } catch (e) {
            result = response.result;
          }
        } else {
          result = response.result;
        }

        if (result && typeof result === 'object' && window.dispatchEvent) {
          if (window.openai) {
            window.openai.toolOutput = result;
          }
          
          const event = new CustomEvent("openai:set_globals", {
            detail: {
              globals: {
                toolOutput: result
              }
            }
          });
          window.dispatchEvent(event);
          
          console.log("Updated toolOutput:", result);
        } else {
          console.warn("Invalid result format:", result);
        }
      }
    } catch (error: any) {
      console.error("Error refreshing tasks:", error);
      if (!silent) {
        alert(`Error refreshing tasks: ${error.message || "Unknown error"}`);
      }
    } finally {
      if (!silent) {
        setIsRefreshing(false);
      }
    }
  };

  const success = data.success !== false;
  const pending = data.pending || [];
  const running = data.running || [];
  const completed = data.completed || [];
  const summary = data.summary || {};
  const error = data.error;

  const handleCancelTask = async (taskId: string) => {
    if (cancellingTasks.has(taskId)) {
      return;
    }

    setCancellingTasks(prev => new Set(prev).add(taskId));

    try {
      if (window.openai?.callTool) {
        const codebaseId = (data && (data.codebase_id || (data.codebaseId as any))) || (window.openai && window.openai.toolOutput && window.openai.toolOutput.codebase_id);
        if (!codebaseId) {
          alert("Unable to cancel task: missing codebase_id");
          return;
        }
        const response: any = await window.openai.callTool("cancel-task", {
          codebase_id: codebaseId,
          task_id: taskId
        });
        
        let result;
        if (response.structuredContent) {
          result = response.structuredContent;
        } else if (typeof response.result === 'string') {
          try {
            result = JSON.parse(response.result);
          } catch (e) {
            console.error("Failed to parse cancel response:", e, response);
            alert(`Failed to cancel task: Invalid response`);
            return;
          }
        } else {
          result = response.result;
        }

        if (result && result.success === false) {
          alert(`Failed to cancel task: ${result.error || "Unknown error"}`);
          return;
        }

        if (result && typeof result === 'object' && window.dispatchEvent) {
          if (window.openai) {
            window.openai.toolOutput = result;
          }
          
          const event = new CustomEvent("openai:set_globals", {
            detail: {
              globals: {
                toolOutput: result
              }
            }
          });
          window.dispatchEvent(event);
          
          console.log("Updated toolOutput after cancel:", result);
        }
      } else {
        alert("Unable to cancel task: Tool calling not available");
      }
    } catch (error: any) {
      console.error("Error cancelling task:", error);
      alert(`Error cancelling task: ${error.message || "Unknown error"}`);
    } finally {
      setCancellingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskId);
        return next;
      });
    }
  };

  const getStatusBadge = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      running: 'bg-blue-100 text-blue-800 border-blue-200',
      completed: 'bg-green-100 text-green-800 border-green-200',
      failed: 'bg-red-100 text-red-800 border-red-200',
      cancelled: 'bg-gray-100 text-gray-800 border-gray-200',
    };
    
    const classes = statusMap[status?.toLowerCase()] || 'bg-gray-100 text-gray-800 border-gray-200';
    return (
      <span className={`px-2 py-1 rounded text-xs font-medium border ${classes}`}>
        {status?.toUpperCase() || 'UNKNOWN'}
      </span>
    );
  };

  const toggleTask = (taskId: string) => {
    setExpanded(prev => ({ ...prev, [taskId]: !prev[taskId] }));
  };

  const isTaskExpanded = (taskId: string) => expanded[taskId];

  const isError = data.isError === true;
  if (isError) {
    return (
      <div className="p-2 sm:p-4 max-w-6xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Task Result Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">An error occurred</p>
        </div>
      </div>
    );
  }

  const errorMessage = error || (data.data && typeof data.data === 'object' && data.data.error);
  if (errorMessage) {
    return (
      <div className="p-2 sm:p-4 max-w-6xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-xl sm:text-2xl font-bold text-red-600 mb-2">
            Task Result Error
          </h2>
        </div>
        <div className="bg-red-50 rounded-lg border border-red-200 p-4">
          <p className="text-red-800 font-mono text-sm">{errorMessage}</p>
        </div>
      </div>
    );
  }

  if (!success) {
    return (
      <div className="p-2 sm:p-4 max-w-6xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
        <div className="flex items-center justify-center p-8">
          <div className="text-center">
            <svg className="animate-spin h-8 w-8 text-teal-600 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="text-gray-600 text-lg">Loading task results...</p>
          </div>
        </div>
      </div>
    );
  }

  // Results rendering
  const hasResults = Array.isArray(data?.data) && data.data.length > 0;
  const results = hasResults ? data.data : [];

  return (
    <div className="p-2 sm:p-4 max-w-7xl mx-auto bg-white max-h-[500px] overflow-x-auto overflow-y-auto">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-4 mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900">Code Analysis Tasks</h2>
        <button
          onClick={() => handleRefresh()}
          disabled={isRefreshing}
          className="px-2 sm:px-3 py-1 sm:py-1.5 text-xs sm:text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 disabled:bg-gray-400 disabled:cursor-not-allowed rounded border border-teal-700 transition-colors flex items-center gap-1 sm:gap-2 w-full sm:w-auto justify-center"
          title="Refresh task list"
        >
          <svg 
            className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {/* Summary */}
      <div className="bg-teal-50 rounded-lg border border-teal-200 p-3 sm:p-4 mb-4 sm:mb-6">
        <div className="grid grid-cols-3 gap-2 sm:gap-4 text-center">
          <div>
            <p className="text-sm text-teal-600 mb-1">Pending</p>
            <p className="text-2xl font-bold text-teal-900">{summary.pending_count || 0}</p>
          </div>
          <div>
            <p className="text-sm text-blue-600 mb-1">Running</p>
            <p className="text-2xl font-bold text-blue-900">{summary.running_count || 0}</p>
          </div>
          <div>
            <p className="text-sm text-green-600 mb-1">Completed</p>
            <p className="text-2xl font-bold text-green-900">{summary.completed_count || 0}</p>
          </div>
        </div>
      </div>

      {/* Running Tasks Section */}
      {running.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center gap-2">
            <span>Running Tasks</span>
            <span className="text-sm font-normal text-blue-600">({running.length})</span>
          </h3>
          <div className="space-y-3">
            {running.map((task: any) => {
              const isExpanded = isTaskExpanded(task.task_id);
              return (
                <div key={task.task_id} className="bg-blue-50 rounded-lg border border-blue-200 overflow-hidden">
                  <div 
                    className="p-4 cursor-pointer hover:bg-blue-100 transition-colors"
                    onClick={() => toggleTask(task.task_id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 flex items-center gap-3">
                        <button
                          className="text-gray-500 hover:text-gray-700 transition-transform"
                          style={{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </button>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <p className="font-mono text-sm font-semibold text-gray-900">{task.task_id}</p>
                            {getStatusBadge(task.status)}
                            <p className="text-sm text-gray-600 font-mono">{task.task_type}</p>
                          </div>
                          {task.started_at && (
                            <p className="text-xs text-gray-500">Started: {new Date(task.started_at).toLocaleString()}</p>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCancelTask(task.task_id);
                        }}
                        disabled={cancellingTasks.has(task.task_id)}
                        className="ml-4 px-3 py-1.5 text-xs font-medium text-white bg-red-600 hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed rounded border border-red-700 transition-colors"
                        title="Cancel this task"
                      >
                        {cancellingTasks.has(task.task_id) ? "Cancelling..." : "Cancel"}
                      </button>
                    </div>
                  </div>
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-0 border-t border-blue-200">
                      {task.progress !== undefined && task.progress !== null && (
                        <div className="mt-3">
                          <div className="flex justify-between text-xs text-gray-600 mb-1">
                            <span>Progress</span>
                            <span>{Math.round(task.progress * 100)}%</span>
                          </div>
                          <div className="w-full bg-blue-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                              style={{ width: `${Math.min(task.progress * 100, 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                      {task.error && (
                        <div className="mt-2 text-xs text-red-600">{task.error}</div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Pending Tasks Section */}
      {pending.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-yellow-900 mb-3 flex items-center gap-2">
            <span>Pending Tasks</span>
            <span className="text-sm font-normal text-yellow-600">({pending.length})</span>
          </h3>
          <div className="space-y-3">
            {pending.map((task: any) => {
              const isExpanded = isTaskExpanded(task.task_id);
              return (
                <div key={task.task_id} className="bg-yellow-50 rounded-lg border border-yellow-200 overflow-hidden">
                  <div 
                    className="p-4 cursor-pointer hover:bg-yellow-100 transition-colors"
                    onClick={() => toggleTask(task.task_id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 flex items-center gap-3">
                        <button
                          className="text-gray-500 hover:text-gray-700 transition-transform"
                          style={{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </button>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <p className="font-mono text-sm font-semibold text-gray-900">{task.task_id}</p>
                            {getStatusBadge(task.status)}
                            <p className="text-sm text-gray-600 font-mono">{task.task_type}</p>
                          </div>
                          {task.created_at && (
                            <p className="text-xs text-gray-500">Created: {new Date(task.created_at).toLocaleString()}</p>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCancelTask(task.task_id);
                        }}
                        disabled={cancellingTasks.has(task.task_id)}
                        className="ml-4 px-3 py-1.5 text-xs font-medium text-white bg-red-600 hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed rounded border border-red-700 transition-colors"
                        title="Cancel this task"
                      >
                        {cancellingTasks.has(task.task_id) ? "Cancelling..." : "Cancel"}
                      </button>
                    </div>
                  </div>
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-0 border-t border-yellow-200">
                      {task.error && (
                        <div className="mt-2 text-xs text-red-600">{task.error}</div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Completed Tasks Section */}
      {completed.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-green-900 mb-3 flex items-center gap-2">
            <span>Completed Tasks</span>
            <span className="text-sm font-normal text-green-600">({completed.length})</span>
          </h3>
          <div className="space-y-4">
            {completed.map((task: any) => {
              const isExpanded = isTaskExpanded(task.task_id);
              return (
                <div key={task.task_id} className="bg-green-50 rounded-lg border border-green-200 overflow-hidden">
                  <div 
                    className="p-4 cursor-pointer hover:bg-green-100 transition-colors"
                    onClick={() => toggleTask(task.task_id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 flex items-center gap-3">
                        <button
                          className="text-gray-500 hover:text-gray-700 transition-transform"
                          style={{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </button>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <p className="font-mono text-sm font-semibold text-gray-900">{task.task_id}</p>
                            {getStatusBadge(task.status)}
                            <p className="text-sm text-gray-600 font-mono">{task.task_type}</p>
                          </div>
                          <div className="flex gap-4 text-xs text-gray-500">
                            {task.created_at && (
                              <span>Created: {new Date(task.created_at).toLocaleString()}</span>
                            )}
                            {task.completed_at && (
                              <span>Completed: {new Date(task.completed_at).toLocaleString()}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-0 border-t border-green-200">
                      {task.error && (
                        <div className="bg-red-50 rounded border border-red-200 p-3 mb-3">
                          <h4 className="text-xs font-semibold text-red-800 mb-1">Error</h4>
                          <p className="text-red-700 text-xs">{task.error}</p>
                        </div>
                      )}

                      {task.result && (
                        <div className="bg-white rounded border border-green-200 p-3">
                          <h4 className="text-xs font-semibold text-green-800 mb-2 flex items-center gap-2">
                            <span>Result</span>
                            <span className="text-xs font-normal text-green-600">({task.task_type})</span>
                          </h4>
                          {(() => {
                            const taskResultComponent = getTaskResultComponent(task.task_type, task.result);
                            if (taskResultComponent) {
                              return taskResultComponent;
                            } else {
                              return (
                                <pre className="bg-gray-50 rounded border border-gray-200 p-2 text-xs overflow-x-auto">
                                  {/* Use NestedObject for better display if result is object, else fallback to JSON */}
                                  {typeof task.result === 'object' && task.result !== null ? (
                                    <NestedObject data={task.result} />
                                  ) : (
                                    formatValue(task.result)
                                  )}
                                </pre>
                              );
                            }
                          })()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty State */}
      {pending.length === 0 && running.length === 0 && completed.length === 0 && (
        <div className="bg-gray-50 rounded-lg border border-gray-200 p-8 text-center">
          <p className="text-gray-600">No tasks found. Submit a code analysis task to get started.</p>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("task-result-root")!).render(
  <TaskResultWidget />
);
