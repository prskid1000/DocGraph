import React from 'react'
import ReactDOM from 'react-dom/client'

// Import all widgets
import SearchEntities from './widgets/SearchEntities'
import GetDefinition from './widgets/GetDefinition'
import FindReferences from './widgets/FindReferences'
import CallGraph from './widgets/CallGraph'
import CodeContext from './widgets/CodeContext'
import Dependencies from './widgets/Dependencies'

// Register widgets globally
(window as any).DocGraphWidgets = {
  SearchEntities,
  GetDefinition,
  FindReferences,
  CallGraph,
  CodeContext,
  Dependencies,
}

export { SearchEntities, GetDefinition, FindReferences, CallGraph, CodeContext, Dependencies }
