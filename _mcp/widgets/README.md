# DocGraph Widgets

React widgets for the DocGraph MCP server. Widgets provide interactive UI components for code exploration and visualization.

## Setup

### Automatic (Recommended)

Widgets are built automatically when the MCP server starts. No manual setup required.

### Manual

If you need to install dependencies manually:

**Windows:**
```cmd
setup.bat
```

**Linux/macOS:**
```bash
./setup.sh
```

Or install directly:
```bash
npm install
```

## Building Widgets

### Development Build

```bash
npm run build
```

This compiles all widgets using the custom build script (`build.mts`) and outputs them to `../widgets-assets/`.

### Build Process

The build system:
1. Uses TypeScript build script (`build.mts`) with Vite programmatic API
2. Creates standalone HTML files with inlined CSS and JavaScript
3. Generates hash-based filenames for cache busting (e.g., `SearchEntities-95754870.html`)
4. Includes global styles and per-widget styles automatically
5. Uses Tailwind CSS v4 via `@tailwindcss/vite` plugin

## Available Widgets

- **SearchEntities**: Search for code entities (functions, classes, variables) in the knowledge graph
- **GetDefinition**: Find and display the definition of a code entity
- **FindReferences**: Locate all references to a specific code entity
- **CallGraph**: Visualize function call relationships
- **CodeContext**: Show contextual information about code elements
- **Dependencies**: Display dependency relationships between modules

## Widget Structure

Each widget is in its own directory under `src/widgets/`:

```
src/widgets/
├── SearchEntities/
│   └── index.tsx
├── GetDefinition/
│   └── index.tsx
├── FindReferences/
│   └── index.tsx
├── CallGraph/
│   └── index.tsx
├── CodeContext/
│   └── index.tsx
└── Dependencies/
    └── index.tsx
```

## React Hooks

Widgets have access to custom React hooks for integration with the OpenAI environment:

### `useWidgetProps<T>(defaultState?)`
Access tool output data passed to the widget.

```tsx
const props = useWidgetProps<{ query: string }>();
```

### `useWidgetState<T>(defaultState?)`
Manage widget state with synchronization to the OpenAI environment.

```tsx
const [state, setState] = useWidgetState({ results: [] });
```

### `useOpenAiGlobal<K>(key)`
Access OpenAI global values (theme, displayMode, locale, etc.)

```tsx
const theme = useOpenAiGlobal('theme'); // 'light' | 'dark'
const displayMode = useOpenAiGlobal('displayMode'); // 'pip' | 'inline' | 'fullscreen'
```

## TypeScript Types

All OpenAI integration types are defined in `src/types.ts`:

- `OpenAiGlobals`: Available global values and functions
- `Theme`, `DisplayMode`, `UserAgent`, `SafeArea`: UI context types
- `CallTool`, `RequestDisplayMode`: Function signatures

## Styling

- **Tailwind CSS v4**: Global utility classes
- **Global Styles**: `src/index.css` (imported automatically)
- **Light Mode**: Enforced by default (no dark mode)
- **Widget-Specific**: Additional styles per widget supported

## Output

Built widgets are output to `../widgets-assets/` as:

- `{widget-name}-{hash}.html` - Standalone HTML file
- `{widget-name}-{hash}.js` - JavaScript bundle
- `{widget-name}-{hash}.css` - CSS bundle

The MCP server loads widgets using a two-tier lookup:
1. Exact match: `widget-name-{hash}.html`
2. Glob fallback: `widget-name-*.html` (if hash changes)

## Development

### Adding a New Widget

1. Create directory: `src/widgets/YourWidget/`
2. Create entry point: `src/widgets/YourWidget/index.tsx`
3. Update `build.mts`:
   - Add to `widgetNameMap`: `YourWidget: "docgraph-your-widget"`
   - Add to `rootElementMap`: `YourWidget: "your-widget-root"`
4. Register in Python: `_mcp/service/tools.py`
5. Build: `npm run build`

### Widget Template

```tsx
import React from 'react';
import { useWidgetProps } from '../../use-widget-props';

interface Props {
  // Define your props
}

export default function YourWidget() {
  const props = useWidgetProps<Props>();
  
  return (
    <div className="p-4">
      {/* Your widget UI */}
    </div>
  );
}
```

## Dependencies

- **React 19**: UI framework
- **Vite 7**: Build tool
- **TypeScript**: Type safety
- **Tailwind CSS v4**: Styling
- **tsx**: TypeScript execution
- **fast-glob**: File pattern matching
- **axios**: HTTP client (for widget data fetching)

## Notes

- Widgets run in the OpenAI environment with specific globals (`window.openai`)
- All widgets use the same React version (19.x)
- CSS is automatically collected and inlined
- No separate build configuration per widget needed
- Hash-based filenames ensure proper cache invalidation
