import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../widgets-assets',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        'search-entities': './src/widgets/SearchEntities/index.tsx',
        'get-definition': './src/widgets/GetDefinition/index.tsx',
        'find-references': './src/widgets/FindReferences/index.tsx',
        'call-graph': './src/widgets/CallGraph/index.tsx',
        'code-context': './src/widgets/CodeContext/index.tsx',
        'dependencies': './src/widgets/Dependencies/index.tsx',
        'task-result': './src/widgets/TaskResult/index.tsx',
      },
      output: [
        {
          dir: '../widgets-assets',
          format: 'iife',
          entryFileNames: '[name]-[hash].js',
          assetFileNames: '[name]-[hash][extname]',
        }
      ]
    }
  }
})
