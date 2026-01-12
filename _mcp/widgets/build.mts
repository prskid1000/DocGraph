import { build, type InlineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fg from "fast-glob";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import tailwindcss from "@tailwindcss/vite";

const entries = fg.sync("src/**/index.{tsx,jsx}");
const outDir = "../widgets-assets";

// Widget name mapping: directory name -> widget identifier
const widgetNameMap: Record<string, string> = {
  "SearchEntities": "docgraph-search-entities",
  "GetDefinition": "docgraph-get-definition",
  "FindReferences": "docgraph-find-references",
  "CallGraph": "docgraph-call-graph",
  "CodeContext": "docgraph-code-context",
  "Dependencies": "docgraph-dependencies",
  "TaskResult": "docgraph-task-result",
};

// Root element ID mapping
const rootElementMap: Record<string, string> = {
  "SearchEntities": "search-entities-root",
  "GetDefinition": "get-definition-root",
  "FindReferences": "find-references-root",
  "CallGraph": "call-graph-root",
  "CodeContext": "code-context-root",
  "Dependencies": "dependencies-root",
  "TaskResult": "task-result-root",
};

const PER_ENTRY_CSS_GLOB = "**/*.{css,pcss,scss,sass}";
const PER_ENTRY_CSS_IGNORE = ["**/*.module.*"];
const GLOBAL_CSS_LIST = [path.resolve("src/index.css")];

const builtWidgets: Array<{ name: string; identifier: string; hash: string }> = [];

// Ensure output directory exists
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// Clean output directory
fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });

// Generate hash from timestamp (for cache busting)
const hash = crypto
  .createHash("sha256")
  .update(Date.now().toString(), "utf8")
  .digest("hex")
  .slice(0, 8);

for (const file of entries) {
  const widgetDirName = path.basename(path.dirname(file));
  const identifier = widgetNameMap[widgetDirName];

  if (!identifier) {
    console.warn(`⚠️  Skipping ${widgetDirName} - no identifier mapping found`);
    continue;
  }

  const entryAbs = path.resolve(file);
  const entryDir = path.dirname(entryAbs);
  const rootElementId = rootElementMap[widgetDirName] || `${widgetDirName.toLowerCase()}-root`;

  // Collect CSS for this entry
  const perEntryCss = fg.sync(PER_ENTRY_CSS_GLOB, {
    cwd: entryDir,
    absolute: true,
    dot: false,
    ignore: PER_ENTRY_CSS_IGNORE,
  });

  const globalCss = GLOBAL_CSS_LIST.filter((p) => fs.existsSync(p));
  const cssToInclude = [...globalCss, ...perEntryCss].filter((p) =>
    fs.existsSync(p)
  );

  const virtualId = `\0virtual-entry:${entryAbs}`;

  const wrapEntryPlugin = {
    name: `virtual-entry-wrapper:${entryAbs}`,
    resolveId(id: string) {
      if (id === virtualId) return id;
      return null;
    },
    load(id: string) {
      if (id !== virtualId) {
        return null;
      }

      const cssImports = cssToInclude
        .map((css) => `import ${JSON.stringify(css)};`)
        .join("\n");

      return `
${cssImports}
export * from ${JSON.stringify(entryAbs)};

import * as __entry from ${JSON.stringify(entryAbs)};
export default (__entry.default ?? __entry.App);

import ${JSON.stringify(entryAbs)};
`;
    },
  };

  const buildConfig: InlineConfig = {
    plugins: [
      wrapEntryPlugin,
      tailwindcss(),
      react(),
      {
        name: "remove-manual-chunks",
        outputOptions(options) {
          if ("manualChunks" in options) {
            delete (options as any).manualChunks;
          }
          return options;
        },
      },
    ],
    esbuild: {
      jsx: "automatic",
      jsxImportSource: "react",
      target: "es2022",
    },
    build: {
      target: "es2022",
      outDir,
      emptyOutDir: false,
      chunkSizeWarningLimit: 2000,
      minify: "esbuild",
      cssCodeSplit: false,
      rollupOptions: {
        input: virtualId,
        output: {
          format: "es",
          entryFileNames: `${widgetDirName}.js`,
          inlineDynamicImports: true,
          assetFileNames: (info) =>
            (info.name || "").endsWith(".css")
              ? `${widgetDirName}.css`
              : `[name]-[hash][extname]`,
        },
        preserveEntrySignatures: "allow-extension",
        treeshake: true,
      },
    },
  };

  console.log(`📦 Building ${widgetDirName} (${identifier})...`);
  try {
    await build(buildConfig);
    console.log(`✅ Built ${widgetDirName}`);
    builtWidgets.push({ name: widgetDirName, identifier, hash });
  } catch (error) {
    console.error(`❌ Failed to build ${widgetDirName}:`, error);
    throw error;
  }
}

// Hash the outputs
console.log(`\n🔐 Hashing outputs with hash: ${hash}`);
for (const widget of builtWidgets) {
  const dir = outDir;
  const baseFiles = [
    { ext: ".js", base: widget.name },
    { ext: ".css", base: widget.name },
  ];

  for (const { ext, base } of baseFiles) {
    const oldPath = path.join(dir, `${base}${ext}`);
    const newPath = path.join(dir, `${base}-${hash}${ext}`);

    if (fs.existsSync(oldPath)) {
      fs.renameSync(oldPath, newPath);
      console.log(`   ${oldPath} → ${newPath}`);
    }
  }
}

// Generate HTML files
console.log(`\n📄 Generating HTML files...`);
for (const widget of builtWidgets) {
  const dir = outDir;
  const htmlPath = path.join(dir, `${widget.name}-${hash}.html`);
  const cssPath = path.join(dir, `${widget.name}-${hash}.css`);
  const jsPath = path.join(dir, `${widget.name}-${hash}.js`);
  const rootElementId = rootElementMap[widget.name] || `${widget.name.toLowerCase()}-root`;

  const css = fs.existsSync(cssPath)
    ? fs.readFileSync(cssPath, { encoding: "utf8" })
    : "";
  const js = fs.existsSync(jsPath)
    ? fs.readFileSync(jsPath, { encoding: "utf8" })
    : "";

  // For production, we'd use CDN URLs. For now, use relative paths.
  // The widget loader will replace these with actual URLs
  const cssBlock = css
    ? `\n  <style>\n${css}\n  </style>\n`
    : "";
  const jsBlock = js ? `\n  <script type="module">\n${js}\n  </script>` : "";

  const html = [
    "<!doctype html>",
    "<html>",
    `<head>${cssBlock}</head>`,
    "<body>",
    `  <div id="${rootElementId}"></div>${jsBlock}`,
    "</body>",
    "</html>",
  ].join("\n");

  fs.writeFileSync(htmlPath, html, { encoding: "utf8" });
  console.log(`   ✓ ${htmlPath}`);
}

console.log(`\n✅ Build complete! Generated ${builtWidgets.length} widgets`);
console.log(`📦 Widgets use two-tier lookup (no manifest.json needed)`);
