// TypeScript type declarations for entity-utils.jsx
import React from "react";

export function getPrimaryIdentifier(item: Record<string, any>): { field: string; value: any } | null;
export function formatKey(key: string): string;
export function formatValue(value: any): React.ReactNode;
export function NestedObject(props: { data: any; level?: number; maxLevel?: number; excludeKeys?: string[] }): JSX.Element;
export function EntityView(props: { item: any; title?: string; variant?: string }): JSX.Element;
export function ExpandableCard(props: { expanded: boolean; onToggle: () => void; header: JSX.Element; children: React.ReactNode }): JSX.Element;
