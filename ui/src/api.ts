// Calls to vektor-server's JSON API (see the README's REST API section).

export interface Stats {
  rag: { size: number } | null;
  embed_model: string;
  llm_model: string | null; // null in low-space mode: /rag/ask is turned off
}

export interface Chunk {
  id: string; // "<document id>#<chunk number>"
  doc_id: string;
  title: string;
  text: string;
  score: number; // cosine similarity, higher is closer
}

export interface SearchReply {
  results: Chunk[];
  took_ms: number;
  embed_ms: number;
}

export interface AskReply {
  answer: string;
  sources: (Chunk & { n: number })[];
  model: string;
  timings: { embed_ms: number; search_ms: number; llm_ms: number };
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ?? `HTTP ${response.status}`);
  }
  return data as T;
}

export const getStats = () => call<Stats>("/stats");

export const searchChunks = (query: string, k: number, efSearch: number, exact: boolean) =>
  call<SearchReply>("/rag/search", { query, k, ef_search: efSearch, exact });

export const ask = (question: string, k: number) =>
  call<AskReply>("/rag/ask", { question, k });

// The chunk text without the title it starts with, cut to about `words` words.
export function snippet(chunk: Chunk, words = 45): string {
  const text = chunk.text.startsWith(chunk.title)
    ? chunk.text.slice(chunk.title.length).replace(/^[.\s]+/, "")
    : chunk.text;
  const parts = text.split(/\s+/);
  return parts.length > words ? parts.slice(0, words).join(" ") + " …" : text;
}

export function ms(value: number): string {
  if (value < 1) {
    return `${value.toFixed(3)} ms`;
  }
  return value < 10 ? `${value.toFixed(2)} ms` : `${Math.round(value).toLocaleString("en")} ms`;
}
