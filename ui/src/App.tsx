import { useEffect, useState } from "react";
import { getStats, type Stats } from "./api";
import AskPage from "./AskPage";
import SearchPage from "./SearchPage";

type Page = "search" | "ask";

export default function App() {
  // ?q=...&ef=40&compare=1 and ?page=ask&q=... open the app in a given state (handy for links).
  const params = new URLSearchParams(window.location.search);
  const [page, setPage] = useState<Page>(params.get("page") === "ask" ? "ask" : "search");
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStats().then(setStats, (err: Error) => setError(err.message));
  }, []);

  const canAsk = stats !== null && stats.llm_model !== null;
  return (
    <main>
      <header>
        <h1>Vektor</h1>
        <nav>
          <button aria-pressed={page === "search"} onClick={() => setPage("search")}>
            Search
          </button>
          {canAsk && (
            <button aria-pressed={page === "ask"} onClick={() => setPage("ask")}>
              Ask
            </button>
          )}
        </nav>
        <p className="muted">
          {stats?.rag ? `${stats.rag.size.toLocaleString("en")} chunks · ` : ""}
          {stats && (canAsk ? `answers by ${stats.llm_model}` : "answers off (low-space mode)")}
        </p>
      </header>
      {error && (
        <p role="alert" className="error">
          Cannot reach vektor-server: {error}
        </p>
      )}
      {stats && !stats.rag && <p>Nothing is ingested yet: run <code>make rag-ingest</code>.</p>}
      {page === "ask" && canAsk ? (
        <AskPage initialQuestion={params.get("q") ?? ""} />
      ) : (
        <SearchPage
          initialQuery={params.get("q") ?? ""}
          initialEf={Number(params.get("ef") ?? 40)}
          initialCompare={params.get("compare") === "1"}
        />
      )}
    </main>
  );
}
