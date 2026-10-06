import { useEffect, useState, type FormEvent } from "react";
import { ms, searchChunks, snippet, type SearchReply } from "./api";

// ef_search values the slider steps through.
export const EF_STEPS = [10, 20, 40, 80, 160, 320];
const K = 5;

interface Props {
  initialQuery?: string;
  initialEf?: number;
  initialCompare?: boolean;
}

export default function SearchPage({ initialQuery = "", initialEf = 40, initialCompare = false }: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [efIndex, setEfIndex] = useState(Math.max(0, EF_STEPS.indexOf(initialEf)));
  const [compare, setCompare] = useState(initialCompare);
  const [hnsw, setHnsw] = useState<SearchReply | null>(null);
  const [exact, setExact] = useState<SearchReply | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(q: string, ef: number, withExact: boolean) {
    if (!q.trim()) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const [h, e] = await fetchBoth(q, ef, withExact);
      setHnsw(h);
      setExact(e);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // A query in the URL (?q=...) runs once when the page opens.
  useEffect(() => {
    if (!initialQuery) {
      return;
    }
    let current = true; // ignore the reply if the page has gone away meanwhile
    fetchBoth(initialQuery, EF_STEPS[efIndex], compare).then(
      ([h, e]) => {
        if (current) {
          setHnsw(h);
          setExact(e);
        }
      },
      (err: Error) => current && setError(err.message),
    );
    return () => {
      current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only once, for the URL's query
  }, []);

  function submit(event: FormEvent) {
    event.preventDefault();
    void run(query, EF_STEPS[efIndex], compare);
  }

  function changeEf(index: number) {
    setEfIndex(index);
    if (hnsw) {
      void run(query, EF_STEPS[index], compare);
    }
  }

  function changeCompare(value: boolean) {
    setCompare(value);
    if (hnsw) {
      void run(query, EF_STEPS[efIndex], value);
    }
  }

  return (
    <section>
      <form className="query" onSubmit={submit}>
        <input
          aria-label="Search query"
          placeholder="Search the papers, e.g. graph-based nearest neighbour search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={busy}>
          Search
        </button>
      </form>
      <div className="controls">
        <label>
          ef_search
          <input
            type="range"
            min={0}
            max={EF_STEPS.length - 1}
            value={efIndex}
            onChange={(e) => changeEf(Number(e.target.value))}
          />
          <output>{EF_STEPS[efIndex]}</output>
        </label>
        <label>
          <input type="checkbox" checked={compare} onChange={(e) => changeCompare(e.target.checked)} />
          Compare with exact search
        </label>
      </div>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <div className={exact ? "columns" : undefined}>
        {hnsw && (
          <Results title={`HNSW, ef_search ${EF_STEPS[efIndex]}`} reply={hnsw} exact={exact} />
        )}
        {exact && <Results title="Exact search (brute force)" reply={exact} />}
      </div>
    </section>
  );
}

// HNSW results, plus exact-search results if withExact.
function fetchBoth(query: string, ef: number, withExact: boolean) {
  return Promise.all([
    searchChunks(query, K, ef, false),
    withExact ? searchChunks(query, K, ef, true) : Promise.resolve(null),
  ]);
}

function Results({ title, reply, exact }: { title: string; reply: SearchReply; exact?: SearchReply | null }) {
  const exactIds = exact ? new Set(exact.results.map((r) => r.id)) : null;
  const matches = exactIds ? reply.results.filter((r) => exactIds.has(r.id)).length : null;
  return (
    <div className="results">
      <h2>{title}</h2>
      <p className="muted">
        search {ms(reply.took_ms)} · embedding the query {ms(reply.embed_ms)}
        {matches !== null && ` · ${matches} of ${reply.results.length} match exact search`}
      </p>
      <ol>
        {reply.results.map((r) => (
          <li key={r.id}>
            <span className="score" title="cosine similarity">
              {r.score.toFixed(3)}
            </span>
            <strong>{r.title}</strong>
            {exactIds && !exactIds.has(r.id) && <span className="badge">not in exact top {K}</span>}
            <p>{snippet(r)}</p>
            <small className="muted">{r.doc_id}</small>
          </li>
        ))}
      </ol>
    </div>
  );
}
