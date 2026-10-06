import { useEffect, useState, type FormEvent } from "react";
import { ask, ms, type AskReply } from "./api";

const K = 5;

export default function AskPage({ initialQuestion = "" }: { initialQuestion?: string }) {
  const [question, setQuestion] = useState(initialQuestion);
  const [reply, setReply] = useState<AskReply | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // A question in the URL (?page=ask&q=...) is asked once when the page opens.
  useEffect(() => {
    if (!initialQuestion) {
      return;
    }
    let current = true; // ignore the reply if the page has gone away meanwhile
    ask(initialQuestion, K).then(
      (r) => current && setReply(r),
      (err: Error) => current && setError(err.message),
    );
    return () => {
      current = false;
    };
  }, [initialQuestion]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setReply(await ask(question, K));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <form className="query" onSubmit={submit}>
        <input
          aria-label="Question"
          placeholder="Ask a question about the papers"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit" disabled={busy}>
          Ask
        </button>
      </form>
      {busy && <p className="muted">Thinking… the language model takes a few seconds.</p>}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {reply && !busy && (
        <>
          <p className="answer">{reply.answer}</p>
          <Timings timings={reply.timings} />
          <h2>Sources</h2>
          <ol className="sources">
            {reply.sources.map((s) => (
              <li key={s.id}>
                <details>
                  <summary>
                    [{s.n}] {s.title} <span className="score">{s.score.toFixed(3)}</span>
                  </summary>
                  <p>{s.text}</p>
                  <small className="muted">{s.doc_id}</small>
                </details>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

// Where the time went: embedding the question, the vector search, and the language model.
function Timings({ timings }: { timings: AskReply["timings"] }) {
  const parts = [
    { name: "embedding", value: timings.embed_ms, className: "embed" },
    { name: "search", value: timings.search_ms, className: "search" },
    { name: "language model", value: timings.llm_ms, className: "llm" },
  ];
  const total = parts.reduce((sum, p) => sum + p.value, 0) || 1;
  return (
    <div className="timings">
      <div className="bar" aria-hidden="true">
        {parts.map((p) => (
          <span key={p.name} className={p.className} style={{ width: `${(100 * p.value) / total}%` }} />
        ))}
      </div>
      <p className="muted">
        {parts.map((p) => `${p.name} ${ms(p.value)}`).join(" · ")}
      </p>
    </div>
  );
}
