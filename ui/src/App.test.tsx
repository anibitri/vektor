import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Chunk } from "./api";
import App from "./App";
import AskPage from "./AskPage";
import SearchPage from "./SearchPage";

const chunk = (id: string, title: string, score: number): Chunk => ({
  id: `${id}#0`,
  doc_id: id,
  title,
  text: `${title}. An abstract about ${title.toLowerCase()}.`,
  score,
});

// Answers fetch() like vektor-server would, from a function of (path, request body).
function fakeServer(reply: (path: string, body: Record<string, unknown>) => unknown) {
  const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(init.body as string) : {};
    return new Response(JSON.stringify(reply(path, body)), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SearchPage", () => {
  it("shows results with scores and timings", async () => {
    const fetchMock = fakeServer(() => ({
      results: [chunk("1", "Graph search", 0.812), chunk("2", "Hashing", 0.5)],
      took_ms: 0.05,
      embed_ms: 8,
    }));
    render(<SearchPage />);
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "graphs" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("Graph search")).toBeTruthy();
    expect(screen.getByText("0.812")).toBeTruthy();
    expect(screen.getByText(/search 0\.050 ms · embedding the query 8\.00 ms/)).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith("/rag/search", expect.anything());
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual({
      query: "graphs",
      k: 5,
      ef_search: 40,
      exact: false,
    });
  });

  it("compares with exact search and marks results exact search did not return", async () => {
    fakeServer((_path, body) => ({
      results: body.exact
        ? [chunk("1", "Graph search", 0.9), chunk("3", "Trees", 0.7)]
        : [chunk("1", "Graph search", 0.9), chunk("2", "Hashing", 0.6)],
      took_ms: body.exact ? 0.4 : 0.05,
      embed_ms: 8,
    }));
    render(<SearchPage initialQuery="graphs" initialCompare />);

    expect(await screen.findByText("Exact search (brute force)")).toBeTruthy();
    expect(screen.getByText("HNSW, ef_search 40")).toBeTruthy();
    expect(screen.getByText("not in exact top 5")).toBeTruthy();
    expect(screen.getByText(/1 of 2 match exact search/)).toBeTruthy();
  });

  it("shows the server's error message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ error: "nothing ingested yet" }), { status: 409 })),
    );
    render(<SearchPage initialQuery="graphs" />);
    expect((await screen.findByRole("alert")).textContent).toContain("nothing ingested yet");
  });
});

describe("AskPage", () => {
  it("shows the answer, numbered sources and the time split", async () => {
    fakeServer(() => ({
      answer: "Graphs make search fast [1].",
      sources: [{ ...chunk("1", "Graph search", 0.9), n: 1 }],
      model: "qwen2.5:1.5b",
      timings: { embed_ms: 8, search_ms: 0.05, llm_ms: 1200 },
    }));
    render(<AskPage />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "Why are graphs fast?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Graphs make search fast [1].")).toBeTruthy();
    expect(screen.getByText(/\[1\] Graph search/)).toBeTruthy();
    expect(screen.getByText(/language model 1,200 ms/)).toBeTruthy();
  });
});

describe("AskPage from a link", () => {
  it("asks the question given in the URL", async () => {
    const fetchMock = fakeServer(() => ({
      answer: "Yes [1].",
      sources: [],
      model: "m",
      timings: { embed_ms: 1, search_ms: 1, llm_ms: 1 },
    }));
    render(<AskPage initialQuestion="Is it fast?" />);
    expect(await screen.findByText("Yes [1].")).toBeTruthy();
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual({ question: "Is it fast?", k: 5 });
  });
});

describe("App", () => {
  const stats = (llm: string | null) => () => ({
    rag: { size: 2034 },
    embed_model: "all-minilm",
    llm_model: llm,
  });

  it("offers the Ask page when the server has a language model", async () => {
    fakeServer(stats("qwen2.5:1.5b"));
    render(<App />);
    expect(await screen.findByRole("button", { name: "Ask" })).toBeTruthy();
    expect(screen.getByText(/2,034 chunks/)).toBeTruthy();
  });

  it("hides the Ask page in low-space mode", async () => {
    fakeServer(stats(null));
    render(<App />);
    expect(await screen.findByText(/answers off \(low-space mode\)/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Ask" })).toBeNull();
  });
});
